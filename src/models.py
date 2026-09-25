import os
import csv
import time
import torch
import matplotlib.pyplot as plt
from typing import Callable, Dict, List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
from .alignment import (
    AlignmentState,
    apply_alignment,
    apply_soft_alignment_with_entropy,
    build_kernel_state,
    compute_logits_entropy,
    build_linear_state,
    build_soft_state,
)

try:
    from vllm import LLM, SamplingParams
    _HAS_VLLM = True
except ImportError:
    _HAS_VLLM = False

try:
    from transformers.cache_utils import Cache
except ImportError:
    Cache = None


SOFT_LATENT_MAX_STEPS = 10000
KERNEL_EARLY_STOPPING_MAX_STEPS = 200
KERNEL_ENTROPY_CHECK_INTERVAL = 10
KERNEL_STABLE_CHANGE_THRESHOLD = 0.1
KERNEL_STABLE_CHANGE_COUNT = 4


def latent_vocab_decode_steps(align_method: str, actual_steps: int) -> int:
    """Count latent steps that evaluate the full vocabulary output head.

    Soft alignment computes ``softmax(h @ W_out.T)`` on every latent step.
    Kernel early stopping computes full-vocabulary logits only at its entropy
    check interval; its other latent steps use the kernel approximation.
    """
    if align_method in ("soft", "text"):
        return actual_steps
    if align_method == "kernel_early_stopping":
        return actual_steps // KERNEL_ENTROPY_CHECK_INTERVAL
    return 0


def _ensure_pad_token(tokenizer: AutoTokenizer) -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
    # Decoder-only batch generation and latent rollout read the final sequence
    # position, so every row must end at its final non-padding token.
    tokenizer.padding_side = "left"


def _past_length(past_key_values: Optional[object]) -> int:
    if past_key_values is None:
        return 0
    if Cache is not None and isinstance(past_key_values, Cache):
        return int(past_key_values.get_seq_length())
    if not past_key_values:
        return 0
    k = past_key_values[0][0]
    return k.shape[-2]


def _sync_cuda(device) -> None:
    """Synchronize only the CUDA device used by the measured model stage."""
    resolved = torch.device(device)
    if resolved.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(resolved)


class _FirstTokenTimer:
    """Transformers logits processor used to mark the end of prefill."""

    def __init__(self, device, started_at: float) -> None:
        self.device = device
        self.started_at = started_at
        self.first_token_at: Optional[float] = None

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if self.first_token_at is None:
            _sync_cuda(self.device)
            self.first_token_at = time.perf_counter()
        return scores


def _completion_token_ids(
    sequences: torch.Tensor,
    prompt_width: int,
    eos_token_id,
) -> List[List[int]]:
    """Return generated IDs, including the first EOS and excluding later padding."""
    eos_ids = (
        {int(eos_token_id)}
        if isinstance(eos_token_id, int)
        else {int(value) for value in (eos_token_id or [])}
    )
    completions: List[List[int]] = []
    for row in sequences[:, prompt_width:].tolist():
        trimmed: List[int] = []
        for token_id in row:
            trimmed.append(int(token_id))
            if int(token_id) in eos_ids:
                break
        completions.append(trimmed)
    return completions


def _vllm_phase_split(outputs, wall_seconds: float) -> Tuple[float, float]:
    """Split vLLM wall time using request TTFT/decode ratios when available."""
    prefill = 0.0
    decode = 0.0
    for output in outputs:
        metrics = getattr(output, "metrics", None)
        arrival = getattr(metrics, "arrival_time", None)
        first = getattr(metrics, "first_token_time", None)
        finished = getattr(metrics, "finished_time", None)
        if arrival is not None and first is not None and finished is not None:
            prefill += max(0.0, float(first) - float(arrival))
            decode += max(0.0, float(finished) - float(first))
    measured = prefill + decode
    if measured <= 0:
        return wall_seconds, 0.0
    prefill_seconds = wall_seconds * prefill / measured
    return prefill_seconds, max(0.0, wall_seconds - prefill_seconds)


class _AlignmentTimer:
    """Accumulate alignment time without synchronizing after every latent step."""

    def __init__(self, device) -> None:
        self.device = torch.device(device)
        self.cpu_seconds = 0.0
        self.events = []

    def measure(self, callback):
        if self.device.type == "cuda" and torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            value = callback()
            end.record()
            self.events.append((start, end))
            return value
        started_at = time.perf_counter()
        value = callback()
        self.cpu_seconds += time.perf_counter() - started_at
        return value

    def seconds(self) -> float:
        gpu_seconds = sum(start.elapsed_time(end) for start, end in self.events) / 1000.0
        return self.cpu_seconds + gpu_seconds


class ModelWrapper:
    def __init__(self, model_name: str, device: torch.device, use_vllm: bool = False, args = None):
        self.model_name = model_name
        self.device = device
        self.use_vllm = use_vllm and _HAS_VLLM
        self.vllm_engine = None
        self.trust_remote_code = bool(getattr(args, "trust_remote_code", False)) if args else False
        self.model_revision = getattr(args, "model_revision", None) if args else None
        revision_kwargs = {"revision": self.model_revision} if self.model_revision else {}
        legacy_realign = bool(getattr(args, "latent_space_realign", False)) if args else False
        self.align_method = getattr(args, "align_method", None) or ("linear" if legacy_realign else "identical")
        self._alignment_states: Dict[Tuple[int, int, str], AlignmentState] = {}
        self.args = args
        self.last_generation_metrics = {}
        self.last_latent_metrics = {}

        # for ablation
        self.pre_aligned = None

        if self.use_vllm:

            tp_size = max(1, int(getattr(args, "tensor_parallel_size", 1)))
            gpu_util = float(getattr(args, "gpu_memory_utilization", 0.9))

            print(f"[vLLM] Using vLLM backend for model {model_name}")
            if args.enable_prefix_caching and args.method == "latent_mas":
                self.vllm_engine = LLM(model=model_name, tensor_parallel_size=tp_size, gpu_memory_utilization=gpu_util, enable_prefix_caching=True, enable_prompt_embeds=True, trust_remote_code=self.trust_remote_code)
            else:
                self.vllm_engine = LLM(model=model_name, tensor_parallel_size=tp_size, gpu_memory_utilization=gpu_util, trust_remote_code=self.trust_remote_code)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=False, trust_remote_code=self.trust_remote_code, **revision_kwargs)

            use_second_hf = bool(getattr(args, "use_second_HF_model", False)) if args else False
            if use_second_hf:
                self.HF_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
                    token=False,
                    trust_remote_code=self.trust_remote_code,
                    **revision_kwargs,
                ).to(args.device2).eval()
                self.embedding_layer = self.HF_model.get_input_embeddings()
                self.HF_device = args.device2
                self._ensure_alignment_state(self.HF_model, self.HF_model)
            elif self.align_method != "identical":
                raise ValueError("Non-identical alignment requires --use_second_HF_model when using vLLM backend.")
            _ensure_pad_token(self.tokenizer)
            return  # skip loading transformers model

        # fallback: normal transformers path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=False, trust_remote_code=self.trust_remote_code, **revision_kwargs)
        _ensure_pad_token(self.tokenizer)
        with torch.no_grad():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
                token=False,
                trust_remote_code=self.trust_remote_code,
                **revision_kwargs,
            )
        if len(self.tokenizer) != self.model.get_input_embeddings().weight.shape[0]:
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(device)
        self.model.eval()
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = True
        self._ensure_alignment_state(self.model, self.model)

    def render_chat(self, messages: List[Dict], add_generation_prompt: bool = True) -> str:
        tpl = getattr(self.tokenizer, "chat_template", None)
        if tpl:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        segments = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            segments.append(f"<|{role}|>\n{content}\n</|{role}|>")
        if add_generation_prompt:
            segments.append("<|assistant|>")
        return "\n".join(segments)

    def prepare_chat_input(
        self, messages: List[Dict], add_generation_prompt: bool = True
    ) -> Tuple[str, torch.Tensor, torch.Tensor, List[str]]:
        prompt_text = self.render_chat(messages, add_generation_prompt=add_generation_prompt)
        encoded = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        active_ids = input_ids[0][attention_mask[0].bool()].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(active_ids)
        return prompt_text, input_ids, attention_mask, tokens

    def prepare_chat_batch(
        self,
        batch_messages: List[List[Dict]],
        add_generation_prompt: bool = True,
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor, List[List[str]]]:
        prompts: List[str] = []
        for messages in batch_messages:
            prompts.append(self.render_chat(messages, add_generation_prompt=add_generation_prompt))
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        tokens_batch: List[List[str]] = []
        for ids_row, mask_row in zip(input_ids, attention_mask):
            active_ids = ids_row[mask_row.bool()].tolist()
            tokens_batch.append(self.tokenizer.convert_ids_to_tokens(active_ids))
        return prompts, input_ids, attention_mask, tokens_batch

    def vllm_generate_text_batch(
        self,
        prompts: List[str],
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
    ) -> List[str]:
        if not self.vllm_engine:
            raise RuntimeError("vLLM engine not initialized. Pass use_vllm=True to ModelWrapper.")
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_tokens=max_new_tokens,
        )
        _sync_cuda(self.device)
        started_at = time.perf_counter()
        outputs = self.vllm_engine.generate(prompts, sampling_params)
        _sync_cuda(self.device)
        wall_seconds = time.perf_counter() - started_at
        generations = [out.outputs[0].text.strip() for out in outputs]
        prefill_seconds, decode_seconds = _vllm_phase_split(outputs, wall_seconds)
        self.last_generation_metrics = {
            "prefill_seconds": prefill_seconds,
            "text_decode_seconds": decode_seconds,
            "output_token_counts": [len(out.outputs[0].token_ids) for out in outputs],
            "timing_source": "vllm_request_metrics",
        }
        return generations

    @staticmethod
    def _embedding_weights(source_model, target_model) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        input_embeds = target_model.get_input_embeddings() if hasattr(target_model, "get_input_embeddings") else None
        output_embeds = source_model.get_output_embeddings() if hasattr(source_model, "get_output_embeddings") else None
        if output_embeds is None:
            output_embeds = getattr(source_model, "lm_head", None)
        if (
            input_embeds is None or output_embeds is None
            or not hasattr(input_embeds, "weight") or not hasattr(output_embeds, "weight")
        ):
            raise RuntimeError("Cannot build alignment: embedding weights not accessible.")
        return output_embeds.weight, input_embeds.weight, getattr(output_embeds, "bias", None)

    def _build_alignment_state(self, source_model, target_model) -> AlignmentState:
        output_weight, input_weight, output_bias = self._embedding_weights(source_model, target_model)
        if self.align_method == "identical":
            if output_weight.shape[1] != input_weight.shape[1]:
                raise ValueError("identical alignment requires source and target hidden dimensions to match")
            return AlignmentState("identical", input_weight.detach().float().norm(dim=1).mean())
        if self.align_method == "linear":
            return build_linear_state(output_weight, input_weight, ridge=float(getattr(self.args, "align_ridge", 1e-5)))
        if self.align_method in ("kernel", "kernel_early_stopping"):
            return build_kernel_state(
                output_weight, input_weight, output_bias,
                feature_count=int(getattr(self.args, "kernel_features", 1024)),
                temperature=float(getattr(self.args, "kernel_temperature", 0.6)),
                seed=int(getattr(self.args, "kernel_seed", getattr(self.args, "seed", 42))),
                chunk_size=int(getattr(self.args, "kernel_chunk_size", 4096)),
                gate_mode=str(getattr(self.args, "kernel_gate_mode", "soft")),
                fixed_feature=int(getattr(self.args, "kernel_fixed_feature", 0)),
                topk=int(getattr(self.args, "kernel_topk", 8)),
            )
        if self.align_method == "soft":
            return build_soft_state(
                output_weight,
                input_weight,
                output_bias,
                temperature=float(getattr(self.args, "soft_temperature", 0.6)),
                query_chunk_size=int(getattr(self.args, "soft_chunk_size", 32)),
                top_k=int(getattr(self.args, "soft_top_k", 30)),
                top_p=float(getattr(self.args, "soft_top_p", 0.95)),
                min_p=float(getattr(self.args, "soft_min_p", 0.001)),
                max_topk=int(getattr(self.args, "soft_max_topk", 10)),
            )
        raise ValueError(f"Unsupported align method: {self.align_method}")

    def _ensure_alignment_state(self, source_model, target_model) -> AlignmentState:
        key = (id(source_model), id(target_model), self.align_method)
        state = self._alignment_states.get(key)
        if state is None:
            state = self._build_alignment_state(source_model, target_model)
            self._alignment_states[key] = state
        return state

    def _apply_latent_realignment(
        self,
        hidden: torch.Tensor,
        model: torch.nn.Module,
        *,
        return_entropy: bool = False,
    ):
        state = self._ensure_alignment_state(model, model)
        if return_entropy:
            if self.align_method == "soft":
                aligned, entropy = apply_soft_alignment_with_entropy(hidden, state)
            elif self.align_method == "kernel_early_stopping":
                aligned = apply_alignment(hidden, state)
                output_weight, _, output_bias = self._embedding_weights(model, model)
                entropy = compute_logits_entropy(
                    hidden,
                    output_weight,
                    output_bias,
                    temperature=float(getattr(self.args, "kernel_temperature", 0.6)),
                    query_chunk_size=32,
                )
            else:
                raise ValueError("Entropy-aware realignment requires an early-stopping alignment method")
            self.pre_aligned = aligned.detach().clone()
            return aligned, entropy
        aligned = apply_alignment(hidden, state)
        self.pre_aligned = aligned.detach().clone()
        return aligned

    def align_hidden_to(self, hidden: torch.Tensor, target: "ModelWrapper") -> torch.Tensor:
        """Align hidden states to target input embeddings; token IDs must match exactly."""
        if self.tokenizer.get_vocab() != target.tokenizer.get_vocab():
            raise ValueError("Alignment currently requires identical token-to-ID vocabularies.")
        source_model = self.HF_model if hasattr(self, "HF_model") else self.model
        target_model = target.HF_model if hasattr(target, "HF_model") else target.model
        aligned = apply_alignment(
            hidden, self._ensure_alignment_state(source_model, target_model)
        )
        target_device = target_model.get_input_embeddings().weight.device
        return aligned.to(target_device)
    @torch.no_grad()
    def generate_text_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple[List[str], Optional[Tuple]]:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        cache_position = None
        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            cache_position = torch.arange(
                past_len,
                past_len + input_ids.shape[-1],
                dtype=torch.long,
                device=self.device,
            )
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)

        _sync_cuda(self.device)
        started_at = time.perf_counter()
        phase_timer = _FirstTokenTimer(self.device, started_at)
        do_sample = temperature > 0
        generation_kwargs = {
            "temperature": temperature,
            "top_p": top_p,
        } if do_sample else {}
        if repetition_penalty != 1.0:
            generation_kwargs["repetition_penalty"] = repetition_penalty
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=False,
            logits_processor=LogitsProcessorList([phase_timer]),
            past_key_values=past_key_values,
            cache_position=cache_position,
            **generation_kwargs,
        )
        _sync_cuda(self.device)
        finished_at = time.perf_counter()
        completion_ids = _completion_token_ids(
            outputs.sequences,
            input_ids.shape[-1],
            self.model.generation_config.eos_token_id,
        )
        generations = [
            self.tokenizer.decode(ids, skip_special_tokens=True).strip()
            for ids in completion_ids
        ]
        first_token_at = phase_timer.first_token_at or finished_at
        self.last_generation_metrics = {
            "prefill_seconds": max(0.0, first_token_at - started_at),
            "text_decode_seconds": max(0.0, finished_at - first_token_at),
            "output_token_counts": [len(ids) for ids in completion_ids],
            "timing_source": "transformers_first_token_boundary",
        }
        return generations, outputs.past_key_values

    @torch.no_grad()
    def generate_text_from_embeds_batch(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
    ) -> Tuple[List[str], Optional[Tuple]]:
        """Generate after prefilling a decoder-only model from input embeddings."""
        if inputs_embeds.dim() != 3:
            raise ValueError("inputs_embeds must be 3D with shape [batch, seq_len, hidden_dim]")
        if attention_mask.shape != inputs_embeds.shape[:2]:
            raise ValueError("attention_mask must match the first two inputs_embeds dimensions")

        inputs_embeds = inputs_embeds.to(self.device)
        attention_mask = attention_mask.to(self.device)
        _sync_cuda(self.device)
        started_at = time.perf_counter()
        phase_timer = _FirstTokenTimer(self.device, started_at)
        do_sample = temperature > 0
        generation_kwargs = {
            "temperature": temperature,
            "top_p": top_p,
        } if do_sample else {}
        if repetition_penalty != 1.0:
            generation_kwargs["repetition_penalty"] = repetition_penalty
        outputs = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=False,
            logits_processor=LogitsProcessorList([phase_timer]),
            **generation_kwargs,
        )
        _sync_cuda(self.device)
        finished_at = time.perf_counter()

        # Decoder-only generation from inputs_embeds returns generated token IDs
        # without synthetic IDs for the embedding-only prefix.
        completion_ids = _completion_token_ids(
            outputs.sequences,
            0,
            self.model.generation_config.eos_token_id,
        )
        generations = [
            self.tokenizer.decode(ids, skip_special_tokens=True).strip()
            for ids in completion_ids
        ]
        first_token_at = phase_timer.first_token_at or finished_at
        self.last_generation_metrics = {
            "prefill_seconds": max(0.0, first_token_at - started_at),
            "text_decode_seconds": max(0.0, finished_at - first_token_at),
            "output_token_counts": [len(ids) for ids in completion_ids],
            "timing_source": "transformers_first_token_boundary",
        }
        return generations, getattr(outputs, "past_key_values", None)

    def tokenize_text(self, text: str) -> torch.Tensor:
        return self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(self.device)

    @torch.no_grad()
    def generate_latent_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        latent_steps: int,
        past_key_values: Optional[Tuple] = None,
        step_observer: Optional[Callable[[int, torch.Tensor], None]] = None,
        hidden_state_transform: Optional[
            Callable[[int, torch.Tensor], torch.Tensor]
        ] = None,
        aligned_embedding_transform: Optional[
            Callable[[int, torch.Tensor], torch.Tensor]
        ] = None,
    ) -> Tuple:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        else:
            attention_mask = attention_mask.to(self.device)

        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)

        _sync_cuda(self.device)
        prefill_started_at = time.perf_counter()
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        _sync_cuda(self.device)
        prefill_seconds = time.perf_counter() - prefill_started_at
        past = outputs.past_key_values
        last_hidden = outputs.hidden_states[-1][:, -1, :]
        if hidden_state_transform is not None:
            last_hidden = hidden_state_transform(0, last_hidden)

        alignment_timer = _AlignmentTimer(self.device)
        latent_started_at = time.perf_counter()
        fixed_budget = bool(getattr(self.args, "fixed_latent_budget", False))
        if fixed_budget and latent_steps < 0:
            raise ValueError("fixed latent budget must be nonnegative")
        early_stopping_enabled = not fixed_budget and self.align_method in ("soft", "kernel_early_stopping")
        if fixed_budget:
            decode_step_limit = latent_steps
        elif self.align_method == "soft":
            decode_step_limit = SOFT_LATENT_MAX_STEPS
        elif self.align_method == "kernel_early_stopping":
            decode_step_limit = KERNEL_EARLY_STOPPING_MAX_STEPS
        else:
            decode_step_limit = latent_steps
        low_entropy_run = torch.zeros(
            input_ids.shape[0], dtype=torch.long, device=last_hidden.device
        )
        stable_entropy_change_run = torch.zeros_like(low_entropy_run)
        previous_sampled_entropy = None
        default_length_threshold = 20 if self.align_method == "kernel_early_stopping" else 256
        default_entropy_threshold = 0.25 if self.align_method == "kernel_early_stopping" else 0.01
        entropy_length_threshold = int(
            getattr(self.args, "early_stopping_length_threshold", default_length_threshold)
        )
        entropy_threshold = float(
            getattr(self.args, "early_stopping_entropy_threshold", default_entropy_threshold)
        )
        low_entropy_checks_required = (
            max(1, (entropy_length_threshold + KERNEL_ENTROPY_CHECK_INTERVAL - 1) // KERNEL_ENTROPY_CHECK_INTERVAL)
            if self.align_method == "kernel_early_stopping"
            else entropy_length_threshold
        )
        actual_steps = 0
        for step in range(decode_step_limit):
            source_model = self.HF_model if hasattr(self, "HF_model") else self.model
            logits_entropy = None
            measure_entropy = (
                self.align_method == "soft"
                or (
                    self.align_method == "kernel_early_stopping"
                    and (step + 1) % KERNEL_ENTROPY_CHECK_INTERVAL == 0
                )
            )
            if self.align_method == "text":
                output_head = source_model.get_output_embeddings()
                if output_head is None:
                    raise RuntimeError("Text feedback requires an output head.")
                next_token = alignment_timer.measure(
                    lambda: output_head(last_hidden).argmax(dim=-1)
                )
                model_inputs = {"input_ids": next_token.unsqueeze(1)}
                batch_size = next_token.shape[0]
            else:
                if measure_entropy:
                    latent_vec, logits_entropy = alignment_timer.measure(
                        lambda: self._apply_latent_realignment(
                            last_hidden, source_model, return_entropy=True
                        )
                    )
                else:
                    latent_vec = alignment_timer.measure(
                        lambda: self._apply_latent_realignment(last_hidden, source_model)
                    )
                if aligned_embedding_transform is not None:
                    transformed = aligned_embedding_transform(step, latent_vec)
                    if (
                        transformed.shape != latent_vec.shape
                        or transformed.device != latent_vec.device
                        or transformed.dtype != latent_vec.dtype
                    ):
                        raise ValueError(
                            "aligned_embedding_transform must preserve tensor shape, "
                            "device, and dtype"
                        )
                    latent_vec = transformed
                latent_embed = latent_vec.unsqueeze(1)
                model_inputs = {"inputs_embeds": latent_embed}
                batch_size = latent_embed.shape[0]
            past_len = _past_length(past)
            latent_mask = torch.ones(
                (batch_size, past_len + 1),
                dtype=torch.long,
                device=self.device,
            )
            outputs = self.model(
                **model_inputs,
                attention_mask=latent_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]
            if hidden_state_transform is not None:
                last_hidden = hidden_state_transform(step + 1, last_hidden)
            actual_steps += 1
            if step_observer is not None:
                step_observer(step + 1, last_hidden)
            if early_stopping_enabled and logits_entropy is not None:
                low_entropy_run = torch.where(
                    logits_entropy < entropy_threshold,
                    low_entropy_run + 1,
                    torch.zeros_like(low_entropy_run),
                )
                stable_change_reached = False
                if self.align_method == "kernel_early_stopping":
                    if previous_sampled_entropy is not None:
                        stable_entropy_change_run = torch.where(
                            (logits_entropy - previous_sampled_entropy).abs()
                            < KERNEL_STABLE_CHANGE_THRESHOLD,
                            stable_entropy_change_run + 1,
                            torch.zeros_like(stable_entropy_change_run),
                        )
                        stable_change_reached = bool(
                            torch.all(
                                stable_entropy_change_run >= KERNEL_STABLE_CHANGE_COUNT
                            ).item()
                        )
                    previous_sampled_entropy = logits_entropy
                low_entropy_reached = bool(
                    torch.all(low_entropy_run >= low_entropy_checks_required).item()
                )
                if low_entropy_reached or stable_change_reached:
                    break
        _sync_cuda(self.device)
        latent_decode_seconds = time.perf_counter() - latent_started_at
        self.last_latent_metrics = {
            "prefill_seconds": prefill_seconds,
            "latent_decode_seconds": latent_decode_seconds,
            "alignment_seconds": alignment_timer.seconds(),
            "latent_output_counts": [actual_steps] * input_ids.shape[0],
            "text_output_counts": [
                latent_vocab_decode_steps(self.align_method, actual_steps)
            ] * input_ids.shape[0],
            "timing_source": "model_stage_boundaries",
        }
        return past

    @torch.no_grad()
    def generate_latent_batch_hidden_state(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        latent_steps: int,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.HF_device)
        else:
            attention_mask = attention_mask.to(self.HF_device)
        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)

        _sync_cuda(self.HF_device)
        prefill_started_at = time.perf_counter()
        outputs = self.HF_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        _sync_cuda(self.HF_device)
        prefill_seconds = time.perf_counter() - prefill_started_at
        past = outputs.past_key_values
        last_hidden = outputs.hidden_states[-1][:, -1, :]
        curr_output_embedding = [outputs.hidden_states[0]]

        alignment_timer = _AlignmentTimer(self.HF_device)
        latent_started_at = time.perf_counter()
        fixed_budget = bool(getattr(self.args, "fixed_latent_budget", False))
        if fixed_budget and latent_steps < 0:
            raise ValueError("fixed latent budget must be nonnegative")
        early_stopping_enabled = not fixed_budget and self.align_method in ("soft", "kernel_early_stopping")
        if fixed_budget:
            decode_step_limit = latent_steps
        elif self.align_method == "soft":
            decode_step_limit = SOFT_LATENT_MAX_STEPS
        elif self.align_method == "kernel_early_stopping":
            decode_step_limit = KERNEL_EARLY_STOPPING_MAX_STEPS
        else:
            decode_step_limit = latent_steps
        low_entropy_run = torch.zeros(
            input_ids.shape[0], dtype=torch.long, device=last_hidden.device
        )
        stable_entropy_change_run = torch.zeros_like(low_entropy_run)
        previous_sampled_entropy = None
        default_length_threshold = 20 if self.align_method == "kernel_early_stopping" else 256
        default_entropy_threshold = 0.25 if self.align_method == "kernel_early_stopping" else 0.01
        entropy_length_threshold = int(
            getattr(self.args, "early_stopping_length_threshold", default_length_threshold)
        )
        entropy_threshold = float(
            getattr(self.args, "early_stopping_entropy_threshold", default_entropy_threshold)
        )
        low_entropy_checks_required = (
            max(1, (entropy_length_threshold + KERNEL_ENTROPY_CHECK_INTERVAL - 1) // KERNEL_ENTROPY_CHECK_INTERVAL)
            if self.align_method == "kernel_early_stopping"
            else entropy_length_threshold
        )
        actual_steps = 0
        for step in range(decode_step_limit):
            source_model = self.HF_model if hasattr(self, "HF_model") else self.model
            logits_entropy = None
            measure_entropy = (
                self.align_method == "soft"
                or (
                    self.align_method == "kernel_early_stopping"
                    and (step + 1) % KERNEL_ENTROPY_CHECK_INTERVAL == 0
                )
            )
            if measure_entropy:
                latent_vec, logits_entropy = alignment_timer.measure(
                    lambda: self._apply_latent_realignment(
                        last_hidden, source_model, return_entropy=True
                    )
                )
            else:
                latent_vec = alignment_timer.measure(
                    lambda: self._apply_latent_realignment(last_hidden, source_model)
                )
            latent_embed = latent_vec.unsqueeze(1)
            past_len = _past_length(past)
            latent_mask = torch.ones(
                (latent_embed.shape[0], past_len + 1),
                dtype=torch.long,
                device=latent_embed.device,
            )
            outputs = self.HF_model(
                inputs_embeds=latent_embed,
                attention_mask=latent_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]
            curr_output_embedding.append(latent_embed.detach())
            actual_steps += 1
            if early_stopping_enabled and logits_entropy is not None:
                low_entropy_run = torch.where(
                    logits_entropy < entropy_threshold,
                    low_entropy_run + 1,
                    torch.zeros_like(low_entropy_run),
                )
                stable_change_reached = False
                if self.align_method == "kernel_early_stopping":
                    if previous_sampled_entropy is not None:
                        stable_entropy_change_run = torch.where(
                            (logits_entropy - previous_sampled_entropy).abs()
                            < KERNEL_STABLE_CHANGE_THRESHOLD,
                            stable_entropy_change_run + 1,
                            torch.zeros_like(stable_entropy_change_run),
                        )
                        stable_change_reached = bool(
                            torch.all(
                                stable_entropy_change_run >= KERNEL_STABLE_CHANGE_COUNT
                            ).item()
                        )
                    previous_sampled_entropy = logits_entropy
                low_entropy_reached = bool(
                    torch.all(low_entropy_run >= low_entropy_checks_required).item()
                )
                if low_entropy_reached or stable_change_reached:
                    break
        _sync_cuda(self.HF_device)
        latent_decode_seconds = time.perf_counter() - latent_started_at
        self.last_latent_metrics = {
            "prefill_seconds": prefill_seconds,
            "latent_decode_seconds": latent_decode_seconds,
            "alignment_seconds": alignment_timer.seconds(),
            "latent_output_counts": [actual_steps] * input_ids.shape[0],
            "text_output_counts": [
                latent_vocab_decode_steps(self.align_method, actual_steps)
            ] * input_ids.shape[0],
            "timing_source": "model_stage_boundaries",
        }
        return past, torch.cat(curr_output_embedding, dim=1)

