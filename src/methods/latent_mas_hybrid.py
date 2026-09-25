from typing import Dict, List, Optional, Tuple
import copy

from . import Agent, default_agents
from ..models import (
    ModelWrapper,
    SOFT_LATENT_MAX_STEPS,
    _AlignmentTimer,
    _past_length,
    _sync_cuda,
    latent_vocab_decode_steps,
)
from ..prompts import build_agent_message_sequential_latent_mas, build_agent_message_hierarchical_latent_mas
from ..reasoning_models import append_manual_reasoning_cue, resolve_manual_think
from ..utils import build_agent_metrics, extract_gsm8k_answer, normalize_answer, extract_markdown_python_block, run_with_timeout
import torch
import argparse
import time
try:
    from vllm import SamplingParams
except ImportError:
    SamplingParams = None
import pdb

try:
    from transformers.cache_utils import Cache
except ImportError:
    Cache = None


def transfer_via_realignment(
    hidden_states: torch.Tensor,
    model_from: ModelWrapper,
    model_to: ModelWrapper,
    lambda_reg: float = 1e-5,
) -> torch.Tensor:
    """Compatibility wrapper for the shared identical/linear/kernel aligner."""
    del lambda_reg  # configured by --align_ridge on the shared aligner
    return model_from.align_hidden_to(hidden_states, model_to)

class LatentMASMethod:
    def __init__(
        self,
        model: ModelWrapper,
        *,
        agent_models: Optional[List[str]] = None,  # NEW: Specify model per agent
        latent_steps: int = 50,
        judger_max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        generate_bs: int = 1,
        args: argparse.Namespace = None,
    ) -> None:
        self.args = args
        self.initial_model = model
        self.latent_steps = latent_steps
        self.judger_max_new_tokens = judger_max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = float(getattr(args, "repetition_penalty", 1.0))
        self.generate_bs = max(1, generate_bs)
        self.method_name = 'latent_mas_hybrid'
        self.vllm_device = args.device
        self.HF_device = args.device2
        self.latent_only = bool(getattr(args, "latent_only", False)) if args else False
        self.sequential_info_only = bool(getattr(args, "sequential_info_only", False)) if args else False

        if self.latent_only:
            self.sequential_info_only = True

        if SamplingParams is not None:
            self.sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=self.repetition_penalty,
                max_tokens=args.max_new_tokens,
            )
        self.task = args.task

        # NEW: Agent-to-model mapping
        if agent_models is None:
            self.agents = default_agents()
            # Default: all agents use same model
            self.agent_models = [model.model_name] * len(self.agents)
        elif len(agent_models) == 2:
            # Analysis-compatible Sender Planner -> Receiver Judger protocol.
            self.agents = [
                Agent(name="Planner", role="planner"),
                Agent(name="Judger", role="judger"),
            ]
            self.agent_models = agent_models
        else:
            self.agents = default_agents()
            if len(agent_models) != len(self.agents):
                raise ValueError("agent_models must contain either two or four model IDs")
            self.agent_models = agent_models

        if model.use_vllm:
            raise ValueError("latent_mas_hybrid full-context communication requires the HF backend")

        # Load all unique models
        self.models: Dict[str, ModelWrapper] = {model.model_name: model}
        self._load_additional_models()
        self._validate_alignment_chain()

        # For compatibility: self.model points to initial model
        # (used in vLLM path which we haven't updated yet)
        self.model = model

    def _load_additional_models(self):
        """Load any models needed by agents that aren't already loaded."""
        unique_models = set(self.agent_models)
        for model_name in unique_models:
            if model_name not in self.models:
                print(f"Loading additional model: {model_name}")
                # Create new ModelWrapper with same args and device
                # Note: Hybrid method doesn't support vLLM yet, so use_vllm=False
                # Use primary vllm_device (args.device) for all models to avoid multi-GPU issues
                new_model = ModelWrapper(
                    model_name,
                    self.vllm_device,  # Use primary device for all models
                    use_vllm=False,  # Hybrid doesn't support vLLM mixing yet
                    args=self.args
                )
                self.models[model_name] = new_model

    def _validate_alignment_chain(self) -> None:
        """Fail early when an adjacent Agent transition cannot be aligned."""
        for source_name, target_name in zip(self.agent_models, self.agent_models[1:]):
            source = self.models[source_name]
            target = self.models[target_name]
            if source.tokenizer.get_vocab() != target.tokenizer.get_vocab():
                raise ValueError(
                    "Hybrid alignment requires identical token-to-ID vocabularies: "
                    f"{source_name} -> {target_name}"
                )
            if source.align_method == "identical":
                source_dim = source.model.get_output_embeddings().weight.shape[1]
                target_dim = target.model.get_input_embeddings().weight.shape[1]
                if source_dim != target_dim:
                    raise ValueError(
                        "identical Hybrid alignment requires equal hidden dimensions: "
                        f"{source_name} ({source_dim}) -> {target_name} ({target_dim})"
                    )

    def _combine_context_and_prompt(
        self,
        agent_model: ModelWrapper,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        context_hidden: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
        context_model: Optional[ModelWrapper],
        alignment_timer: _AlignmentTimer,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Map prior context to this model and append native prompt embeddings."""
        prompt_ids = prompt_ids.to(agent_model.device)
        prompt_mask = prompt_mask.to(agent_model.device)
        prompt_embeds = agent_model.model.get_input_embeddings()(prompt_ids)
        if context_hidden is None:
            return prompt_embeds, prompt_mask
        if context_mask is None or context_model is None:
            raise ValueError("context_mask and context_model are required with context_hidden")

        aligned_context = alignment_timer.measure(
            lambda: context_model.align_hidden_to(context_hidden, agent_model)
        )
        aligned_context = aligned_context.to(
            device=prompt_embeds.device,
            dtype=prompt_embeds.dtype,
        )
        context_mask = context_mask.to(agent_model.device)
        combined_embeds = torch.cat([aligned_context, prompt_embeds], dim=1)
        combined_mask = torch.cat([context_mask, prompt_mask], dim=1)
        return combined_embeds, combined_mask

    def _prefill_and_latent(
        self,
        agent_model: ModelWrapper,
        combined_embeds: torch.Tensor,
        combined_mask: torch.Tensor,
        prompt_width: int,
        alignment_timer: _AlignmentTimer,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple]:
        """Prefill the full context+prompt sequence, then produce new latent states."""
        _sync_cuda(agent_model.device)
        prefill_started_at = time.perf_counter()
        outputs = agent_model.model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            past_key_values=None,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        _sync_cuda(agent_model.device)
        prefill_seconds = time.perf_counter() - prefill_started_at
        past = outputs.past_key_values
        prefill_hidden = outputs.hidden_states[-1]
        last_hidden = prefill_hidden[:, -1, :]

        latent_hidden_list = []
        running_mask = combined_mask
        latent_started_at = time.perf_counter()
        soft_early_stopping = agent_model.align_method == "soft"
        decode_step_limit = SOFT_LATENT_MAX_STEPS if soft_early_stopping else self.latent_steps
        low_entropy_run = torch.zeros(
            combined_embeds.shape[0], dtype=torch.long, device=last_hidden.device
        )
        entropy_length_threshold = int(
            getattr(self.args, "early_stopping_length_threshold", 256)
        )
        entropy_threshold = float(
            getattr(self.args, "early_stopping_entropy_threshold", 0.01)
        )
        actual_steps = 0
        for _ in range(decode_step_limit):
            logits_entropy = None
            if soft_early_stopping:
                latent_vec, logits_entropy = alignment_timer.measure(
                    lambda: agent_model._apply_latent_realignment(
                        last_hidden, agent_model.model, return_entropy=True
                    )
                )
            else:
                latent_vec = alignment_timer.measure(
                    lambda: agent_model._apply_latent_realignment(
                        last_hidden, agent_model.model
                    )
                )
            latent_embed = latent_vec.unsqueeze(1)
            next_mask = torch.ones(
                (latent_embed.shape[0], 1),
                dtype=torch.long,
                device=latent_embed.device,
            )
            running_mask = torch.cat([running_mask, next_mask], dim=1)
            outputs = agent_model.model(
                inputs_embeds=latent_embed,
                attention_mask=running_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]
            # These are h_1...h_K. The prompt's final h_0 is already present in
            # prefill_hidden and must not be duplicated in the latent segment.
            latent_hidden_list.append(last_hidden.unsqueeze(1))
            actual_steps += 1
            if logits_entropy is not None:
                low_entropy_run = torch.where(
                    logits_entropy < entropy_threshold,
                    low_entropy_run + 1,
                    torch.zeros_like(low_entropy_run),
                )
                if bool(torch.all(low_entropy_run >= entropy_length_threshold).item()):
                    break

        if latent_hidden_list:
            latent_hidden_states = torch.cat(latent_hidden_list, dim=1)
        else:
            latent_hidden_states = prefill_hidden[:, 0:0, :]
        latent_mask = torch.ones(
            latent_hidden_states.shape[:2],
            dtype=combined_mask.dtype,
            device=combined_mask.device,
        )

        if self.latent_only:
            next_context = latent_hidden_states
            next_context_mask = latent_mask
        elif self.sequential_info_only:
            prompt_hidden = prefill_hidden[:, -prompt_width:, :]
            prompt_mask = combined_mask[:, -prompt_width:]
            next_context = torch.cat([prompt_hidden, latent_hidden_states], dim=1)
            next_context_mask = torch.cat([prompt_mask, latent_mask], dim=1)
        else:
            # context_B = prefill_hidden_B || output_B, preserving the complete
            # history re-expressed by model B plus its new latent CoT states.
            next_context = torch.cat([prefill_hidden, latent_hidden_states], dim=1)
            next_context_mask = torch.cat([combined_mask, latent_mask], dim=1)

        _sync_cuda(agent_model.device)
        latent_decode_seconds = time.perf_counter() - latent_started_at
        agent_model.last_latent_metrics = {
            "prefill_seconds": prefill_seconds,
            "latent_decode_seconds": latent_decode_seconds,
            "alignment_seconds": alignment_timer.seconds(),
            "latent_output_counts": [actual_steps] * combined_embeds.shape[0],
            "text_output_counts": [
                latent_vocab_decode_steps(agent_model.align_method, actual_steps)
            ]
            * combined_embeds.shape[0],
            "timing_source": "model_stage_boundaries",
        }
        return next_context, next_context_mask, past

    def _test_helper(self):
        return None

    @staticmethod
    def _slice_tensor(tensor: torch.Tensor, tokens_to_keep: int) -> torch.Tensor:
        if tokens_to_keep <= 0:
            return tensor[..., 0:0, :].contiguous()
        keep = min(tokens_to_keep, tensor.shape[-2])
        start = tensor.shape[-2] - keep
        return tensor[..., start:, :].contiguous()

    def _truncate_past(self, past_kv: Optional[Tuple], tokens_to_keep: int) -> Optional[Tuple]:
        if past_kv is None or tokens_to_keep <= 0:
            return None
        if Cache is not None and isinstance(past_kv, Cache):
            legacy = past_kv.to_legacy_cache()
            trimmed_legacy = tuple(
                tuple(self._slice_tensor(t, tokens_to_keep) for t in layer)
                for layer in legacy
            )
            return past_kv.__class__.from_legacy_cache(trimmed_legacy)
        trimmed_layers = []
        for layer in past_kv:
            if isinstance(layer, tuple):
                trimmed_layers.append(tuple(self._slice_tensor(t, tokens_to_keep) for t in layer))
            elif torch.is_tensor(layer):
                trimmed_layers.append(self._slice_tensor(layer, tokens_to_keep))
            else:
                trimmed_layers.append(layer)
        return tuple(trimmed_layers)

    def _format_results(
        self,
        items: List[Dict],
        final_texts: List[str],
        agent_traces: List[List[Dict]],
    ) -> List[Dict]:
        results: List[Dict] = []
        for idx, item in enumerate(items):
            final_text = final_texts[idx]
            if self.task in ["mbppplus", "humanevalplus"]:
                pred = extract_markdown_python_block(final_text)
                gold = item.get("gold", "")
                if pred is None:
                    ok = False
                else:
                    ok, _ = run_with_timeout(pred + "\n" + gold, timeout=10)
            elif self.task in ["aime2024", "aime2025"]:
                pred = normalize_answer(extract_gsm8k_answer(final_text))
                gold = str(item.get("gold", "")).strip()
                try:
                    ok = pred not in (None, "") and int(pred) == int(gold)
                except (TypeError, ValueError):
                    ok = False
            else:
                pred = normalize_answer(extract_gsm8k_answer(final_text))
                gold = item.get("gold", "")
                ok = (pred == gold) if (pred and gold) else False

            results.append(
                {
                    "question": item["question"],
                    "gold": gold,
                    "solution": item["solution"],
                    "prediction": pred,
                    "raw_prediction": final_text,
                    "agents": agent_traces[idx],
                    "correct": ok,
                }
            )
        return results

    @torch.no_grad()
    def run_batch(self, items: List[Dict]) -> List[Dict]:
        """Run the full-context Hybrid hidden-state communication chain."""
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        batch_size = len(items)
        context_hidden: Optional[torch.Tensor] = None
        context_mask: Optional[torch.Tensor] = None
        context_model: Optional[ModelWrapper] = None
        agent_traces: List[List[Dict]] = [[] for _ in range(batch_size)]
        final_texts = ["" for _ in range(batch_size)]
        requested_think = getattr(
            self.args,
            "think_requested",
            getattr(self.args, "think", None),
        )

        for agent_idx, agent in enumerate(self.agents):
            agent_model_name = self.agent_models[agent_idx]
            agent_model = self.models[agent_model_name]
            manual_think = resolve_manual_think(agent_model_name, requested_think)
            prompt_args = copy.copy(self.args)
            prompt_args.model_name = agent_model_name
            context_input_counts = (
                [0] * batch_size
                if context_mask is None
                else [int(row.sum().item()) for row in context_mask]
            )

            if self.args.prompt == "sequential":
                batch_messages = [
                    build_agent_message_sequential_latent_mas(
                        role=agent.role,
                        question=item["question"],
                        context="",
                        method=self.method_name,
                        args=prompt_args,
                    )
                    for item in items
                ]
            else:
                batch_messages = [
                    build_agent_message_hierarchical_latent_mas(
                        role=agent.role,
                        question=item["question"],
                        context="",
                        method=self.method_name,
                        args=prompt_args,
                    )
                    for item in items
                ]

            prompts, _, _, _ = agent_model.prepare_chat_batch(
                batch_messages, add_generation_prompt=True
            )
            wrapped_prompts = [
                append_manual_reasoning_cue(
                    prompt, agent_model_name, manual_think
                )
                for prompt in prompts
            ]
            encoded = agent_model.tokenizer(
                wrapped_prompts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
            prompt_ids = encoded["input_ids"].to(agent_model.device)
            prompt_mask = encoded["attention_mask"].to(agent_model.device)
            prompt_tokens_batch = []
            prompt_id_rows = []
            for ids_row, mask_row in zip(prompt_ids, prompt_mask):
                active_ids = ids_row[mask_row.bool()].to("cpu").tolist()
                prompt_id_rows.append(active_ids)
                prompt_tokens_batch.append(
                    agent_model.tokenizer.convert_ids_to_tokens(active_ids)
                )

            alignment_timer = _AlignmentTimer(agent_model.device)
            combined_embeds, combined_mask = self._combine_context_and_prompt(
                agent_model,
                prompt_ids,
                prompt_mask,
                context_hidden,
                context_mask,
                context_model,
                alignment_timer,
            )

            if agent.role != "judger":
                context_hidden, context_mask, _ = self._prefill_and_latent(
                    agent_model,
                    combined_embeds,
                    combined_mask,
                    prompt_ids.shape[1],
                    alignment_timer,
                )
                context_hidden = context_hidden.detach()
                context_mask = context_mask.detach()
                context_model = agent_model
                phase_metrics = agent_model.last_latent_metrics
                for idx in range(batch_size):
                    agent_traces[idx].append(
                        {
                            "name": agent.name,
                            "role": agent.role,
                            "model": agent_model_name,
                            "manual_think": manual_think,
                            "input": wrapped_prompts[idx],
                            "input_ids": prompt_id_rows[idx],
                            "input_tokens": prompt_tokens_batch[idx],
                            "latent_steps": phase_metrics["latent_output_counts"][idx],
                            "output": "",
                            "metrics": build_agent_metrics(
                                text_input_tokens=len(prompt_id_rows[idx]),
                                latent_input_tokens=context_input_counts[idx],
                                text_output_tokens=phase_metrics["text_output_counts"][idx],
                                latent_output_tokens=phase_metrics["latent_output_counts"][idx],
                                phase_metrics=phase_metrics,
                                batch_size=batch_size,
                            ),
                        }
                    )
            else:
                generated_batch, _ = agent_model.generate_text_from_embeds_batch(
                    combined_embeds,
                    combined_mask,
                    max_new_tokens=self.judger_max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    repetition_penalty=self.repetition_penalty,
                )
                phase_metrics = dict(agent_model.last_generation_metrics)
                phase_metrics["alignment_seconds"] = alignment_timer.seconds()
                for idx in range(batch_size):
                    final_text = generated_batch[idx].strip()
                    final_texts[idx] = final_text
                    agent_traces[idx].append(
                        {
                            "name": agent.name,
                            "role": agent.role,
                            "model": agent_model_name,
                            "manual_think": manual_think,
                            "input": wrapped_prompts[idx],
                            "input_ids": prompt_id_rows[idx],
                            "input_tokens": prompt_tokens_batch[idx],
                            "output": final_text,
                            "metrics": build_agent_metrics(
                                text_input_tokens=len(prompt_id_rows[idx]),
                                latent_input_tokens=context_input_counts[idx],
                                text_output_tokens=phase_metrics["output_token_counts"][idx],
                                phase_metrics=phase_metrics,
                                batch_size=batch_size,
                            ),
                        }
                    )

        return self._format_results(items, final_texts, agent_traces)

    def run_batch_vllm(self, items: List[Dict]) -> List[Dict]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        batch_size = len(items)
        past_kv: Optional[Tuple] = None
        agent_traces: List[List[Dict]] = [[] for _ in range(batch_size)]
        final_texts = ["" for _ in range(batch_size)]

        embedding_record = []
        for agent in self.agents:

            if self.args.prompt == "sequential":
                batch_messages = [
                    build_agent_message_sequential_latent_mas(role=agent.role, question=item["question"], context="", method=self.method_name, args=self.args)
                    for item in items
                ]
            elif self.args.prompt == "hierarchical":
                batch_messages = [
                    build_agent_message_hierarchical_latent_mas(role=agent.role, question=item["question"], context="", method=self.method_name, args=self.args)
                    for item in items
                ]

            prompts, input_ids, attention_mask, tokens_batch = self.model.prepare_chat_batch(
                batch_messages, add_generation_prompt=True
            )

            if agent.role != "judger":
                prev_past_len = _past_length(past_kv)

                # to wrap all latent thoughts from previous agents
                wrapped_prompts = [
                    append_manual_reasoning_cue(
                        prompt, self.model.model_name, self.args.think
                    )
                    for prompt in prompts
                ]

                wrapped_encoded = self.model.tokenizer(
                    wrapped_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                wrapped_ids = wrapped_encoded["input_ids"].to(self.model.HF_device)
                wrapped_mask = wrapped_encoded["attention_mask"].to(self.model.HF_device)
                wrapped_tokens_batch: List[List[str]] = []
                for ids_row, mask_row in zip(wrapped_ids, wrapped_mask):
                    active_ids = ids_row[mask_row.bool()].tolist()
                    wrapped_tokens_batch.append(self.model.tokenizer.convert_ids_to_tokens(active_ids))

                past_kv, previous_hidden_embedding = self.model.generate_latent_batch_hidden_state(
                    wrapped_ids,
                    attention_mask=wrapped_mask,
                    latent_steps=self.latent_steps,
                    past_key_values=past_kv,
                )
                if self.sequential_info_only or self.latent_only:
                    new_past_len = _past_length(past_kv)
                    tokens_added = new_past_len - prev_past_len
                    tokens_to_keep = self.latent_steps if self.latent_only else tokens_added
                    past_kv = self._truncate_past(past_kv, tokens_to_keep)

                if self.latent_only:
                    if self.latent_steps > 0:
                        previous_hidden_embedding = previous_hidden_embedding[:, -self.latent_steps:, :]
                    else:
                        previous_hidden_embedding = previous_hidden_embedding[:, 0:0, :]

                embedding_record.append(previous_hidden_embedding)

                if self.sequential_info_only or self.latent_only:
                    embedding_record = embedding_record[-1:]

                for idx in range(batch_size):
                    mask = wrapped_mask[idx].bool()
                    trimmed_ids = wrapped_ids[idx][mask].to("cpu").tolist()
                    agent_traces[idx].append(
                        {
                            "name": agent.name,
                            "role": agent.role,
                            "input": wrapped_prompts[idx],
                            "input_ids": trimmed_ids,
                            "input_tokens": wrapped_tokens_batch[idx],
                            "latent_steps": self.latent_steps,
                            "output": "",
                        }
                    )
            else:

                # A stack of [B, L_i, H]
                past_embedding = torch.cat(embedding_record, dim=1).to(self.vllm_device)

                judger_prompts = [
                    append_manual_reasoning_cue(
                        prompt, self.model.model_name, self.args.think
                    )
                    for prompt in prompts
                ]

                judger_encoded = self.model.tokenizer(
                    judger_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                judger_encoded = judger_encoded["input_ids"].to(self.model.HF_device)
                # Get current prompt embedding
                curr_prompt_emb = self.model.embedding_layer(judger_encoded).squeeze(0).to(self.vllm_device)

                # assert Qwen model
                assert "Qwen" in self.args.model_name or "qwen" in self.args.model_name, "latent_embedding_position is only supported for Qwen models currently."

                # handle latent embedding insertion position
                len_of_left = []
                for p in judger_prompts:
                    idx = p.find("<|im_start|>user\n")
                    # Get the text up to and including "<|im_start|>user\n"
                    left = p[: idx + len("<|im_start|>user\n")]
                    len_of_left.append(len(self.model.tokenizer(left)['input_ids']))

                B, L, H = curr_prompt_emb.shape
                _, Lp, H = past_embedding.shape  # assume shape consistency

                whole_prompt_emb_list = []
                for i in range(B):
                    insert_idx = len_of_left[i]
                    left_emb = curr_prompt_emb[i, :insert_idx, :]
                    right_emb = curr_prompt_emb[i, insert_idx:, :]
                    combined = torch.cat([left_emb, past_embedding[i], right_emb], dim=0)
                    whole_prompt_emb_list.append(combined)

                # Pad back to max length if needed
                max_len = max(x.shape[0] for x in whole_prompt_emb_list)
                whole_prompt_emb = torch.stack([
                    torch.cat([x, torch.zeros(max_len - x.shape[0], H, device=x.device)], dim=0)
                    for x in whole_prompt_emb_list
                ])

                # else:
                    # Get full prompt embedding from cat with previous ones
                    # B L H B L H
                    # whole_prompt_emb = torch.cat([past_embedding, curr_prompt_emb], dim=1)

                # pdb.set_trace()

                # Use vLLM
                prompt_embeds_list = [
                    {
                        "prompt_embeds": embeds
                    } for embeds in whole_prompt_emb
                ]


                outputs = self.model.vllm_engine.generate(
                    prompt_embeds_list,
                    self.sampling_params,
                )

                generated_texts = [out.outputs[0].text.strip() for out in outputs]

                for idx in range(batch_size):
                    text_out = generated_texts[idx].strip()
                    final_texts[idx] = text_out
                    agent_traces[idx].append(
                        {
                            "name": agent.name,
                            "role": agent.role,
                            "input": judger_prompts[idx],
                            "output": text_out,
                        }
                    )


        results: List[Dict] = []
        for idx, item in enumerate(items):
            final_text = final_texts[idx]
            pred = normalize_answer(extract_gsm8k_answer(final_text))
            gold = item["gold"]
            ok = (pred == gold) if (pred and gold) else False
            results.append(
                {
                    "question": item["question"],
                    "gold": gold,
                    "solution": item["solution"],
                    "prediction": pred,
                    "raw_prediction": final_text,
                    "agents": agent_traces[idx],
                    "correct": ok,
                }
            )
        return results

    def run_item(self, item: Dict) -> Dict:
        return self.run_batch([item])[0]
