from typing import Callable, Dict, List, Optional, Tuple

from . import default_agents
from ..models import ModelWrapper, _past_length, _sync_cuda, _vllm_phase_split
from ..prompts import build_agent_message_sequential_latent_mas, build_agent_message_hierarchical_latent_mas
from ..reasoning_models import append_manual_reasoning_cue
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

class LatentMASMethod:
    def __init__(
        self,
        model: ModelWrapper,
        *,
        latent_steps: int = 50,
        judger_max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        generate_bs: int = 1,
        args: argparse.Namespace = None,
        latent_step_observer: Optional[
            Callable[[str, int, torch.Tensor], None]
        ] = None,
        latent_hidden_transform: Optional[
            Callable[[str, int, torch.Tensor], torch.Tensor]
        ] = None,
        latent_embedding_transform: Optional[
            Callable[[str, int, torch.Tensor], torch.Tensor]
        ] = None,
        latent_roles_only: bool = False,
    ) -> None:
        self.args = args
        self.model = model
        self.latent_steps = latent_steps
        self.judger_max_new_tokens = judger_max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = float(getattr(args, "repetition_penalty", 1.0))
        self.generate_bs = max(1, generate_bs)
        self.agents = default_agents()
        self.method_name = 'latent_mas'
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
        self.latent_step_observer = latent_step_observer
        self.latent_hidden_transform = latent_hidden_transform
        self.latent_embedding_transform = latent_embedding_transform
        self.latent_roles_only = bool(latent_roles_only)

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

    @torch.no_grad()
    def run_batch(self, items: List[Dict]) -> List[Dict]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        batch_size = len(items)
        past_kv: Optional[Tuple] = None
        agent_traces: List[List[Dict]] = [[] for _ in range(batch_size)]
        final_texts = ["" for _ in range(batch_size)]
        # Per-role input metrics use the complete input visible to that role:
        # text_input is all retained textual prompt tokens, while latent_input
        # is the full past_key_values length supplied before the current prompt.
        cached_text_tokens = [0] * batch_size

        for agent in self.agents:
            role_kv_input_tokens = _past_length(past_kv)

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
                wrapped_ids = wrapped_encoded["input_ids"].to(self.model.device)
                wrapped_mask = wrapped_encoded["attention_mask"].to(self.model.device)
                wrapped_tokens_batch: List[List[str]] = []
                for ids_row, mask_row in zip(wrapped_ids, wrapped_mask):
                    active_ids = ids_row[mask_row.bool()].tolist()
                    wrapped_tokens_batch.append(self.model.tokenizer.convert_ids_to_tokens(active_ids))

                step_observer = None
                hidden_transform = None
                embedding_transform = None
                if self.latent_hidden_transform is not None:
                    hidden_transform = (
                        lambda step, hidden, role=agent.role:
                        self.latent_hidden_transform(role, step, hidden)
                    )
                if self.latent_step_observer is not None:
                    step_observer = (
                        lambda step, hidden, role=agent.role:
                        self.latent_step_observer(role, step, hidden)
                    )
                if self.latent_embedding_transform is not None:
                    embedding_transform = (
                        lambda step, embedding, role=agent.role:
                        self.latent_embedding_transform(role, step, embedding)
                    )
                past_kv = self.model.generate_latent_batch(
                    wrapped_ids,
                    attention_mask=wrapped_mask,
                    latent_steps=self.latent_steps,
                    past_key_values=past_kv,
                    step_observer=step_observer,
                    hidden_state_transform=hidden_transform,
                    aligned_embedding_transform=embedding_transform,
                )
                latent_metrics = self.model.last_latent_metrics
                actual_latent_steps = max(latent_metrics["latent_output_counts"], default=0)
                if self.sequential_info_only or self.latent_only:
                    new_past_len = _past_length(past_kv)
                    tokens_added = new_past_len - prev_past_len
                    tokens_to_keep = actual_latent_steps if self.latent_only else tokens_added
                    past_kv = self._truncate_past(past_kv, tokens_to_keep)

                for idx in range(batch_size):
                    mask = wrapped_mask[idx].bool()
                    trimmed_ids = wrapped_ids[idx][mask].to("cpu").tolist()
                    current_text_tokens = len(trimmed_ids)
                    total_text_tokens = cached_text_tokens[idx] + current_text_tokens
                    agent_traces[idx].append(
                        {
                            "name": agent.name,
                            "role": agent.role,
                            "input": wrapped_prompts[idx],
                            "input_ids": trimmed_ids,
                            "input_tokens": wrapped_tokens_batch[idx],
                            "latent_steps": latent_metrics["latent_output_counts"][idx],
                            "output": "",
                            "metrics": build_agent_metrics(
                                text_input_tokens=total_text_tokens,
                                latent_input_tokens=role_kv_input_tokens,
                                text_output_tokens=latent_metrics["text_output_counts"][idx],
                                latent_output_tokens=latent_metrics["latent_output_counts"][idx],
                                phase_metrics=latent_metrics,
                                batch_size=batch_size,
                            ),
                        }
                    )
                    if self.latent_only:
                        cached_text_tokens[idx] = 0
                    elif self.sequential_info_only:
                        cached_text_tokens[idx] = current_text_tokens
                    else:
                        cached_text_tokens[idx] = total_text_tokens
            else:
                if self.latent_roles_only:
                    break

                past_for_decoding = past_kv if _past_length(past_kv) > 0 else None
                role_kv_input_tokens = _past_length(past_for_decoding)

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
                judger_ids = judger_encoded["input_ids"].to(self.model.device)
                judger_mask = judger_encoded["attention_mask"].to(self.model.device)
                judger_tokens_batch: List[List[str]] = []
                for ids_row, mask_row in zip(judger_ids, judger_mask):
                    active_ids = ids_row[mask_row.bool()].tolist()
                    judger_tokens_batch.append(self.model.tokenizer.convert_ids_to_tokens(active_ids))
                generated_batch, _ = self.model.generate_text_batch(
                    judger_ids,
                    judger_mask,
                    max_new_tokens=self.judger_max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    repetition_penalty=self.repetition_penalty,
                    past_key_values=past_for_decoding,
                )
                generation_metrics = self.model.last_generation_metrics
                for idx in range(batch_size):
                    final_text = generated_batch[idx].strip()
                    final_texts[idx] = final_text
                    mask = judger_mask[idx].bool()
                    trimmed_ids = judger_ids[idx][mask].to("cpu").tolist()
                    total_text_tokens = cached_text_tokens[idx] + len(trimmed_ids)
                    agent_traces[idx].append(
                        {
                            "name": agent.name,
                            "role": agent.role,
                            "input": judger_prompts[idx],
                            "input_ids": trimmed_ids,
                            "input_tokens": judger_tokens_batch[idx],
                            "output": final_text,
                            "metrics": build_agent_metrics(
                                text_input_tokens=total_text_tokens,
                                latent_input_tokens=role_kv_input_tokens,
                                text_output_tokens=generation_metrics["output_token_counts"][idx],
                                phase_metrics=generation_metrics,
                                batch_size=batch_size,
                            ),
                        }
                    )

        if self.latent_roles_only:
            return [
                {
                    "question": item["question"],
                    "gold": item.get("gold", ""),
                    "solution": item.get("solution", ""),
                    "prediction": None,
                    "raw_prediction": "",
                    "agents": agent_traces[idx],
                    "correct": False,
                    "error": None,
                }
                for idx, item in enumerate(items)
            ]

        results: List[Dict] = []
        for idx, item in enumerate(items):
            final_text = final_texts[idx]
            if self.task in ['mbppplus', 'humanevalplus']:
                pred = extract_markdown_python_block(final_text)
                gold = item.get("gold", "")

                if pred is None:
                    ok = False
                    error_msg = "python error: No python code block found"
                else:
                    python_code_to_exe = pred + "\n" + gold
                    ok, error_msg = run_with_timeout(python_code_to_exe, timeout=10)

                print(f'=========================================')
                print(f'Question {idx}')
                print(f'error_msg: {error_msg}')
                # print(f'=========================================')

            elif self.task in ["aime2024", "aime2025"]:
                pred = normalize_answer(extract_gsm8k_answer(final_text))
                gold = str(item.get("gold", "")).strip()
                try:
                    pred_int = int(pred)
                    gold_int = int(gold)
                    ok = (pred_int == gold_int)
                    error_msg = None
                # A degenerate generation (for example, repeated text without
                # a final numeric answer) yields ``pred is None``. ``int``
                # raises TypeError for that case, so treat it as an incorrect
                # prediction rather than aborting the entire evaluation run.
                except (TypeError, ValueError):
                    ok = False
                    error_msg = f'Value error in parsing answer. Pred: {pred}, Gold: {gold}'

            else:
                pred = normalize_answer(extract_gsm8k_answer(final_text))
                gold = item.get("gold", "")
                ok = (pred == gold) if (pred and gold) else False
                error_msg = None

            results.append(
                {
                    "question": item["question"],
                    "gold": gold,
                    "solution": item["solution"],
                    "prediction": pred,
                    "raw_prediction": final_text,
                    "agents": agent_traces[idx],
                    "correct": ok,
                    "error": error_msg,
                }
            )
        return results

    def run_batch_vllm(self, items: List[Dict]) -> List[Dict]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        batch_size = len(items)
        past_kv: Optional[Tuple] = None
        agent_traces: List[List[Dict]] = [[] for _ in range(batch_size)]
        final_texts = ["" for _ in range(batch_size)]

        embedding_record = []
        cached_text_tokens = [0] * batch_size
        for agent in self.agents:
            role_kv_input_tokens = _past_length(past_kv)

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
                latent_metrics = self.model.last_latent_metrics
                actual_latent_steps = max(latent_metrics["latent_output_counts"], default=0)
                if self.sequential_info_only or self.latent_only:
                    new_past_len = _past_length(past_kv)
                    tokens_added = new_past_len - prev_past_len
                    tokens_to_keep = actual_latent_steps if self.latent_only else tokens_added
                    past_kv = self._truncate_past(past_kv, tokens_to_keep)

                if self.latent_only:
                    if actual_latent_steps > 0:
                        previous_hidden_embedding = previous_hidden_embedding[:, -actual_latent_steps:, :]
                    else:
                        previous_hidden_embedding = previous_hidden_embedding[:, 0:0, :]

                embedding_record.append(previous_hidden_embedding)

                if self.sequential_info_only or self.latent_only:
                    embedding_record = embedding_record[-1:]

                for idx in range(batch_size):
                    mask = wrapped_mask[idx].bool()
                    trimmed_ids = wrapped_ids[idx][mask].to("cpu").tolist()
                    current_text_tokens = len(trimmed_ids)
                    total_text_tokens = cached_text_tokens[idx] + current_text_tokens
                    agent_traces[idx].append(
                        {
                            "name": agent.name,
                            "role": agent.role,
                            "input": wrapped_prompts[idx],
                            "input_ids": trimmed_ids,
                            "input_tokens": wrapped_tokens_batch[idx],
                            "latent_steps": latent_metrics["latent_output_counts"][idx],
                            "output": "",
                            "metrics": build_agent_metrics(
                                text_input_tokens=total_text_tokens,
                                latent_input_tokens=role_kv_input_tokens,
                                text_output_tokens=latent_metrics["text_output_counts"][idx],
                                latent_output_tokens=latent_metrics["latent_output_counts"][idx],
                                phase_metrics=latent_metrics,
                                batch_size=batch_size,
                            ),
                        }
                    )
                    if self.latent_only:
                        cached_text_tokens[idx] = 0
                    elif self.sequential_info_only:
                        cached_text_tokens[idx] = current_text_tokens
                    else:
                        cached_text_tokens[idx] = total_text_tokens
            else:

                # A stack of [B, L_i, H]
                past_embedding = torch.cat(embedding_record, dim=1).to(self.vllm_device)
                # vLLM receives retained history as prompt embeddings rather
                # than an external HF cache. This is the equivalent KV history
                # prefetched before Judger decoding.
                role_kv_input_tokens = past_embedding.shape[1]

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


                _sync_cuda(self.vllm_device)
                started_at = time.perf_counter()
                outputs = self.model.vllm_engine.generate(
                    prompt_embeds_list,
                    self.sampling_params,
                )
                _sync_cuda(self.vllm_device)
                wall_seconds = time.perf_counter() - started_at
                prefill_seconds, decode_seconds = _vllm_phase_split(outputs, wall_seconds)
                generation_metrics = {
                    "prefill_seconds": prefill_seconds,
                    "text_decode_seconds": decode_seconds,
                    "output_token_counts": [len(out.outputs[0].token_ids) for out in outputs],
                    "timing_source": "vllm_request_metrics",
                }
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
                            "metrics": build_agent_metrics(
                                text_input_tokens=(
                                    cached_text_tokens[idx]
                                    + len(self.model.tokenizer(judger_prompts[idx], add_special_tokens=False)["input_ids"])
                                ),
                                latent_input_tokens=role_kv_input_tokens,
                                text_output_tokens=generation_metrics["output_token_counts"][idx],
                                phase_metrics=generation_metrics,
                                batch_size=batch_size,
                            ),
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
