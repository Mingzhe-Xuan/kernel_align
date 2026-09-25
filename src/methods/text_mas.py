import copy
from typing import Dict, List, Optional

from . import Agent, default_agents
from ..models import ModelWrapper
from ..prompts import build_agent_messages_hierarchical_text_mas, build_agent_messages_sequential_text_mas
from ..utils import build_agent_metrics, extract_gsm8k_answer, normalize_answer, extract_markdown_python_block, run_with_timeout
import argparse
import pdb

class TextMASMethod:
    def __init__(
        self,
        model: ModelWrapper,
        *,
        agent_models: Optional[List[str]] = None,
        max_new_tokens_each: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        generate_bs: int = 1,
        args: argparse.Namespace = None,
    ) -> None:
        self.model = model
        self.max_new_tokens_each = max_new_tokens_each
        self.max_new_tokens_judger = max_new_tokens_each
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = float(getattr(args, "repetition_penalty", 1.0))
        self.generate_bs = max(1, generate_bs)
        self.args = args
        self.method_name = "text_mas"
        self.task = args.task

        if agent_models is None:
            self.agents = default_agents()
            self.agent_models = [model.model_name] * len(self.agents)
        elif len(agent_models) == 2:
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

        if model.use_vllm and len(set(self.agent_models)) > 1:
            raise ValueError("heterogeneous TextMAS requires the HF backend")

        self.models: Dict[str, ModelWrapper] = {model.model_name: model}
        for model_name in set(self.agent_models):
            if model_name not in self.models:
                print(f"Loading additional model: {model_name}")
                self.models[model_name] = ModelWrapper(
                    model_name,
                    args.device,
                    use_vllm=False,
                    args=args,
                )

    def run_batch(self, items: List[Dict]) -> List[Dict]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        batch_size = len(items)
        contexts = ["" for _ in range(batch_size)]
        history_contexts = ["" for _ in range(batch_size)]
        agent_traces: List[List[Dict]] = [[] for _ in range(batch_size)]
        final_texts = ["" for _ in range(batch_size)]

        for agent_index, agent in enumerate(self.agents):
            agent_model_name = self.agent_models[agent_index]
            agent_model = self.models[agent_model_name]
            prompt_args = copy.copy(self.args)
            prompt_args.model_name = agent_model_name

            if self.args.prompt == "hierarchical":
                batch_messages = [
                    build_agent_messages_hierarchical_text_mas(
                        role=agent.role,
                        question=item["question"],
                        context=contexts[idx],
                        method=self.method_name,
                        args=prompt_args,
                    )
                    for idx, item in enumerate(items)
                ]
            else:
                batch_messages = [
                    build_agent_messages_sequential_text_mas(
                        role=agent.role,
                        question=item["question"],
                        context=contexts[idx],
                        method=self.method_name,
                        args=prompt_args,
                    )
                    for idx, item in enumerate(items)
                ]

            prompts, input_ids, attention_mask, tokens_batch = agent_model.prepare_chat_batch(
                batch_messages, add_generation_prompt=True
            )

            if agent_model.use_vllm:
                generated_texts = agent_model.vllm_generate_text_batch(
                    prompts,
                    max_new_tokens=self.max_new_tokens_each,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    repetition_penalty=self.repetition_penalty,
                )
            else:
                generated_texts, _ = agent_model.generate_text_batch(
                    input_ids,
                    attention_mask,
                    max_new_tokens=self.max_new_tokens_each,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    repetition_penalty=self.repetition_penalty,
                )

            generation_metrics = agent_model.last_generation_metrics
            agent_name_map_for_prompt_hierarchical = {
                "Planner": "Math Agent",
                "Critic": "Science Agent",
                "Refiner": "Code Agent",
                "Judger": "Task Summrizer",
                "planner": "Math Agent",
                "critic": "Science Agent",
                "refiner": "Code Agent",
                "judger": "Task Summrizer",
            }

            for idx in range(batch_size):

                text_out = generated_texts[idx].strip()

                if self.args.prompt == "hierarchical":
                    formatted_output = f"[{agent_name_map_for_prompt_hierarchical[agent.name]}]:\n{text_out}\n\n"
                else:
                    formatted_output = f"[{agent.name}]:\n{text_out}\n\n"

                if agent.role != "judger":

                    contexts[idx] = f"{contexts[idx]}{formatted_output}"
                    history_contexts[idx] = f"{history_contexts[idx]}{formatted_output}"
                else:
                    final_texts[idx] = text_out
                mask = attention_mask[idx].bool()
                trimmed_ids = input_ids[idx][mask].to("cpu").tolist()
                agent_traces[idx].append(
                    {
                        "name": agent.name,
                        "role": agent.role,
                        "model": agent_model_name,
                        "input": prompts[idx],
                        "input_ids": trimmed_ids,
                        "input_tokens": tokens_batch[idx],
                        "output": text_out,
                        "metrics": build_agent_metrics(
                            text_input_tokens=len(trimmed_ids),
                            text_output_tokens=generation_metrics["output_token_counts"][idx],
                            phase_metrics=generation_metrics,
                            batch_size=batch_size,
                        ),
                    }
                )
            # import pdb; pdb.set_trace()

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

            elif self.task in ["aime2024", "aime2025"]:
                pred = normalize_answer(extract_gsm8k_answer(final_text))
                gold = str(item.get("gold", "")).strip()
                try:
                    pred_int = int(pred)
                    gold_int = int(gold)
                    ok = (pred_int == gold_int)
                    error_msg = None
                # `pred` can be None when no numeric answer is generated.
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
                    "context": history_contexts[idx],
                    "prediction": pred,
                    "raw_prediction": final_text,
                    "agents": agent_traces[idx],
                    "correct": ok,
                }
            )
        return results

    def run_item(self, item: Dict) -> Dict:
        return self.run_batch([item])[0]
