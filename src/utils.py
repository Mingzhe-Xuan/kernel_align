"""Shared runtime, evaluation, and result-aggregation utilities."""

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def auto_device(device: Optional[str] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# this is to extract answer in \boxed{}
def extract_gsm8k_answer(text: str) -> Optional[str]:
    boxes = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxes:
        content = boxes[-1]
        number = re.search(r"[-+]?\d+(?:\.\d+)?", content)
        return number.group(0) if number else content.strip()

    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1]
    return None


def extract_gold(text: str) -> Optional[str]:
    match = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", text)
    return match.group(1) if match else None


def normalize_answer(ans: Optional[str]) -> Optional[str]:
    if ans is None:
        return None
    return ans.strip().lower()


def extract_markdown_python_block(text: str) -> Optional[str]:
    pattern = r"```python(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return None


# to run python
import traceback
from multiprocessing import Process, Manager
def run_with_timeout(code, timeout):
    def worker(ns, code):
        try:
            local_ns = {}
            exec(code, local_ns)
            ns['ok'] = True
            ns['error'] = None
        except Exception:
            ns['ok'] = False
            ns['error'] = traceback.format_exc()
    with Manager() as manager:
        ns = manager.dict()
        p = Process(target=worker, args=(ns, code))
        p.start()
        p.join(timeout)
        if p.is_alive():
            p.terminate()
            ns['ok'] = False
            ns['error'] = f"TimeoutError: Execution exceeded {timeout} seconds"
        return ns.get('ok', False), ns.get('error', None)

def build_agent_metrics(
    *,
    text_input_tokens: int,
    latent_input_tokens: int = 0,
    text_output_tokens: int = 0,
    latent_output_tokens: int = 0,
    phase_metrics=None,
    batch_size: int = 1,
):
    """Build per-problem role metrics from counts and batch-level timings.

    Latent-state callers define ``text_input_tokens`` as all retained textual
    prompt tokens visible to the role and ``latent_input_tokens`` as the full
    KV-cache history supplied before the role's current prompt.
    """
    phase_metrics = phase_metrics or {}
    divisor = max(1, int(batch_size))
    timing = {
        key: float(phase_metrics[key]) / divisor
        for key in (
            "prefill_seconds",
            "latent_decode_seconds",
            "alignment_seconds",
            "text_decode_seconds",
        )
        if key in phase_metrics
    }
    source = phase_metrics.get("timing_source")
    if source:
        timing["source"] = source
    return {
        "tokens": {
            "text_input": int(text_input_tokens),
            "latent_input": int(latent_input_tokens),
            "text_output": int(text_output_tokens),
            "latent_output": int(latent_output_tokens),
        },
        "timing": timing,
    }


def is_number(value: Any) -> bool:
    """Return whether value is a non-boolean integer or float."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def average_values(values: list[Any]) -> Any:
    """Recursively average numeric leaves and retain identical metadata."""
    if values and all(is_number(value) for value in values):
        return round(sum(values) / len(values), 6)

    if values and all(isinstance(value, dict) for value in values):
        common_keys = set(values[0])
        for value in values[1:]:
            common_keys.intersection_update(value)
        return {
            key: average_values([value[key] for value in values])
            for key in values[0]
            if key in common_keys
        }

    if values and all(value == values[0] for value in values[1:]):
        return values[0]

    return values


def build_average(
    documents: list[dict[str, Any]],
    source_files: list[Path],
) -> dict[str, Any]:
    """Build an aggregate document from repeated run summaries."""
    if not documents:
        raise ValueError("No repeated experiment summaries were found")

    averaged = average_values(documents)
    seeds = [document.get("run", {}).get("seed") for document in documents]
    if isinstance(averaged.get("run"), dict):
        averaged["run"].pop("seed", None)

    return {
        "aggregation": {
            "repetitions": len(documents),
            "seeds": seeds,
            "source_files": [str(path) for path in source_files],
        },
        "average": averaged,
    }


def aggregate_result_files(
    inputs: list[Path],
    output_path: Path,
    expected_repetitions: Optional[int] = None,
) -> None:
    """Read, aggregate, and write repeated run summaries."""
    documents = [json.loads(path.read_text(encoding="utf-8-sig")) for path in inputs]
    if expected_repetitions is not None and len(documents) != expected_repetitions:
        raise ValueError(
            f"Expected {expected_repetitions} repeated summaries, found {len(documents)}"
        )

    output = build_average(documents, inputs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Averaged {len(documents)} runs into {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Repository utility commands")
    subparsers = parser.add_subparsers(dest="command", required=True)
    aggregate_parser = subparsers.add_parser(
        "aggregate-results",
        help="average numeric fields from repeated run summaries",
    )
    aggregate_parser.add_argument("--output", type=Path, required=True)
    aggregate_parser.add_argument("--expected-repetitions", type=int)
    aggregate_parser.add_argument("inputs", type=Path, nargs="+")
    args = parser.parse_args()

    if args.command == "aggregate-results":
        aggregate_result_files(
            args.inputs,
            args.output,
            expected_repetitions=args.expected_repetitions,
        )


if __name__ == "__main__":
    main()
