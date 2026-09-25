import json
from pathlib import Path
from typing import Dict, Iterable, Optional

from datasets import load_dataset

from .utils import extract_gold, normalize_answer


DEV_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "dev"


def load_local_dev(task: str) -> Iterable[Dict]:
    """Load the repository's fixed five-example development set for a task."""
    path = DEV_DATA_DIR / f"{task}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Local dev set not found: {path}")

    items = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise ValueError(f"Local dev set must contain a JSON list: {path}")

    for item in items:
        if task in {"aime2024", "aime2025", "gsm8k"}:
            question = item.get("question", item.get("problem", "")).strip()
            answer = str(item["answer"]).strip()
            gold = normalize_answer(answer)
        elif task in {"arc_easy", "arc_challenge"}:
            stem = item.get("question", item.get("problem", "")).strip()
            raw_choices = item["choices"]
            if isinstance(raw_choices, dict):
                labels = [str(label).strip() for label in raw_choices["label"]]
                choices = [str(text).strip() for text in raw_choices["text"]]
            else:
                choices = [str(text).strip() for text in raw_choices]
                labels = [chr(ord("A") + index) for index in range(len(choices))]
            question = stem + "\n" + "\n".join(
                f"{label}: {text}" for label, text in zip(labels, choices)
            )
            raw_answer = str(item["answer"]).strip()
            answer = labels[choices.index(raw_answer)] if raw_answer in choices else raw_answer
            gold = normalize_answer(answer)
        elif task == "gpqa":
            labels = ("A", "B", "C", "D")
            choices = [str(choice).strip() for choice in item["choices"]]
            question = item["question"].strip() + "\n" + "\n".join(
                f"{label}: {choice}" for label, choice in zip(labels, choices)
            )
            answer_text = str(item["answer"]).strip()
            try:
                answer = labels[choices.index(answer_text)]
            except ValueError as exc:
                raise ValueError(f"GPQA dev answer is absent from choices in {item.get('id')}") from exc
            gold = normalize_answer(answer)
        elif task == "medqa":
            question = item.get("query", item["question"]).strip()
            answer = str(item.get("answer_key", item["answer"])).strip()
            gold = normalize_answer(answer)
        elif task == "humanevalplus":
            question = (
                "Please provide a self-contained Python script that solves the following problem "
                "in a markdown code block:\n```python\nYOUR_PYTHON_CODE\n```:\n"
                + item["prompt"]
            )
            answer = str(item["test"]).replace("candidate", item["entry_point"])
            answer += f'\n\ncheck({item["entry_point"]})'
            gold = answer
        elif task == "mbppplus":
            tests = [str(test) for test in item["test_list"]]
            question = (
                "Please provide a self-contained Python script that solves the following problem "
                "in a markdown code block:\n```python\nYOUR_PYTHON_CODE\n```:\n"
                + item["prompt"]
                + "\nYour answer will be tested on test cases like:\n"
                + "\n".join(tests[:3])
            )
            answer = "\n".join(tests)
            gold = answer
        else:
            raise ValueError(f"No local dev adapter for task: {task}")

        yield {
            "question": question,
            "solution": item.get("solution", answer),
            "gold": gold,
        }


def load_gsm8k(split: str = "test", cache_dir: Optional[str] = None) -> Iterable[Dict]:
    ds = load_dataset("gsm8k", "main", split=split, cache_dir=cache_dir)
    for item in ds:
        question = item["question"].strip()
        solution = item["answer"]
        gold = normalize_answer(extract_gold(solution))
        yield {
            "question": question,
            "solution": solution,
            "gold": gold,
        }


def load_aime2025(split: str = "train", cache_dir: Optional[str] = None) -> Iterable[Dict]:
    ds = load_dataset("yentinglin/aime_2025", split=split, cache_dir=cache_dir)
    for item in ds:
        problem = item["problem"].strip()
        answer = str(item["answer"]).strip()
        gold = normalize_answer(answer)
        yield {
            "question": problem,
            "solution": answer,
            "gold": gold,
        }


def load_aime2024(split: str = "train", cache_dir: Optional[str] = None) -> Iterable[Dict]:
    ds = load_dataset("HuggingFaceH4/aime_2024", split=split, cache_dir=cache_dir)
    for item in ds:
        problem = item["problem"].strip()
        answer = str(item["answer"]).strip()
        gold = normalize_answer(answer)
        yield {
            "question": problem,
            "solution": answer,
            "gold": gold,
        }


def load_gpqa_diamond(split: str = "test", cache_dir: Optional[str] = None) -> Iterable[Dict]:
    ds = load_dataset("fingertap/GPQA-Diamond", split=split, cache_dir=cache_dir)
    for item in ds:
        question = item["question"].strip()
        answer = item["answer"].strip()
        gold = normalize_answer(answer)
        yield {
            "question": question,
            "solution": answer,
            "gold": gold,
        }


def load_arc_easy(split: str = "test", cache_dir: Optional[str] = None) -> Iterable[Dict]:
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split=split, cache_dir=cache_dir)
    for item in ds:
        stem = item["question"].strip()
        choices = item["choices"]
        labels = choices["label"]
        texts = choices["text"]
        label_map = {"1": "a", "2": "b", "3": "c", "4": "d"}

        def map_label(l: str) -> str:
            s = str(l).strip()
            if s in label_map:
                return label_map[s]
            return s.lower()

        # Map choices
        formatted_choices = {}
        mapped_order = []
        for label, text in zip(labels, texts):
            mlabel = map_label(label)
            formatted_choices[mlabel] = text.strip()
            mapped_order.append(mlabel)

        ordered_lines = [f"{lab}: {formatted_choices[lab]}" for lab in mapped_order]
        question = stem + "\n" + "\n".join(ordered_lines)

        # Map answers
        raw_answer = item.get("answerKey", "").strip()
        mapped_answer = map_label(raw_answer) if raw_answer else ""
        gold = normalize_answer(mapped_answer)
        yield {
            "question": question,
            "solution": mapped_answer,
            "gold": gold,
        }


def load_arc_challenge(split: str = "test", cache_dir: Optional[str] = None) -> Iterable[Dict]:
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split, cache_dir=cache_dir)
    for item in ds:
        stem = item["question"].strip()
        choices = item["choices"]
        labels = choices["label"]
        texts = choices["text"]
        label_map = {"1": "a", "2": "b", "3": "c", "4": "d"}

        def map_label(l: str) -> str:
            s = str(l).strip()
            if s in label_map:
                return label_map[s]
            return s.lower()

        formatted_choices = {}
        mapped_order = []
        for label, text in zip(labels, texts):
            mlabel = map_label(label)
            formatted_choices[mlabel] = text.strip()
            mapped_order.append(mlabel)

        ordered_lines = [f"{lab}: {formatted_choices[lab]}" for lab in mapped_order]
        question = stem + "\n" + "\n".join(ordered_lines)

        raw_answer = item.get("answerKey", "").strip()
        mapped_answer = map_label(raw_answer) if raw_answer else ""
        gold = normalize_answer(mapped_answer)
        yield {
            "question": question,
            "solution": mapped_answer,
            "gold": gold,
        }


def load_winogrande(
    split: str = "validation",
    subset: str = "winogrande_debiased",
    cache_dir: Optional[str] = None,
) -> Iterable[Dict]:
    ds = load_dataset("allenai/winogrande", subset, split=split, cache_dir=cache_dir)
    for item in ds:
        ask_str = 'Pickout proper choice that fits the _ in the following sentence:'
        sentence = item["sentence"].strip()
        option1 = str(item["option1"]).strip()
        option2 = str(item["option2"]).strip()
        question = f"{ask_str}\n{sentence}\n1: {option1}\n2: {option2}"
        answer = str(item["answer"])
        gold = normalize_answer(answer)
        yield {
            "question": question,
            "solution": answer,
            "gold": gold,
        }


def load_mbppplus(
    split: str = "test",
    subset: str = None,
    cache_dir: Optional[str] = None,
) -> Iterable[Dict]:
    ds = load_dataset("evalplus/mbppplus", subset, split=split, cache_dir=cache_dir)
    for item in ds:
        question = f"""Please provide a self-contained Python script that solves the following problem in a markdown code block:\n```python\nYOUR_PYTHON_CODE\n```:
{item["prompt"]}
Your answer will be tested on test cases like:
{item["test_list"][0]}
{item["test_list"][1]}
{item["test_list"][2]}
"""

        answer = str(item["test"])
        gold = answer
        yield {
            "question": question,
            "solution": answer,
            "gold": gold,
        }


def load_humanevalplus(
    split: str = "test",
    subset: str = None,
    cache_dir: Optional[str] = None,
) -> Iterable[Dict]:
    ds = load_dataset("evalplus/humanevalplus", subset, split=split, cache_dir=cache_dir)
    for item in ds:
        question = f"""Please provide a self-contained Python script that solves the following problem in a markdown code block:\n```python\nYOUR_PYTHON_CODE\n```:
{item["prompt"]}
"""
        raw_answer = str(item["test"])
        answer = raw_answer.replace('candidate', item['entry_point'])
        answer += f'\n\ncheck({item["entry_point"]})'
        gold = answer
        yield {
            "question": question,
            "solution": answer,
            "gold": gold,
        }


# qa data from https://github.com/lupantech/AgentFlow/tree/main
from typing import Iterable, Dict, Optional
from datasets import load_dataset

def load_medqa(split=None, subset=None, cache_dir=None):

    ds = load_dataset("json", data_files="./data/medqa.json", split='train')
    for item in ds:
        question = item["query"]
        raw_answer = str(item["answer"])

        choice_map = {"0":"A", "1":"B", "2":"C", "3":"D"}

        for idx, op in enumerate(item['options']):
            if raw_answer in op:
                answer = choice_map[str(idx)].lower()
                break

        gold = normalize_answer(answer)

        yield {
            "question": question,
            "solution": answer,
            "gold": gold,
        }

