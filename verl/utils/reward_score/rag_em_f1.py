import re
import string
import collections
from typing import Union


def _normalize_answer(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize_answer(prediction).split()
    truth_tokens = _normalize_answer(ground_truth).split()
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return float(pred_tokens == truth_tokens)
    common = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def _em_score(prediction: str, ground_truth: str) -> float:
    return float(_normalize_answer(prediction) == _normalize_answer(ground_truth))


def _extract_answer(text: str) -> str:
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL)
    return m.group(1).strip() if m else ""


def compute_score(
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
    **kwargs,
) -> dict:
    answers: list[str] = []
    answers = ground_truth

    pred = _extract_answer(solution_str)
    if not pred or not answers:
        return {"score": 0.0, "f1": 0.0, "em": 0.0, "pred": pred}

    max_f1 = max(_f1_score(pred, gt) for gt in answers)
    max_em = max(_em_score(pred, gt) for gt in answers)
    score = 0.7 * max_f1 + 0.3 * max_em

    return {"score": score, "f1": max_f1, "em": max_em, "pred": pred}
