from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from pyrag.plan_agent import SYSTEM_PROMPT as PLAN_SYSTEM_PROMPT
from pyrag.plan_agent import CODE_EXAMPLE
from eval import compute_exact, extract_answer, metric_max_over_ground_truths

DATA_SOURCE = "pyrag_plan"


def build_plan_user_prompt(original_query: str, sub_queries: List[str]) -> str:
    return f"""
Original question:
{original_query}

Sub-queries to resolve:
{sub_queries}

Reference example (do NOT copy, write code for the actual question above):
{CODE_EXAMPLE}

Now write the Python code for the original question.
End with: final_answer = answer(f"Given: <facts>. Answer the question: {original_query}")
""".strip()


def em_of(pred: str, gold_answers: List[str]) -> float:
    p = extract_answer(pred or "")
    return metric_max_over_ground_truths(compute_exact, p, gold_answers or [])


def row_from_sample(
    sample: Dict[str, Any],
    index: int,
    only_correct: bool,
) -> Optional[Dict[str, Any]]:
    if sample.get("error"):
        return None
    question = (sample.get("question") or "").strip()
    gold = list(sample.get("gold_answers") or sample.get("answers") or [])
    subs = sample.get("sub_queries")
    if not question or not gold or not subs or not isinstance(subs, list):
        return None

    if len(subs) == 1 and subs[0].strip().lower() == question.strip().lower():
        return None

    is_correct = em_of(sample.get("pred_answer") or "", gold) >= 1.0
    if only_correct and not is_correct:
        return None

    prompt = [
        {"role": "system", "content": PLAN_SYSTEM_PROMPT},
        {"role": "user",   "content": build_plan_user_prompt(question, subs)},
    ]
    return {
        "data_source": DATA_SOURCE,
        "prompt": prompt,
        "ability": "code-gen",
        "reward_model": {"style": "rule", "ground_truth": gold},
        "extra_info": {
            "index": index,
            "split": "all",
            "src_id": sample.get("id"),
            "question": question,
            "sub_queries": subs,
            "topk": 5,
            "is_correct_trace": bool(is_correct),
            "baseline_pred": sample.get("pred_answer"),
            "baseline_code": sample.get("generated_code"),
        },
    }


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", nargs="+", required=True,
                   help="jsonl file(s) from eval_hotpotqa.py or precompute_decompose.py")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--val_ratio", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--only_correct", action="store_true",
                   help="Keep only trajectories whose baseline EM==1.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    rows: List[Dict[str, Any]] = []
    stats = {"total": 0, "kept": 0, "no_subs": 0, "fallback_subs": 0,
             "error": 0, "incorrect_filtered": 0, "correct": 0}
    for path in args.input:
        print(f"[Load] {path}")
        samples = load_jsonl(path)
        print(f"       {len(samples)} samples")
        for s in samples:
            stats["total"] += 1
            if s.get("error"):
                stats["error"] += 1
                continue
            subs = s.get("sub_queries")
            if not subs or not isinstance(subs, list):
                stats["no_subs"] += 1
                continue
            question = (s.get("question") or "").strip()
            if len(subs) == 1 and subs[0].strip().lower() == question.strip().lower():
                stats["fallback_subs"] += 1
                continue

            gold = list(s.get("gold_answers") or s.get("answers") or [])
            correct = em_of(s.get("pred_answer") or "", gold) >= 1.0
            if correct:
                stats["correct"] += 1
            if args.only_correct and not correct:
                stats["incorrect_filtered"] += 1
                continue

            r = row_from_sample(s, index=len(rows), only_correct=args.only_correct)
            if r is not None:
                rows.append(r)
                stats["kept"] += 1

    print("\n=== Stats ===")
    for k, v in stats.items():
        print(f"  {k:22s} : {v}")

    random.shuffle(rows)
    n_val = int(len(rows) * args.val_ratio) if args.val_ratio > 0 else 0
    val_rows, train_rows = rows[:n_val], rows[n_val:]
    for i, r in enumerate(train_rows):
        r["extra_info"]["index"] = i
        r["extra_info"]["split"] = "train"
    for i, r in enumerate(val_rows):
        r["extra_info"]["index"] = i
        r["extra_info"]["split"] = "val"

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train_rows).to_parquet(out_dir / "train.parquet", index=False)
    print(f"\n[Saved] {out_dir / 'train.parquet'}   rows={len(train_rows)}")
    if val_rows:
        pd.DataFrame(val_rows).to_parquet(out_dir / "val.parquet", index=False)
        print(f"[Saved] {out_dir / 'val.parquet'}     rows={len(val_rows)}")

    if train_rows:
        preview = train_rows[0]
        print("\n=== Preview (first train row) ===")
        print(f"question     : {preview['extra_info']['question']}")
        print(f"sub_queries  : {preview['extra_info']['sub_queries']}")
        print(f"gold         : {preview['reward_model']['ground_truth']}")
        print(f"is_correct   : {preview['extra_info']['is_correct_trace']}")
        print(f"--- user (truncated) ---")
        print(preview["prompt"][1]["content"][:600])


if __name__ == "__main__":
    main()
