from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from pyrag.tools import ANSWER_SYSTEM_PROMPT_NO_DOCS
from pyrag.utils import format_docs_for_prompt
from eval import compute_exact, extract_answer, metric_max_over_ground_truths, normalize_answer

DATA_SOURCE = "pyrag_answer_no_docs"


_YESNO_STARTS = ("is ", "are ", "was ", "were ", "do ", "does ", "did ",
                 "has ", "have ", "had ", "can ", "could ", "will ",
                 "would ", "should ")


def infer_qtype(q: str) -> str:
    q_low = (q or "").strip().lower()
    if not q_low:
        return "other"
    if q_low.startswith(_YESNO_STARTS) or "yes or no" in q_low or "are both" in q_low:
        return "yes-no"
    for kw, t in [("who", "who"), ("when", "when"), ("where", "where"),
                  ("which", "which"), ("what", "what"),
                  ("how many", "howmany"), ("how", "how"), ("why", "why")]:
        if re.search(rf"\b{kw}\b", q_low):
            return t
    return "other"


_GIVEN_RE = re.compile(
    r"given\s*:\s*(.*?)\s*answer the question\s*:\s*(.*)",
    re.IGNORECASE | re.DOTALL,
)


def find_last_synthesis_call(execution_log: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for entry in reversed(execution_log or []):
        if entry.get("type") == "answer" and not entry.get("docs"):
            return entry
    return None


def split_given_answer(synth_query: Any) -> Tuple[str, str]:
    if not isinstance(synth_query, str):
        synth_query = "" if synth_query is None else str(synth_query)
    m = _GIVEN_RE.search(synth_query)
    if not m:
        return synth_query, ""
    return m.group(1).strip(), m.group(2).strip()


def facts_support_gold(facts: str, gold_answers: List[str]) -> bool:
    if not facts:
        return False
    nf = normalize_answer(facts)
    if not nf:
        return False
    return any(normalize_answer(g) and normalize_answer(g) in nf for g in gold_answers)


def build_answer_user_prompt(query: str) -> str:
    return (
        "=== QUESTION ===\n"
        f"{query}\n"
        "=== END QUESTION ===\n\n"
        "=== RETRIEVED DOCUMENTS ===\n"
        f"{format_docs_for_prompt([])}\n"
        "=== END DOCUMENTS ==="
    )


def row_from_trace(sample: Dict[str, Any], keep_categories: set, index: int) -> Optional[Dict[str, Any]]:
    if sample.get("error"):
        return None

    log = sample.get("execution_log") or []
    synth = find_last_synthesis_call(log)
    if synth is None:
        return None

    synth_query = synth.get("query") or ""
    if not isinstance(synth_query, str):
        synth_query = str(synth_query)
    synth_pred  = synth.get("answer_returned") or extract_answer(synth.get("answer_raw") or "")
    if not isinstance(synth_pred, str):
        synth_pred = "" if synth_pred is None else str(synth_pred)
    gold_answers = list(sample.get("gold_answers") or sample.get("answers") or [])
    if not gold_answers or not synth_query:
        return None

    em = metric_max_over_ground_truths(compute_exact, synth_pred, gold_answers)
    correct = em >= 1.0

    facts, original_from_synth = split_given_answer(synth_query)
    sup = facts_support_gold(facts, gold_answers)

    if correct:
        category = "correct"
    elif sup:
        category = "wrong_facts_ok"
    else:
        category = "wrong_facts_bad"

    if category not in keep_categories:
        return None

    original_question = sample.get("question") or original_from_synth or ""
    qtype = infer_qtype(original_question)

    prompt = [
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT_NO_DOCS},
        {"role": "user",   "content": build_answer_user_prompt(synth_query)},
    ]

    return {
        "data_source": DATA_SOURCE,
        "prompt": prompt,
        "ability": "synthesis",
        "reward_model": {
            "style": "rule",
            "ground_truth": gold_answers,
        },
        "extra_info": {
            "index": index,
            "split": "all",
            "src_id": sample.get("id"),
            "original_question": original_question,
            "synthesis_query": synth_query,
            "qtype": qtype,
            "category": category,
            "baseline_pred": synth_pred,
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
                   help="JSONL outputs of eval_hotpotqa.py (with execution_log)")
    p.add_argument("--output_dir", required=True)
    p.add_argument(
        "--keep_categories", nargs="+",
        default=["correct", "wrong_facts_ok"],
        choices=["correct", "wrong_facts_ok", "wrong_facts_bad"],
        help="Only keep samples falling in these categories.",
    )
    p.add_argument("--val_ratio", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    keep = set(args.keep_categories)

    stats = {"total": 0, "correct": 0, "wrong_facts_ok": 0, "wrong_facts_bad": 0,
             "no_synth": 0, "error": 0, "qtype": {}}
    rows: List[Dict[str, Any]] = []

    global_idx = 0
    for path in args.input:
        print(f"[Load] {path}")
        samples = load_jsonl(path)
        print(f"       {len(samples)} samples")
        for s in samples:
            stats["total"] += 1
            if s.get("error"):
                stats["error"] += 1
                continue
            synth = find_last_synthesis_call(s.get("execution_log") or [])
            if synth is None:
                stats["no_synth"] += 1
                continue

            row = row_from_trace(s, keep_categories=keep, index=global_idx)
            if row is None:
                gold = s.get("gold_answers") or s.get("answers") or []
                pred = synth.get("answer_returned") or ""
                em = metric_max_over_ground_truths(compute_exact, pred, gold)
                if em >= 1.0:
                    cat = "correct"
                else:
                    facts, _ = split_given_answer(synth.get("query") or "")
                    cat = "wrong_facts_ok" if facts_support_gold(facts, gold) else "wrong_facts_bad"
                stats[cat] = stats.get(cat, 0) + 1
                continue

            stats[row["extra_info"]["category"]] += 1
            stats["qtype"][row["extra_info"]["qtype"]] = (
                stats["qtype"].get(row["extra_info"]["qtype"], 0) + 1
            )
            rows.append(row)
            global_idx += 1

    print("\n=== Stats ===")
    for k, v in stats.items():
        print(f"  {k:20s} : {v}")
    print(f"  kept              : {len(rows)}")

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
        print(f"category     : {preview['extra_info']['category']}")
        print(f"qtype        : {preview['extra_info']['qtype']}")
        print(f"gold         : {preview['reward_model']['ground_truth']}")
        print(f"baseline_pred: {preview['extra_info']['baseline_pred']}")
        print(f"--- user ---\n{preview['prompt'][1]['content']}")


if __name__ == "__main__":
    main()
