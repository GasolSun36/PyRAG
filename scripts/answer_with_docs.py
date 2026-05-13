from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from pyrag.tools import ANSWER_SYSTEM_PROMPT_WITH_DOCS
from pyrag.utils import format_docs_for_prompt

DATA_SOURCE = "pyrag_answer_with_docs"


def _strip_title_prefix(body: str, title: str) -> str:
    if not body or not title:
        return body
    for candidate in (f'"{title}"', title):
        if body.startswith(candidate):
            return body[len(candidate):].lstrip(" \n")
    return body


def format_ctxs_as_docs(ctxs: List[Dict[str, Any]], topk: int) -> List[str]:
    docs: List[str] = []
    for idx, ctx in enumerate(ctxs[:topk]):
        title = (ctx.get("title") or "").strip().strip('"')
        body = _strip_title_prefix((ctx.get("doc") or "").strip(), title)
        docs.append(f"Doc {idx+1} (Title: {title})\n{body}")
    return docs


def build_answer_user_prompt(query: str, docs: List[str]) -> str:
    return (
        "=== QUESTION ===\n"
        f"{query}\n"
        "=== END QUESTION ===\n\n"
        "=== RETRIEVED DOCUMENTS ===\n"
        f"{format_docs_for_prompt(docs)}\n"
        "=== END DOCUMENTS ==="
    )


def row_from_sample(sample: Dict[str, Any], topk: int, index: int, split: str) -> Dict[str, Any] | None:
    question = (sample.get("question") or "").strip()
    answers = sample.get("answers") or sample.get("golden_answers") or []
    if not question or not answers:
        return None

    ctxs = sample.get("ctxs") or []
    if not ctxs:
        return None

    docs = format_ctxs_as_docs(ctxs, topk=topk)

    prompt = [
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT_WITH_DOCS},
        {"role": "user", "content": build_answer_user_prompt(question, docs)},
    ]

    return {
        "data_source": DATA_SOURCE,
        "prompt": prompt,
        "ability": "span-qa",
        "reward_model": {
            "style": "rule",
            "ground_truth": list(answers),
        },
        "extra_info": {
            "index": index,
            "split": split,
            "src_id": sample.get("id"),
            "question": question,
            "topk": topk,
            "num_ctxs_available": len(ctxs),
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
    p.add_argument(
        "--input", nargs="+", required=True,
        help="One or more jsonl files (nq.jsonl / triviaqa.jsonl / popqa.jsonl ...)",
    )
    p.add_argument("--output_dir", required=True)
    p.add_argument("--topk", type=int, default=5,
                   help="Used only when --topk_choices is not set.")
    p.add_argument("--topk_choices", type=int, nargs="+", default=None,
                   help="If given, each sample draws topk uniformly from this set "
                        "(e.g. --topk_choices 3 5 8 10). Samples whose ctxs < min(choices) are "
                        "capped to len(ctxs).")
    p.add_argument("--data_source_filter", nargs="+", default=None,
                   help="If given, keep only rows whose `data_source` field is in this set "
                        "(e.g. --data_source_filter nq).")
    p.add_argument("--val_ratio", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_per_input", type=int, default=-1, help="-1 means no cap")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    ds_filter = set(args.data_source_filter) if args.data_source_filter else None
    topk_choices = list(args.topk_choices) if args.topk_choices else None

    kept_by_ds: Dict[str, int] = {}
    all_rows: List[Dict[str, Any]] = []
    for path in args.input:
        print(f"[Load] {path}")
        samples = load_jsonl(path)
        if args.max_per_input > 0:
            samples = samples[: args.max_per_input]
        print(f"       {len(samples)} samples")
        for i, s in enumerate(samples):
            if ds_filter is not None and s.get("data_source") not in ds_filter:
                continue
            if topk_choices is not None:
                n_avail = len(s.get("ctxs") or [])
                if n_avail <= 0:
                    continue
                viable = [k for k in topk_choices if k <= n_avail] or [n_avail]
                tk = random.choice(viable)
            else:
                tk = args.topk
            row = row_from_sample(s, topk=tk, index=i, split="all")
            if row is not None:
                all_rows.append(row)
                kept_by_ds[s.get("data_source") or "unknown"] = kept_by_ds.get(
                    s.get("data_source") or "unknown", 0) + 1

    if kept_by_ds:
        print(f"[Kept by data_source] {kept_by_ds}")

    print(f"[Total rows] {len(all_rows)}")
    random.shuffle(all_rows)

    n_val = int(len(all_rows) * args.val_ratio) if args.val_ratio > 0 else 0
    val_rows = all_rows[:n_val]
    train_rows = all_rows[n_val:]
    for i, r in enumerate(train_rows):
        r["extra_info"]["index"] = i
        r["extra_info"]["split"] = "train"
    for i, r in enumerate(val_rows):
        r["extra_info"]["index"] = i
        r["extra_info"]["split"] = "val"

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(train_rows).to_parquet(out_dir / "train.parquet", index=False)
    print(f"[Saved] {out_dir / 'train.parquet'}   rows={len(train_rows)}")
    if val_rows:
        pd.DataFrame(val_rows).to_parquet(out_dir / "val.parquet", index=False)
        print(f"[Saved] {out_dir / 'val.parquet'}     rows={len(val_rows)}")

    preview = train_rows[0]
    print("\n=== Preview (first train row) ===")
    print(f"data_source  : {preview['data_source']}")
    print(f"ability      : {preview['ability']}")
    print(f"ground_truth : {preview['reward_model']['ground_truth']}")
    print(f"extra_info   : { {k: v for k, v in preview['extra_info'].items() if k != 'docs'} }")
    print(f"--- system ---\n{preview['prompt'][0]['content'][:200]}...")
    print(f"--- user ---\n{preview['prompt'][1]['content'][:600]}...")


if __name__ == "__main__":
    main()
