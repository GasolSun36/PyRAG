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

from pyrag.decompose_agent import SYSTEM_PROMPT as DECOMPOSE_SYSTEM_PROMPT
from pyrag.decompose_agent import build_decompose_user_prompt

DATA_SOURCE = "pyrag_decompose"

def row_from_sample(sample: Dict[str, Any], index: int) -> Optional[Dict[str, Any]]:
    question = (sample.get("question") or "").strip()
    gold = list(
        sample.get("gold_answers")
        or sample.get("golden_answers")
        or sample.get("answers")
        or []
    )
    if not question or not gold:
        return None
    prompt = [
        {"role": "system", "content": DECOMPOSE_SYSTEM_PROMPT},
        {"role": "user",   "content": build_decompose_user_prompt(question)},
    ]
    return {
        "data_source": DATA_SOURCE,
        "prompt": prompt,
        "ability": "decompose",
        "reward_model": {"style": "rule", "ground_truth": gold},
        "extra_info": {
            "index": index,
            "split": "all",
            "src_id": sample.get("id"),
            "question": question,
            "topk": 5,
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
                   help="HotpotQA-like jsonl with question + answers fields")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--val_ratio", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    rows: List[Dict[str, Any]] = []
    for path in args.input:
        print(f"[Load] {path}")
        samples = load_jsonl(path)
        print(f"       {len(samples)} samples")
        for s in samples:
            r = row_from_sample(s, index=len(rows))
            if r is not None:
                rows.append(r)

    print(f"[Kept] {len(rows)}")
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
    print(f"[Saved] {out_dir / 'train.parquet'}   rows={len(train_rows)}")
    if val_rows:
        pd.DataFrame(val_rows).to_parquet(out_dir / "val.parquet", index=False)
        print(f"[Saved] {out_dir / 'val.parquet'}     rows={len(val_rows)}")


if __name__ == "__main__":
    main()
