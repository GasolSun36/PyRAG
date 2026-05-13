#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

HOTPOTQA_JSONL="eval_data/hotpotqa.jsonl"
TRACE_JSONL="outputs/result.jsonl"
OUT_ROOT="verl_data"

NQ_INPUTS="train_data/nq.jsonl"

echo "=========================================================="
echo "  nq shards glob : ${NQ_SHARDS_GLOB}"
echo "  nq shard files : ${#NQ_INPUTS[@]}"
echo "  hotpotqa       : ${HOTPOTQA_JSONL}"
echo "  trace          : ${TRACE_JSONL}"
echo "  output root    : ${OUT_ROOT}"
echo "=========================================================="

mkdir -p "${OUT_ROOT}"

echo ""
echo "[1/4] answer_with_docs  (from NQ train, topk randomized in {3,5,8,10})"
python scripts/answer_with_docs.py \
    --input      "${NQ_INPUTS[@]}" \
    --data_source_filter nq \
    --topk_choices 3 5 8 10 \
    --output_dir "${OUT_ROOT}/answer_with_docs" \
    --val_ratio 0

echo ""
echo "[2/4] answer_no_docs    (from HotpotQA trace — only pipeline-correct synthesis steps)"
python scripts/answer_no_docs.py \
    --input      "${TRACE_JSONL}" \
    --output_dir "${OUT_ROOT}/answer_no_docs" \
    --keep_categories correct \
    --val_ratio 0

echo ""
echo "[3/4] plan              (from HotpotQA trace, sub_queries + gold)"
python scripts/plan.py \
    --input      "${TRACE_JSONL}" \
    --output_dir "${OUT_ROOT}/plan" \
    --val_ratio 0

echo ""
echo "[4/4] decompose         (from HotpotQA raw question + gold)"
python scripts/decompose.py \
    --input      "${HOTPOTQA_JSONL}" \
    --output_dir "${OUT_ROOT}/decompose" \
    --val_ratio 0

echo ""
echo "=========================================================="
echo "[Done] datasets (train-only, no val split):"
echo "  ${OUT_ROOT}/answer_with_docs/train.parquet"
echo "  ${OUT_ROOT}/answer_no_docs/train.parquet"
echo "  ${OUT_ROOT}/plan/train.parquet"
echo "  ${OUT_ROOT}/decompose/train.parquet"
echo "=========================================================="
