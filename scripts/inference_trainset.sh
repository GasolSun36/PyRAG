#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

INPUT="train_data/hotpotqa.jsonl"
OUTPUT_DIR="outputs"
TOPK="5"
RETRIEVAL_HOST="127.0.0.1"
RETRIEVAL_PORT="8008"

mkdir -p "${OUTPUT_DIR}"

TOTAL=$(wc -l < "${INPUT}")
if [[ "${TOTAL}" -le 0 ]]; then
    echo "[ERR] empty input: ${INPUT}" >&2
    exit 1
fi

OUT_FILE="${OUTPUT_DIR}/result.jsonl"
LOG_FILE="${OUTPUT_DIR}/run.log"

echo "=========================================================="
echo "  input      : ${INPUT}"
echo "  total rows : ${TOTAL}"
echo "  output     : ${OUT_FILE}"
echo "  log        : ${LOG_FILE}"
echo "  topk       : ${TOPK}"
echo "  retriever  : ${RETRIEVAL_HOST}:${RETRIEVAL_PORT}"
echo "=========================================================="

python scripts/eval.py \
    --input  "${INPUT}" \
    --output "${OUT_FILE}" \
    --start  0 \
    --end    "${TOTAL}" \
    --topk   "${TOPK}" \
    --retrieval_host "${RETRIEVAL_HOST}" \
    --retrieval_port "${RETRIEVAL_PORT}" \
    2>&1 | tee "${LOG_FILE}"

OUT_LINES=$(wc -l < "${OUT_FILE}")
echo ""
echo "=========================================================="
echo "[Done]  rows: ${OUT_LINES}  (expected: ${TOTAL})"
echo "=========================================================="
