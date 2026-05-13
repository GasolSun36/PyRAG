import argparse
import json
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm

from pyrag import HttpRetrievalAgent, OpenAILLM, RAGProgramRunner, env_enable_thinking


def parse_args():
    p = argparse.ArgumentParser(description="Batch eval on HotpotQA with CodeRAG")
    p.add_argument("--input",  default="eval_data/hotpotqa.jsonl")
    p.add_argument("--output", default="./outputs/hotpotqa_eval.jsonl")
    p.add_argument("--topk",   type=int, default=5)
    p.add_argument("--start",  type=int, default=0,  help="first sample index (inclusive)")
    p.add_argument("--end",    type=int, default=-1, help="last sample index (exclusive), -1 = all")
    p.add_argument(
        "--llm_model",
        default=os.environ.get(
            "LLM_MODEL",
            "Qwen/Qwen2.5-7B-Instruct",
        ),
        help="Instruct model: decompose + answer() QA",
    )
    p.add_argument(
        "--llm_base_url",
        default=os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8337/v1"),
        help="OpenAI-compatible base URL for instruct model",
    )
    p.add_argument(
        "--plan_llm_model",
        default=os.environ.get(
            "PLAN_LLM_MODEL",
            "Qwen/Qwen2.5-7B-Instruct",
        ),
        help="Code model: generate_code / fix_code only",
    )
    p.add_argument(
        "--plan_llm_base_url",
        default=os.environ.get("PLAN_LLM_BASE_URL", "http://127.0.0.1:8336/v1"),
        help="OpenAI-compatible base URL for plan (code) model",
    )
    p.add_argument("--retrieval_host", default="127.0.0.1")
    p.add_argument("--retrieval_port", type=int, default=8008)
    return p.parse_args()


def load_done_ids(output_path: str) -> set:
    done = set()
    if not os.path.exists(output_path):
        return done
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "id" in obj:
                    done.add(obj["id"])
            except json.JSONDecodeError:
                pass
    return done


def main():
    args = parse_args()
    print(f"[Info] instruct LLM (decompose + answer): {args.llm_model}\n       {args.llm_base_url}")
    print(f"[Info] plan LLM (generate/fix code):      {args.plan_llm_model}\n       {args.plan_llm_base_url}")

    instruct_llm = OpenAILLM(
        model=args.llm_model,
        base_url=args.llm_base_url,
        enable_thinking=env_enable_thinking(),
    )
    plan_llm = OpenAILLM(
        model=args.plan_llm_model,
        base_url=args.plan_llm_base_url,
        enable_thinking=env_enable_thinking(),
    )
    retrieval_agent = HttpRetrievalAgent(host=args.retrieval_host, port=args.retrieval_port)
    runner = RAGProgramRunner(
        llm=instruct_llm,
        plan_llm=plan_llm,
        retrieval_agent=retrieval_agent,
    )

    samples = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    end = len(samples) if args.end == -1 else args.end
    samples = samples[args.start:end]
    total = len(samples)
    print(f"[Info] Loaded {total} samples (index {args.start}~{end - 1 if args.end != -1 else len(samples) - 1 + args.start})")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    done_ids = load_done_ids(args.output)
    if done_ids:
        print(f"[Resume] {len(done_ids)} samples already done, skipping them.")

    out_f = open(args.output, "a", encoding="utf-8")

    success = 0
    fail    = 0

    try:
        pbar = tqdm(samples, total=total, desc="Evaluating", dynamic_ncols=True)
        for idx, sample in enumerate(pbar):
            sid      = sample.get("id", str(idx + args.start))
            question = sample["question"]
            gold     = sample.get("answers") or sample.get("golden_answers") or []

            if sid in done_ids:
                continue

            pbar.set_postfix({"success": success, "fail": fail, "id": str(sid)[:20]})

            record = {
                "id":              sid,
                "question":        question,
                "gold_answers":    gold,
                "pred_answer":     None,
                "sub_queries":     None,
                "generated_code":  None,
                "execution_log":   [],
                "error":           None,
            }

            try:
                result = runner.run(question, topk=args.topk)
                record["pred_answer"]    = result.get("final_answer", "")
                record["sub_queries"]    = result.get("sub_queries")
                record["generated_code"] = result.get("generated_code")
                record["execution_log"]  = result.get("execution_log", [])
                success += 1
                tqdm.write(f"  -> pred: {record['pred_answer']}  |  gold: {gold}")
            except Exception as e:
                record["error"] = traceback.format_exc()
                fail += 1
                tqdm.write(f"  [ERROR] {e}", file=sys.stderr)

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

    finally:
        out_f.close()

    print(f"\n[Done] success={success}  fail={fail}  output={args.output}")


if __name__ == "__main__":
    main()
