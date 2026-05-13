import json
import os

from pyrag import HttpRetrievalAgent, OpenAILLM, RAGProgramRunner, env_enable_thinking

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./outputs")


def _make_run_dir(output_dir: str) -> str:
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, ts)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _format_execution_log(execution_log: list) -> str:
    lines = []
    for entry in execution_log:
        lines.append(f"{'='*60}")
        lines.append(f"[Step {entry['step']}]  type={entry['type']}")
        lines.append(f"  query : {entry['query']}")
        if entry["type"] == "retrieve":
            lines.append(f"  topk  : {entry['topk']}")
            for i, doc in enumerate(entry["docs"], 1):
                lines.append(f"  doc{i}  : {doc}")
        else:
            short = entry.get("answer_returned", entry.get("answer", ""))
            raw = entry.get("answer_raw", entry.get("answer", ""))
            lines.append(f"  returned (chained): {short}")
            if raw and raw.strip() != short.strip():
                lines.append(f"  raw model output: {raw}")
            for i, doc in enumerate(entry["docs"], 1):
                lines.append(f"  doc{i}  : {doc}")
        lines.append("")
    return "\n".join(lines)


def save_result(result: dict, output_dir: str) -> None:
    run_dir = _make_run_dir(output_dir)

    code_path = os.path.join(run_dir, "generated_code.py")
    with open(code_path, "w", encoding="utf-8") as f:
        f.write(result["generated_code"])
    print(f"[Saved] generated code  → {code_path}")

    log_path = os.path.join(run_dir, "execution_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(_format_execution_log(result["execution_log"]))
    print(f"[Saved] execution log   → {log_path}")

    result_path = os.path.join(run_dir, "result.json")
    payload = {k: v for k, v in result.items() if k != "generated_code"}
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[Saved] result          → {result_path}")


def main():
    instruct_llm = OpenAILLM(
        model=os.environ.get(
            "LLM_MODEL",
            "Qwen2.5-7B-Instruct",
        ),
        base_url=os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8337/v1"),
        enable_thinking=env_enable_thinking(),
    )
    plan_llm = OpenAILLM(
        model=os.environ.get(
            "PLAN_LLM_MODEL",
            "Qwen2.5-7B-Instruct",
        ),
        base_url=os.environ.get("PLAN_LLM_BASE_URL", "http://127.0.0.1:8336/v1"),
        enable_thinking=env_enable_thinking(),
    )

    retrieval_agent = HttpRetrievalAgent(host="127.0.0.1", port=8008)

    runner = RAGProgramRunner(
        llm=instruct_llm,
        plan_llm=plan_llm,
        retrieval_agent=retrieval_agent,
    )

    query = "Who is older, Jed Hoyer or John William Henry II?"
    result = runner.run(query, topk=5)

    save_result(result, OUTPUT_DIR)

    print("\n=== Final Answer ===")
    print(result["final_answer"])


if __name__ == "__main__":
    main()
