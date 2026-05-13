import logging
from typing import Any, Dict, Optional, Set, Tuple

from pyrag.code_executor import CodeExecutor
from pyrag.decompose_agent import DecomposeAgent
from pyrag.llm import OpenAILLM
from pyrag.plan_agent import PlanAgent
from pyrag.retrieval_agent import RetrievalAgent
from pyrag.tools import make_tools

logger = logging.getLogger(__name__)

MAX_FIX_ROUNDS = 3

_INSUFFICIENT_ANSWER_MARKERS = (
    "not enough information",
    "insufficient information",
    "no enough information",
    "cannot answer",
    "unable to answer",
    "信息不足",
    "无法根据",
    "无法回答",
    "unknown",
    "Unknown",
    "not found",
)


def _answer_indicates_insufficient_info(text: str) -> bool:
    if not text or not isinstance(text, str):
        return False
    t = text.strip().lower()
    return any(m.lower() in t for m in _INSUFFICIENT_ANSWER_MARKERS)


def _retrieve_indices_for_insufficient_answers(execution_log: list) -> Set[int]:
    last_retrieve_idx = 0
    out: Set[int] = set()
    for entry in execution_log:
        if entry["type"] == "retrieve":
            last_retrieve_idx += 1
        elif entry["type"] == "answer":
            if _answer_indicates_insufficient_info(entry.get("answer_returned", "")):
                if last_retrieve_idx > 0:
                    out.add(last_retrieve_idx)
    return out


def _topk_used_at_retrieve_index(execution_log: list, retrieve_index: int) -> Optional[int]:
    count = 0
    for entry in execution_log:
        if entry["type"] == "retrieve":
            count += 1
            if count == retrieve_index:
                return entry.get("topk")
    return None


def _last_retrieve_index(execution_log: list) -> int:
    return sum(1 for e in execution_log if e["type"] == "retrieve")


def _build_retrieve_topk_boost(
    execution_log: list,
    target_topk: int = 10,
) -> Dict[int, int]:
    indices = _retrieve_indices_for_insufficient_answers(execution_log)
    boost: Dict[int, int] = {}
    for idx in indices:
        used = _topk_used_at_retrieve_index(execution_log, idx)
        if used is not None and used < target_topk:
            boost[idx] = target_topk
    return boost


class RAGProgramRunner:
    def __init__(
        self,
        llm: OpenAILLM,
        retrieval_agent: RetrievalAgent,
        plan_llm: Optional[OpenAILLM] = None,
    ):
        self.llm = llm
        self.plan_llm = plan_llm if plan_llm is not None else llm
        self.retrieval_agent = retrieval_agent
        self.decompose_agent = DecomposeAgent(llm)
        self.plan_agent = PlanAgent(self.plan_llm)
        self.executor = CodeExecutor()

    def _print_trace(self, result: Dict[str, Any]) -> None:
        print("\n=== Execution Trace ===")
        for entry in result["execution_log"]:
            if entry["type"] == "retrieve":
                print(f"[Step {entry['step']}] retrieve({entry['query']!r}, topk={entry['topk']})")
                for i, doc in enumerate(entry["docs"], 1):
                    print(f"    doc{i}: {doc[:80]}{'...' if len(doc) > 80 else ''}")
            else:
                print(f"[Step {entry['step']}] answer({entry['query']!r})")
                raw = entry.get("answer_raw", "")
                out = entry.get("answer_returned", entry.get("answer", ""))
                if raw:
                    print(f"    raw: {raw}")
                print(f"    → {out}")

    def _execute_code_with_fixes(
        self,
        query: str,
        code: str,
        topk: int,
        retrieve_topk_boost: Optional[Dict[int, int]] = None,
    ) -> Tuple[Dict[str, Any], str]:
        for fix_round in range(MAX_FIX_ROUNDS + 1):
            retrieve_fn, answer_fn, execution_log = make_tools(
                self.retrieval_agent,
                self.llm,
                default_topk=topk,
                retrieve_topk_boost=retrieve_topk_boost,
            )
            try:
                result = self.executor.execute(code, retrieve_fn, answer_fn, execution_log)
                return result, code
            except RuntimeError as e:
                error_msg = str(e)
                if fix_round == MAX_FIX_ROUNDS:
                    raise
                logger.warning(
                    "Execution failed (round %d/%d), requesting fix from LLM.\nError: %s",
                    fix_round + 1, MAX_FIX_ROUNDS, error_msg,
                )
                print(
                    f"\n[Fix round {fix_round + 1}/{MAX_FIX_ROUNDS}] "
                    f"Execution error, asking LLM to fix...\n  {error_msg.splitlines()[0]}"
                )
                code = self.plan_agent.fix_code(
                    original_query=query,
                    failed_code=code,
                    error_msg=error_msg,
                )
                print("=== Fixed Code ===")
                print(code)
        raise RuntimeError("_execute_code_with_fixes: exhausted fix rounds without return")

    def run(self, query: str, topk: int = 5) -> Dict[str, Any]:
        sub_queries = self.decompose_agent.decompose(query)
        print("\n=== Sub-queries ===")
        for i, q in enumerate(sub_queries, 1):
            print(f"  {i}. {q}")

        code = self.plan_agent.generate_code(query, sub_queries)
        print("\n=== Generated Code ===")
        print(code)

        result, code = self._execute_code_with_fixes(query, code, topk)

        retried_with_topk10 = False
        boost = _build_retrieve_topk_boost(result["execution_log"])
        if not boost and _answer_indicates_insufficient_info(
            str(result.get("final_answer", ""))
        ):
            last_r = _last_retrieve_index(result["execution_log"])
            if last_r > 0:
                used = _topk_used_at_retrieve_index(result["execution_log"], last_r)
                if used is not None and used < 10:
                    boost[last_r] = 10

        if boost:
            print(
                "\n[Retry] Insufficient-information answer at some step(s); "
                f"re-executing the same code with retrieve topk boost: {boost}"
            )
            result, code = self._execute_code_with_fixes(
                query, code, topk, retrieve_topk_boost=boost
            )
            retried_with_topk10 = True

        self._print_trace(result)

        print("\n=== Final Answer ===")
        print(result["final_answer"])

        return {
            "original_query": query,
            "sub_queries": sub_queries,
            "generated_code": code,
            "execution_log": result["execution_log"],
            "variables": result["variables"],
            "final_answer": result["final_answer"],
            "retried_with_topk10": retried_with_topk10,
        }
