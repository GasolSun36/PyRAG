from __future__ import annotations

import collections
import os
import re
import signal
import string
import sys
import threading
import traceback
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[3]
CODERAG_DIR = PROJECT_ROOT / "CodeRAG"
for import_dir in (CODERAG_DIR, THIS_DIR):
    if import_dir.exists() and str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))


MAX_FIX_ROUNDS = 3
_ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
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
    "not found",
)
_PRINTED_LOG_KEYS: set[str] = set()
_DEBUG_SAMPLE_COUNTER = 0


def _print_once(key: str, *args: Any, **kwargs: Any) -> None:
    if key in _PRINTED_LOG_KEYS:
        return
    _PRINTED_LOG_KEYS.add(key)
    print(*args, **kwargs)


def _log_exception_once(where: str, exc: BaseException, **context: Any) -> None:
    key = f"exception:{where}"
    if key in _PRINTED_LOG_KEYS:
        return
    _PRINTED_LOG_KEYS.add(key)
    print(
        f"[decompose_reward] exception at {where}: {type(exc).__name__}: {exc!r}",
        file=sys.stderr,
        flush=True,
    )
    if context:
        print(
            f"[decompose_reward] exception context at {where}: {context!r}",
            file=sys.stderr,
            flush=True,
        )
    traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)


def _debug_enabled() -> bool:
    return os.environ.get("CODERAG_PLAN_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}


def _debug_max_chars() -> int:
    return _env_int("CODERAG_PLAN_DEBUG_MAX_CHARS", 1000)


def _short(value: Any, max_chars: Optional[int] = None) -> str:
    text = repr(value)
    limit = _debug_max_chars() if max_chars is None else max_chars
    if limit > 0 and len(text) > limit:
        return text[:limit] + f"...<truncated {len(text) - limit} chars>"
    return text


def _debug_print(sample_id: int, message: str) -> None:
    if not _debug_enabled():
        return
    print(f"[decompose_reward][sample={sample_id}] {message}", flush=True)


class _PlanExecTimeout(Exception):
    pass


class _CallBudgetExceeded(Exception):
    pass


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, "").strip() or default)
    except ValueError:
        return default


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


def _as_list(ground_truth: Any) -> List[str]:
    if ground_truth is None:
        return []
    if isinstance(ground_truth, list):
        return [str(x) for x in ground_truth if x is not None and str(x).strip()]
    return [str(ground_truth)] if str(ground_truth).strip() else []


def _extract_answer(text: Any) -> str:
    if text is None:
        return ""
    s = text if isinstance(text, str) else str(text)
    m = _ANSWER_TAG_RE.search(s)
    return m.group(1).strip() if m else s.strip()


def _parse_sub_queries(solution_str: str) -> Optional[List[str]]:
    from utils import extract_json_block

    try:
        obj = extract_json_block(solution_str or "")
    except Exception:
        return None
    if not isinstance(obj, list):
        return None
    if not obj or not all(isinstance(x, str) and x.strip() for x in obj):
        return None
    return [x.strip() for x in obj]


def _answer_indicates_insufficient_info(text: str) -> bool:
    if not text or not isinstance(text, str):
        return False
    t = text.strip().lower()
    return any(marker.lower() in t for marker in _INSUFFICIENT_ANSWER_MARKERS)


def _retrieve_indices_for_insufficient_answers(execution_log: list) -> set[int]:
    last_retrieve_idx = 0
    out: set[int] = set()
    for entry in execution_log:
        if entry.get("type") == "retrieve":
            last_retrieve_idx += 1
        elif entry.get("type") == "answer":
            if _answer_indicates_insufficient_info(entry.get("answer_returned", "")):
                if last_retrieve_idx > 0:
                    out.add(last_retrieve_idx)
    return out


def _topk_used_at_retrieve_index(execution_log: list, retrieve_index: int) -> Optional[int]:
    count = 0
    for entry in execution_log:
        if entry.get("type") == "retrieve":
            count += 1
            if count == retrieve_index:
                return entry.get("topk")
    return None


def _last_retrieve_index(execution_log: list) -> int:
    return sum(1 for e in execution_log if e.get("type") == "retrieve")


def _build_retrieve_topk_boost(execution_log: list, target_topk: int = 10) -> Dict[int, int]:
    boost: Dict[int, int] = {}
    for idx in _retrieve_indices_for_insufficient_answers(execution_log):
        used = _topk_used_at_retrieve_index(execution_log, idx)
        if used is not None and used < target_topk:
            boost[idx] = target_topk
    return boost


def _signal_timeout(seconds: int):
    class _Timer:
        def __enter__(self_inner):
            if threading.current_thread() is threading.main_thread():
                def _handler(signum, frame):
                    raise _PlanExecTimeout(f"exec exceeded {seconds}s")

                self_inner._prev = signal.signal(signal.SIGALRM, _handler)
                signal.alarm(seconds)
            else:
                self_inner._prev = None
            return self_inner

        def __exit__(self_inner, *exc):
            if self_inner._prev is not None:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, self_inner._prev)
            return False

    return _Timer()


@lru_cache(maxsize=1)
def _get_plan_agent():
    from llm import OpenAILLM, env_enable_thinking
    from plan_agent import PlanAgent

    plan_url = os.environ.get(
        "CODERAG_REWARD_PLAN_URL",
        os.environ.get("CODERAG_FROZEN_PLAN_URL", "http://127.0.0.1:8346/v1"),
    )
    plan_model = os.environ.get(
        "CODERAG_REWARD_PLAN_MODEL",
        os.environ.get(
            "CODERAG_FROZEN_PLAN_MODEL",
            "Qwen/Qwen2.5-7B-Instruct",
        ),
    )
    llm = OpenAILLM(model=plan_model, base_url=plan_url, enable_thinking=env_enable_thinking())
    return PlanAgent(llm=llm, max_retries=3)


@lru_cache(maxsize=1)
def _get_answer_llm_and_retriever():
    from llm import OpenAILLM, env_enable_thinking
    from retrieval_agent import HttpRetrievalAgent

    answer_url = os.environ.get(
        "CODERAG_REWARD_ANSWER_URL",
        os.environ.get("CODERAG_FROZEN_ANSWER_URL", "http://127.0.0.1:8338/v1"),
    )
    answer_model = os.environ.get(
        "CODERAG_REWARD_ANSWER_MODEL",
        os.environ.get(
            "CODERAG_FROZEN_ANSWER_MODEL",
            "Qwen/Qwen2.5-7B-Instruct",
        ),
    )
    retr_host = os.environ.get("CODERAG_RETRIEVAL_HOST", "127.0.0.1")
    retr_port = _env_int("CODERAG_RETRIEVAL_PORT", 8008)
    answer_llm = OpenAILLM(model=answer_model, base_url=answer_url, enable_thinking=env_enable_thinking())
    retriever = HttpRetrievalAgent(host=retr_host, port=retr_port, timeout=15)
    return answer_llm, retriever


@lru_cache(maxsize=20000)
def _cached_plan_generate(question: str, sub_queries_tuple: tuple[str, ...]) -> str:
    sub_queries = list(sub_queries_tuple)
    try:
        return _get_plan_agent().generate_code(question, sub_queries)
    except Exception as exc:
        _log_exception_once(
            "plan_agent.generate_code",
            exc,
            question=question,
            sub_queries=sub_queries,
        )
        raise


def _build_budgeted_tools(
    default_topk: int,
    retrieve_topk_boost: Optional[Dict[int, int]] = None,
    sample_id: int = 0,
):
    from tools import make_tools

    answer_llm, retriever = _get_answer_llm_and_retriever()
    retrieve_fn, answer_fn, execution_log = make_tools(
        retriever,
        answer_llm,
        default_topk=default_topk,
        retrieve_topk_boost=retrieve_topk_boost,
    )
    max_calls = _env_int("CODERAG_PLAN_MAX_CALLS", 20)
    calls = {"n": 0}

    def _tick():
        calls["n"] += 1
        if calls["n"] > max_calls:
            raise _CallBudgetExceeded(f"exceeded {max_calls} retrieve/answer calls")

    def retrieve(query: str, topk: int = default_topk):
        _tick()
        query_str = str(query)
        _debug_print(
            sample_id,
            f"step={len(execution_log) + 1} retrieve call begin: query={_short(query_str)} topk={int(topk)}",
        )
        try:
            docs = retrieve_fn(query_str, topk=topk)
        except Exception as exc:
            _log_exception_once("retriever.retrieve", exc, query=query_str, topk=topk)
            raise
        _debug_print(
            sample_id,
            f"step={len(execution_log)} retrieve call end: num_docs={len(docs) if docs is not None else 0} docs={_short(docs)}",
        )
        return docs

    def answer(query: str, docs: Optional[List[str]] = None):
        _tick()
        query_str = str(query)
        docs_list = list(docs) if docs else []
        _debug_print(
            sample_id,
            f"step={len(execution_log) + 1} answer call begin: query={_short(query_str)} num_docs={len(docs_list)} docs={_short(docs_list)}",
        )
        try:
            returned = answer_fn(query_str, docs_list)
        except Exception as exc:
            _log_exception_once("answer_llm.generate", exc, query=query_str, docs=docs_list)
            raise
        _debug_print(sample_id, f"step={len(execution_log)} answer call end: answer={_short(returned)}")
        return returned

    return retrieve, answer, execution_log, calls


def _execute_code_once(
    code: str,
    topk: int,
    retrieve_topk_boost: Optional[Dict[int, int]] = None,
    sample_id: int = 0,
) -> Tuple[Dict[str, Any], str]:
    timeout = _env_int("CODERAG_PLAN_EXEC_TIMEOUT", 30)
    retrieve_fn, answer_fn, execution_log, calls = _build_budgeted_tools(
        default_topk=topk,
        retrieve_topk_boost=retrieve_topk_boost,
        sample_id=sample_id,
    )
    namespace: Dict[str, Any] = {
        "retrieve": retrieve_fn,
        "answer": answer_fn,
    }
    try:
        compiled = compile(code, "<decompose-plan-rollout>", "exec")
        _debug_print(
            sample_id,
            f"exec begin: topk={topk} timeout={timeout} retrieve_topk_boost={retrieve_topk_boost!r}",
        )
        with _signal_timeout(timeout):
            exec(compiled, namespace)  # noqa: S102
    except (_PlanExecTimeout, _CallBudgetExceeded):
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Code execution failed ({type(exc).__name__}: {exc})\n--- Generated code ---\n{code}"
        ) from exc

    variables = {
        k: v
        for k, v in namespace.items()
        if not k.startswith("__") and k not in ("retrieve", "answer")
    }
    _debug_print(
        sample_id,
        f"exec end: num_calls={calls['n']} final_answer={_short(namespace.get('final_answer'))} "
        f"variables={_short(variables)} execution_log={_short(execution_log)}",
    )
    return {
        "final_answer": namespace.get("final_answer", ""),
        "variables": variables,
        "execution_log": execution_log,
        "num_calls": calls["n"],
    }, code


def _execute_code_with_fixes(
    question: str,
    code: str,
    topk: int,
    boost: Optional[Dict[int, int]] = None,
    sample_id: int = 0,
):
    last_error = ""
    for fix_round in range(MAX_FIX_ROUNDS + 1):
        try:
            _debug_print(sample_id, f"execute code attempt begin: fix_round={fix_round} code={_short(code)}")
            return _execute_code_once(code, topk=topk, retrieve_topk_boost=boost, sample_id=sample_id)
        except (_PlanExecTimeout, _CallBudgetExceeded):
            raise
        except RuntimeError as exc:
            last_error = str(exc)
            _debug_print(sample_id, f"execute code attempt failed: fix_round={fix_round} error={_short(last_error)}")
            if fix_round == MAX_FIX_ROUNDS:
                raise
            try:
                _debug_print(sample_id, f"fix_code begin: fix_round={fix_round}")
                code = _get_plan_agent().fix_code(
                    original_query=question,
                    failed_code=code,
                    error_msg=last_error,
                )
                _debug_print(sample_id, f"fix_code end: fix_round={fix_round} fixed_code={_short(code)}")
            except Exception as fix_exc:
                _log_exception_once(
                    "plan_agent.fix_code",
                    fix_exc,
                    question=question,
                    failed_code=code,
                    error_msg=last_error,
                    fix_round=fix_round,
                )
                raise
    raise RuntimeError(f"exhausted fix rounds: {last_error}")


def _run_pipeline_from_subqueries(question: str, sub_queries: List[str], topk: int, sample_id: int = 0):
    _debug_print(
        sample_id,
        f"plan generate begin: question={_short(question)} sub_queries={_short(sub_queries)}",
    )
    code = _cached_plan_generate(question, tuple(sub_queries))
    _debug_print(sample_id, f"plan generate end: generated_code={_short(code)}")
    result, code = _execute_code_with_fixes(question, code, topk=topk, sample_id=sample_id)

    retried_with_topk10 = False
    boost = _build_retrieve_topk_boost(result.get("execution_log", []))
    _debug_print(sample_id, f"first execution result={_short(result)} topk10_boost_candidates={boost!r}")
    if not boost and _answer_indicates_insufficient_info(str(result.get("final_answer", ""))):
        last_r = _last_retrieve_index(result.get("execution_log", []))
        if last_r > 0:
            used = _topk_used_at_retrieve_index(result.get("execution_log", []), last_r)
            if used is not None and used < 10:
                boost[last_r] = 10

    if boost:
        _debug_print(sample_id, f"retry with topk10 boost={boost!r}")
        result, code = _execute_code_with_fixes(question, code, topk=topk, boost=boost, sample_id=sample_id)
        retried_with_topk10 = True
        _debug_print(sample_id, f"retry execution result={_short(result)}")

    result["retried_with_topk10"] = retried_with_topk10
    result["generated_code"] = code
    return result


def _zero(err_type: str, pred: str = "", num_calls: int = 0) -> Dict[str, Any]:
    return {
        "score": 0.0,
        "f1": 0.0,
        "em": 0.0,
        "pred": pred,
        "err_type": err_type,
        "num_calls": float(num_calls),
        "retried_with_topk10": 0.0,
        "num_sub_queries": 0.0,
    }


def score_decompose_em_f1(
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[Dict[str, Any]] = None,
    sample_id: int = 0,
) -> Dict[str, Any]:
    extra_info = extra_info or {}
    question = (extra_info.get("question") or "").strip()
    topk = int(extra_info.get("topk", 5))
    _debug_print(
        sample_id,
        f"score begin: question={_short(question)} topk={topk} ground_truth={_short(ground_truth)} extra_info={_short(extra_info)}",
    )
    _debug_print(sample_id, f"raw rollout solution={_short(solution_str)}")
    if not question:
        _debug_print(sample_id, "return zero: missing_question")
        return _zero("missing_question")

    sub_queries = _parse_sub_queries(solution_str)
    if sub_queries is None:
        _debug_print(sample_id, "return zero: bad_subquery_json")
        return _zero("bad_subquery_json")
    _debug_print(sample_id, f"parsed sub_queries={_short(sub_queries)}")

    try:
        result = _run_pipeline_from_subqueries(question, sub_queries, topk=topk, sample_id=sample_id)
    except _PlanExecTimeout as exc:
        _log_exception_once(
            "run decompose pipeline timeout",
            exc,
            question=question,
            sub_queries=sub_queries,
            topk=topk,
        )
        out = _zero("timeout")
        out["num_sub_queries"] = float(len(sub_queries))
        _debug_print(sample_id, f"return zero: timeout output={out!r}")
        return out
    except _CallBudgetExceeded as exc:
        _log_exception_once(
            "run decompose pipeline call_budget",
            exc,
            question=question,
            sub_queries=sub_queries,
            topk=topk,
        )
        out = _zero("call_budget")
        out["num_sub_queries"] = float(len(sub_queries))
        _debug_print(sample_id, f"return zero: call_budget output={out!r}")
        return out
    except Exception as exc:
        _log_exception_once(
            "run decompose pipeline error",
            exc,
            question=question,
            sub_queries=sub_queries,
            topk=topk,
        )
        out = _zero("pipeline_error")
        out["num_sub_queries"] = float(len(sub_queries))
        _debug_print(sample_id, f"return zero: pipeline_error output={out!r}")
        return out

    pred = _extract_answer(result.get("final_answer"))
    golds = _as_list(ground_truth)
    _debug_print(sample_id, f"pipeline result={_short(result)}")
    _debug_print(sample_id, f"parsed prediction={_short(pred)} golds={_short(golds)}")
    if not pred or not golds:
        out = _zero("empty_pred_or_gold", pred=pred, num_calls=int(result.get("num_calls", 0)))
        out["num_sub_queries"] = float(len(sub_queries))
        out["retried_with_topk10"] = 1.0 if result.get("retried_with_topk10") else 0.0
        _debug_print(sample_id, f"return zero: empty_pred_or_gold output={out!r}")
        return out

    max_f1 = max(_f1_score(pred, gt) for gt in golds)
    max_em = max(_em_score(pred, gt) for gt in golds)
    score = 0.7 * max_f1 + 0.3 * max_em
    _debug_print(sample_id, f"score end: pred={_short(pred)} max_f1={max_f1} max_em={max_em} score={score}")

    return {
        "score": score,
        "f1": max_f1,
        "em": max_em,
        "pred": pred,
        "err_type": "ok",
        "num_calls": float(result.get("num_calls", 0)),
        "retried_with_topk10": 1.0 if result.get("retried_with_topk10") else 0.0,
        "num_sub_queries": float(len(sub_queries)),
    }


def compute_score(
    *args: Any,
    extra_info: Optional[Dict[str, Any]] = None,
    data_source: str = "coderag_decompose",
    **kwargs: Any,
) -> Dict[str, Any]:
    global _DEBUG_SAMPLE_COUNTER
    _DEBUG_SAMPLE_COUNTER += 1
    sample_id = _DEBUG_SAMPLE_COUNTER
    solution_str = kwargs.pop("solution_str", None)
    ground_truth = kwargs.pop("ground_truth", None)
    if len(args) == 2:
        solution_str, ground_truth = args
    elif len(args) == 3:
        if isinstance(args[2], dict) or args[2] is None:
            solution_str, ground_truth, extra_info = args
        else:
            data_source, solution_str, ground_truth = args
    elif len(args) == 4:
        data_source, solution_str, ground_truth, extra_info = args
    elif args:
        raise TypeError(f"compute_score expected 2 to 4 positional arguments, got {len(args)}")
    if solution_str is None or ground_truth is None:
        raise TypeError("compute_score requires solution_str and ground_truth")

    _debug_print(
        sample_id,
        "compute_score input: "
        f"data_source={data_source!r}, "
        f"solution_str={_short(solution_str)}, "
        f"ground_truth={_short(ground_truth)}, "
        f"extra_info={_short(extra_info)}, "
        f"kwargs={_short(kwargs)}",
    )
    if data_source != "coderag_decompose":
        output = _zero(f"unsupported_data_source:{data_source}")
        _debug_print(sample_id, f"return zero: unsupported_data_source data_source={data_source!r}")
    else:
        output = score_decompose_em_f1(solution_str, ground_truth, extra_info, sample_id=sample_id)
    _debug_print(sample_id, f"compute_score output: {output!r}")
    return output

