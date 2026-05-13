from __future__ import annotations

import builtins as _builtins
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

# Modules that are dangerous to allow inside rollout code execution sandbox
_BLOCKED_IMPORT_MODULES = frozenset({
    "os", "sys", "subprocess", "socket", "signal", "shutil",
    "pathlib", "ctypes", "multiprocessing", "threading",
    "pty", "tty", "termios", "fcntl", "resource", "importlib",
})


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    top = name.split(".")[0]
    if top in _BLOCKED_IMPORT_MODULES:
        raise ImportError(f"import of '{name}' is blocked in plan rollout sandbox")
    return _builtins.__import__(name, globals, locals, fromlist, level)


_SANDBOX_BUILTINS = {**vars(_builtins), "__import__": _safe_import}


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[3]
CODERAG_DIR = PROJECT_ROOT / "CodeRAG"
for import_dir in (CODERAG_DIR, THIS_DIR):
    if import_dir.exists() and str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))


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
        f"[plan_reward] exception at {where}: {type(exc).__name__}: {exc!r}",
        file=sys.stderr,
        flush=True,
    )
    if context:
        print(f"[plan_reward] exception context at {where}: {context!r}", file=sys.stderr, flush=True)
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
    print(f"[plan_reward][sample={sample_id}] {message}", flush=True)


def _reward_disable_proxy() -> bool:
    return os.environ.get("CODERAG_REWARD_DISABLE_PROXY", "1").strip().lower() not in {"0", "false", "no", "off"}


def _disable_proxy_env_for_reward_services() -> None:
    if not _reward_disable_proxy():
        return
    removed = {key: os.environ.pop(key) for key in _PROXY_ENV_KEYS if key in os.environ}
    os.environ.setdefault("NO_PROXY", "*")
    os.environ.setdefault("no_proxy", "*")
    if removed:
        _print_once(
            "disabled_proxy_env",
            f"[plan_reward] disabled proxy env for reward services: {sorted(removed)}",
            flush=True,
        )


class _PlanExecTimeout(Exception):
    pass


class _PlanCallBudgetExceeded(Exception):
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


def _extract_answer(text: Any) -> str:
    if text is None:
        return ""
    s = text if isinstance(text, str) else str(text)
    m = _ANSWER_TAG_RE.search(s)
    return m.group(1).strip() if m else s.strip()


def _as_list(ground_truth: Any) -> List[str]:
    if ground_truth is None:
        return []
    if isinstance(ground_truth, list):
        return [str(x) for x in ground_truth if x is not None and str(x).strip()]
    return [str(ground_truth)] if str(ground_truth).strip() else []


def _extract_python_code(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"```python\s*(.*?)\s*```", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


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
def _get_answer_and_retriever():
    _disable_proxy_env_for_reward_services()

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

    answer_llm = OpenAILLM(
        model=answer_model,
        base_url=answer_url,
        enable_thinking=env_enable_thinking(),
    )
    retriever = HttpRetrievalAgent(host=retr_host, port=retr_port, timeout=15)
    return answer_llm, retriever


@lru_cache(maxsize=20000)
def _cached_retrieve(query: str, topk: int) -> tuple[str, ...]:
    _, retriever = _get_answer_and_retriever()
    try:
        return tuple(retriever.retrieve(query, topk=topk))
    except Exception as exc:
        _log_exception_once("retriever.retrieve", exc, query=query, topk=topk)
        raise


@lru_cache(maxsize=20000)
def _cached_answer(query: str, docs_key: tuple[str, ...]) -> str:
    from tools import answer_system_prompt_for_docs
    from utils import extract_answer_tag, format_docs_for_prompt

    answer_llm, _ = _get_answer_and_retriever()
    docs = list(docs_key)
    system_prompt = answer_system_prompt_for_docs(docs)
    user_prompt = (
        "=== QUESTION ===\n"
        f"{query}\n"
        "=== END QUESTION ===\n\n"
        "=== RETRIEVED DOCUMENTS ===\n"
        f"{format_docs_for_prompt(docs)}\n"
        "=== END DOCUMENTS ==="
    )
    try:
        raw = answer_llm.generate(system_prompt, user_prompt)
    except Exception as exc:
        _log_exception_once(
            "answer_llm.generate",
            exc,
            query=query,
            docs=docs,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        raise
    try:
        return extract_answer_tag(raw)
    except Exception as exc:
        _log_exception_once("extract_answer_tag", exc, query=query, raw=raw)
        raise


def _build_tools(
    default_topk: int,
    retrieve_topk_boost: Optional[Dict[int, int]] = None,
    sample_id: int = 0,
):
    max_calls = _env_int("CODERAG_PLAN_MAX_CALLS", 20)
    calls = {"n": 0}
    execution_log: List[Dict[str, Any]] = []
    boost = retrieve_topk_boost or {}
    retrieve_call_idx = [0]

    def _tick():
        calls["n"] += 1
        if calls["n"] > max_calls:
            raise _PlanCallBudgetExceeded(f"exceeded {max_calls} retrieve/answer calls")

    def retrieve(query: str, topk: int = default_topk):
        _tick()
        retrieve_call_idx[0] += 1
        idx = retrieve_call_idx[0]
        if idx in boost:
            topk = max(topk, boost[idx])
        query_str = str(query)
        _debug_print(sample_id, f"step={len(execution_log) + 1} retrieve call begin: query={_short(query_str)} topk={int(topk)}")
        docs = list(_cached_retrieve(query_str, int(topk)))
        execution_log.append({
            "step": len(execution_log) + 1,
            "type": "retrieve",
            "query": query_str,
            "topk": int(topk),
            "docs": docs,
        })
        _debug_print(sample_id, f"step={len(execution_log)} retrieve call end: num_docs={len(docs)} docs={_short(docs)}")
        return docs

    def answer(query: str, docs: Optional[List[str]] = None):
        _tick()
        docs_list = list(docs) if docs else []
        query_str = str(query)
        _debug_print(
            sample_id,
            f"step={len(execution_log) + 1} answer call begin: query={_short(query_str)} num_docs={len(docs_list)} docs={_short(docs_list)}",
        )
        returned = _cached_answer(query_str, tuple(docs_list))
        execution_log.append({
            "step": len(execution_log) + 1,
            "type": "answer",
            "query": query_str,
            "docs": docs_list,
            "answer_returned": returned,
        })
        _debug_print(sample_id, f"step={len(execution_log)} answer call end: answer={_short(returned)}")
        return returned

    return retrieve, answer, execution_log, calls


def _execute_code(
    compiled: Any,
    topk: int,
    timeout: int,
    retrieve_topk_boost: Optional[Dict[int, int]] = None,
    sample_id: int = 0,
) -> Tuple[Dict[str, Any], int]:
    retrieve_fn, answer_fn, execution_log, calls = _build_tools(
        topk,
        retrieve_topk_boost=retrieve_topk_boost,
        sample_id=sample_id,
    )
    namespace: Dict[str, Any] = {
        "retrieve": retrieve_fn,
        "answer": answer_fn,
        "__builtins__": _SANDBOX_BUILTINS,
    }
    _debug_print(sample_id, f"exec begin: topk={topk} timeout={timeout} retrieve_topk_boost={retrieve_topk_boost!r}")
    with _signal_timeout(timeout):
        exec(compiled, namespace)  # noqa: S102
    _debug_print(
        sample_id,
        f"exec end: num_calls={calls['n']} final_answer={_short(namespace.get('final_answer'))} execution_log={_short(execution_log)}",
    )
    return {
        "final_answer": namespace.get("final_answer"),
        "execution_log": execution_log,
    }, calls["n"]


def _zero(err_type: str, pred: str = "", num_calls: int = 0) -> Dict[str, Any]:
    return {
        "score": 0.0,
        "f1": 0.0,
        "em": 0.0,
        "pred": pred,
        "err_type": err_type,
        "num_calls": float(num_calls),
        "retried_with_topk10": 0.0,
    }


def score_plan_em_f1(
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[Dict[str, Any]] = None,
    sample_id: int = 0,
) -> Dict[str, Any]:
    extra_info = extra_info or {}
    topk = int(extra_info.get("topk", 5))
    timeout = _env_int("CODERAG_PLAN_EXEC_TIMEOUT", 30)
    _debug_print(sample_id, f"score begin: topk={topk} timeout={timeout} ground_truth={_short(ground_truth)} extra_info={_short(extra_info)}")
    _debug_print(sample_id, f"raw rollout solution={_short(solution_str)}")
    code = _extract_python_code(solution_str)
    _debug_print(sample_id, f"extracted python code={_short(code)}")
    if not code:
        _debug_print(sample_id, "return zero: no_code")
        return _zero("no_code")

    try:
        compiled = compile(code, "<plan-rollout>", "exec")
    except SyntaxError as exc:
        _log_exception_once("compile plan rollout", exc, code=code)
        _debug_print(sample_id, f"return zero: syntax_error exc={exc!r}")
        return _zero("syntax_error")

    try:
        result, num_calls = _execute_code(compiled, topk=topk, timeout=timeout, sample_id=sample_id)
    except _PlanExecTimeout as exc:
        _log_exception_once("execute plan rollout timeout", exc, topk=topk, timeout=timeout, code=code)
        _debug_print(sample_id, f"return zero: timeout exc={exc!r}")
        return _zero("timeout")
    except _PlanCallBudgetExceeded as exc:
        _log_exception_once("execute plan rollout call_budget", exc, topk=topk, timeout=timeout, code=code)
        _debug_print(sample_id, f"return zero: call_budget exc={exc!r}")
        return _zero("call_budget")
    except SystemExit as exc:
        _debug_print(sample_id, f"return zero: sys_exit exc={exc!r}")
        return _zero("sys_exit")
    except Exception as exc:
        _log_exception_once("execute plan rollout runtime_error", exc, topk=topk, timeout=timeout, code=code)
        _debug_print(sample_id, f"return zero: runtime_error exc={exc!r}")
        return _zero("runtime_error")

    retried_with_topk10 = False
    boost = _build_retrieve_topk_boost(result.get("execution_log", []))
    _debug_print(sample_id, f"first execution result={_short(result)} num_calls={num_calls} topk10_boost_candidates={boost!r}")
    if not boost and _answer_indicates_insufficient_info(
        str(result.get("final_answer", ""))
    ):
        last_r = _last_retrieve_index(result.get("execution_log", []))
        if last_r > 0:
            used = _topk_used_at_retrieve_index(result.get("execution_log", []), last_r)
            if used is not None and used < 10:
                boost[last_r] = 10

    if boost:
        _debug_print(sample_id, f"retry with topk10 boost={boost!r}")
        try:
            result, num_calls = _execute_code(
                compiled,
                topk=topk,
                timeout=timeout,
                retrieve_topk_boost=boost,
                sample_id=sample_id,
            )
            retried_with_topk10 = True
            _debug_print(sample_id, f"retry execution result={_short(result)} num_calls={num_calls}")
        except _PlanExecTimeout as exc:
            _log_exception_once(
                "execute plan rollout after topk10 timeout",
                exc,
                topk=topk,
                timeout=timeout,
                retrieve_topk_boost=boost,
                code=code,
            )
            _debug_print(sample_id, f"return zero: timeout_after_topk10 exc={exc!r}")
            return _zero("timeout_after_topk10", num_calls=num_calls)
        except _PlanCallBudgetExceeded as exc:
            _log_exception_once(
                "execute plan rollout after topk10 call_budget",
                exc,
                topk=topk,
                timeout=timeout,
                retrieve_topk_boost=boost,
                code=code,
            )
            _debug_print(sample_id, f"return zero: call_budget_after_topk10 exc={exc!r}")
            return _zero("call_budget_after_topk10", num_calls=num_calls)
        except SystemExit as exc:
            _debug_print(sample_id, f"return zero: sys_exit_after_topk10 exc={exc!r}")
            return _zero("sys_exit_after_topk10", num_calls=num_calls)
        except Exception as exc:
            _log_exception_once(
                "execute plan rollout after topk10 runtime_error",
                exc,
                topk=topk,
                timeout=timeout,
                retrieve_topk_boost=boost,
                code=code,
            )
            _debug_print(sample_id, f"return zero: runtime_error_after_topk10 exc={exc!r}")
            return _zero("runtime_error_after_topk10", num_calls=num_calls)

    pred = _extract_answer(result.get("final_answer"))
    golds = _as_list(ground_truth)
    _debug_print(sample_id, f"parsed prediction={_short(pred)} golds={_short(golds)}")
    if not pred or not golds:
        out = _zero("empty_pred_or_gold", pred=pred, num_calls=num_calls)
        out["retried_with_topk10"] = 1.0 if retried_with_topk10 else 0.0
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
        "num_calls": float(num_calls),
        "retried_with_topk10": 1.0 if retried_with_topk10 else 0.0,
    }


def compute_score(
    *args: Any,
    extra_info: Optional[Dict[str, Any]] = None,
    data_source: str = "coderag_plan",
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
    if data_source != "coderag_plan":
        output = _zero(f"unsupported_data_source:{data_source}")
        _debug_print(sample_id, f"return zero: unsupported_data_source data_source={data_source!r}")
    else:
        output = score_plan_em_f1(solution_str, ground_truth, extra_info, sample_id=sample_id)
    _debug_print(sample_id, f"compute_score output: {output!r}")
    return output

