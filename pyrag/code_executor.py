from typing import Any, Callable, Dict, List


class CodeExecutor:
    def execute(
        self,
        code: str,
        retrieve_fn: Callable,
        answer_fn: Callable,
        execution_log: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        namespace: Dict[str, Any] = {
            "retrieve": retrieve_fn,
            "answer": answer_fn,
        }

        try:
            exec(code, namespace)
        except Exception as e:
            raise RuntimeError(
                f"Code execution failed ({type(e).__name__}: {e})\n--- Generated code ---\n{code}"
            ) from e

        final_answer = namespace.get("final_answer", "[ERROR: final_answer not set]")
        if callable(final_answer):
            final_answer = None

        variables = {
            k: v
            for k, v in namespace.items()
            if not k.startswith("__") and k not in ("retrieve", "answer")
        }

        return {
            "final_answer": final_answer,
            "variables": variables,
            "execution_log": execution_log,
        }
