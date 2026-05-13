import logging
from typing import List

from pyrag.llm import OpenAILLM
from pyrag.utils import extract_json_block

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a question decomposition agent for multi-hop QA. "
    "Break a complex question into a minimal list of atomic single-hop sub-queries. "
    "Each sub-query should be independently answerable by a search engine. "
    "Return a JSON list of strings ONLY. No explanation, no markdown, no extra text."
)

DECOMPOSE_EXAMPLE = """\
Example:
Original question:
In what year was the director of Inception born?

JSON output:
["Who directed Inception?", "When was Christopher Nolan born?"]"""


def build_decompose_user_prompt(query: str) -> str:
    return f"""
Original question:
{query}

{DECOMPOSE_EXAMPLE}

Now return a JSON list of atomic sub-queries for the original question above.
Return ONLY the JSON list of strings.
""".strip()


class DecomposeAgent:
    def __init__(self, llm: OpenAILLM, max_retries: int = 3):
        self.llm = llm
        self.max_retries = max_retries

    def decompose(self, query: str) -> List[str]:
        user_prompt = build_decompose_user_prompt(query)

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                raw = self.llm.generate(SYSTEM_PROMPT, user_prompt)
                result = extract_json_block(raw)
                if not isinstance(result, list) or not all(isinstance(s, str) for s in result):
                    raise ValueError(f"Expected list[str], got: {type(result).__name__} → {result!r}")
                if len(result) == 0:
                    raise ValueError("Decompose returned empty list")
                return result
            except Exception as e:
                last_exc = e
                logger.warning("decompose attempt %d/%d failed: %s", attempt, self.max_retries, e)

        logger.error("decompose failed after %d retries, falling back to original query. Last error: %s",
                     self.max_retries, last_exc)
        return [query]
