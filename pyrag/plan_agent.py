import logging
from typing import List

from pyrag.llm import OpenAILLM
from pyrag.utils import extract_python_block

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a planning agent that writes Python code to answer questions via RAG.

You have exactly two functions available:

  retrieve(query: str) -> List[str]
      Searches a retrieval system and returns a list of relevant document strings.
      Do NOT pass a topk argument; the retrieval count is controlled externally.

  answer(query: str, docs: List[str]) -> str
      Calls an LLM to answer a question given a list of documents.
      Returns ONLY the short text inside the model's <answer>...</answer> block (the runtime extracts
      it). Use that return value directly in f-strings — it is a short phrase, not the full model reply.

  answer(query: str) -> str          (no docs — aggregation / synthesis mode)
      Same as above but with NO documents. The model answers using ONLY the information already
      present in the query string. Use this for the FINAL aggregation step.

Rules:
- Use retrieve() + answer() step by step to answer each sub-query.
- If you call retrieve() for a sub-question, pass that return value into the matching answer() for
  that sub-question. Do not pass [] for that step unless the answer is intentionally retrieval-free.
- Build on intermediate answers by interpolating them into later queries using f-strings.
- Variable names MUST be valid Python identifiers: only letters, digits, and underscores, NO spaces.
- Every variable you reference MUST be assigned BEFORE it is used. Never use a name that was not
  explicitly assigned in a previous line.
- Do NOT parse answer() return values in generated code (no .split(), regex, or indexing). The runtime
  already returns the short <answer> span.
- If you use a for/if/while block that might conditionally set final_answer, initialise final_answer = ""
  BEFORE the block so it is always defined.
- The last line of your code MUST be: final_answer = answer(<synthesis question>)
  Do NOT pass docs to this final answer() call — it is aggregation-only.
- NEVER use an f-string or string concatenation as final_answer. final_answer MUST ALWAYS be the
  return value of an answer() call so the result goes through the model and has a proper <answer> tag.
- CRITICAL: The <synthesis question> must use this TWO-PART format:
    "Given: <fact1>, <fact2>, ... Answer the question: <ORIGINAL QUESTION>"
  where the first part lists intermediate results as background facts, and the second part REPEATS
  the ORIGINAL question VERBATIM after "Answer the question:".
  Do NOT rephrase the original question or embed intermediate answers INTO the question itself.
  BAD:  f"Which American director, {name}, who is {nationality}, hosted ...?"  (answer leaks into the question → LLM says "yes")
  GOOD: f"Given: {name} hosted the event, {name} is {nationality}. Answer the question: Which American director hosted ...?"
- Valid Python 3 only: use ASCII double or single quotes for strings; never put backticks around
  variable names or expressions.
- Return ONLY the Python code, no explanation, no imports, no markdown prose."""

CODE_EXAMPLE = """\
# Example for: "Who was born earlier, the director of Inception or Jurassic Park?"
docs1 = retrieve("Who directed Inception?")
director1 = answer("Who directed Inception?", docs1)

docs2 = retrieve("Who directed Jurassic Park?")
director2 = answer("Who directed Jurassic Park?", docs2)

docs3 = retrieve(f"When was {director1} born?")
birth1 = answer(f"When was {director1} born?", docs3)

docs4 = retrieve(f"When was {director2} born?")
birth2 = answer(f"When was {director2} born?", docs4)

# ALWAYS end with answer() — list facts first, then repeat the ORIGINAL question verbatim
final_answer = answer(
    f"Given: {director1} directed Inception and was born {birth1}, "
    f"{director2} directed Jurassic Park and was born {birth2}. "
    f"Answer the question: Who was born earlier, the director of Inception or Jurassic Park?"
)"""


FIX_PROMPT_TEMPLATE = """\
The following Python code was generated to answer the question but raised a runtime error.
Fix the code so it runs correctly. Return ONLY the corrected Python code.

=== Original question ===
{original_query}

=== Failing code ===
{failed_code}

=== Runtime error ===
{error_msg}

=== Fix instructions ===
- Do not parse answer() return values in generated code; the runtime returns short <answer> text.
- Every variable you reference must be explicitly assigned earlier.
- If retrieve() was called, pass its docs into answer(); do not use answer(..., []) for that step.
- Initialise final_answer = "" before any loop/conditional that might set it.
- The last line MUST be: final_answer = answer(<synthesis question>) — call answer() with NO docs
  to aggregate all intermediate results. NEVER use an f-string or string as final_answer directly.
"""


SYNTAX_FEEDBACK_TEMPLATE = """
---
Your previous output is NOT valid Python (syntax error).
Details: {error_detail}

Fix it. Use only ASCII ' or \" for strings; never wrap variable names in backticks; output code only.

Previous attempt:
```python
{failed_code}
```
Return ONLY corrected Python code.
"""


class PlanAgent:
    def __init__(self, llm: OpenAILLM, max_retries: int = 3):
        self.llm = llm
        self.max_retries = max_retries

    def _build_initial_prompt(self, original_query: str, sub_queries: List[str]) -> str:
        return f"""
Original question:
{original_query}

Sub-queries to resolve:
{sub_queries}

Reference example (do NOT copy, write code for the actual question above):
{CODE_EXAMPLE}

Now write the Python code for the original question.
End with: final_answer = answer(f"Given: <facts>. Answer the question: {original_query}")
""".strip()

    def _try_compile(self, raw: str) -> str:
        code = extract_python_block(raw)
        compile(code, "<generated>", "exec")
        return code

    def generate_code(self, original_query: str, sub_queries: List[str]) -> str:
        user_prompt = self._build_initial_prompt(original_query, sub_queries)

        last_exc: Exception | None = None
        last_raw = ""
        extra = ""
        for attempt in range(1, self.max_retries + 1):
            try:
                raw = self.llm.generate(SYSTEM_PROMPT, user_prompt + extra)
                last_raw = raw
                return self._try_compile(raw)
            except SyntaxError as e:
                last_exc = e
                failed = extract_python_block(last_raw) if last_raw else ""
                loc = f"line {e.lineno}" if e.lineno else "unknown"
                detail = f"{e.msg} ({loc})"
                logger.warning(
                    "generate_code attempt %d/%d SyntaxError: %s",
                    attempt, self.max_retries, detail,
                )
                extra = SYNTAX_FEEDBACK_TEMPLATE.format(
                    error_detail=detail,
                    failed_code=failed[:6000] if failed else "(empty)",
                )
            except Exception as e:
                last_exc = e
                logger.warning("generate_code attempt %d/%d failed: %s", attempt, self.max_retries, e)

        tail = last_raw[:2500] if last_raw else ""
        raise RuntimeError(
            f"generate_code failed after {self.max_retries} retries. Last error: {last_exc}\n"
            f"--- Last model output (truncated) ---\n{tail}"
        )

    def fix_code(
        self,
        original_query: str,
        failed_code: str,
        error_msg: str,
        max_fix_retries: int = 3,
    ) -> str:
        user_prompt = FIX_PROMPT_TEMPLATE.format(
            original_query=original_query,
            failed_code=failed_code,
            error_msg=error_msg,
        )
        last_exc: Exception | None = None
        for attempt in range(1, max_fix_retries + 1):
            try:
                raw = self.llm.generate(SYSTEM_PROMPT, user_prompt)
                return self._try_compile(raw)
            except SyntaxError as e:
                last_exc = e
                logger.warning("fix_code attempt %d/%d SyntaxError: %s", attempt, max_fix_retries, e)
            except Exception as e:
                last_exc = e
                logger.warning("fix_code attempt %d/%d failed: %s", attempt, max_fix_retries, e)

        raise RuntimeError(
            f"fix_code failed after {max_fix_retries} retries. Last error: {last_exc}"
        )
