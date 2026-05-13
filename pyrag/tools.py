from typing import Any, Callable, Dict, List, Optional, Tuple

from pyrag.llm import OpenAILLM
from pyrag.retrieval_agent import RetrievalAgent
from pyrag.utils import extract_answer_tag, format_docs_for_prompt

ANSWER_SYSTEM_PROMPT_WITH_DOCS = (
    "You are given a question and retrieved documents.\n"
    "You MUST answer using ONLY information from the retrieved documents.\n"
    "Even for yes/no questions, decide yes or no by reasoning from facts in the documents.\n\n"
    "Output format (STRICT):\n"
    "<redacted_thinking> ... </redacted_thinking>\n"
    "<answer> ... </answer>\n\n"
    "Evidence citation rule:\n"
    "- Whenever you use evidence from the documents in your reasoning, you MUST cite it inline as Doc [i] "
    "(matching the document indices shown in the retrieved block, e.g. [Doc 1] → Doc [1]).\n"
    "- Only cite documents that are actually relevant.\n"
    "- Keep <redacted_thinking> concise (1–3 sentences).\n\n"
    "Answer rules:\n"
    "- The <answer> should be a short phrase, preferably taken directly from the documents when possible.\n"
    "- Match the answer TYPE to the QUESTION: WHO / which person / 谁先 / born first / earlier → a person's "
    "NAME in <answer>, not only a date; WHEN / 何时 → date or time; yes/no → exactly yes / no / unknown when "
    "the documents do not support a definite answer.\n"
    "- Do NOT output anything outside <redacted_thinking> and <answer>.\n\n"
    "Example (do NOT copy the content, only follow the style):\n"
    "<redacted_thinking>Doc [1] states that Future Ted serves as the narrator, and Doc [4] confirms the voice actor.</redacted_thinking>\n"
    "<answer> Ted Mosby </answer>\n"
)

ANSWER_SYSTEM_PROMPT_NO_DOCS = (
    "There are NO retrieved documents. The question text itself contains background facts "
    "(after 'Given:') and the actual question to answer (after 'Answer the question:').\n"
    "You MUST use the provided facts to answer the ACTUAL QUESTION.\n\n"
    "CRITICAL: Your job is to ANSWER the question, NOT to confirm whether the facts are true.\n"
    "- If the question asks WHO / WHICH person → reply with a person's NAME.\n"
    "- If the question asks WHEN → reply with a date or time.\n"
    "- If the question asks WHERE → reply with a place.\n"
    "- ONLY answer yes/no when the question is explicitly a yes/no question (e.g. 'Are both ...?', 'Is it true ...?').\n"
    "- NEVER answer 'yes' or 'no' to a WHO/WHICH/WHEN/WHERE question.\n\n"
    "Output format (STRICT):\n"
    "<redacted_thinking> ... </redacted_thinking>\n"
    "<answer> ... </answer>\n\n"
    "- In <redacted_thinking>, identify the actual question type and combine the given facts to produce the answer (1–2 sentences).\n"
    "- The <answer> must directly answer the question — a name, date, place, etc. — NOT 'yes' or 'no' unless the question is truly yes/no.\n"
    "- Do NOT output anything outside <redacted_thinking> and <answer>.\n"
)

ANSWER_SYSTEM_PROMPT = ANSWER_SYSTEM_PROMPT_WITH_DOCS


def answer_system_prompt_for_docs(docs: List[str]) -> str:
    return ANSWER_SYSTEM_PROMPT_WITH_DOCS if docs else ANSWER_SYSTEM_PROMPT_NO_DOCS


def make_tools(
    retrieval_agent: RetrievalAgent,
    llm: OpenAILLM,
    default_topk: int = 5,
    retrieve_topk_boost: Optional[Dict[int, int]] = None,
) -> Tuple[Callable, Callable, List[Dict[str, Any]]]:
    execution_log: List[Dict[str, Any]] = []
    boost = retrieve_topk_boost or {}
    retrieve_call_idx = [0]

    def retrieve(query: str, topk: int = default_topk) -> List[str]:
        retrieve_call_idx[0] += 1
        idx = retrieve_call_idx[0]
        if idx in boost:
            topk = max(topk, boost[idx])
        docs = retrieval_agent.retrieve(query, topk=topk)
        execution_log.append({
            "step": len(execution_log) + 1,
            "type": "retrieve",
            "query": query,
            "topk": topk,
            "docs": docs,
        })
        return docs

    def answer(query: str, docs: Optional[List[str]] = None) -> str:
        if docs is None:
            docs = []
        user_prompt = (
            f"=== QUESTION ===\n"
            f"{query}\n"
            f"=== END QUESTION ===\n\n"
            f"=== RETRIEVED DOCUMENTS ===\n"
            f"{format_docs_for_prompt(docs)}\n"
            f"=== END DOCUMENTS ==="
        )
        result = llm.generate(answer_system_prompt_for_docs(docs), user_prompt)
        returned = extract_answer_tag(result)
        execution_log.append({
            "step": len(execution_log) + 1,
            "type": "answer",
            "query": query,
            "docs": docs,
            "answer_raw": result,
            "answer_returned": returned,
        })
        return returned

    return retrieve, answer, execution_log
