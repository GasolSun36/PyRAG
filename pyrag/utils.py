import json
import re
from typing import Any, Dict, List


def extract_json_block(text: str) -> Any:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    fence = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        return json.loads(fence.group(1).strip())
    obj = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if obj:
        return json.loads(obj.group(1).strip())
    raise ValueError(f"Could not parse JSON from model output:\n{text}")


def extract_answer_tag(text: str) -> str:
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.strip()


def normalize_python_source(code: str) -> str:
    if not code:
        return code
    trans = str.maketrans({
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
        "\u00a0": " ",
    })
    code = code.translate(trans)
    code = re.sub(r"[\u200b-\u200d\ufeff]", "", code)
    return code


def extract_python_block(text: str) -> str:
    text = text.strip()
    fence = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        return normalize_python_source(fence.group(1).strip())
    fence = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if fence:
        return normalize_python_source(fence.group(1).strip())
    return normalize_python_source(text)


def format_docs_for_prompt(docs: List[str]) -> str:
    if not docs:
        return "No retrieved documents."
    return "\n\n".join([f"[Doc {i+1}]\n{doc}" for i, doc in enumerate(docs)])
