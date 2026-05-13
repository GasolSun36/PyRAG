import os
import re

from openai import OpenAI


def env_enable_thinking(default: bool = False) -> bool:
    v = os.environ.get("LLM_ENABLE_THINKING", "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "y", "on")

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class OpenAILLM:
    def __init__(
        self,
        model: str,
        base_url: str = "http://127.0.0.1:8000/v1",
        api_key: str = "EMPTY",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        enable_thinking: bool = False,
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        extra_body: dict = {}
        extra_body["chat_template_kwargs"] = {"enable_thinking": bool(self.enable_thinking)}

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            extra_body=extra_body or None,
        )
        content = response.choices[0].message.content or ""
        if not self.enable_thinking:
            content = _THINK_RE.sub("", content).strip()
        else:
            content = content.strip()
        return content
