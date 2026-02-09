"""LLM provider classes with a shared base to eliminate duplication.

All OpenAI-compatible APIs (OpenAI, DeepSeek, Groq, Together) share a single
implementation via ``OpenAICompatibleLLM``.  Provider-specific classes only
override the defaults (URL, model, env-var name).
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from typing import Any

import requests


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseLLM(ABC):
    """Common interface every LLM provider must implement."""

    @abstractmethod
    def answer(self, prompt: str) -> str:
        """Return the model's response to *prompt*."""


# ---------------------------------------------------------------------------
# OpenAI-compatible chat-completions base
# ---------------------------------------------------------------------------

class OpenAICompatibleLLM(BaseLLM):
    """Shared implementation for any API that follows the OpenAI
    ``/chat/completions`` request/response schema."""

    # Subclasses override these defaults:
    _default_api_url: str = ""
    _default_model: str = ""
    _env_key_name: str = ""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        api_url: str | None = None,
        max_tokens: int = 128,
    ) -> None:
        self.api_key = api_key or os.getenv(self._env_key_name, "")
        self.model = model or self._default_model
        self.api_url = api_url or self._default_api_url
        self.max_tokens = max_tokens

    def answer(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
        }
        resp = requests.post(self.api_url, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Concrete OpenAI-compatible providers
# ---------------------------------------------------------------------------

class OpenAILLM(OpenAICompatibleLLM):
    _default_api_url = "https://api.openai.com/v1/chat/completions"
    _default_model = "gpt-3.5-turbo"
    _env_key_name = "OPENAI_API_KEY"


class DeepSeekLLM(OpenAICompatibleLLM):
    """DeepSeek API — free tier available with generous limits."""
    _default_api_url = "https://api.deepseek.com/v1/chat/completions"
    _default_model = "deepseek-chat"
    _env_key_name = "DEEPSEEK_API_KEY"


class GroqLLM(OpenAICompatibleLLM):
    """Groq API — free tier with fast inference on Llama, Mixtral, etc."""
    _default_api_url = "https://api.groq.com/openai/v1/chat/completions"
    _default_model = "llama-3.3-70b-versatile"
    _env_key_name = "GROQ_API_KEY"


class TogetherLLM(OpenAICompatibleLLM):
    """Together AI — free tier with many open-source models."""
    _default_api_url = "https://api.together.xyz/v1/chat/completions"
    _default_model = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    _env_key_name = "TOGETHER_API_KEY"


# ---------------------------------------------------------------------------
# Non-OpenAI providers
# ---------------------------------------------------------------------------

class AnthropicLLM(BaseLLM):
    """Anthropic Messages API."""

    def __init__(self, api_key: str | None = None, model: str = "claude-2") -> None:
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.model = model
        self.api_url = "https://api.anthropic.com/v1/messages"

    def answer(self, prompt: str) -> str:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "max_tokens": 128,
            "messages": [{"role": "user", "content": prompt}],
        }
        resp = requests.post(self.api_url, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()["content"][0]["text"].strip()


class GoogleLLM(BaseLLM):
    """Google Generative Language (Gemini) API."""

    def __init__(self, api_key: str | None = None, model: str = "models/gemini-pro") -> None:
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY", "")
        self.model = model

    def answer(self, prompt: str) -> str:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        payload: dict[str, Any] = {"contents": [{"parts": [{"text": prompt}]}]}
        resp = requests.post(url, headers={"Content-Type": "application/json"}, json=payload)
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()


class MetaLlamaLLM(BaseLLM):
    """Self-hosted Llama endpoint (llama.cpp / HuggingFace TGI)."""

    def __init__(self, endpoint: str | None = None, model: str = "llama-2-70b-chat") -> None:
        self.endpoint = endpoint or os.getenv("META_LLM_ENDPOINT", "")
        self.model = model

    def answer(self, prompt: str) -> str:
        payload = {"model": self.model, "prompt": prompt, "max_tokens": 128}
        resp = requests.post(self.endpoint, json=payload)
        resp.raise_for_status()
        return resp.json().get("choices", [{}])[0].get("text", "").strip()


class OllamaLLM(BaseLLM):
    """Ollama — run models locally for free (requires Ollama installed)."""

    def __init__(self, endpoint: str | None = None, model: str = "deepseek-r1:8b") -> None:
        self.endpoint = endpoint or os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
        self.model = model

    def answer(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        resp = requests.post(f"{self.endpoint}/api/chat", json=payload)
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()


class MiniMaxLLM(BaseLLM):
    """MiniMax chat-completion API."""

    def __init__(self, api_key: str | None = None, model: str = "abab6.5s-chat") -> None:
        self.api_key = api_key or os.getenv("MINIMAX_API_KEY", "")
        self.model = model
        self.api_url = "https://api.minimax.chat/v1/text/chatcompletion_v2"

    def answer(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 128,
            "temperature": 0.7,
        }
        resp = requests.post(self.api_url, headers=headers, json=payload)
        resp.raise_for_status()
        result = resp.json()

        # MiniMax has several possible response shapes.
        if "choices" in result and result["choices"]:
            choice = result["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                return choice["message"]["content"].strip()
            if "reply" in choice:
                return choice["reply"].strip()
        if "reply" in result:
            return result["reply"].strip()
        return f"Error: unexpected MiniMax response: {result}"


# ---------------------------------------------------------------------------
# Dummy / offline LLM (returns best-matching sentence from context)
# ---------------------------------------------------------------------------

class DummyLLM(BaseLLM):
    """Offline LLM that picks the most relevant sentence from the prompt's
    ``Context:`` section using simple token overlap."""

    def answer(self, prompt: str) -> str:
        if "Context:" in prompt and "Question:" in prompt:
            ctx = prompt.split("Context:", 1)[1].split("Question:", 1)[0].strip()
            question = prompt.split("Question:", 1)[1].strip()
            return _best_sentence(question, ctx)
        return "I don't know."


def _best_sentence(question: str, context: str) -> str:
    """Return the sentence in *context* with the highest token overlap to *question*."""
    if not context:
        return "I don't know."
    q_tokens = set(re.findall(r"\b\w+\b", question.lower()))
    sentences = re.split(r"(?<=[.!?])\s+", context.strip())
    best_score, best_sent = 0, ""
    for sentence in sentences:
        score = len(q_tokens & set(re.findall(r"\b\w+\b", sentence.lower())))
        if score > best_score:
            best_score, best_sent = score, sentence
    return best_sent or "I don't know."


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_LLM_REGISTRY: dict[str, type[BaseLLM]] = {
    "openai": OpenAILLM,
    "anthropic": AnthropicLLM,
    "google": GoogleLLM,
    "meta": MetaLlamaLLM,
    "deepseek": DeepSeekLLM,
    "groq": GroqLLM,
    "together": TogetherLLM,
    "ollama": OllamaLLM,
    "minimax": MiniMaxLLM,
    "dummy": DummyLLM,
}


def get_llm(provider: str, api_key: str | None = None) -> BaseLLM:
    """Instantiate an LLM by provider name.

    Parameters
    ----------
    provider : str
        One of the keys in ``_LLM_REGISTRY`` (case-insensitive).
    api_key : str | None
        Optional API key override.  Ignored for providers that don't use one.
    """
    provider = provider.lower()
    cls = _LLM_REGISTRY.get(provider)
    if cls is None:
        raise ValueError(
            f"Unknown LLM provider: {provider!r}. "
            f"Choose from: {', '.join(sorted(_LLM_REGISTRY))}"
        )

    # Providers that don't accept api_key in their constructor:
    if provider in {"ollama", "meta", "dummy"}:
        return cls()
    return cls(api_key=api_key) if api_key else cls()


