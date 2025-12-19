# LLM provider classes for OpenAI, Anthropic, Google, Meta
import os
import requests

class OpenAILLM:
    def __init__(self, api_key=None, model="gpt-3.5-turbo"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.api_url = "https://api.openai.com/v1/chat/completions"
    def answer(self, context):
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": context}],
            "max_tokens": 128,
        }
        resp = requests.post(self.api_url, headers=headers, json=data)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

class AnthropicLLM:
    def __init__(self, api_key=None, model="claude-2"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self.api_url = "https://api.anthropic.com/v1/messages"
    def answer(self, context):
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}
        data = {
            "model": self.model,
            "max_tokens": 128,
            "messages": [{"role": "user", "content": context}],
        }
        resp = requests.post(self.api_url, headers=headers, json=data)
        resp.raise_for_status()
        return resp.json()["content"][0]["text"].strip()

class GoogleLLM:
    def __init__(self, api_key=None, model="models/gemini-pro"):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.model = model
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/{self.model}:generateContent?key={self.api_key}"
    def answer(self, context):
        headers = {"Content-Type": "application/json"}
        data = {"contents": [{"parts": [{"text": context}]}]}
        resp = requests.post(self.api_url, headers=headers, json=data)
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()

class MetaLlamaLLM:
    def __init__(self, endpoint=None, model="llama-2-70b-chat"):
        self.endpoint = endpoint or os.getenv("META_LLM_ENDPOINT")
        self.model = model
    def answer(self, context):
        # Example: expects a local endpoint compatible with Llama.cpp or HuggingFace text-generation-inference
        data = {"model": self.model, "prompt": context, "max_tokens": 128}
        resp = requests.post(self.endpoint, json=data)
        resp.raise_for_status()
        return resp.json().get("choices", [{}])[0].get("text", "").strip()
