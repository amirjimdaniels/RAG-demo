"""CLI entry-point: evaluate RAG vs. no-RAG accuracy for a chosen LLM."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from data import DATA, GROUND_TRUTH, QUESTIONS
from evaluation import NoRetriever, evaluate
from llms import get_llm
from retriever_minimal import MinimalRetriever
from retriever_numpy import NumpyRetriever

APP_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = APP_ROOT / ".env"
load_dotenv(dotenv_path=ENV_PATH)


def main() -> None:
    provider = sys.argv[1] if len(sys.argv) > 1 else os.getenv("LLM_PROVIDER", "dummy")
    print(f"Evaluating with LLM provider: {provider}", flush=True)

    llm = get_llm(provider)

    # WITHOUT RAG
    _, acc = evaluate(llm, NoRetriever(DATA), QUESTIONS, GROUND_TRUTH, verbose=True)
    print(f"LLM-only accuracy: {acc:.2f}\n", flush=True)

    # WITH RAG — NumPy retriever
    print("Evaluating RAG (Numpy Retriever):", flush=True)
    _, acc_rag = evaluate(llm, NumpyRetriever(DATA), QUESTIONS, GROUND_TRUTH, verbose=True)
    print(f"RAG (numpy) accuracy: {acc_rag:.2f}\n", flush=True)

    # WITH RAG — Minimal retriever
    print("Evaluating RAG (Minimal Retriever):", flush=True)
    _, acc_min = evaluate(llm, MinimalRetriever(DATA), QUESTIONS, GROUND_TRUTH, verbose=True)
    print(f"RAG (minimal) accuracy: {acc_min:.2f}\n", flush=True)


if __name__ == "__main__":
    main()


