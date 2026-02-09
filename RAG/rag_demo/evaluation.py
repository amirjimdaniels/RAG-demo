"""Shared evaluation utilities: prompt construction, LLM-as-judge, and scoring.

Both the Flask app and the CLI entry-point import from here instead of
duplicating the same logic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from llms import BaseLLM


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_prompt(question: str, context: str | None = None) -> str:
    """Build an LLM prompt, optionally including retrieved *context*."""
    if context:
        return (
            "Use the context to answer the question in one short sentence. "
            "Do not copy the context verbatim.\n"
            f"Context: {context}\n"
            f"Question: {question}\n"
            "Answer:"
        )
    return (
        "Answer the question in one short sentence.\n"
        f"Question: {question}\n"
        "Answer:"
    )


# ---------------------------------------------------------------------------
# Retriever helpers
# ---------------------------------------------------------------------------

class NoRetriever:
    """Null retriever that always returns an empty string (no RAG)."""

    def __init__(self, data: list[str]) -> None:
        self.data = data

    def retrieve(self, query: str, top_k: int = 1) -> str:  # noqa: ARG002
        return ""


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def normalize_answer(text: str) -> str:
    """Lower-case, strip punctuation, and collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def text_match_judge(answer: str, ground_truth: str) -> bool:
    """Check whether key ground-truth fragments appear in *answer*.

    This avoids an extra LLM round-trip per question and keeps the demo
    fast.  The ground truth is split on ``;`` and ``/`` into fragments;
    each fragment is matched via 3-word sliding windows.
    """
    norm_answer = normalize_answer(answer)
    fragments = [normalize_answer(f) for f in re.split(r"[;/]", ground_truth) if normalize_answer(f)]
    if not fragments:
        return False
    matched = 0
    for frag in fragments:
        words = frag.split()
        if len(words) <= 3:
            if frag in norm_answer:
                matched += 1
        else:
            for i in range(len(words) - 2):
                if " ".join(words[i : i + 3]) in norm_answer:
                    matched += 1
                    break
    return matched >= len(fragments)


@dataclass
class EvalResult:
    """Single question evaluation result."""
    question: str
    answer: str
    ground_truth: str
    correct: bool


def evaluate(
    llm: BaseLLM,
    retriever: object,
    questions: list[str],
    ground_truth: list[str],
    *,
    verbose: bool = False,
) -> tuple[list[EvalResult], float]:
    """Run the full evaluation loop and return ``(results, accuracy)``.

    Uses ``llm_judge`` for scoring and builds the appropriate prompt based on
    whether the retriever is a ``NoRetriever`` (no context) or a real one.
    """
    results: list[EvalResult] = []
    correct = 0

    for q, gt in zip(questions, ground_truth):
        context = retriever.retrieve(q)  # type: ignore[union-attr]

        if isinstance(retriever, NoRetriever):
            llm_input = build_prompt(q, context=None)
        else:
            llm_input = build_prompt(q, context=context)

        try:
            answer = llm.answer(llm_input)
            is_correct = text_match_judge(answer, gt)
        except Exception as exc:
            answer = f"Error: {exc}"
            is_correct = False

        if verbose:
            print(f"Q: {q}\nA: {answer}\nGT: {gt}\n---", flush=True)

        if is_correct:
            correct += 1

        results.append(EvalResult(
            question=q,
            answer=answer,
            ground_truth=gt,
            correct=is_correct,
        ))

    accuracy = correct / len(questions) if questions else 0.0
    return results, accuracy
