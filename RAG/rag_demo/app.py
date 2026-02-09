"""Flask web UI for side-by-side RAG vs. no-RAG comparison."""

from __future__ import annotations

import io
import math
import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from pypdf import PdfReader
from docx import Document as DocxDocument

from data import BRIEF, DATA, GROUND_TRUTH, QUESTIONS
from embeddings import DocumentStore, split_into_sentences
from evaluation import build_prompt, text_match_judge
from llms import get_llm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

APP_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = APP_ROOT / ".env"
load_dotenv(dotenv_path=ENV_PATH)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Active document store (swappable at runtime)
# ---------------------------------------------------------------------------

_store = DocumentStore(raw_text=BRIEF)
_is_custom_doc = False
_is_summarized = False
_last_page_count = 1

MAX_PDF_PAGES = 20
EST_CHARS_PER_PAGE = 3000
SUMMARY_CHAR_THRESHOLD = EST_CHARS_PER_PAGE
SUMMARY_CHUNK_CHARS = 3000
SUMMARY_REDUCE_CHAR_THRESHOLD = 6000

# ---------------------------------------------------------------------------
# API-key helpers
# ---------------------------------------------------------------------------


def _is_local_request() -> bool:
    host = request.host.split(":")[0].lower()
    return host in {"127.0.0.1", "localhost"}


def _env_key_name(provider: str) -> str:
    return f"{provider.upper()}_API_KEY"


def _get_api_key(provider: str) -> str:
    """Resolve the API key from query-param -> cookie -> env."""
    api_key = request.args.get("api_key", "").strip()
    if api_key:
        return api_key
    cookie_key = request.cookies.get(f"api_key_{provider}")
    if cookie_key:
        return cookie_key
    return os.getenv(_env_key_name(provider), "")


def _save_api_key_to_env(provider: str, api_key: str) -> None:
    """Persist the key to the local ``.env`` file (only on localhost)."""
    if not api_key or not _is_local_request():
        return
    env_key = _env_key_name(provider)
    lines: list[str] = []
    if ENV_PATH.exists():
        lines = ENV_PATH.read_text(encoding="utf-8").splitlines()
    updated = False
    for i, line in enumerate(lines):
        if line.startswith(f"{env_key}="):
            lines[i] = f"{env_key}={api_key}"
            updated = True
            break
    if not updated:
        lines.append(f"{env_key}={api_key}")
    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_llm(params):
    """Return (provider, llm) from request query-params."""
    provider = params.get("llm", os.getenv("LLM_PROVIDER", "dummy"))
    api_key = _get_api_key(provider)
    if api_key:
        _save_api_key_to_env(provider, api_key)
    return provider, get_llm(provider, api_key or None)


def _estimate_pages_from_text(text: str) -> int:
    return max(1, math.ceil(len(text) / EST_CHARS_PER_PAGE))


def _estimate_pages_from_sentences(sentences: list[str]) -> int:
    return _estimate_pages_from_text(" ".join(sentences))


_last_page_count = _estimate_pages_from_sentences(DATA)


def _chunk_sentences(sentences: list[str], max_chars: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for sentence in sentences:
        if size + len(sentence) + 1 > max_chars and current:
            chunks.append(" ".join(current))
            current = []
            size = 0
        current.append(sentence)
        size += len(sentence) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


def _summarize_chunk(text: str, llm) -> str:
    prompt = (
        "Summarize the following content for factual QA. "
        "Preserve names, dates, numbers, and key terms. "
        "Write concise sentences without losing important details.\n\n"
        f"Content:\n{text}"
    )
    return llm.answer(prompt).strip()


def _summarize_text(text: str, llm) -> str:
    sentences = split_into_sentences(text)
    chunks = _chunk_sentences(sentences, SUMMARY_CHUNK_CHARS)
    summaries = [_summarize_chunk(chunk, llm) for chunk in chunks]
    combined = "\n".join(summaries)
    if len(combined) > SUMMARY_REDUCE_CHAR_THRESHOLD and len(summaries) > 1:
        combined = _summarize_chunk(combined, llm)
    return combined


def _maybe_summarize(text: str, provider: str, llm) -> tuple[str, bool, str | None]:
    if len(text) <= SUMMARY_CHAR_THRESHOLD:
        return text, False, None
    if provider == "dummy":
        return text, False, "Summarization skipped (dummy LLM)."
    try:
        return _summarize_text(text, llm), True, None
    except Exception as exc:
        return text, False, f"Summarization failed: {exc}"


# ---------------------------------------------------------------------------
# Two-pass retrieval & map-reduce helpers
# ---------------------------------------------------------------------------

MAP_REDUCE_CHUNK_THRESHOLD = 15


def _expand_query(question: str, initial_answer: str, llm) -> str:
    """Use the LLM to produce an enriched search query from the question + initial answer."""
    prompt = (
        "Given the original question and an initial answer, generate an improved search query "
        "that would help find more relevant information. Return ONLY the search query, "
        "nothing else.\n\n"
        f"Question: {question}\n"
        f"Initial answer: {initial_answer}\n"
        "Improved search query:"
    )
    try:
        expanded = llm.answer(prompt).strip()
        return f"{question} {expanded}"
    except Exception:
        return question


def _summarize_chunk_for_question(chunk: str, question: str, llm) -> str:
    """Summarize a single chunk, focusing on information relevant to *question*."""
    prompt = (
        "Summarize the following text, focusing specifically on information "
        "relevant to the question. Preserve names, dates, numbers, and key facts. "
        "If the text contains no relevant information, respond with 'No relevant information.'\n\n"
        f"Question: {question}\n"
        f"Text: {chunk}\n"
        "Focused summary:"
    )
    try:
        return llm.answer(prompt).strip()
    except Exception:
        return chunk


def _map_reduce_retrieve(
    store: DocumentStore,
    question: str,
    llm,
    min_similarity: float = 0.3,
    max_chunks: int = 15,
) -> str:
    """Map-reduce retrieval: retrieve relevant chunks, summarize each for the
    question (map), then combine summaries (reduce) into final context."""
    relevant = store.retrieve_all_above_threshold(
        question, min_similarity=min_similarity, max_chunks=max_chunks
    )

    if len(relevant) <= 3:
        return " ".join(text for text, _score in relevant)

    # MAP: summarize each chunk with focus on the question
    summaries = []
    for chunk_text, _score in relevant:
        summary = _summarize_chunk_for_question(chunk_text, question, llm)
        if summary.lower() != "no relevant information.":
            summaries.append(summary)

    if not summaries:
        return relevant[0][0]

    combined = " ".join(summaries)

    # REDUCE: if combined summaries are still very long, do a final summarization
    if len(combined) > 4000 and len(summaries) > 3:
        reduce_prompt = (
            "Combine and condense the following summaries into a single coherent passage "
            "that answers the question. Preserve all key facts.\n\n"
            f"Question: {question}\n"
            f"Summaries:\n{combined}\n"
            "Combined summary:"
        )
        try:
            combined = llm.answer(reduce_prompt).strip()
        except Exception:
            pass

    return combined


def _extract_text_from_upload(file_storage) -> tuple[str, int]:
    filename = file_storage.filename or ""
    ext = Path(filename).suffix.lower()
    if ext not in {".txt", ".md", ".pdf", ".docx"}:
        raise ValueError("Unsupported file type. Use .txt, .md, .pdf, or .docx")

    data = file_storage.read()
    if not data:
        raise ValueError("Uploaded file is empty")

    if ext in {".txt", ".md"}:
        text = data.decode("utf-8", errors="ignore")
        page_count = _estimate_pages_from_text(text)
        if page_count > MAX_PDF_PAGES:
            raise ValueError(
                f"Document estimated at {page_count} pages. Max is {MAX_PDF_PAGES} pages"
            )
        return text, page_count

    if ext == ".pdf":
        reader = PdfReader(io.BytesIO(data))
        page_count = len(reader.pages)
        if page_count > MAX_PDF_PAGES:
            raise ValueError(f"PDF is {page_count} pages. Max is {MAX_PDF_PAGES} pages")
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text, page_count

    doc = DocxDocument(io.BytesIO(data))
    text = "\n".join(p.text for p in doc.paragraphs)
    page_count = _estimate_pages_from_text(text)
    if page_count > MAX_PDF_PAGES:
        raise ValueError(
            f"Document estimated at {page_count} pages. Max is {MAX_PDF_PAGES} pages"
        )
    return text, page_count


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/questions")
def questions():
    """Return the list of example questions and ground truth."""
    return jsonify([
        {"question": q, "ground_truth": gt}
        for q, gt in zip(QUESTIONS, GROUND_TRUTH)
    ])


@app.route("/api/document", methods=["GET"])
def get_document():
    """Return info about the current document store."""
    return jsonify({
        "is_custom": _is_custom_doc,
        "summarized": _is_summarized,
        "sentence_count": _store.sentence_count,
        "preview": _store.text_preview,
        "page_count": _last_page_count,
    })


@app.route("/api/document", methods=["POST"])
def set_document():
    """Replace the document store with user-supplied text."""
    global _store, _is_custom_doc, _is_summarized, _last_page_count

    body = request.get_json(silent=True) or {}
    text = body.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    provider, llm = _resolve_llm(request.args)
    _last_page_count = _estimate_pages_from_text(text)
    processed_text, summarized, summary_error = _maybe_summarize(text, provider, llm)

    _store = DocumentStore(raw_text=processed_text)
    if _store.chunk_count < 1:
        return jsonify({"error": "Document too short — need at least 1 chunk"}), 400

    _is_custom_doc = True
    _is_summarized = summarized

    return jsonify({
        "is_custom": True,
        "summarized": summarized,
        "sentence_count": _store.sentence_count,
        "preview": _store.text_preview,
        "page_count": _last_page_count,
        "summary_error": summary_error,
    })


@app.route("/api/document/file", methods=["POST"])
def set_document_file():
    """Replace the document store with uploaded file text."""
    global _store, _is_custom_doc, _is_summarized, _last_page_count

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if not file or not file.filename:
        return jsonify({"error": "Empty file"}), 400

    try:
        text, page_count = _extract_text_from_upload(file)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    provider, llm = _resolve_llm(request.args)
    _last_page_count = page_count
    processed_text, summarized, summary_error = _maybe_summarize(text, provider, llm)

    _store = DocumentStore(raw_text=processed_text)
    if _store.chunk_count < 1:
        return jsonify({"error": "Document too short — need at least 1 chunk"}), 400

    _is_custom_doc = True
    _is_summarized = summarized

    return jsonify({
        "is_custom": True,
        "summarized": summarized,
        "sentence_count": _store.sentence_count,
        "preview": _store.text_preview,
        "page_count": _last_page_count,
        "summary_error": summary_error,
    })


@app.route("/api/document/reset", methods=["POST"])
def reset_document():
    """Reset back to the built-in demo document."""
    global _store, _is_custom_doc, _is_summarized, _last_page_count
    _store = DocumentStore(raw_text=BRIEF)
    _is_custom_doc = False
    _is_summarized = False
    _last_page_count = _estimate_pages_from_sentences(DATA)
    return jsonify({
        "is_custom": False,
        "summarized": False,
        "sentence_count": _store.sentence_count,
        "preview": _store.text_preview,
        "page_count": _last_page_count,
    })


@app.route("/api/ask")
def ask():
    """Answer a single question with and without RAG.

    Supports two-pass retrieval (query expansion) and map-reduce for large docs.
    """
    question = request.args.get("q", "").strip()
    if not question:
        return jsonify({"error": "No question provided"}), 400

    provider, llm = _resolve_llm(request.args)
    ground_truth = request.args.get("gt", "")

    is_real_llm = provider != "dummy"
    use_two_pass = (
        request.args.get("two_pass", "true").lower() in {"1", "true", "yes"}
        and is_real_llm
    )
    use_map_reduce = (
        is_real_llm
        and _store.chunk_count > MAP_REDUCE_CHUNK_THRESHOLD
    )

    # --- Retrieval step (the whole point of RAG) ---
    effective_query = question

    if use_two_pass:
        initial_context = _store.retrieve(question)
        initial_prompt = build_prompt(question, context=initial_context)
        try:
            initial_answer = llm.answer(initial_prompt)
            effective_query = _expand_query(question, initial_answer, llm)
        except Exception:
            pass  # Keep original query on failure

    if use_map_reduce:
        retrieved_context = _map_reduce_retrieve(_store, effective_query, llm)
        strategy = "map_reduce"
    elif use_two_pass:
        retrieved_context = _store.retrieve(effective_query)
        strategy = "two_pass"
    else:
        retrieved_context = _store.retrieve(question)
        strategy = "single_pass"

    # --- Without RAG: LLM sees only the question ---
    no_rag_prompt = build_prompt(question, context=None)
    try:
        no_rag_answer = llm.answer(no_rag_prompt)
    except Exception as exc:
        no_rag_answer = f"Error: {exc}"

    # --- With RAG: LLM sees retrieved context + question ---
    rag_prompt = build_prompt(question, context=retrieved_context)
    try:
        rag_answer = llm.answer(rag_prompt)
    except Exception as exc:
        rag_answer = f"Error: {exc}"

    # Score against ground truth if provided
    no_rag_correct = text_match_judge(no_rag_answer, ground_truth) if ground_truth else None
    rag_correct = text_match_judge(rag_answer, ground_truth) if ground_truth else None

    return jsonify({
        "provider": provider,
        "question": question,
        "retrieved_context": retrieved_context,
        "retrieval_strategy": strategy,
        "no_rag_answer": no_rag_answer,
        "rag_answer": rag_answer,
        "ground_truth": ground_truth or None,
        "no_rag_correct": no_rag_correct,
        "rag_correct": rag_correct,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
