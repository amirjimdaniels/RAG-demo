# Minimal RAG Demo

A simple demonstration of Retrieval Augmented Generation (RAG) comparing retrieval-based answers vs naive keyword matching. Includes a **web UI** for side-by-side comparison.

## 🚀 Quick Start

### Docker (Recommended)
```bash
# Build the image (run from the RAG/ directory)
docker build -t rag-demo .

# Run with the dummy LLM (no API key required)
docker run -p 5000:5000 rag-demo
```

To pass API keys at runtime, supply them as environment variables:
```bash
docker run -p 5000:5000 \
  -e OPENAI_API_KEY=your_key \
  rag-demo
```

Then open http://localhost:5000 in your browser.

### Deploy to Render

The repository includes a `render.yaml` for one-click deployment.

#### Using the Blueprint (recommended)

1. Push this repository to GitHub (or fork it).
2. In the [Render dashboard](https://dashboard.render.com/), click **New → Blueprint** and connect your repo.
3. Render will detect `render.yaml` and create the `rag-demo` web service automatically.
4. Set your LLM API keys in **Environment → Environment Variables** in the Render dashboard (e.g. `OPENAI_API_KEY`). The app works without any key using the built-in dummy LLM.
5. Click **Deploy** — the app will be live at the URL Render assigns.

> **Note:** Render injects a `PORT` environment variable at runtime. The app (via Gunicorn) automatically binds to that port, so no manual port configuration is needed.

#### Manual setup / "What is the start command?"

If you create the service manually in the Render dashboard, use these settings:

| Field | Value |
|-------|-------|
| **Environment** | Docker |
| **Dockerfile path** | `RAG/Dockerfile` |
| **Docker context** | `RAG` |
| **Start Command** | `gunicorn --bind "0.0.0.0:$PORT" --workers 2 --timeout 120 app:app` |

For a **native Python** (non-Docker) service instead:

| Field | Value |
|-------|-------|
| **Environment** | Python |
| **Root directory** | `RAG` |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `cd rag_demo && gunicorn --bind "0.0.0.0:$PORT" --workers 2 --timeout 120 app:app` |

### Local Development (Without Docker)
```bash
pip install -r requirements.txt
cd rag_demo
python app.py
```
Then open http://localhost:5000 in your browser.

### Command Line
```bash
python rag_demo/main.py          # Uses dummy LLM (no API key needed)
python rag_demo/main.py deepseek # Uses DeepSeek API
python rag_demo/main.py groq     # Uses Groq API
```

## 📊 What This Demo Shows

The demo compares:
- **Without RAG**: Naive keyword matching (often gets wrong answers)
- **With RAG**: Semantic retrieval using embeddings (finds the right context)

Currently uses a **DummyLLM** that returns retrieved context as-is - no actual LLM API calls are made by default. This is intentional to show that the *retrieval* is the key differentiator.

## 🆓 Free LLM Options

| Provider | Free Tier | API Key | Notes |
|----------|-----------|---------|-------|
| **DeepSeek** | ✅ Yes | [Get Key](https://platform.deepseek.com/) | Generous free tier, great for testing |
| **Groq** | ✅ Yes | [Get Key](https://console.groq.com/) | Very fast inference, Llama 3.3 70B free |
| **Together AI** | ✅ $5 credit | [Get Key](https://api.together.xyz/) | Many open-source models |
| **Ollama** | ✅ 100% Free | None (local) | Run models locally, requires [Ollama](https://ollama.ai/) |
| **MiniMax** | ✅ Free tier | [Get Key](https://api.minimax.chat/) | MiniMax-M2.1 model |
| **Google Gemini** | ✅ Free tier | [Get Key](https://makersuite.google.com/app/apikey) | 60 requests/min free |

### Using Free LLMs

```bash
# DeepSeek (recommended - truly free)
export DEEPSEEK_API_KEY=your_key
python rag_demo/main.py deepseek

# Groq (fast & free)
export GROQ_API_KEY=your_key
python rag_demo/main.py groq

# MiniMax (free tier)
export MINIMAX_API_KEY=your_key
python rag_demo/main.py minimax

# Together AI ($5 free credit)
export TOGETHER_API_KEY=your_key
python rag_demo/main.py together

# Ollama (local, completely free)
# First: ollama pull llama3.2
python rag_demo/main.py ollama
```

## 💰 Paid LLM Options

```bash
# OpenAI (GPT-3.5/4)
export OPENAI_API_KEY=your_key
python rag_demo/main.py openai

# Anthropic (Claude)
export ANTHROPIC_API_KEY=your_key
python rag_demo/main.py anthropic

# Google (Gemini Pro)
export GOOGLE_API_KEY=your_key
python rag_demo/main.py google
```

## 📁 Project Structure

```
rag_demo/
├── __init__.py          # Package marker
├── app.py               # Flask web UI for comparison
├── main.py              # CLI demo script
├── data.py              # Sample dataset, questions & ground truth
├── embeddings.py        # TF-IDF vectorization utilities
├── evaluation.py        # Shared prompt building, LLM-as-judge, scoring
├── llms.py              # LLM provider classes (base class + all providers)
├── retriever_numpy.py   # Vector search with NumPy
├── retriever_minimal.py # Lightweight pure-Python cosine similarity
└── templates/
    └── index.html       # Web UI template
```

## 🔧 How It Works

This demo implements several modern RAG techniques:

1. **Embedding**: Questions and documents are converted to vector representations. By default, a SentenceTransformer model (e.g., `BAAI/bge-small-en-v1.5`) is used for semantic search, but it can fall back to TF-IDF if unavailable.
2. **Chunking**: Documents are split into overlapping chunks or sentences to improve retrieval granularity.
3. **Retrieval**: Cosine similarity is used to find the most relevant chunks for a given question. NumPy-accelerated and pure-Python retrievers are both available.
4. **Reranking (Cross-Encoder)**: Optionally, a cross-encoder model (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) reranks the top retrieved chunks for higher accuracy.
5. **Two-Pass Retrieval (Query Expansion)**: For real LLMs, the system can use a two-pass approach: it first retrieves context and gets an initial answer, then asks the LLM to generate an improved search query, and retrieves again with this expanded query.
6. **Map-Reduce Retrieval**: For large documents, the system retrieves all relevant chunks above a similarity threshold, summarizes each chunk with the LLM (map), and then combines or further summarizes these summaries (reduce) to produce a focused context.
7. **Generation**: The LLM (or DummyLLM) uses the retrieved context to answer the question.

The key insight: **good retrieval = good answers**, even with a simple LLM. Advanced strategies like reranking and multi-step retrieval further boost answer quality, especially on long or complex documents.

## Requirements

- Python 3.8+
- flask
- numpy
- python-dotenv
- requests
- scikit-learn

## 🔐 API Key Storage (Local Only)

The web UI stores API keys **in a client-side cookie** (per provider) so you don't have to re-enter them.
When you run the app locally (http://localhost:5000), the server also **writes the key to a local .env** file
the first time you use it in the UI. This keeps keys on your machine and avoids committing secrets.

**Important:**
- Do not commit .env to source control.
- Cookies are stored locally in your browser (SameSite=Lax).

## 🔄 Restarting the Server

To restart the web server after making changes or if you encounter issues:

1. **Stop the server**: Press `Ctrl+C` in the terminal where the server is running.
2. **(Optional) Clear cache**: Delete any `__pycache__` folders if you want a clean state.
3. **Start the server**:
   ```bash
   cd rag_demo
   python app.py
   ```
   Then open [http://localhost:5000](http://localhost:5000) in your browser.

If you see errors on startup:
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- If you deleted or modified files in `rag_demo/templates/`, restore them or re-copy from backup.
- Check that your Python environment is activated (if using a virtualenv).
