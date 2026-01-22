# Minimal RAG Demo

A simple demonstration of Retrieval Augmented Generation (RAG) comparing retrieval-based answers vs naive keyword matching. Includes a **web UI** for side-by-side comparison.

## 🚀 Quick Start

### Web UI (Recommended)
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
| **Google Gemini** | ✅ Free tier | [Get Key](https://makersuite.google.com/app/apikey) | 60 requests/min free |

### Using Free LLMs

```bash
# DeepSeek (recommended - truly free)
export DEEPSEEK_API_KEY=your_key
python rag_demo/main.py deepseek

# Groq (fast & free)
export GROQ_API_KEY=your_key
python rag_demo/main.py groq

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
├── app.py              # Flask web UI for comparison
├── main.py             # CLI demo script
├── data.py             # Sample dataset & embeddings
├── llms.py             # LLM provider classes (all options)
├── retriever_numpy.py  # Vector search with numpy
├── retriever_minimal.py # Lightweight cosine similarity
└── templates/
    └── index.html      # Web UI template
```

## 🔧 How It Works

1. **Embedding**: Questions and documents are converted to bag-of-words vectors
2. **Retrieval**: Cosine similarity finds the most relevant document
3. **Generation**: The LLM (or DummyLLM) uses the retrieved context to answer

The key insight: **good retrieval = good answers**, even with a simple LLM.

## Requirements

- Python 3.8+
- numpy
- requests
- flask (for web UI)
