# Minimal RAG Demo

This project demonstrates Retrieval Augmented Generation (RAG) using a lightweight LLM and two retrieval methods: numpy-based in-memory search and the most minimal alternative. It compares LLM-only answers to RAG-augmented answers for accuracy.

## Features
- Minimal dependencies
- Simple dataset and retrieval
- Easy to run and understand


## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Run the demo with the default dummy LLM:
	```
	python rag_demo/main.py
	```
3. To use a real LLM, set the provider via environment variable or command-line argument:
	```
	# OpenAI (GPT-3.5/4)
	export OPENAI_API_KEY=your_key
	python rag_demo/main.py openai

	# Anthropic (Claude)
	export ANTHROPIC_API_KEY=your_key
	python rag_demo/main.py anthropic

	# Google (Gemini/PaLM)
	export GOOGLE_API_KEY=your_key
	python rag_demo/main.py google

	# Meta (Llama, local endpoint)
	export META_LLM_ENDPOINT=http://localhost:8000/v1/completions
	python rag_demo/main.py meta
	```
	You can also set the provider with the `LLM_PROVIDER` environment variable.

## Files
- `rag_demo/main.py`: Main script for running the demo
- `rag_demo/retriever_numpy.py`: Numpy-based vector search
- `rag_demo/retriever_minimal.py`: Most lightweight retrieval (brute-force cosine)
- `rag_demo/data.py`: Sample dataset

## Requirements
- Python 3.8+
- See `requirements.txt` for dependencies
