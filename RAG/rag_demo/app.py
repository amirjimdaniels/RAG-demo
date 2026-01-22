import os
from flask import Flask, render_template, jsonify, request
from retriever_numpy import NumpyRetriever
from retriever_minimal import MinimalRetriever
from data import DATA, QUESTIONS, GROUND_TRUTH
from llms import DeepSeekLLM, GroqLLM, TogetherLLM, OllamaLLM

app = Flask(__name__)

# Dummy LLM that returns context as-is
class DummyLLM:
    def answer(self, context):
        return context

def get_llm(provider: str):
    """Get LLM instance based on provider name"""
    provider = provider.lower()
    if provider == "deepseek":
        return DeepSeekLLM()
    elif provider == "groq":
        return GroqLLM()
    elif provider == "together":
        return TogetherLLM()
    elif provider == "ollama":
        return OllamaLLM()
    else:
        return DummyLLM()

# No retrieval - just returns all docs concatenated (simulates no RAG)
class NoRetriever:
    def __init__(self, data):
        self.data = data
    def retrieve(self, query, top_k=1):
        # Without RAG: return first doc that has any word match (naive)
        for doc in self.data:
            if any(word in doc.lower() for word in query.lower().split()):
                return doc
        return self.data[0]  # fallback to first doc

def build_prompt(question, context):
    """Build a prompt for the LLM with question and context"""
    return f"""Based on the following context, answer the question concisely.

Context: {context}

Question: {question}

Answer:"""

def evaluate(llm, retriever, questions, ground_truth, use_prompt=False):
    results = []
    correct = 0
    for q, gt in zip(questions, ground_truth):
        context = retriever.retrieve(q)
        try:
            if use_prompt:
                prompt = build_prompt(q, context)
                answer = llm.answer(prompt)
            else:
                answer = llm.answer(context)
        except Exception as e:
            answer = f"Error: {str(e)}"
        is_correct = gt.lower() in answer.lower()
        if is_correct:
            correct += 1
        results.append({
            "question": q,
            "answer": answer,
            "ground_truth": gt,
            "correct": is_correct
        })
    accuracy = correct / len(questions) if questions else 0
    return results, accuracy

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/compare')
def compare():
    # Get LLM provider from query param or environment
    provider = request.args.get('llm', os.getenv('LLM_PROVIDER', 'dummy'))
    llm = get_llm(provider)
    use_real_llm = provider != 'dummy'
    
    # Without RAG (naive keyword matching)
    no_rag_results, no_rag_accuracy = evaluate(llm, NoRetriever(DATA), QUESTIONS, GROUND_TRUTH, use_prompt=use_real_llm)
    
    # With RAG (numpy retriever with embeddings)
    rag_results, rag_accuracy = evaluate(llm, NumpyRetriever(DATA), QUESTIONS, GROUND_TRUTH, use_prompt=use_real_llm)
    
    return jsonify({
        "provider": provider,
        "no_rag": {
            "results": no_rag_results,
            "accuracy": no_rag_accuracy
        },
        "rag": {
            "results": rag_results,
            "accuracy": rag_accuracy
        }
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
