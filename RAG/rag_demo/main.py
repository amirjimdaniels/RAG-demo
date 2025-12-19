
import sys
import os
from retriever_numpy import NumpyRetriever
from retriever_minimal import MinimalRetriever
from data import DATA, QUESTIONS, GROUND_TRUTH
from llms import OpenAILLM, AnthropicLLM, GoogleLLM, MetaLlamaLLM

# Dummy LLM: returns first sentence containing a keyword
class DummyLLM:
    def __init__(self, data):
        self.data = data
    def answer(self, question):
        for doc in self.data:
            if any(word in doc.lower() for word in question.lower().split()):
                return doc
        return "I don't know."

def evaluate(llm, retriever, questions, ground_truth):
    correct = 0
    for q, gt in zip(questions, ground_truth):
        context = retriever.retrieve(q)
        answer = llm.answer(context)
        print(f"Q: {q}\nA: {answer}\nGT: {gt}\n---", flush=True)
        if gt.lower() in answer.lower():
            correct += 1
    return correct / len(questions)


def get_llm(provider: str):
    provider = provider.lower()
    if provider == "openai":
        return OpenAILLM()
    elif provider == "anthropic":
        return AnthropicLLM()
    elif provider == "google":
        return GoogleLLM()
    elif provider == "meta":
        return MetaLlamaLLM()
    elif provider == "dummy":
        return DummyLLM(DATA)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")

def main():
    provider = os.getenv("LLM_PROVIDER", "dummy")
    if len(sys.argv) > 1:
        provider = sys.argv[1]
    print(f"Evaluating with LLM provider: {provider}", flush=True)
    llm = get_llm(provider)
    acc = evaluate(llm, MinimalRetriever(DATA), QUESTIONS, GROUND_TRUTH)
    print(f"LLM-only accuracy: {acc:.2f}\n", flush=True)

    print("Evaluating RAG (Numpy Retriever):", flush=True)
    rag_llm = get_llm(provider)
    acc_rag = evaluate(rag_llm, NumpyRetriever(DATA), QUESTIONS, GROUND_TRUTH)
    print(f"RAG (numpy) accuracy: {acc_rag:.2f}\n", flush=True)

    print("Evaluating RAG (Minimal Retriever):", flush=True)
    acc_min = evaluate(rag_llm, MinimalRetriever(DATA), QUESTIONS, GROUND_TRUTH)
    print(f"RAG (minimal) accuracy: {acc_min:.2f}\n", flush=True)

if __name__ == "__main__":
    main()


