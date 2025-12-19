# Simple dataset and embedding
DATA = [
    "The capital of France is Paris.",
    "The largest planet is Jupiter.",
    "Water boils at 100 degrees Celsius.",
    "The author of 1984 is George Orwell.",
    "The speed of light is about 299,792 km per second."
]
QUESTIONS = [
    "What is the capital of France?",
    "Who wrote 1984?",
    "What is the boiling point of water?",
    "What is the largest planet?",
    "How fast is light?"
]
GROUND_TRUTH = [
    "Paris",
    "George Orwell",
    "100 degrees Celsius",
    "Jupiter",
    "299,792 km per second"
]

def embed(text):
    # Simple bag-of-words embedding (for demo only)
    vocab = ["paris", "france", "jupiter", "planet", "water", "boils", "100", "degrees", "celsius", "author", "1984", "george", "orwell", "speed", "light", "299792", "km", "second"]
    vec = [0]*len(vocab)
    for i, word in enumerate(vocab):
        if word in text.lower():
            vec[i] = 1
    return vec

EMBEDDINGS = [embed(d) for d in DATA]
