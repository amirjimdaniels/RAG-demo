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
    # Extended vocabulary to better match questions to documents
    vocab = [
        "paris", "france", "capital",  # France question
        "jupiter", "planet", "largest",  # Jupiter question
        "water", "boils", "boiling", "100", "degrees", "celsius", "point",  # Water question
        "author", "1984", "george", "orwell", "wrote", "write",  # 1984 question
        "speed", "light", "fast", "299", "792", "km", "second"  # Light question
    ]
    vec = [0]*len(vocab)
    for i, word in enumerate(vocab):
        if word in text.lower():
            vec[i] = 1
    return vec

EMBEDDINGS = [embed(d) for d in DATA]
