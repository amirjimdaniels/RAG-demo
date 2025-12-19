import numpy as np
from data import EMBEDDINGS, embed

class NumpyRetriever:
    def __init__(self, data):
        self.data = data
        self.embeddings = np.array([embed(d) for d in data])
    def retrieve(self, query, top_k=1):
        q_emb = np.array(embed(query))
        sims = self.embeddings @ q_emb / (np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(q_emb) + 1e-8)
        idx = np.argmax(sims)
        return self.data[idx]
