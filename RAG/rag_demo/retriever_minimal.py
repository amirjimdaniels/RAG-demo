from data import embed
import math

class MinimalRetriever:
    def __init__(self, data):
        self.data = data
        self.embeddings = [embed(d) for d in data]
    def retrieve(self, query, top_k=1):
        q_emb = embed(query)
        def cosine(a, b):
            dot = sum(x*y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x*x for x in a))
            norm_b = math.sqrt(sum(x*x for x in b))
            return dot / (norm_a * norm_b + 1e-8)
        sims = [cosine(q_emb, e) for e in self.embeddings]
        idx = sims.index(max(sims))
        return self.data[idx]
