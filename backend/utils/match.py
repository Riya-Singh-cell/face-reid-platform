from numpy import dot
from numpy.linalg import norm

def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))

def is_match(emb1, emb2, threshold=0.7):
    sim = cosine_similarity(emb1, emb2)
    return sim > threshold, sim