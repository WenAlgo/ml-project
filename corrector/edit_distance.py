import editdistance

class SimpleCandidateGenerator:
    def __init__(self, vocabulary):
        self.vocab = list(set(w for w in vocabulary if w and len(w)>0))

    def candidates(self, token, max_cand=10):
        t = token.lower()
        scored = []
        for w in self.vocab:
            if abs(len(w)-len(t)) > 6:
                continue
            scored.append((editdistance.eval(t, w.lower()), w))
        scored.sort(key=lambda x: x[0])
        return [w for d,w in scored[:max_cand]]
