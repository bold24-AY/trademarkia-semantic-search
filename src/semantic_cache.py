from sklearn.metrics.pairwise import cosine_similarity

class SemanticCache:

    def __init__(self, threshold=0.85):

        self.entries = []
        self.threshold = threshold
        self.hit = 0
        self.miss = 0

    def lookup(self, embedding):

        for entry in self.entries:

            sim = cosine_similarity(
                embedding.reshape(1,-1),
                entry["embedding"].reshape(1,-1)
            )[0][0]

            if sim > self.threshold:

                self.hit += 1
                return entry

        self.miss += 1
        return None


    def add(self, query, embedding, result):

        self.entries.append({
            "query": query,
            "embedding": embedding,
            "result": result
        })


    def stats(self):

        total = self.hit + self.miss

        return {
            "total_entries": len(self.entries),
            "hit_count": self.hit,
            "miss_count": self.miss,
            "hit_rate": self.hit / total if total else 0
        }