import faiss
import numpy as np

class VectorStore:

    def __init__(self, embeddings, documents):

        self.documents = documents

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dimension)

        self.index.add(embeddings)

    def search(self, query_embedding, top_k=5):

        D, I = self.index.search(
            np.array([query_embedding]),
            top_k
        )

        results = [self.documents[i] for i in I[0]]

        return results