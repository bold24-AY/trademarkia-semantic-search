from src.embeddings import embed_text
import numpy as np


class SearchEngine:

    def __init__(self, vector_store):
        self.vector_store = vector_store

    def search(self, query, top_k=5):

        query_embedding = embed_text(query)

        results = self.vector_store.search(query_embedding, top_k)

        return results