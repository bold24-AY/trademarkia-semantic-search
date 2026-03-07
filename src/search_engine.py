from src.embeddings import embed_text
import re
import numpy as np
from src.embeddings import embed_text


def extract_snippet(query, document):

    document = document.replace("\n", " ")

    # split into sentences
    sentences = re.split(r'[.!?]', document)

    # remove very short sentences
    sentences = [s.strip() for s in sentences if len(s.strip()) > 40]

    if len(sentences) == 0:
        return document[:250]

    query_embedding = embed_text(query)

    best_sentence = sentences[0]
    best_score = -1

    for sentence in sentences:

        sentence_embedding = embed_text(sentence)

        score = np.dot(query_embedding, sentence_embedding)

        if score > best_score:
            best_score = score
            best_sentence = sentence

    return best_sentence
class SearchEngine:

    def __init__(self, vector_store):
        self.vector_store = vector_store
    def search(self, query, top_k=5):

        query_embedding = embed_text(query)

        results = self.vector_store.search(query_embedding, top_k)

        snippets = []

        for doc in results:
            snippet = extract_snippet(query, doc)
            snippets.append(snippet)

        return snippets 