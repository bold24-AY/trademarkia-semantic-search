from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.mixture import GaussianMixture
import joblib

from src.embeddings import embed_text
from src.vector_store import VectorStore
from src.search_engine import SearchEngine
from src.semantic_cache import SemanticCache


# -----------------------------------
# Initialize FastAPI
# -----------------------------------

app = FastAPI(
    title="Semantic Search API",
    description="Semantic search with FAISS, fuzzy clustering, and semantic cache",
    version="1.0"
)


# -----------------------------------
# Load stored data
# -----------------------------------

print("Loading embeddings...")

embeddings = np.load("data/embeddings.npy")

print("Loading documents...")

with open("data/documents.txt", "r", encoding="latin1") as f:
    documents = [line.strip() for line in f]


# -----------------------------------
# Load clustering model
# -----------------------------------

print("Loading clustering model...")

gmm = joblib.load("data/gmm_model.pkl")


# -----------------------------------
# Initialize system components
# -----------------------------------

print("Initializing vector store...")

vector_store = VectorStore(embeddings, documents)

print("Initializing search engine...")

search_engine = SearchEngine(vector_store)

print("Initializing semantic cache...")

cache = SemanticCache()


# -----------------------------------
# Request schema
# -----------------------------------

class QueryRequest(BaseModel):
    query: str


# -----------------------------------
# Query endpoint
# -----------------------------------

@app.post("/query")
def query_api(request: QueryRequest):

    query = request.query

    # embed query
    embedding = embed_text(query)

    # predict cluster
    cluster_probs = gmm.predict_proba([embedding])
    dominant_cluster = int(np.argmax(cluster_probs))

    # check cache
    cached = cache.lookup(embedding)

    if cached:

        similarity = float(
            cosine_similarity(
                embedding.reshape(1,-1),
                cached["embedding"].reshape(1,-1)
            )[0][0]
        )

        return {
            "query": query,
            "cache_hit": True,
            "matched_query": cached["query"],
            "similarity_score": similarity,
            "dominant_cluster": dominant_cluster,
            "result": cached["result"]
        }

    # cache miss → run search
    results = search_engine.search(query)

    cache.add(query, embedding, results)

    return {
        "query": query,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": None,
        "dominant_cluster": dominant_cluster,
        "result": results
    }


# -----------------------------------
# Cache statistics endpoint
# -----------------------------------

@app.get("/cache/stats")
def cache_stats():

    return cache.stats()


# -----------------------------------
# Clear cache endpoint
# -----------------------------------

@app.delete("/cache")
def clear_cache():

    cache.entries = []

    cache.hit = 0
    cache.miss = 0

    return {"message": "Cache cleared"}