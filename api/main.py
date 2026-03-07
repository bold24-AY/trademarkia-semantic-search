from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

from src.embeddings import embed_text
from src.vector_store import VectorStore
from src.search_engine import SearchEngine
from src.semantic_cache import SemanticCache


# ---------------------------------------
# Initialize FastAPI app
# ---------------------------------------

app = FastAPI(
    title="Semantic Search API",
    description="Semantic search with FAISS + semantic cache",
    version="1.0"
)


# ---------------------------------------
# Load saved embeddings and documents
# ---------------------------------------

print("Loading embeddings...")

embeddings = np.load("data/embeddings.npy")

print("Loading documents...")

with open("data/documents.txt", "r", encoding="latin1") as f:
    documents = [line.strip() for line in f]


# ---------------------------------------
# Initialize system components
# ---------------------------------------

print("Initializing vector store...")

vector_store = VectorStore(embeddings, documents)

print("Initializing search engine...")

search_engine = SearchEngine(vector_store)

print("Initializing semantic cache...")

cache = SemanticCache()


# ---------------------------------------
# Request schema
# ---------------------------------------

class QueryRequest(BaseModel):
    query: str


# ---------------------------------------
# Query endpoint
# ---------------------------------------

@app.post("/query")
def query_api(request: QueryRequest):

    query = request.query

    # create embedding
    embedding = embed_text(query)

    # check cache
    cached = cache.lookup(embedding)

    if cached:
        return {
            "query": query,
            "cache_hit": True,
            "matched_query": cached["query"],
            "result": cached["result"]
        }

    # run semantic search
    results = search_engine.search(query)

    # store in cache
    cache.add(query, embedding, results)

    return {
        "query": query,
        "cache_hit": False,
        "result": results
    }


# ---------------------------------------
# Cache statistics endpoint
# ---------------------------------------

@app.get("/cache/stats")
def cache_stats():

    return cache.stats()


# ---------------------------------------
# Clear cache endpoint
# ---------------------------------------

@app.delete("/cache")
def clear_cache():

    cache.entries = []

    return {"message": "Cache cleared"}