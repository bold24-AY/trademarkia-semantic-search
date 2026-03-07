from sentence_transformers import SentenceTransformer

# Load embedding model once
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(text):
    """
    Convert text query into embedding vector
    """
    return model.encode([text])[0]