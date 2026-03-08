import joblib

def load_clustering_model(path="data/gmm_model.pkl"):
    """
    Load the trained Gaussian Mixture Model used
    for fuzzy clustering of document embeddings.
    """
    return joblib.load(path)


def predict_cluster(model, embedding):
    """
    Predict cluster probabilities for a query embedding.
    """
    probs = model.predict_proba(embedding.reshape(1, -1))
    return probs