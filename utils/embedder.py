from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list:
        embeddings = self.model.encode(texts)
        # .tolist() converts numpy arrays → plain Python lists
        # (needed for ChromaDB and JSON serialization)
        return embeddings.tolist()