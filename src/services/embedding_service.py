from functools import lru_cache

from sentence_transformers import SentenceTransformer

from src.config import settings


@lru_cache(maxsize=1)
def load_embedding_model():
    return SentenceTransformer(settings.EMBEDDING_MODEL)


class EmbeddingService:
    def __init__(self):
        self.model = load_embedding_model()

    def create_embeddings(self, texts):
        if not texts:
            raise ValueError("No text was provided for embedding.")

        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        ).astype("float32")

    def create_query_embedding(self, question):
        if not question or not question.strip():
            raise ValueError("Please enter a question.")

        return self.model.encode(
            question.strip(),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")