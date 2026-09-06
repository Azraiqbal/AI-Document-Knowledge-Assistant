import faiss
import numpy as np


class VectorStoreService:
    def __init__(self):
        self.index = None
        self.chunks = []

    def build_index(self, embeddings, chunks):
        if embeddings is None or len(embeddings) == 0:
            raise ValueError("No embeddings were provided.")

        if not chunks:
            raise ValueError("No document chunks were provided.")

        if len(embeddings) != len(chunks):
            raise ValueError(
                "The number of embeddings and chunks must be equal."
            )

        vectors = np.asarray(
            embeddings,
            dtype="float32",
        )

        vector_dimension = vectors.shape[1]

        self.index = faiss.IndexFlatIP(vector_dimension)
        self.index.add(vectors)

        self.chunks = chunks

    def search(self, query_embedding, top_k=4):
        if self.index is None or self.index.ntotal == 0:
            raise ValueError(
                "Process a document before asking questions."
            )

        query_vector = np.asarray(
            [query_embedding],
            dtype="float32",
        )

        result_count = min(
            top_k,
            self.index.ntotal,
        )

        scores, indices = self.index.search(
            query_vector,
            result_count,
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0],
        ):
            if index == -1:
                continue

            result = self.chunks[index].copy()
            result["score"] = float(score)

            results.append(result)

        return results

    def clear(self):
        self.index = None
        self.chunks = []