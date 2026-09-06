from google import genai

from src.config import settings
from src.services.embedding_service import EmbeddingService
from src.services.vector_store_service import VectorStoreService


class RAGService:
    def __init__(self):
        if not settings.GEMINI_API_KEY:
            raise ValueError(
                "Gemini API key is not configured in the .env file."
            )

        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStoreService()

        self.client = genai.Client(
            api_key=settings.GEMINI_API_KEY
        )

        self.document_name = None

    def index_document(self, chunks):
        if not chunks:
            raise ValueError("No document chunks were provided.")

        texts = [chunk["text"] for chunk in chunks]

        embeddings = self.embedding_service.create_embeddings(texts)

        self.vector_store.build_index(
            embeddings=embeddings,
            chunks=chunks,
        )

        self.document_name = chunks[0]["source"]

    def ask_question(self, question):
        if not question or not question.strip():
            raise ValueError("Please enter a question.")

        question = question.strip()

        query_embedding = (
            self.embedding_service.create_query_embedding(question)
        )

        retrieved_chunks = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=settings.TOP_K_RESULTS,
        )

        if not retrieved_chunks:
            return {
                "answer": (
                    "I could not find this information "
                    "in the uploaded document."
                ),
                "sources": [],
            }

        context_parts = []

        for number, chunk in enumerate(retrieved_chunks, start=1):
            context_parts.append(
                f"[Source {number} | "
                f"File: {chunk['source']} | "
                f"Page: {chunk['page']}]\n"
                f"{chunk['text']}"
            )

        context = "\n\n".join(context_parts)

        summary_phrases = [
            "main idea",
            "main topic",
            "summary",
            "summarize",
            "overview",
            "document about",
        ]

        is_summary_question = any(
        phrase in question.lower()
        for phrase in summary_phrases
        )

        if is_summary_question:
            task_instruction = (
                "The user is asking for the document's overall meaning. "
                "Summarize the central subject and key points from the context. "
                "This question is answerable, so provide a summary."
            )
        else:
            task_instruction = (
                "Answer the question using information supported by the context."
            )

        prompt = f"""
        You are an AI Document Knowledge Assistant.

        TASK:
        {task_instruction}

        RULES:
        1. Use only the supplied document context.
        2. You may summarize and combine facts directly supported by the context.
        3. Do not require the exact question words to appear in the document.
        4. Do not use outside knowledge or invent information.
        5. Only when the context is genuinely unrelated, respond:
            "I could not find this information in the uploaded document."
        6. Answer in the same language as the question.
        7. Keep the answer clear and concise.

        DOCUMENT CONTEXT:
        {context}

        QUESTION:
        {question}

        ANSWER:
        """
        try:
            response = self.client.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=prompt,
                config={
                    "temperature": 0,
                    "max_output_tokens": 600,
                    "http_options": {
                        "timeout": 20000
                    },
                },
            )

            answer = response.text.strip() if response.text else ""

            if not answer:
                answer = (
                    "I could not find this information "
                    "in the uploaded document."
                )

            unsupported_message = (
                "I could not find this information "
                "in the uploaded document."
            )

            if unsupported_message.lower() in answer.lower():
                answer = unsupported_message
                sources = []
            else:
                sources = [
                    {
                        "source": chunk["source"],
                        "page": chunk["page"],
                        "chunk": chunk["chunk"],
                        "score": chunk["score"],
                        "text": chunk["text"],
                    }
                    for chunk in retrieved_chunks
                ]

            return {
                "answer": answer,
                "sources": sources,
            }

        except Exception as error:
            raise RuntimeError(
                f"Gemini error: {type(error).__name__}: {error}"
            ) from error

    def clear_document(self):
        self.vector_store.clear()
        self.document_name = None
