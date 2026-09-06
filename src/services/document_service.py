from io import BytesIO
from pathlib import Path

import pymupdf
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings


class DocumentService:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def extract_pdf(self, file_bytes):
        pages = []

        with pymupdf.open(stream=file_bytes, filetype="pdf") as document:
            for page_number, page in enumerate(document, start=1):
                text = page.get_text("text").strip()

                if text:
                    pages.append(
                        {
                            "text": text,
                            "page": page_number,
                        }
                    )

        return pages

    def extract_docx(self, file_bytes):
        document = Document(BytesIO(file_bytes))

        text = "\n".join(
            paragraph.text
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        ).strip()

        if not text:
            return []

        return [
            {
                "text": text,
                "page": 1,
            }
        ]

    def extract_txt(self, file_bytes):
        try:
            text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = file_bytes.decode("latin-1")

        text = text.strip()

        if not text:
            return []

        return [
            {
                "text": text,
                "page": 1,
            }
        ]

    def extract_text(self, file_bytes, filename):
        extension = Path(filename).suffix.lower()

        if extension not in settings.ALLOWED_EXTENSIONS:
            raise ValueError(
                "Unsupported file type. Please upload a PDF, DOCX or TXT file."
            )

        if extension == ".pdf":
            return self.extract_pdf(file_bytes)

        if extension == ".docx":
            return self.extract_docx(file_bytes)

        return self.extract_txt(file_bytes)

    def create_chunks(self, pages, filename):
        chunks = []
        chunk_number = 1

        for page_data in pages:
            page_chunks = self.text_splitter.split_text(page_data["text"])

            for chunk_text in page_chunks:
                chunks.append(
                    {
                        "text": chunk_text,
                        "source": filename,
                        "page": page_data["page"],
                        "chunk": chunk_number,
                    }
                )

                chunk_number += 1

        return chunks

    def process_document(self, file_bytes, filename):
        if not file_bytes:
            raise ValueError("The uploaded document is empty.")

        pages = self.extract_text(file_bytes, filename)

        if not pages:
            raise ValueError(
                "No readable text was found in the uploaded document."
            )

        chunks = self.create_chunks(pages, filename)

        if not chunks:
            raise ValueError("Text chunks could not be created.")

        return {
            "filename": filename,
            "page_count": len(pages),
            "chunk_count": len(chunks),
            "chunks": chunks,
        }