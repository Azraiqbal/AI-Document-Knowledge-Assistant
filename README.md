# AI Document / Knowledge Assistant

An AI-powered document question-answering application built using Retrieval-Augmented Generation (RAG). Users can upload custom documents and receive context-aware answers with page-level source references.

## Features

- Upload PDF, DOCX and TXT documents
- Extract text from uploaded documents
- Divide extracted text into overlapping chunks
- Generate semantic embeddings
- Store and search embeddings using FAISS
- Retrieve relevant document context
- Generate answers using Google Gemini
- Display filenames, page numbers and relevance scores
- Handle unsupported questions without inventing information
- Maintain question-and-answer history
- Secure API-key management
- User-friendly Streamlit interface
- Proper validation and error handling

## RAG Workflow

1. The user uploads a document.
2. Text is extracted from the document.
3. Extracted text is divided into smaller chunks.
4. Sentence Transformers convert the chunks into embeddings.
5. FAISS stores the embeddings for similarity search.
6. The user's question is converted into an embedding.
7. The most relevant document chunks are retrieved.
8. Gemini generates an answer using only the retrieved context.
9. The answer is displayed with source references.

## Technology Stack

- Python
- Streamlit
- PyMuPDF
- python-docx
- LangChain Text Splitters
- Sentence Transformers
- FAISS
- Google Gemini API
- python-dotenv

## Project Structure

```text
AI-Document-Knowledge-Assistant/
├── app.py
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── uploads/
│   └── .gitkeep
└── src/
    ├── __init__.py
    ├── config.py
    └── services/
        ├── __init__.py
        ├── document_service.py
        ├── embedding_service.py
        ├── vector_store_service.py
        └── rag_service.py
```

## Installation

Clone the repository:

```bash
git clone https://github.com/Azraiqbal/AI-Document-Knowledge-Assistant.git
cd AI-Document-Knowledge-Assistant
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Environment Configuration

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

The `.env` file is excluded from GitHub through `.gitignore`.

## Run the Application

```bash
streamlit run app.py
```

Open the displayed local URL in your browser:

```text
http://localhost:8501
```

## Usage

1. Upload a PDF, DOCX or TXT document.
2. Click **Process Document**.
3. Wait for text processing and indexing to complete.
4. Enter a question related to the document.
5. Review the generated answer and source references.
6. Use **Clear Document & Chat** to start again.

## Unsupported Questions

The assistant is instructed to answer only from the uploaded document. If relevant information is unavailable, it responds:

```text
I could not find this information in the uploaded document.
```

## Error Handling

The application handles:

- Missing document uploads
- Unsupported file formats
- Empty documents
- Scanned PDFs without extractable text
- Missing API keys
- Empty questions
- Document-processing failures
- Gemini API failures
- Vector-indexing errors

## Limitations

- Image-only scanned PDFs require OCR and are not currently supported.
- Processing time depends on document length and system performance.
- Gemini API usage is subject to API quotas and availability.
- Uploaded documents are processed one at a time.

## Future Improvements

- OCR support for scanned documents
- Multiple-document processing
- Persistent vector database
- User authentication
- Downloadable chat reports
- Multilingual document analysis
- Voice-based questions

## Author

**Azra Iqbal**<br>
B.Tech CSE — Artificial Intelligence and Machine Learning<br>
Allenhouse Institute of Technology, Kanpur

## Internship

Developed for **Innovation Hacks AI Internship 2026 — Week 02**.