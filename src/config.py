import os
from pathlib import Path

from dotenv import load_dotenv


# Project ke main folder ka path
BASE_DIR = Path(__file__).resolve().parent.parent

# .env file load karna
load_dotenv(BASE_DIR / ".env")


class Settings:
    APP_NAME = "AI Document / Knowledge Assistant"

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    GEMINI_MODEL = "gemini-2.5-flash"

    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    UPLOAD_DIR = BASE_DIR / "uploads"

    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 150

    TOP_K_RESULTS = 3

    ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}


settings = Settings()

# Upload folder available hona chahiye
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)