from pathlib import Path
import os
from dotenv import load_dotenv

# --------------------------------------------------
# Load environment variables
# --------------------------------------------------
load_dotenv()

# --------------------------------------------------
# Base Project Paths
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

PDF_DIR = RAW_DATA_DIR / "pdfs"
TEXT_DIR = RAW_DATA_DIR / "text"
MARKDOWN_DIR = RAW_DATA_DIR / "markdown"

INDEX_DIR = BASE_DIR / "indexes" / "faiss"

# Ensure directories exist
for path in [
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    PDF_DIR,
    TEXT_DIR,
    MARKDOWN_DIR,
    INDEX_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# API Keys
# --------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GROQ_API_KEY:
    raise EnvironmentError("❌ GROQ_API_KEY not found in .env file")

# Tavily is optional (only for web / hybrid search)
TAVILY_ENABLED = True if TAVILY_API_KEY else False

# --------------------------------------------------
# LLM Configuration (Groq)
# --------------------------------------------------
GROQ_MODEL_NAME = "llama3-70b-8192"
LLM_TEMPERATURE = 0.2
MAX_TOKENS = 2048

# --------------------------------------------------
# Chunking Configuration
# --------------------------------------------------
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --------------------------------------------------
# Retrieval Configuration
# --------------------------------------------------
TOP_K_DOCUMENTS = 5
TOP_K_WEB_RESULTS = 5

# --------------------------------------------------
# UI Defaults
# --------------------------------------------------
DEFAULT_SEARCH_MODE = "hybrid"  
# options: "document", "web", "hybrid"

ENABLE_STREAMING = True
DARK_THEME = True

# --------------------------------------------------
# FAISS Settings
# --------------------------------------------------
FAISS_INDEX_NAME = "index"
FAISS_INDEX_PATH = INDEX_DIR / FAISS_INDEX_NAME

# --------------------------------------------------
# Debug / Logging
# --------------------------------------------------
DEBUG = True
