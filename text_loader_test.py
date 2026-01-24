from pathlib import Path
from src.ingestion.text_loader import load_text_files_from_directory

docs = load_text_files_from_directory(Path("data/raw/text"))
print(docs[0].title)
print(docs[0].content)
