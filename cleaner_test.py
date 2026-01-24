from src.preprocessing.cleaner import clean_text

raw = "Hello   world\n\n\nThis is   a test.\x00"
print(clean_text(raw))