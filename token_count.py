import sys
import os
import fitz  # PyMuPDF
import tiktoken

# Set terminal encoding to UTF-8 (for Windows)
sys.stdout.reconfigure(encoding='utf-8')

# Choose the encoding for the model you're using (e.g., "gpt-4", "gpt-3.5-turbo")
encoding = tiktoken.encoding_for_model("gpt-4")

# Folder containing PDFs
pdf_folder = r"C:\Users\lenovo\Downloads\OneDrive_2024-12-09\SOP to review"


total_tokens = 0

for filename in os.listdir(pdf_folder):
    if filename.lower().endswith(".pdf"):
        filepath = os.path.join(pdf_folder, filename)
        doc = fitz.open(filepath)
        text = ""
        for page in doc:
            text += page.get_text()
        tokens = len(encoding.encode(text))
        print(f"{filename}: {tokens} tokens\n\n")
        total_tokens += tokens
        

print(f"\n Total tokens in all PDFs: {total_tokens}")
