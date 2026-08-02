import os
import re
import pickle
import faiss
import numpy as np
from typing import List, Dict, Any
import pypdf  # pip install pypdf
from sentence_transformers import SentenceTransformer

# ============================================
# CONFIGURATION & PATHS
# ============================================

PDF_DIR = r"C:\Users\riaze\Desktop\TechCorp-Financial-Enterprise-RAG\apple_pipeline\pdfs"
VECTOR_FOLDER = r"C:\Users\riaze\Desktop\TechCorp-Financial-Enterprise-RAG\apple_pipeline\vector_store"
INDEX_PATH = os.path.join(VECTOR_FOLDER, "apple.index")
METADATA_PATH = os.path.join(VECTOR_FOLDER, "metadata.pkl")

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 800       # Words/Characters window
CHUNK_OVERLAP = 150    # Overlap to prevent context loss

# Ensure directories exist
os.makedirs(VECTOR_FOLDER, exist_ok=True)

# ============================================
# ENTERPRISE CHUNKER & PARSER
# ============================================

class FinancialDocumentProcessor:
    def __init__(self, model_name: str):
        print("Loading embedding model for indexing...")
        self.model = SentenceTransformer(model_name)
        self.chunks_metadata = []
        self.text_chunks = []

    def extract_year_from_filename(self, filename: str) -> str:
        match = re.search(r'\b(20\d{2})\b', filename)
        return match.group(1) if match else "Unknown"

    def clean_text(self, text: str) -> str:
        # Basic regex cleaning to keep table structures readable
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        return text.strip()

    def process_pdf(self, pdf_path: str, company_name: str = "APPLE"):
        filename = os.path.basename(pdf_path)
        year = self.extract_year_from_filename(filename)
        print(f"Processing: {filename} | Company: {company_name} | Year: {year}")

        reader = pypdf.PdfReader(pdf_path)
        full_text = ""
        
        # Extract page by page
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text:
                full_text += f"\n--- Page {page_num} ---\n" + text

        cleaned_text = self.clean_text(full_text)
        
        # Financial Section Ingestion (Detecting main headings)
        sections = re.split(r'(\nITEM\s+[0-9A-Z]+\.|\nCONSOLIDATED STATEMENTS OF [A-Z ]+)', cleaned_text, flags=re.IGNORECASE)
        
        current_section = "General Financial Information"
        chunk_counter = len(self.chunks_metadata)

        for i in range(0, len(sections)):
            part = sections[i].strip()
            if not part:
                continue

            # Update section header if matched
            if part.upper().startswith("ITEM") or part.upper().startswith("CONSOLIDATED"):
                current_section = part.replace('\n', ' ')
                continue

            # Slit section text into overlapping chunks
            words = part.split()
            for j in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
                chunk_words = words[j:j + CHUNK_SIZE]
                if len(chunk_words) < 20: # Ignore tiny fragments
                    continue

                chunk_text = " ".join(chunk_words)
                
                # Context Enrichment Prefix (Injected before Embedding)
                enriched_text = f"[Company: {company_name} | Year: {year} | Section: {current_section}] {chunk_text}"

                metadata_entry = {
                    "chunk_id": chunk_counter,
                    "company": company_name,
                    "year": year,
                    "section": current_section,
                    "parent_section": current_section,
                    "text": enriched_text
                }

                self.text_chunks.append(enriched_text)
                self.chunks_metadata.append(metadata_entry)
                chunk_counter += 1

    def build_and_save_index(self, index_path: str, metadata_path: str):
        print(f"\nTotal Chunks Created: {len(self.text_chunks)}")
        print("Generating Dense Embeddings...")
        
        embeddings = self.model.encode(
            self.text_chunks, 
            normalize_embeddings=True, 
            show_progress_bar=True
        ).astype("float32")

        # Create FAISS L2/Cosine Index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension) # Inner Product for Normalized Vectors (Cosine Sim)
        index.add(embeddings)

        print(f"Saving FAISS Index to: {index_path}")
        faiss.write_index(index, index_path)

        print(f"Saving Metadata Pickle to: {metadata_path}")
        with open(metadata_path, "wb") as f:
            pickle.dump(self.chunks_metadata, f)

        print("✅ Pipeline Extraction & Vector Indexing Completed Successfully!")

# ============================================
# EXECUTION PIPELINE
# ============================================

if __name__ == "__main__":
    processor = FinancialDocumentProcessor(EMBEDDING_MODEL_NAME)

    if not os.path.exists(PDF_DIR):
        print(f"Directory Error: '{PDF_DIR}' does not exist. Please place Apple PDFs in this folder.")
    else:
        pdf_files = [os.path.join(PDF_DIR, f) for f in os.listdir(PDF_DIR) if f.endswith('.pdf')]
        
        if not pdf_files:
            print(f"No PDFs found in {PDF_DIR}. Please add Apple 10-K PDF files.")
        else:
            for pdf_path in pdf_files:
                processor.process_pdf(pdf_path, company_name="APPLE")

            processor.build_and_save_index(INDEX_PATH, METADATA_PATH)