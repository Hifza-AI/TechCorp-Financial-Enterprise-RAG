import json
import pickle
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class EmbeddingIndexBuilder:
    """
    Loads ALL chunk files (across every company/report already
    chunked), embeds them with BGE-base-en-v1.5, and builds a FAISS
    index. A separate metadata sidecar list is saved alongside the
    index so that a FAISS result (a row number) can be mapped back
    to its actual text + company/year/section_path/page_numbers.

    BGE models are trained with an asymmetric convention: passages
    (the chunks we're indexing) are embedded as-is, but QUERIES at
    retrieval time need a specific instruction prefix prepended --
    this is documented behavior for BGE and materially affects
    retrieval quality if skipped. See `embed_query()` below.
    """

    MODEL_NAME = "BAAI/bge-base-en-v1.5"

    QUERY_INSTRUCTION = (
        "Represent this sentence for searching relevant passages: "
    )

    def __init__(self):

        print(f"Loading embedding model: {self.MODEL_NAME} ...")

        self.model = SentenceTransformer(self.MODEL_NAME)

        self.dimension = self.model.get_sentence_embedding_dimension()

        print(f"Model loaded. Embedding dimension: {self.dimension}")

    # =========================================================
    # LOAD ALL CHUNKS (every company, every report)
    # =========================================================

    def load_all_chunks(self, chunks_dir="STAGE_1/chunks"):

        chunks_dir = Path(chunks_dir)

        all_chunks = []

        for company_dir in sorted(chunks_dir.iterdir()):

            if not company_dir.is_dir():
                continue

            for json_file in sorted(company_dir.glob("*_chunks.json")):

                with open(json_file, "r", encoding="utf-8") as f:

                    chunks = json.load(f)

                    all_chunks.extend(chunks)

        return all_chunks

    # =========================================================
    # EMBED PASSAGES (the chunks being indexed)
    # =========================================================

    def embed_passages(self, texts, batch_size=32):

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,  # required for cosine similarity via inner product
            convert_to_numpy=True,
        )

        return embeddings.astype("float32")

    # =========================================================
    # EMBED A QUERY (asymmetric -- needs the instruction prefix)
    # =========================================================

    def embed_query(self, query_text):

        prefixed = self.QUERY_INSTRUCTION + query_text

        embedding = self.model.encode(
            [prefixed],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return embedding.astype("float32")

    # =========================================================
    # BUILD FAISS INDEX
    # =========================================================

    def build_index(self, embeddings):

        # Flat index with inner product = cosine similarity, since
        # embeddings are normalized above. For a few thousand chunks
        # (our scale), an exact flat index is fast enough -- no need
        # for an approximate index (IVF/HNSW) at this size.
        index = faiss.IndexFlatIP(self.dimension)

        index.add(embeddings)

        return index

    # =========================================================
    # SAVE INDEX + METADATA SIDECAR
    # =========================================================

    def save(
        self,
        index,
        chunks,
        output_dir="STAGE_1/vector_store",
    ):

        output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(index, str(output_dir / "index.faiss"))

        # The metadata sidecar is a plain list, in the SAME ORDER the
        # chunks were added to the index -- so FAISS row i always
        # corresponds to metadata[i]. This is the join key between
        # "a vector matched" and "here's the actual text/citation".
        with open(output_dir / "metadata.pkl", "wb") as f:
            pickle.dump(chunks, f)

        print(f"\nSaved FAISS index and metadata to: {output_dir}")


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":

    print("\n====================================")
    print(" Embedding Index Builder Started")
    print("====================================\n")

    builder = EmbeddingIndexBuilder()

    print("\nLoading all chunks...")

    chunks = builder.load_all_chunks()

    print(f"Loaded {len(chunks)} chunks total.")

    if not chunks:

        print("No chunks found. Run ChunkBuilder first.")

    else:

        texts = [c["text"] for c in chunks]

        print("\nEmbedding chunks (this may take a few minutes)...")

        embeddings = builder.embed_passages(texts)

        print(f"\nEmbeddings shape: {embeddings.shape}")

        print("\nBuilding FAISS index...")

        index = builder.build_index(embeddings)

        print(f"Index contains {index.ntotal} vectors.")

        builder.save(index, chunks)

        print("\n====================================")
        print(" Embedding Index Building Completed")
        print("====================================")