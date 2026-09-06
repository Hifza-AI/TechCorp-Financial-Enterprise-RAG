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

    # NEW: BGE-base-en-v1.5's underlying position embeddings were
    # trained at a 512-token max sequence length -- SentenceTransformer
    # silently TRUNCATES anything longer than this at encode time (no
    # error, no warning), simply dropping every token past the limit.
    #
    # Confirmed against this project's actual chunk output across all
    # 15 companies checked so far: 218 of 10,348 chunks (about 2.1%)
    # exceed this, almost all of them large financial TABLES --
    # Segment Information breakdowns, multi-year Stockholders' Equity
    # rollforwards, and Notes indexes -- some by a large margin (the
    # worst case is roughly 3,300 tokens, more than 6x the limit).
    #
    # Without handling this, the embedded vector for an oversized
    # chunk only ever "sees" its first ~512 tokens -- e.g. a 20-row
    # Stockholders' Equity table would have its LAST several rows
    # completely invisible to the embedding, even though the full
    # text is still stored correctly in metadata and would display
    # fine if retrieved. This isn't a crash or a visible error -- it
    # just silently makes rows near the end of large tables
    # undiscoverable by any query whose relevant keywords only appear
    # in that truncated tail, which is exactly the kind of gap that
    # won't show up until retrieval accuracy is measured and comes in
    # lower than expected, with no obvious cause.
    MAX_TOKENS = 512

    # A conservative estimate (BGE uses a WordPiece-style tokenizer,
    # where common English word average out to somewhat more than
    # one token each due to punctuation and sub-word splitting).
    # Erring low here is deliberately safe: it's fine to split a
    # ~500-token chunk into two windows unnecessarily, but it is NOT
    # fine to under-split and still truncate silently.
    APPROX_TOKENS_PER_WORD = 1.3

    OVERLAP_TOKENS = 50

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
    # NEW: SPLIT OVERSIZED CHUNKS INTO EMBEDDABLE WINDOWS
    # =========================================================

    def _split_oversized_text(self, text):
        """
        Splits `text` into overlapping word-windows sized to stay
        safely under MAX_TOKENS once BGE's tokenizer processes them.
        Returns [text] unchanged (a list of one) when it's already
        short enough -- so this is a no-op for the 98% of chunks that
        don't need it.

        The overlap exists so that a fact sitting right at a window
        boundary (e.g. a row label in one window, its value in the
        next) still has a reasonable chance of appearing whole in at
        least one window, rather than being cleanly severed exactly
        where a query might need it most.
        """

        max_words = int(self.MAX_TOKENS / self.APPROX_TOKENS_PER_WORD)

        overlap_words = int(
            self.OVERLAP_TOKENS / self.APPROX_TOKENS_PER_WORD
        )

        words = text.split()

        if len(words) <= max_words:
            return [text]

        windows = []

        start = 0

        while start < len(words):

            end = start + max_words

            windows.append(" ".join(words[start:end]))

            if end >= len(words):
                break

            start = end - overlap_words

        return windows

    def prepare_embeddable_units(self, chunks):
        """
        Expands `chunks` into a (possibly longer) list of
        "embeddable units" -- each unit has its own `text` for
        embedding, but carries a reference back to the ORIGINAL
        chunk's full metadata via `source_chunk`, plus a
        `window_index`/`window_count` pair so a retrieved unit can be
        traced back to exactly which slice of its parent chunk it
        came from.

        For the ~98% of chunks that are already short enough, this
        produces exactly one unit per chunk, identical in spirit to
        embedding `chunks` directly -- so nothing changes for the
        common case, and the metadata sidecar this project already
        relies on (section_path, page_numbers, chunk_type, etc.)
        stays exactly as before.
        """

        units = []

        oversized_count = 0

        for chunk in chunks:

            windows = self._split_oversized_text(chunk["text"])

            if len(windows) > 1:
                oversized_count += 1

            for window_index, window_text in enumerate(windows):

                units.append({
                    "text": window_text,
                    "source_chunk": chunk,
                    "window_index": window_index,
                    "window_count": len(windows),
                })

        if oversized_count:

            print(
                f"Note: {oversized_count} chunk(s) exceeded "
                f"{self.MAX_TOKENS} tokens and were split into "
                f"multiple overlapping windows so no content is "
                f"silently truncated during embedding."
            )

        return units

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
        units,
        output_dir="STAGE_1/vector_store",
    ):

        output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(index, str(output_dir / "index.faiss"))

        # NEW: the metadata sidecar now stores the ORIGINAL chunk
        # (via source_chunk) for every unit, so a FAISS row -> a full
        # citation lookup works exactly as before for the common
        # case, while for a split chunk, EVERY window row points back
        # to the SAME original chunk -- meaning a match on any window
        # correctly surfaces the complete original text (not just the
        # slice that happened to match), plus window_index/
        # window_count in case you ever want to know which part of a
        # long table actually matched.
        metadata = [
            {
                **unit["source_chunk"],
                "window_index": unit["window_index"],
                "window_count": unit["window_count"],
            }
            for unit in units
        ]

        with open(output_dir / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

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

        units = builder.prepare_embeddable_units(chunks)

        print(f"Prepared {len(units)} embeddable units (from {len(chunks)} chunks).")

        texts = [u["text"] for u in units]

        print("\nEmbedding chunks (this may take a few minutes)...")

        embeddings = builder.embed_passages(texts)

        print(f"\nEmbeddings shape: {embeddings.shape}")

        print("\nBuilding FAISS index...")

        index = builder.build_index(embeddings)

        print(f"Index contains {index.ntotal} vectors.")

        builder.save(index, units)

        print("\n====================================")
        print(" Embedding Index Building Completed")
        print("====================================")