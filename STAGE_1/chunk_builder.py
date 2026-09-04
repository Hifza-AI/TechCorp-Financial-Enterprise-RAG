import json
import re
from pathlib import Path

class ChunkBuilder:
    """
    Walks the nested hierarchy tree (headings -> paragraphs/tables ->
    children) and flattens it into retrieval-ready chunks. Each chunk
    carries metadata (company, year, section path, page numbers) so
    the LLM can cite exactly where an answer came from.

    Two kinds of chunks are produced:
      - "text" chunks: paragraphs under a heading, grouped up to
        `max_chunk_chars`, with `chunk_overlap_chars` of trailing
        context repeated into the next chunk so a fact split across
        a chunk boundary doesn't get orphaned.
      - "table" chunks: one chunk per table, kept intact (never split
        mid-table, since a half-table is useless for a numeric answer).
    """

    def __init__(
        self,
        max_chunk_chars=1200,
        chunk_overlap_chars=150,
        min_chunk_chars=250,
    ):
        self.max_chunk_chars = max_chunk_chars
        self.chunk_overlap_chars = chunk_overlap_chars
        self.min_chunk_chars = min_chunk_chars

    def build_chunks(self, company, year, file_name, sections):

        chunks = []

        self._walk(
            nodes=sections,
            section_path=[],
            company=company,
            year=year,
            file_name=file_name,
            chunks=chunks,
        )

        chunks = self._merge_small_chunks(chunks)

        return chunks

    def _merge_small_chunks(self, chunks):

        merged = []

        buffer = None

        for chunk in chunks:

            if chunk["chunk_type"] != "text":

                if buffer is not None:
                    merged.append(buffer)
                    buffer = None

                merged.append(chunk)
                continue

            if buffer is None:
                buffer = dict(chunk)
                buffer["page_numbers"] = list(chunk.get("page_numbers", []))
                buffer["section_path"] = [chunk["section_path"]]
                continue

            same_context = (
                buffer["company"] == chunk["company"]
                and buffer["year"] == chunk["year"]
            )

            current_len = len(buffer["text"])

            combined_len = current_len + len(chunk["text"]) + 1

            if (
                same_context
                and current_len < self.min_chunk_chars
                and combined_len <= self.max_chunk_chars
            ):

                buffer["text"] = buffer["text"] + " " + chunk["text"]

                if chunk["section_path"] not in buffer["section_path"]:
                    buffer["section_path"].append(chunk["section_path"])

                for page in chunk.get("page_numbers", []):
                    if page not in buffer["page_numbers"]:
                        buffer["page_numbers"].append(page)

            else:

                buffer["section_path"] = " | ".join(buffer["section_path"])
                buffer["page_numbers"] = sorted(buffer["page_numbers"])

                merged.append(buffer)

                buffer = dict(chunk)
                buffer["page_numbers"] = list(chunk.get("page_numbers", []))
                buffer["section_path"] = [chunk["section_path"]]

        if buffer is not None:
            buffer["section_path"] = " | ".join(buffer["section_path"])
            buffer["page_numbers"] = sorted(buffer["page_numbers"])
            merged.append(buffer)

        return merged

    def _walk(self, nodes, section_path, company, year, file_name, chunks):

        for node in nodes:

            current_path = section_path + [node["title"].strip()]

            if node["paragraphs"]:

                text_chunks = self._chunk_paragraphs(
                    node["paragraphs"],
                    section_path=current_path,
                    company=company,
                    year=year,
                    file_name=file_name,
                )

                chunks.extend(text_chunks)

            for table in node["tables"]:

                if not self._looks_like_real_table(table):
                    continue

                chunks.append(
                    self._build_table_chunk(
                        table,
                        section_path=current_path,
                        company=company,
                        year=year,
                        file_name=file_name,
                    )
                )

            self._walk(
                nodes=node["children"],
                section_path=current_path,
                company=company,
                year=year,
                file_name=file_name,
                chunks=chunks,
            )

    def _chunk_paragraphs(self, paragraphs, section_path, company, year, file_name):

        chunks = []

        current_text_parts = []
        current_length = 0
        current_pages = set()

        def flush():

            if not current_text_parts:
                return None

            text = " ".join(current_text_parts).strip()

            if not text:
                return None

            return {
                "chunk_type": "text",
                "text": text,
                "company": company,
                "year": year,
                "file_name": file_name,
                "section_path": " > ".join(section_path),
                "page_numbers": sorted(current_pages),
            }

        for paragraph in paragraphs:

            text = paragraph["text"].strip()

            if not text:
                continue

            page = paragraph.get("page_number")

            for piece in self._split_long_text(text):

                if (
                    current_length + len(piece) > self.max_chunk_chars
                    and current_text_parts
                ):

                    finished = flush()

                    if finished:
                        chunks.append(finished)

                    overlap_text = (
                        finished["text"][-self.chunk_overlap_chars:]
                        if finished else ""
                    )

                    current_text_parts = [overlap_text] if overlap_text else []
                    current_length = len(overlap_text)
                    current_pages = set()

                current_text_parts.append(piece)
                current_length += len(piece)

                if page is not None:
                    current_pages.add(page)

        finished = flush()

        if finished:
            chunks.append(finished)

        chunks = [c for c in chunks if len(c["text"].strip()) >= 15]

        return chunks

    def _split_long_text(self, text):

        if len(text) <= self.max_chunk_chars:
            return [text]

        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", text)

        pieces = []
        current = ""

        for sentence in sentences:

            if len(current) + len(sentence) > self.max_chunk_chars and current:
                pieces.append(current.strip())
                current = sentence
            else:
                current = (current + " " + sentence).strip()

        if current:
            pieces.append(current.strip())

        return pieces if pieces else [text]

    def _looks_like_real_table(self, table, min_numeric_row_ratio=0.4):

        rows = table.get("rows", [])

        if not rows:
            return False

        numeric_row_count = 0

        for row in rows:

            has_numeric = False

            if "values" in row:

                for value in row["values"].values():

                    if value and self._looks_numeric(value):
                        has_numeric = True
                        break

            elif "cells" in row:

                for cell in row["cells"]:

                    if self._looks_numeric(cell.get("text", "")):
                        has_numeric = True
                        break

            if has_numeric:
                numeric_row_count += 1

        ratio = numeric_row_count / len(rows)

        return ratio >= min_numeric_row_ratio

    def _looks_numeric(self, text):

        cleaned = text.strip()

        cleaned = cleaned.replace(",", "")
        cleaned = cleaned.replace("$", "")
        cleaned = cleaned.replace("%", "")
        cleaned = cleaned.replace("(", "")
        cleaned = cleaned.replace(")", "")

        if not cleaned or cleaned in ("-", "--", "\u2014"):
            return False

        try:
            float(cleaned)
            return True
        except ValueError:
            return False

    def _build_table_chunk(self, table, section_path, company, year, file_name):

        table_text = self._render_table_as_text(table, section_path, company, year)

        return {
            "chunk_type": "table",
            "text": table_text,
            "company": company,
            "year": year,
            "file_name": file_name,
            "section_path": " > ".join(section_path),
            "page_numbers": (
                [table["page_number"]] if table.get("page_number") is not None else []
            ),
            "header_detected": table.get("header_detected", False),
        }

    def _render_table_as_text(self, table, section_path=None, company=None, year=None):
        """
        Renders a parsed table back into a readable text block so the
        LLM can read it directly from the chunk (rather than needing
        a separate structured-lookup path). Kept simple and literal --
        this is what actually gets embedded and shown to the LLM.

        NEW: prepends a short, natural-language lead-in sentence
        (using the table's own immediate section title, plus
        company/year) before the raw "Columns: ..." rendering.
        Confirmed on real retrieval tests: a natural-language query
        like "What was Apple's total net sales in fiscal 2024?" was
        failing to surface the correct table (the one literally
        containing "Total net sales -- 2024: 391,035") in the top
        results -- both BM25 and dense embeddings match noticeably
        better against ordinary prose than against a compact
        "Columns: X, Y, Z" grid with no surrounding sentence. This
        lead-in doesn't change any parsing/structure -- it's purely
        an additive natural-language hint layered on top of the
        exact same rendered data.
        """

        lines = []

        if section_path:

            topic = section_path[-1]

            lead_in_parts = []

            if company:
                lead_in_parts.append(company)

            if year:
                lead_in_parts.append(f"{year}")

            who_when = " ".join(str(p) for p in lead_in_parts)

            if who_when:
                lines.append(
                    f"The following table shows {topic} data for {who_when}:"
                )
            else:
                lines.append(f"The following table shows {topic} data:")

        header = table.get("header", [])

        if header:
            lines.append("Columns: " + ", ".join(header))

        for row in table.get("rows", []):

            if "label" in row:

                label = row.get("label", "").strip()

                values = row.get("values", {})

                value_str = ", ".join(
                    f"{col}: {val}" for col, val in values.items() if val is not None
                )

                lines.append(f"{label} -- {value_str}" if value_str else label)

            elif "cells" in row:

                cell_texts = [cell["text"] for cell in row["cells"]]

                lines.append(" | ".join(cell_texts))

        return "\n".join(lines)


def load_hierarchy(company, stem, base_dir="STAGE_1/hierarchy"):

    path = Path(base_dir) / company / f"{stem}_hierarchy.json"

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def discover_stems(hierarchy_dir="STAGE_1/hierarchy"):

    hierarchy_dir = Path(hierarchy_dir)

    stems = []

    for company_dir in sorted(hierarchy_dir.iterdir()):

        if not company_dir.is_dir():
            continue

        for json_file in sorted(company_dir.glob("*_hierarchy.json")):

            stem = json_file.stem.replace("_hierarchy", "")

            stems.append((company_dir.name, stem))

    return stems


def extract_year(stem):

    for part in stem.split("_"):

        if part.isdigit() and len(part) == 4:

            year = int(part)

            if 1990 <= year <= 2100:
                return year

    return None


def save_chunks(company, stem, chunks, output_dir="STAGE_1/chunks"):

    company_dir = Path(output_dir) / company

    company_dir.mkdir(parents=True, exist_ok=True)

    output_file = company_dir / f"{stem}_chunks.json"

    with open(output_file, "w", encoding="utf-8") as f:

        json.dump(chunks, f, indent=4, ensure_ascii=False, default=str)

    print(f"Saved Chunks: {output_file} ({len(chunks)} chunks)")


if __name__ == "__main__":

    print("\n====================================")
    print(" Chunk Builder Started")
    print("====================================\n")

    stems = discover_stems()

    if not stems:

        print("No hierarchy files found.")
        print("Run HierarchyBuilder first.")

    else:

        builder = ChunkBuilder()

        total_chunks = 0
        total_text_chunks = 0
        total_table_chunks = 0

        for company, stem in stems:

            print(f"Chunking: {company}/{stem}")

            hierarchy = load_hierarchy(company, stem)

            year = extract_year(stem)

            chunks = builder.build_chunks(
                company=company,
                year=year,
                file_name=stem,
                sections=hierarchy["sections"],
            )

            save_chunks(company, stem, chunks)

            total_chunks += len(chunks)
            total_text_chunks += sum(1 for c in chunks if c["chunk_type"] == "text")
            total_table_chunks += sum(1 for c in chunks if c["chunk_type"] == "table")

        print("\n====================================")
        print(" Chunk Building Completed")
        print("====================================")
        print(f"Total Chunks       : {total_chunks}")
        print(f"Text Chunks        : {total_text_chunks}")
        print(f"Table Chunks       : {total_table_chunks}")
        print("\nOutput:")
        print("STAGE_1/chunks")