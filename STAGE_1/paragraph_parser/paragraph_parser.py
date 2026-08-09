import json
from pathlib import Path
print(">>> RUNNING UPDATED HEADING DETECTOR - v2 <<<")

class ParagraphParser:
    """
    Takes the THREE outputs that already exist for each report:
      1. Cleaned JSON        (STAGE_1/cleaned)            -> raw lines
      2. Heading Detection   (STAGE_1/heading_detection)   -> is_heading per line
      3. Table Analysis      (STAGE_1/table_analysis)      -> is_candidate per line

    ...and produces ONE clean, ordered sequence of "blocks" per page:
      - heading blocks   (one per detected heading line)
      - paragraph blocks (consecutive normal text lines merged together)

    Table-candidate lines are SKIPPED here on purpose -- they are
    already being handled by TableParser separately. Re-including
    them here would duplicate/garble table content inside paragraphs
    (the same "double-processing" bug we already fixed once in
    section_builder.py).
    """

    def __init__(
        self,
        max_paragraph_gap=14.0,
        max_indent_shift=25.0,
    ):
        # If the vertical gap between two consecutive normal-text
        # lines is bigger than this, we treat it as a paragraph break
        # (e.g. blank space between two separate paragraphs).
        self.max_paragraph_gap = max_paragraph_gap

        # If the left (x) position jumps by more than this between
        # two lines, we also treat it as a new paragraph (e.g. a new
        # bullet point or indented block starting).
        self.max_indent_shift = max_indent_shift

    # =========================================================
    # MAIN
    # =========================================================

    def parse(self, merged_reports):

        parsed_reports = []

        for report in merged_reports:

            parsed_reports.append(
                self._parse_report(report)
            )

        return parsed_reports

    # =========================================================
    # REPORT
    # =========================================================

    def _parse_report(self, report):

        parsed_pages = []

        for page in report["pages"]:

            blocks = self._parse_page(page)

            parsed_pages.append({
                "page_number": page["page_number"],
                "blocks": blocks,
            })

        # -------------------------------------------------------
        # Report-level cleanup: repeated header/footer removal
        #
        # Cleaning didn't fully strip repeating boilerplate like
        # "Apple Inc. | 2024 Form 10-K | 21" (page footer), so it
        # leaks through as its own stray paragraph block on nearly
        # every page. Instead of hardcoding this exact text (which
        # would break for every other company -- each has a
        # different footer wording/format), we detect it generically:
        # any PARAGRAPH block whose text repeats on a large fraction
        # of this report's pages is almost certainly a footer/header,
        # not real content, regardless of what company it's from.
        # -------------------------------------------------------

        parsed_pages = self._remove_repeated_boilerplate(parsed_pages)

        return {
            "company": report["company"],
            "file_name": report["file_name"],
            "file_path": report.get("file_path"),
            "total_pages": report.get("total_pages"),
            "pages": parsed_pages,
        }

    # =========================================================
    # REPEATED HEADER/FOOTER REMOVAL (company-agnostic)
    # =========================================================

    def _remove_repeated_boilerplate(
        self,
        parsed_pages,
        min_repeat_ratio=0.3,
        min_occurrences=5,
    ):

        total_pages = len(parsed_pages)

        if total_pages == 0:
            return parsed_pages

        text_page_counts = {}

        for page in parsed_pages:

            seen_on_this_page = set()

            for block in page["blocks"]:

                if block["block_type"] != "paragraph":
                    continue

                text = block["text"].strip()

                if not text or text in seen_on_this_page:
                    continue

                seen_on_this_page.add(text)

                text_page_counts[text] = text_page_counts.get(text, 0) + 1

        threshold = max(
            min_occurrences,
            total_pages * min_repeat_ratio,
        )

        boilerplate_texts = {
            text
            for text, count in text_page_counts.items()
            if count >= threshold
        }

        if not boilerplate_texts:
            return parsed_pages

        for page in parsed_pages:

            page["blocks"] = [
                block
                for block in page["blocks"]
                if not (
                    block["block_type"] == "paragraph"
                    and block["text"].strip() in boilerplate_texts
                )
            ]

        return parsed_pages

    # =========================================================
    # PAGE
    # =========================================================

    def _parse_page(self, page):

        lines = page["lines"]

        heading_flags = {
            c["line_index"]: c
            for c in page.get("heading_analysis", {}).get("candidates", [])
        }

        table_flags = {
            c["line_index"]: c
            for c in page.get("table_analysis", {}).get("candidate_lines", [])
        }

        blocks = []

        pending_paragraph_lines = []

        for index, line in enumerate(lines):

            text = (line.get("text") or "").strip()

            if not text:
                continue

            is_table_line = table_flags.get(index, {}).get(
                "is_candidate", False
            )

            is_heading_line = heading_flags.get(index, {}).get(
                "is_heading", False
            )

            # -------------------------------------------------
            # TABLE LINE -> skip entirely (handled by TableParser)
            # -------------------------------------------------

            if is_table_line:

                # Flush whatever paragraph was building up before
                # hitting this table line -- a table interrupting
                # a paragraph means the paragraph is done.
                self._flush_paragraph(
                    pending_paragraph_lines,
                    blocks,
                    page["page_number"],
                )
                pending_paragraph_lines = []

                continue

            # -------------------------------------------------
            # HEADING LINE -> flush pending paragraph, emit heading
            # -------------------------------------------------

            if is_heading_line:

                self._flush_paragraph(
                    pending_paragraph_lines,
                    blocks,
                    page["page_number"],
                )
                pending_paragraph_lines = []

                heading_info = heading_flags[index]

                blocks.append({
                    "block_type": "heading",
                    "text": text,
                    "level": heading_info.get("level", 0),
                    "page_number": page["page_number"],
                    "bbox": line.get("bbox"),
                })

                continue

            # -------------------------------------------------
            # NORMAL LINE -> decide if it continues current
            # paragraph or starts a new one
            # -------------------------------------------------

            if pending_paragraph_lines:

                previous_line = pending_paragraph_lines[-1]

                if self._is_new_paragraph(previous_line, line):

                    self._flush_paragraph(
                        pending_paragraph_lines,
                        blocks,
                        page["page_number"],
                    )
                    pending_paragraph_lines = []

            pending_paragraph_lines.append(line)

        # Flush anything left at the end of the page
        self._flush_paragraph(
            pending_paragraph_lines,
            blocks,
            page["page_number"],
        )

        return blocks

    # =========================================================
    # PARAGRAPH BREAK DECISION
    # =========================================================

    def _is_new_paragraph(self, previous_line, current_line):

        prev_bbox = previous_line.get("bbox")
        curr_bbox = current_line.get("bbox")

        if not prev_bbox or not curr_bbox:
            return False

        prev_bottom = prev_bbox[3]
        curr_top = curr_bbox[1]

        vertical_gap = curr_top - prev_bottom

        if vertical_gap > self.max_paragraph_gap:
            return True

        prev_left = prev_bbox[0]
        curr_left = curr_bbox[0]

        indent_shift = abs(curr_left - prev_left)

        if indent_shift > self.max_indent_shift:
            return True

        return False

    # =========================================================
    # FLUSH PARAGRAPH BUFFER INTO A BLOCK
    # =========================================================

    def _flush_paragraph(self, pending_lines, blocks, page_number):

        if not pending_lines:
            return

        text = " ".join(
            (line.get("text") or "").strip()
            for line in pending_lines
        ).strip()

        if not text:
            return

        bboxes = [
            line["bbox"] for line in pending_lines if line.get("bbox")
        ]

        merged_bbox = self._merge_bboxes(bboxes)

        blocks.append({
            "block_type": "paragraph",
            "text": text,
            "page_number": page_number,
            "bbox": merged_bbox,
            "line_count": len(pending_lines),
        })

    # =========================================================
    # BBOX MERGE
    # =========================================================

    def _merge_bboxes(self, bboxes):

        if not bboxes:
            return None

        return [
            min(b[0] for b in bboxes),
            min(b[1] for b in bboxes),
            max(b[2] for b in bboxes),
            max(b[3] for b in bboxes),
        ]


# =============================================================
# MERGE THE 3 SEPARATE PIPELINE OUTPUTS FOR ONE REPORT
# =============================================================

def merge_report_sources(cleaned_report, heading_report, table_report):
    """
    Cleaned, Heading Detection, and Table Analysis were produced by
    THREE separate scripts, each reading STAGE_1/cleaned and writing
    their own output folder. Since none of them modify line order or
    line count, `line_index` lines up perfectly across all three --
    so we can safely zip them back together here by page_number and
    line_index.
    """

    heading_pages = {
        p["page_number"]: p for p in heading_report["pages"]
    }

    table_pages = {
        p["page_number"]: p for p in table_report["pages"]
    }

    merged_pages = []

    for page in cleaned_report["pages"]:

        page_number = page["page_number"]

        merged_page = dict(page)  # shallow copy is enough here

        heading_page = heading_pages.get(page_number, {})
        table_page = table_pages.get(page_number, {})

        merged_page["heading_analysis"] = heading_page.get(
            "heading_analysis", {"candidates": []}
        )

        merged_page["table_analysis"] = table_page.get(
            "table_analysis", {"candidate_lines": []}
        )

        merged_pages.append(merged_page)

    return {
        "company": cleaned_report["company"],
        "file_name": cleaned_report["file_name"],
        "file_path": cleaned_report.get("file_path"),
        "total_pages": cleaned_report.get("total_pages"),
        "pages": merged_pages,
    }


# =============================================================
# LOADERS
# =============================================================

def load_report_triplet(company, stem):

    cleaned_path = Path(
        f"STAGE_1/cleaned/{company}/{stem}_cleaned.json"
    )
    heading_path = Path(
        f"STAGE_1/heading_detection/{company}/{stem}_headings.json"
    )
    table_path = Path(
        f"STAGE_1/table_analysis/{company}/{stem}_table_analyzed.json"
    )

    with open(cleaned_path, "r", encoding="utf-8") as f:
        cleaned_report = json.load(f)

    with open(heading_path, "r", encoding="utf-8") as f:
        heading_report = json.load(f)

    with open(table_path, "r", encoding="utf-8") as f:
        table_report = json.load(f)

    return cleaned_report, heading_report, table_report


def discover_report_stems(cleaned_dir="STAGE_1/cleaned"):

    cleaned_dir = Path(cleaned_dir)

    stems = []

    for company_dir in sorted(cleaned_dir.iterdir()):

        if not company_dir.is_dir():
            continue

        for json_file in sorted(company_dir.glob("*_cleaned.json")):

            stem = json_file.stem.replace("_cleaned", "")

            stems.append((company_dir.name, stem))

    return stems


# =============================================================
# SAVE
# =============================================================

def save_parsed_reports(parsed_reports, output_dir="STAGE_1/paragraphs"):

    output_dir = Path(output_dir)

    for report in parsed_reports:

        company_dir = output_dir / report["company"]

        company_dir.mkdir(parents=True, exist_ok=True)

        output_file = company_dir / (
            Path(report["file_name"]).stem + "_paragraphs.json"
        )

        with open(output_file, "w", encoding="utf-8") as f:

            json.dump(
                report,
                f,
                indent=4,
                ensure_ascii=False,
                default=str,
            )

        print(f"Saved Paragraphs: {output_file}")


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":

    print("\n====================================")
    print(" Paragraph Parser Started")
    print("====================================\n")

    stems = discover_report_stems()

    if not stems:

        print("No cleaned reports found.")
        print("Run PDFExtractor -> TextCleaner first.")

    else:

        parser = ParagraphParser()

        parsed_reports = []

        for company, stem in stems:

            print(f"Parsing: {company}/{stem}")

            cleaned_report, heading_report, table_report = load_report_triplet(
                company, stem
            )

            merged_report = merge_report_sources(
                cleaned_report, heading_report, table_report
            )

            parsed_report = parser._parse_report(merged_report)

            parsed_reports.append(parsed_report)

        save_parsed_reports(parsed_reports)

        total_headings = 0
        total_paragraphs = 0

        for report in parsed_reports:
            for page in report["pages"]:
                for block in page["blocks"]:
                    if block["block_type"] == "heading":
                        total_headings += 1
                    else:
                        total_paragraphs += 1

        print("\n====================================")
        print(" Paragraph Parsing Completed")
        print("====================================")
        print(f"Reports Parsed     : {len(parsed_reports)}")
        print(f"Heading Blocks     : {total_headings}")
        print(f"Paragraph Blocks   : {total_paragraphs}")
        print("\nOutput:")
        print("STAGE_1/paragraphs")