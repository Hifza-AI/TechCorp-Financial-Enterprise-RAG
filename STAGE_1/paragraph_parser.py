import json
import re
from pathlib import Path


class ParagraphParser:
    """
    Merges cleaned lines + heading_analysis + table_analysis into
    ordered heading/paragraph blocks per page.

    All cross-referencing between the three pipeline outputs
    (cleaned, heading_detection, table_analysis) is done by
    "page_number" -- the current extractor.py produces a single,
    simple, sequential page_number (over KEPT/non-blank pages only),
    which is guaranteed unique per report, so it's safe to use
    directly as the join-key here.
    """

    def __init__(
        self,
        max_paragraph_gap=14.0,
        max_indent_shift=25.0,
    ):
        self.max_paragraph_gap = max_paragraph_gap
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

        page_number = page["page_number"]

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

            heading_info = heading_flags.get(index, {})

            is_heading_line = heading_info.get("is_heading", False)

            # -------------------------------------------------
            # BOLD-SENTENCE-RUN FRAGMENT -> skip entirely.
            #
            # NEW: heading_detector.py can now recognize a heading
            # that wraps across MULTIPLE physical PDF lines (common
            # in SEC Risk-Factors sections) and merge it into ONE
            # complete heading, attached to the line where the
            # sentence actually ends. The earlier fragment line(s)
            # are tagged "in_bold_run": True to say "this content is
            # already fully captured in that later merged heading --
            # don't treat it as anything else."
            #
            # Without this check, those earlier fragments (which are
            # NOT marked as headings themselves) would fall through
            # to the normal paragraph-accumulation logic below and
            # get glued onto whatever paragraph happened to precede
            # them -- producing a grammatically broken paragraph
            # (ending mid-sentence) AND, since the fragment's words
            # are ALSO inside the later merged heading, the same
            # text would effectively be duplicated across two blocks.
            # -------------------------------------------------

            if heading_info.get("in_bold_run", False):

                self._flush_paragraph(
                    pending_paragraph_lines,
                    blocks,
                    page_number,
                )
                pending_paragraph_lines = []

                continue

            # -------------------------------------------------
            # HEADING LINE -> flush pending paragraph, emit heading
            #
            # NEW (checked BEFORE the table-line check below): a line
            # that heading_detector.py has already confirmed as a
            # genuine heading (bold/italic + size + scoring signals)
            # must never be silently dropped just because
            # table_analyzer.py's speculative rescue-passes ALSO
            # flagged it as a table candidate. Confirmed on Apple
            # 2016: short section titles that sit directly above a
            # table -- "iPhone", "Mac", "Services", "Price Range of
            # Common Stock", "Selected Financial Data" -- happen to
            # sit at the exact same left-margin x-position as the
            # table's own row-labels below them (e.g. "Net sales"),
            # so table_analyzer's Pass 3 (header-zone rescue) was
            # matching them as if they were wrapped column-header
            # fragments. Since the OLD check order tested is_table_line
            # first, every one of these genuine headings was being
            # skipped entirely -- never shown as a heading, and never
            # recovered anywhere else either (TableParser has no real
            # row/column structure to place a single stray heading
            # line into, so it silently vanished).
            #
            # heading_detector.py's signals are independently strong
            # and don't rely on table-adjacent x-alignment guesswork,
            # so giving heading-status priority here is safe: a line
            # that's genuinely a heading is essentially never also
            # genuinely a piece of real table data.
            # -------------------------------------------------

            if is_heading_line:

                self._flush_paragraph(
                    pending_paragraph_lines,
                    blocks,
                    page_number,
                )
                pending_paragraph_lines = []

                # Use the heading's OWN recorded text, not the raw
                # text of this single physical line. For a heading
                # that heading_detector.py merged from multiple
                # wrapped lines, `heading_info["text"]` holds the FULL
                # combined sentence -- the local `text` variable here
                # would only be this one closing line's own fragment
                # (e.g. "and the Company may be unable to compete
                # effectively in these markets." instead of the
                # complete "Global markets for the Company's products
                # and services are highly competitive..."). Falling
                # back to `text` keeps single-line headings (the
                # majority case) working exactly as before.
                heading_text = heading_info.get("text", text)

                blocks.append({
                    "block_type": "heading",
                    "text": heading_text,
                    "level": heading_info.get("level", 0),
                    "page_number": page_number,
                    "bbox": line.get("bbox"),
                })

                continue

            # -------------------------------------------------
            # TABLE LINE -> skip entirely (handled by TableParser)
            # -------------------------------------------------

            if is_table_line:

                self._flush_paragraph(
                    pending_paragraph_lines,
                    blocks,
                    page_number,
                )
                pending_paragraph_lines = []

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
                        page_number,
                    )
                    pending_paragraph_lines = []

            pending_paragraph_lines.append(line)

        self._flush_paragraph(
            pending_paragraph_lines,
            blocks,
            page_number,
        )

        return blocks

    # =========================================================
    # PARAGRAPH BREAK DECISION
    # =========================================================

    def _is_new_paragraph(self, previous_line, current_line):

        # NEW: a line that IS a standalone footnote marker (e.g.
        # "(1)", "(2)") always starts a fresh, separate footnote
        # item -- even when the gap/indent from the previous line
        # is small. Confirmed on Apple 2016 page 22: three distinct
        # footnotes explaining share-repurchase details were being
        # merged into ONE 1500+ character paragraph, because
        # consecutive footnote items are typeset with the same
        # tight line-spacing and left margin as normal body text --
        # neither the vertical-gap nor indent-shift checks below
        # ever fire between them. A lone "(N)" marker is a reliable,
        # company-agnostic signal that a new, unrelated item is
        # starting right here, regardless of layout spacing.
        current_text = (current_line.get("text") or "").strip()

        if re.fullmatch(r"\(\d{1,2}\)", current_text):
            return True

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
    THREE separate scripts. We zip them back together here by
    "page_number" -- which the current extractor.py already makes
    unique and sequential for every company (blank pages already
    filtered out at extraction time).
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

    # TEMP: only process Apple for now (other companies' table_analysis
    # files were removed while table_parser.py fixes were being
    # verified). Remove this filter once ready to process everyone.
    stems = [(company, stem) for company, stem in stems if company == "Apple"]

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