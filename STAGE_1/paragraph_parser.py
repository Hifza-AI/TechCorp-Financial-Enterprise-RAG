import json
import re
from pathlib import Path


def _is_browser_print_artifact(text):
    """
    True for the browser-injected "Print to PDF" header seen at the
    top of every page in some of these files -- a timestamp
    ("5/16/26, 9:56 AM") and/or the literal word "Document". Mirrors
    the same check in table_analyzer.py (kept as a plain module-level
    function here since paragraph_parser.py doesn't import that
    module). See ParagraphParser._parse_page() for the full
    rationale -- this is never genuine 10-K content.
    """

    stripped = text.strip()

    if stripped == "Document":
        return True

    if re.fullmatch(
        r"\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}\s*[AP]M",
        stripped,
        re.IGNORECASE,
    ):
        return True

    return False


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

        # NEW (Chipotle 2016, confirmed via real chunks.json output):
        # computed ONCE per report, before any page is parsed. See
        # _find_page_repeated_boilerplate()'s docstring for the full
        # rationale -- this is the SAME per-page-watermark detection
        # added to table_analyzer.py, but applied here too because a
        # repeated watermark line does NOT need to be its own,
        # exactly-matching paragraph block to cause damage: it just
        # needs to sit close enough to real body text (by the normal
        # y-gap/indent-shift rules below) to get silently GLUED into
        # the middle of an otherwise-real, uniquely-worded paragraph
        # -- e.g. real confirmed output: "...tried to increase, 5
        # 20161231 10K FY_Taxonomy2015 where necessary, the number
        # of suppliers..." -- with the watermark text spliced
        # directly into a genuine sentence.
        #
        # Because the SURROUNDING text differs on every page, each
        # resulting merged paragraph is a UNIQUE string globally --
        # so the existing _remove_repeated_boilerplate() below (which
        # only catches an entire block's text repeating VERBATIM
        # across many pages) can never catch this: it only ever sees
        # 83 different, one-off-looking paragraphs, never noticing
        # they all share one common repeated substring. The fix has
        # to happen earlier, at the per-LINE level, before the
        # watermark line is ever allowed to merge into anything.
        page_repeated_boilerplate = self._find_page_repeated_boilerplate(
            report
        )

        parsed_pages = []

        for page in report["pages"]:

            blocks = self._parse_page(page, page_repeated_boilerplate)

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
    # PAGE-LEVEL REPEATED WATERMARK DETECTION
    # =========================================================

    def _find_page_repeated_boilerplate(
        self,
        report,
        min_page_ratio=0.5,
        min_pages=10,
    ):
        """
        NEW (Chipotle 2016, confirmed via real cleaned.json /
        chunks.json output): some SEC-filing PDF conversions stamp
        EVERY page with an identical per-page watermark/metadata
        line -- confirmed here as the literal text "20161231 10K
        FY_Taxonomy2015" (a filing-date + form-type +
        XBRL-taxonomy-version stamp), appearing as its own standalone
        line on essentially all of this filing's pages.

        This is the SAME detection added to table_analyzer.py (see
        that file's identical method for the fuller rationale) --
        duplicated here as a plain, self-contained method (rather
        than importing table_analyzer.py) since paragraph_parser.py
        doesn't otherwise depend on that module. Both copies use the
        exact same frequency-based logic already proven safe by
        paragraph_parser's own _remove_repeated_boilerplate() for
        whole-block text; this just applies it one level earlier, at
        individual LINES, before they ever get a chance to merge into
        a paragraph.

        A genuine, one-off table value, row-label, or sentence
        fragment is never identical, word-for-word, across half or
        more of an entire filing's pages -- real content varies page
        to page by definition -- so this frequency threshold is safe.
        """

        pages = report.get("pages", [])

        page_count = len(pages)

        if page_count == 0:
            return set()

        text_page_counts = {}

        for page in pages:

            seen_on_this_page = set()

            for line in page.get("lines", []):

                text = (line.get("text") or "").strip()

                if not text or text in seen_on_this_page:
                    continue

                seen_on_this_page.add(text)

                text_page_counts[text] = (
                    text_page_counts.get(text, 0) + 1
                )

        threshold = max(min_pages, page_count * min_page_ratio)

        return {
            text for text, count in text_page_counts.items()
            if count >= threshold
        }

    # =========================================================
    # REPEATED HEADER/FOOTER REMOVAL (company-agnostic)
    # =========================================================

    def _remove_repeated_boilerplate(
        self,
        parsed_pages,
        min_repeat_ratio=0.3,
        min_occurrences=5,
        min_occurrences_short=3,
        short_word_limit=2,
    ):
        """
        NEW: now checks HEADING blocks too, not just paragraphs.

        Confirmed on Nvidia 2026: "NVIDIA Corporation and
        Subsidiaries" is a page-header that repeats at the top of
        nearly every page in the financial-statements section (31
        occurrences) -- but because it's short and bold, it scores
        as a genuine HEADING rather than a paragraph, so it was
        completely bypassing this removal (which only ever checked
        block_type == "paragraph"). Each occurrence became its own
        spurious, empty heading node cluttering the hierarchy tree.

        Applying the identical frequency-based logic to headings is
        safe: a GENUINE document-structure heading (an Item, a Note,
        a statement title) only ever appears ONCE per report by
        definition -- it's only page-header-style boilerplate
        (company name, repeated filing captions) that could
        realistically repeat across 30%+ of pages or 5+ times.

        NEW: VERY SHORT (<= 2 word) heading text gets its own, much
        LOWER occurrence-floor (default 3) instead of the standard
        threshold. Confirmed on Intel 2025 (139 pages): generic
        table row/column-labels -- "Total", "Revenue", "Assets",
        "Liabilities" -- are short, bold text sitting inside segment
        and financial-statement tables, so they score as genuine
        headings the same way "NVIDIA Corporation..." did. But
        because these are individually much SHORTER, more
        common/generic phrases, they don't need anywhere near the
        same repetition count to be confidently recognized as
        recurring table-labels rather than a genuine, unique
        section title -- "Total" appeared as a heading on 14 of
        Intel's 139 pages, comfortably below the standard threshold
        (capped at 20 for a document this long) but still an
        unambiguous, non-coincidental repeat for a 1-word candidate.
        A longer, more specific phrase needs the higher bar precisely
        because it's inherently less likely to repeat by coincidence
        -- a short generic word doesn't carry that same protection,
        so it's safe to flag it sooner.

        This does NOT catch every recurring table-label -- some
        (e.g. Intel's segment names "CCG", "DCAI") appear as a
        heading-candidate only on the 1-2 specific pages containing
        that particular segment-summary table, never crossing even
        this lower floor. That narrower pattern needs a different,
        context-based fix (recognizing table-adjacency directly)
        rather than a frequency-count approach, and is intentionally
        NOT addressed by this change.
        """

        total_pages = len(parsed_pages)

        if total_pages == 0:
            return parsed_pages

        text_page_counts = {}

        for page in parsed_pages:

            seen_on_this_page = set()

            for block in page["blocks"]:

                if block["block_type"] not in ("paragraph", "heading"):
                    continue

                text = block["text"].strip()

                if not text or text in seen_on_this_page:
                    continue

                seen_on_this_page.add(text)

                text_page_counts[text] = text_page_counts.get(text, 0) + 1

        # NEW: cap the percentage-based portion at 20 occurrences.
        # Confirmed on Nvidia 2026 (87 pages): "Notes to the
        # Consolidated Financial Statements" repeats as a page-header
        # 25 times -- clearly boilerplate -- but 30% of 87 pages is
        # 26.1, just ABOVE 25, so the uncapped threshold would have
        # let this one slip through. The 30%-of-total-pages rule
        # makes sense as a safety net for SHORT documents, but for
        # LONGER filings (80-130+ pages), boilerplate is often
        # specific to just one section (e.g. the ~25-30 pages of
        # financial statements + notes), not the whole document --
        # so scaling the threshold to the FULL page count makes it
        # unreasonably strict exactly when it needs to be more
        # lenient. Capping keeps the floor (min_occurrences=5)
        # meaningful for short documents while preventing the
        # percentage rule from becoming impossibly strict for long
        # ones.
        standard_threshold = max(
            min_occurrences,
            min(total_pages * min_repeat_ratio, 20),
        )

        boilerplate_texts = set()

        for text, count in text_page_counts.items():

            word_count = len(text.split())

            threshold = (
                min_occurrences_short
                if word_count <= short_word_limit
                else standard_threshold
            )

            if count >= threshold:
                boilerplate_texts.add(text)

        if not boilerplate_texts:
            return parsed_pages

        for page in parsed_pages:

            page["blocks"] = [
                block
                for block in page["blocks"]
                if not (
                    block["block_type"] in ("paragraph", "heading")
                    and block["text"].strip() in boilerplate_texts
                )
            ]

        return parsed_pages

    # =========================================================
    # PAGE
    # =========================================================

    def _parse_page(self, page, page_repeated_boilerplate=None):

        page_repeated_boilerplate = page_repeated_boilerplate or set()

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

            # NEW: some of these PDFs were exported via a browser's
            # own "Print to PDF" (not the original SEC-filed PDF) --
            # every page carries a browser-injected print header: a
            # timestamp like "5/16/26, 9:56 AM" and/or the literal
            # word "Document" (the browser tab's generic title), both
            # sitting at the very top of the page, above any real
            # content. table_analyzer.py already excludes this from
            # ever becoming a table candidate, but when it ISN'T
            # swept into a table it was falling through to here and
            # being treated as ordinary paragraph text -- sometimes
            # landing mid-sentence in the middle of a real paragraph
            # (confirmed on Apple 2018: "...submit and post such
            # files). Document Yes No" -- the word "Document" spliced
            # directly into unrelated real content). This is never
            # genuine 10-K content, so it's skipped entirely here too,
            # regardless of whether it would otherwise be a heading,
            # table line, or plain paragraph text.
            if _is_browser_print_artifact(text):
                continue

            # NEW (Chipotle 2016, confirmed via real chunks.json
            # output): a line whose exact text repeats across a
            # large share of this report's pages (computed once per
            # report by _find_page_repeated_boilerplate(), see that
            # method's docstring for the full rationale) is a
            # page-level watermark/stamp, never genuine 10-K content
            # -- exactly like the browser-print-artifact check just
            # above. Skipped here, at the very same point in the
            # per-line loop, so it can never be glued into the
            # middle of a real paragraph via the normal
            # paragraph-continuation logic below, regardless of how
            # close it happens to sit (by y-gap or indent) to
            # genuine body text on that page.
            if text in page_repeated_boilerplate:
                continue

            is_table_line = table_flags.get(index, {}).get(
                "is_candidate", False
            )

            heading_info = heading_flags.get(index, {})

            is_heading_line = heading_info.get("is_heading", False)

            # -------------------------------------------------
            # BOLD-SENTENCE-RUN FRAGMENT -> skip entirely.
            #
            # heading_detector.py can recognize a heading that wraps
            # across MULTIPLE physical PDF lines (common in SEC
            # Risk-Factors sections) and merge it into ONE complete
            # heading, attached to the line where the sentence
            # actually ends. The earlier fragment line(s) are tagged
            # "in_bold_run": True to say "this content is already
            # fully captured in that later merged heading -- don't
            # treat it as anything else."
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
            # Checked BEFORE the table-line check below: a line that
            # heading_detector.py has already confirmed as a genuine
            # heading (bold/italic + size + scoring signals) must
            # never be silently dropped just because
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
                    "is_note_marker": heading_info.get("is_note_marker", False),
                    "is_top_level_marker": heading_info.get(
                        "is_top_level_marker", False
                    ),
                    "is_prominent_boundary": heading_info.get(
                        "is_prominent_boundary", False
                    ),
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

        # A line that IS a standalone footnote marker (e.g. "(1)",
        # "(2)") always starts a fresh, separate footnote item --
        # even when the gap/indent from the previous line is small.
        # Confirmed on Apple 2016 page 22: three distinct footnotes
        # explaining share-repurchase details were being merged into
        # ONE 1500+ character paragraph, because consecutive footnote
        # items are typeset with the same tight line-spacing and left
        # margin as normal body text -- neither the vertical-gap nor
        # indent-shift checks below ever fire between them. A lone
        # "(N)" marker is a reliable, company-agnostic signal that a
        # new, unrelated item is starting right here, regardless of
        # layout spacing.
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