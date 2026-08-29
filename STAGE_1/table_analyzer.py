import json
import re
from copy import deepcopy
from pathlib import Path


class TableAnalyzer:

    def __init__(
        self,
        min_numeric_ratio=0.30,
        min_table_lines=2,
        x_alignment_tolerance=12,
        y_alignment_tolerance=6,
    ):
        self.min_numeric_ratio = min_numeric_ratio
        self.min_table_lines = min_table_lines
        self.x_alignment_tolerance = x_alignment_tolerance
        self.y_alignment_tolerance = y_alignment_tolerance

    def analyze(self, cleaned_reports, heading_reports=None):
        """
        heading_reports (optional): a list of already-computed
        heading_detector.py outputs, one per report, in the SAME
        order as cleaned_reports. When provided, any line that
        heading_detector has already confirmed as a genuine heading
        is excluded from ALL rescue-passes below (Pass 2, 3, 4) --
        both as something that can itself be promoted, AND as an
        anchor that a NEARBY line can be rescued against.
        Confirmed on Apple 2016: short bold/italic section titles
        ("iPhone", "Mac", "Services", "Debt", "Price Range of Common
        Stock") sit at the exact same left-margin x as the row-labels
        of the table just below them, so Pass 3's column-alignment
        heuristic was matching them as if they were wrapped
        column-header fragments -- and because Pass 4 cascades
        backward from any confirmed candidate, that single false
        promotion then swallowed the TAIL of whatever genuine
        paragraph happened to sit right before the heading too.

        This parameter is entirely optional and backward-compatible:
        if omitted (or a report has no matching heading data), this
        behaves exactly as before.
        """

        analyzed_reports = []

        for index, report in enumerate(cleaned_reports):

            heading_report = (
                heading_reports[index]
                if heading_reports and index < len(heading_reports)
                else None
            )

            analyzed_report = self._analyze_report(report, heading_report)

            analyzed_reports.append(analyzed_report)

        return analyzed_reports

    def _analyze_report(self, report, heading_report=None):

        analyzed_report = deepcopy(report)

        heading_pages = {}

        if heading_report:

            for heading_page in heading_report.get("pages", []):

                page_number = heading_page.get("page_number")

                candidates = heading_page.get(
                    "heading_analysis", {}
                ).get("candidates", [])

                heading_pages[page_number] = {
                    c["line_index"]
                    for c in candidates
                    if c.get("is_heading")
                }

        analyzed_pages = []

        for page in report.get("pages", []):

            heading_line_indices = heading_pages.get(
                page.get("page_number"), set()
            )

            analyzed_page = self._analyze_page(page, heading_line_indices)

            analyzed_pages.append(analyzed_page)

        analyzed_report["pages"] = analyzed_pages

        return analyzed_report

    def _analyze_page(self, page, heading_line_indices=None):

        heading_line_indices = heading_line_indices or set()

        analyzed_page = deepcopy(page)

        lines = page.get("lines", [])

        line_analysis = []

        for index, line in enumerate(lines):

            analysis = self._analyze_line(
                line,
                index,
                lines,
            )

            line_analysis.append(analysis)

        confirmed_ys = [
            item["_y"]
            for item in line_analysis
            if item["is_candidate"] and item["_y"] is not None
            and item["line_index"] not in heading_line_indices
        ]

        # NEW: genuine row-labels are always short ("iPhone", "Net
        # sales", "Adjustment for net (gains)/losses..."). A full,
        # multi-sentence PARAGRAPH is never a real row-label, even if
        # it happens to sit at the same y-position as a confirmed
        # numeric candidate. Confirmed on Apple 2016 page 22: a
        # footnote reference marker "(1)" (a lone digit in
        # parentheses, itself numeric-looking enough to become its
        # own Pass-1 candidate) sits INLINE with the start of its
        # full footnote paragraph -- "In April 2016, the Company's
        # Board of Directors increased the Company's share
        # repurchase program authorization..." (19 words). Without
        # this guard, that entire footnote sentence gets promoted as
        # if it were "(1)"'s row-label, silently removing it from
        # paragraph output (TableParser has no real row/column
        # structure to put it in either, so it's lost completely).
        # Reused from the same MAX_LABEL_WORDS convention already
        # established in table_parser.py for the same "is this
        # actually a label" judgment call.
        MAX_ROW_LABEL_WORDS = 15

        for item in line_analysis:

            if item["is_candidate"]:
                continue

            if item["line_index"] in heading_line_indices:
                continue

            if item["_y"] is None:
                continue

            word_count = len(item["text"].split())

            if word_count > MAX_ROW_LABEL_WORDS:
                continue

            is_row_mate = any(
                abs(item["_y"] - y) <= self.y_alignment_tolerance
                for y in confirmed_ys
            )

            if is_row_mate:
                item["is_candidate"] = True
                item["promoted_as_row_label"] = True

        confirmed_x_positions = []

        for item in line_analysis:
            if item["is_candidate"] and item["line_index"] not in heading_line_indices:
                confirmed_x_positions.extend(item.get("x_positions", []))

        HEADER_ZONE_WINDOW = 150
        HEADER_X_TOLERANCE = 20

        for item in line_analysis:

            if item["is_candidate"]:
                continue

            if item["line_index"] in heading_line_indices:
                continue

            if item["_y"] is None:
                continue

            word_count = len(item["text"].split())

            if word_count == 0 or word_count > 5:
                continue

            if item["text"].strip().endswith("."):
                continue

            is_above_a_table = any(
                0 < (y - item["_y"]) <= HEADER_ZONE_WINDOW
                for y in confirmed_ys
            )

            if not is_above_a_table:
                continue

            is_column_aligned = any(
                abs(x - cx) <= HEADER_X_TOLERANCE
                for x in item.get("x_positions", [])
                for cx in confirmed_x_positions
            )

            if is_column_aligned:
                item["is_candidate"] = True
                item["promoted_as_header_fragment"] = True

        ADJACENT_Y_WINDOW = 40
        MAX_LABEL_WORDS = 15

        # NEW: a copyright/legal notice ("Copyright (c) 2016 S&P, a
        # division of McGraw Hill Financial. All rights reserved.")
        # is NEVER a genuine table row-label, no matter how short or
        # how close it sits to a real table. Confirmed on Apple
        # 2016 page 23: the real stock-performance table's header
        # happened to sit close enough (within ADJACENT_Y_WINDOW) to
        # a preceding copyright line that the copyright line got
        # promoted as if it were a wrapped label continuation for
        # that table -- and because Pass 4 processes lines in
        # REVERSE and cascades (by design, so a genuine 3+-line
        # wrapped label rescues correctly in one pass), that single
        # false promotion then chained BACKWARD through two more
        # unrelated lines (a second copyright notice, and the tail
        # of an unrelated footnote sentence), none of which had
        # anything to do with the table that triggered the chain.
        # Blocking the copyright line from ever being promoted stops
        # the cascade at its source, so it can no longer drag in
        # unrelated narrative text before it.
        def _looks_like_copyright_notice(text):
            return bool(re.search(r"copyright|\u00a9", text, re.IGNORECASE))

        for index in range(len(line_analysis) - 1, -1, -1):

            item = line_analysis[index]

            if item["is_candidate"]:
                continue

            if item["line_index"] in heading_line_indices:
                continue

            if index + 1 >= len(line_analysis):
                continue

            next_item = line_analysis[index + 1]

            if not next_item["is_candidate"]:
                continue

            # NEW: never cascade FROM a confirmed heading either -- if
            # the "anchor" that triggered this promotion is itself a
            # genuine heading (e.g. "Debt", "Mac"), the line before it
            # is almost certainly the TAIL of an unrelated paragraph,
            # not that heading's own wrapped row-label. Confirmed on
            # Apple 2016 page 33: "Debt" (a heading) sat close enough
            # to the end of the preceding "Capital Assets" paragraph
            # that its last sentence ("facilities and infrastructure,
            # ... retail store facilities.") was being swallowed as if
            # it were "Debt"'s own wrapped label.
            if next_item["line_index"] in heading_line_indices:
                continue

            word_count = len(item["text"].split())

            if word_count == 0 or word_count > MAX_LABEL_WORDS:
                continue

            # NEW: a genuine wrapped row-label fragment never forms a
            # complete grammatical sentence -- it just continues onto
            # the next physical line without natural sentence-ending
            # punctuation (e.g. "Open market and privately negotiated"
            # / "purchases"). A line that ends in "." is far more
            # likely the TAIL of an unrelated narrative paragraph that
            # simply happens to sit close to a real table's start.
            # Confirmed on Apple 2016 page 21: "...Company's common
            # stock on the NASDAQ during each quarter of the two most
            # recent years." (a complete sentence, 15 words) was being
            # swept in as if it were a wrapped label for the price-
            # range table's own header row directly below it.
            if item["text"].strip().endswith("."):
                continue

            if _looks_like_copyright_notice(item["text"]):
                continue

            if item["_y"] is None or next_item.get("_y") is None:
                continue

            y_gap = abs(next_item["_y"] - item["_y"])

            if y_gap <= ADJACENT_Y_WINDOW:
                item["is_candidate"] = True
                item["promoted_as_label_continuation"] = True

        for item in line_analysis:
            item.pop("_y", None)

        analyzed_page["table_analysis"] = {
            "candidate_lines": line_analysis,
            "candidate_count": sum(
                1
                for item in line_analysis
                if item["is_candidate"]
            ),
        }

        return analyzed_page

    def _analyze_line(self, line, index, lines):

        text = line.get("text", "").strip()

        spans = line.get("spans", [])

        y = self._line_y(line)

        if not text:

            return {
                "line_index": index,
                "text": text,
                "is_candidate": False,
                "numeric_ratio": 0.0,
                "numeric_count": 0,
                "token_count": 0,
                "x_positions": [],
                "y_positions": [],
                "_y": y,
            }

        # NEW: some of these PDFs were exported via a browser's own
        # "Print to PDF" (not the original SEC-filed PDF) -- every
        # page carries a browser-injected print header: a timestamp
        # like "5/16/26, 9:56 AM" and the literal word "Document"
        # (the browser tab's generic title), both sitting at the very
        # TOP of the page, above any real content.
        #
        # Confirmed on Apple 2016 page 46: the isolated word
        # "Document" got promoted via Pass 3 (short, doesn't end in
        # a period, happened to x-align with one of the Cash Flow
        # table's columns) and then swept into that table's own
        # header-row grouping by table_parser.py's wrapped-label
        # merge -- which corrupted the table's overall bbox (pulling
        # its recorded top all the way up to y=15, far above where
        # the table visually starts). That corrupted bbox then made
        # hierarchy_builder.py think the table appeared BEFORE its
        # own "CONSOLIDATED STATEMENTS OF CASH FLOWS" heading, so it
        # attached the table to the PREVIOUS section instead.
        #
        # This is never genuine 10-K content, so it's excluded here
        # unconditionally -- it can never become a table candidate
        # through any pass, in any table, on any page.
        if self._is_browser_print_artifact(text):

            return {
                "line_index": index,
                "text": text,
                "is_candidate": False,
                "numeric_ratio": 0.0,
                "numeric_count": 0,
                "token_count": 0,
                "x_positions": [],
                "y_positions": [],
                "_y": y,
            }

        numeric_count = 0
        token_count = 0

        x_positions = []
        y_positions = []

        for span in spans:

            span_text = span.get("text", "").strip()

            if not span_text:
                continue

            token_count += 1

            if self._is_numeric_token(span_text):

                numeric_count += 1

            bbox = span.get("bbox")

            if bbox and len(bbox) >= 4:

                x_positions.append(
                    round(float(bbox[0]), 2)
                )

                y_positions.append(
                    round(float(bbox[1]), 2)
                )

        numeric_ratio = (
            numeric_count / token_count
            if token_count
            else 0.0
        )

        nearby_numeric_lines = self._count_nearby_numeric_lines(
            index,
            lines,
        )

        has_multiple_positions = len(x_positions) >= 2

        is_candidate = (

            numeric_ratio >= self.min_numeric_ratio

            and (

                nearby_numeric_lines >= self.min_table_lines - 1

                or has_multiple_positions

            )

        )

        return {

            "line_index": index,

            "text": text,

            "is_candidate": is_candidate,

            "numeric_ratio": round(
                numeric_ratio,
                3,
            ),

            "numeric_count": numeric_count,

            "token_count": token_count,

            "x_positions": x_positions,

            "y_positions": y_positions,

            "nearby_numeric_lines": nearby_numeric_lines,

            "_y": y,

        }

    def _is_browser_print_artifact(self, text):
        """
        True for the browser-injected "Print to PDF" header seen at
        the top of every page in some of these files -- a timestamp
        ("5/16/26, 9:56 AM") and/or the literal word "Document". See
        the caller for the full rationale (Apple 2016 page 46's Cash
        Flow table bbox corruption).
        """

        import re

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

    def _is_numeric_token(self, text):

        cleaned = text.strip()

        if not cleaned:

            return False

        cleaned = re.sub(r"[\$,\.\-\+\(\)%\s]", "", cleaned)

        if not cleaned:
            return False

        return cleaned.isdigit()

    def _count_nearby_numeric_lines(
        self,
        index,
        lines,
    ):

        count = 0

        current_line = lines[index]

        current_y = self._line_y(current_line)

        if current_y is None:

            return 0

        start = max(
            0,
            index - 2,
        )

        end = min(
            len(lines),
            index + 3,
        )

        for other_index in range(start, end):

            if other_index == index:

                continue

            other_line = lines[other_index]

            other_y = self._line_y(
                other_line
            )

            if other_y is None:

                continue

            if abs(other_y - current_y) <= 80:

                if self._line_numeric_ratio(
                    other_line
                ) >= self.min_numeric_ratio:

                    count += 1

        return count

    def _line_y(self, line):

        bbox = line.get("bbox")

        if not bbox or len(bbox) < 4:

            return None

        try:

            return float(bbox[1])

        except (TypeError, ValueError):

            return None

    def _line_numeric_ratio(self, line):

        spans = line.get("spans", [])

        total = 0
        numeric = 0

        for span in spans:

            text = span.get("text", "").strip()

            if not text:

                continue

            total += 1

            if self._is_numeric_token(text):

                numeric += 1

        if total == 0:

            return 0.0

        return numeric / total


def load_cleaned_reports(input_dir):

    input_dir = Path(input_dir)

    reports = []

    if not input_dir.exists():

        print(
            f"Folder not found: {input_dir}"
        )

        return reports

    for company_dir in sorted(
        input_dir.iterdir()
    ):

        if not company_dir.is_dir():

            continue

        for json_file in sorted(
            company_dir.glob("*_cleaned.json")
        ):

            with open(
                json_file,
                "r",
                encoding="utf-8",
            ) as f:

                reports.append(
                    json.load(f)
                )

    return reports


def load_matching_heading_report(
    report,
    heading_dir="STAGE_1/heading_detection",
):
    """
    Loads the heading_detector.py output that corresponds to a given
    cleaned report (same company + same file stem), so TableAnalyzer
    can cross-reference confirmed headings and avoid ever promoting
    them as table candidates. Returns None (graceful degradation) if
    heading_detection hasn't been run yet for this report -- analysis
    then simply falls back to the previous, heading-unaware behavior
    for that one report.
    """

    company = report.get("company")

    stem = Path(report.get("file_name", "")).stem

    heading_path = Path(heading_dir) / company / f"{stem}_headings.json"

    if not heading_path.exists():
        return None

    with open(heading_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_analyzed_reports(
    analyzed_reports,
    output_dir="STAGE_1/table_analysis",
):

    output_dir = Path(output_dir)

    for report in analyzed_reports:

        company = report["company"]

        company_dir = (
            output_dir / company
        )

        company_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_file = (
            company_dir
            /
            (
                Path(
                    report["file_name"]
                ).stem
                +
                "_table_analyzed.json"
            )
        )

        with open(
            output_file,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                report,
                f,
                indent=4,
                ensure_ascii=False,
                default=str,
            )

        print(
            f"Saved Table Analysis: "
            f"{output_file}"
        )


if __name__ == "__main__":

    INPUT_DIR = (
        "STAGE_1/cleaned"
    )

    OUTPUT_DIR = (
        "STAGE_1/table_analysis"
    )

    print(
        "\n===================================="
    )

    print(
        " Table Analyzer Started"
    )

    print(
        "====================================\n"
    )

    cleaned_reports = (
        load_cleaned_reports(
            INPUT_DIR
        )
    )

    if not cleaned_reports:

        print(
            "No cleaned JSON files found."
        )

        print(
            "Run TextCleaner first."
        )

    else:

        analyzer = TableAnalyzer()

        # Cross-reference heading_detector.py's results so confirmed
        # headings never get mistaken for table candidates. Falls
        # back gracefully (per-report) if a report's heading_detection
        # output isn't available yet.
        heading_reports = [
            load_matching_heading_report(report)
            for report in cleaned_reports
        ]

        analyzed_reports = (
            analyzer.analyze(
                cleaned_reports,
                heading_reports,
            )
        )

        save_analyzed_reports(
            analyzed_reports,
            OUTPUT_DIR,
        )

        print(
            "\n===================================="
        )

        print(
            " Table Analysis Completed"
        )

        print(
            "===================================="
        )