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

    # =========================================================
    # MAIN
    # =========================================================

    def analyze(self, cleaned_reports):

        analyzed_reports = []

        for report in cleaned_reports:

            analyzed_report = self._analyze_report(report)

            analyzed_reports.append(analyzed_report)

        return analyzed_reports

    # =========================================================
    # REPORT
    # =========================================================

    def _analyze_report(self, report):

        analyzed_report = deepcopy(report)

        analyzed_pages = []

        for page in report.get("pages", []):

            analyzed_page = self._analyze_page(page)

            analyzed_pages.append(analyzed_page)

        analyzed_report["pages"] = analyzed_pages

        return analyzed_report

    # =========================================================
    # PAGE
    # =========================================================

    def _analyze_page(self, page):

        analyzed_page = deepcopy(page)

        lines = page.get("lines", [])

        # -----------------------------------------------------
        # PASS 1: numeric-based candidate detection
        # (same as before)
        # -----------------------------------------------------

        line_analysis = []

        for index, line in enumerate(lines):

            analysis = self._analyze_line(
                line,
                index,
                lines,
            )

            line_analysis.append(analysis)

        # -----------------------------------------------------
        # PASS 2: row-mate rescue
        #
        # A line with ZERO numeric content (e.g. "iPhone") is
        # never picked up by Pass 1, because its own numeric_ratio
        # is 0.0 -- even when it sits on the exact same visual row
        # as a confirmed table row (same y-position, just a
        # different column/x). Without this pass, row LABELS are
        # silently dropped and the table becomes unusable
        # ("iPhone $137,781" loses its "iPhone").
        #
        # Fix: for every non-candidate line, check whether any
        # CONFIRMED candidate line shares its y-position within
        # y_alignment_tolerance. If so, promote it to candidate
        # too -- it's a row-mate (almost certainly the row label).
        # -----------------------------------------------------

        confirmed_ys = [
            item["_y"]
            for item in line_analysis
            if item["is_candidate"] and item["_y"] is not None
        ]

        for item in line_analysis:

            if item["is_candidate"]:
                continue

            if item["_y"] is None:
                continue

            is_row_mate = any(
                abs(item["_y"] - y) <= self.y_alignment_tolerance
                for y in confirmed_ys
            )

            if is_row_mate:
                item["is_candidate"] = True
                item["promoted_as_row_label"] = True

        # -----------------------------------------------------
        # PASS 3: header-zone rescue
        #
        # Multi-line WRAPPED text-column-headers (e.g. "Total
        # Number" / "of Shares" / "Purchased" -- each its own line,
        # stacked above the actual numeric data) sit at a DIFFERENT
        # y than the data rows below them, so Pass 2's same-y
        # row-mate rescue never catches them. Confirmed on real data
        # (Apple 2016's "Share Repurchase" table): "Total Number",
        # "of Shares", "Average", "Price", "Paid Per" all stayed
        # is_candidate=False and never reached TableParser at all.
        #
        # Fix: for every non-candidate line, check if it's (a) SHORT
        # (<=5 words -- header fragments are short; real paragraphs
        # aren't), (b) positioned ABOVE a confirmed-candidate line
        # within a reasonable vertical window (header zone sits
        # just above the table body), AND (c) x-aligned with some
        # confirmed candidate's column position (this is the key
        # guard against false positives -- a random paragraph
        # sitting above the table won't share the table's exact
        # column x-positions, but genuine wrapped header fragments
        # will).
        # -----------------------------------------------------

        confirmed_x_positions = []

        for item in line_analysis:
            if item["is_candidate"]:
                confirmed_x_positions.extend(item.get("x_positions", []))

        HEADER_ZONE_WINDOW = 150  # points above a confirmed row to search
        HEADER_X_TOLERANCE = 20

        for item in line_analysis:

            if item["is_candidate"]:
                continue

            if item["_y"] is None:
                continue

            word_count = len(item["text"].split())

            if word_count == 0 or word_count > 5:
                continue

            if item["text"].strip().endswith("."):
                continue  # looks like end of a sentence, not a header fragment

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

        # Drop the internal-only helper key before saving
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

    # =========================================================
    # LINE ANALYSIS
    # =========================================================

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

            "_y": y,  # internal-only, stripped before saving

        }

    # =========================================================
    # NUMERIC TOKEN
    # =========================================================

    def _is_numeric_token(self, text):

        cleaned = text.strip()

        if not cleaned:

            return False

        # NOTE: We intentionally do NOT treat a lone "-"/"—" as numeric
        # here. Bulleted lists (e.g. "- Updated MacBook Air...") also
        # start with a dash, and counting it as "numeric" was enough
        # to push short bullet lines over the candidate threshold --
        # misclassifying entire bullet-point paragraphs as tables.
        # Real financial tables that use "-" for a zero/blank cell
        # always have OTHER genuine numeric cells in the same row,
        # which already trigger candidate detection on their own --
        # so this special case wasn't actually needed for real tables,
        # only harmful for bullet lists.

        # Remove common accounting wrappers/symbols AND whitespace.
        #
        # We used to try float(cleaned) on the WHOLE remaining string,
        # which only works for a SINGLE clean number. But a single
        # PDF span can sometimes contain a RANGE like
        # "$116.18 - $91.50" (two numbers, one dash, one space) as
        # one continuous text-run -- float() on that fails entirely
        # (ValueError), so the whole line scored numeric_ratio=0.0
        # and a real financial table (quarterly stock price ranges)
        # was silently dropped as "not a table" -- confirmed on
        # Apple 2016's "Price Range of Common Stock" table.
        #
        # Fix: instead of requiring the WHOLE token to parse as one
        # float, strip every character that's normal in financial
        # formatting ($, comma, period, dash, plus, parens, percent,
        # whitespace) and check if what's LEFT is purely digits. This
        # correctly recognizes both single numbers ("252") AND
        # compound numeric content like ranges ("$116.18 - $91.50")
        # as numeric, without needing them to be one valid float.

        cleaned = re.sub(r"[\$,\.\-\+\(\)%\s]", "", cleaned)

        if not cleaned:
            return False

        return cleaned.isdigit()

    # =========================================================
    # NEARBY NUMERIC LINES
    # =========================================================

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

        # Look around the current line

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

    # =========================================================
    # LINE Y POSITION
    # =========================================================

    def _line_y(self, line):

        bbox = line.get("bbox")

        if not bbox or len(bbox) < 4:

            return None

        try:

            return float(bbox[1])

        except (TypeError, ValueError):

            return None

    # =========================================================
    # LINE NUMERIC RATIO
    # =========================================================

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


# =============================================================
# LOAD CLEANED JSON
# =============================================================

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


# =============================================================
# SAVE ANALYZED JSON
# =============================================================

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


# =============================================================
# MAIN
# =============================================================

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

        analyzed_reports = (
            analyzer.analyze(
                cleaned_reports
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