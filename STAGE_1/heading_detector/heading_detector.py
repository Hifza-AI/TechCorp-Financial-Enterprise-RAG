import json
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path

print(">>> RUNNING UPDATED HEADING DETECTOR - v2 <<<")
class HeadingDetector:
    """
    Detects headings using font-based signals (size, bold, italic)
    relative to the document's own body-text baseline -- NOT
    hardcoded text patterns like "Item" or "Note" or Title-Case
    checks. This makes it company-agnostic: it works the same way
    whether the PDF uses ALL CAPS headings, Title Case headings,
    or anything else, because it never looks at casing to decide.

    Key signals used (confirmed from real extracted data):
      - Bold (flags & 16)      -> strong heading signal
      - Italic (flags & 2)     -> sub-heading signal (seen on
                                   MD&A subsection labels like
                                   "iPhone", "Mac", "iPad")
      - Size relative to the page/report's own body-text baseline
        (NOT an absolute number, since baseline differs per PDF)
      - Short line length
      - Doesn't end in a period (headings rarely do; sentences do)
    """

    BOLD_FLAG = 16
    ITALIC_FLAG = 2

    def __init__(
        self,
        heading_score_threshold=5,
        max_heading_words=12,
    ):
        self.heading_score_threshold = heading_score_threshold
        self.max_heading_words = max_heading_words

    # =========================================================
    # MAIN
    # =========================================================

    def detect(self, cleaned_reports):

        detected_reports = []

        for report in cleaned_reports:

            detected_reports.append(
                self._detect_report(report)
            )

        return detected_reports

    # =========================================================
    # REPORT
    # =========================================================

    def _detect_report(self, report):

        detected_report = deepcopy(report)

        # Baseline is computed ONCE per report (not per page),
        # because body text size is usually consistent across
        # an entire 10-K filing. This also makes headings that
        # repeat with slightly different per-page noise still
        # comparable against one stable reference point.
        baseline_size = self._compute_baseline_size(report)

        detected_pages = []

        for page in report.get("pages", []):

            detected_page = self._detect_page(page, baseline_size)

            detected_pages.append(detected_page)

        detected_report["pages"] = detected_pages
        detected_report["baseline_body_size"] = baseline_size

        return detected_report

    # =========================================================
    # BASELINE BODY-TEXT SIZE
    # =========================================================

    def _compute_baseline_size(self, report):
        """
        The baseline is the most common font size among NORMAL
        (non-bold, non-italic) spans across the whole report.
        This is almost always the body-paragraph text size.
        Using the report's own statistics (instead of a fixed
        number like 9.0) is what makes this generalize across
        companies whose base font size differs.
        """

        size_counter = Counter()

        for page in report.get("pages", []):

            for line in page.get("lines", []):

                for span in line.get("spans", []):

                    flags = span.get("flags", 0) or 0

                    is_bold = bool(flags & self.BOLD_FLAG)
                    is_italic = bool(flags & self.ITALIC_FLAG)

                    if is_bold or is_italic:
                        continue

                    text = (span.get("text") or "").strip()

                    if not text:
                        continue

                    size = span.get("size")

                    if size is None:
                        continue

                    # Round to 1 decimal to absorb tiny float noise
                    # (PyMuPDF sizes are rarely perfectly identical)
                    size_counter[round(float(size), 1)] += 1

        if not size_counter:
            return 9.0  # sane fallback if a report has no normal text at all

        return size_counter.most_common(1)[0][0]

    # =========================================================
    # PAGE
    # =========================================================

    def _detect_page(self, page, baseline_size):

        detected_page = deepcopy(page)

        lines = page.get("lines", [])

        heading_candidates = []

        # Tracks whether we are currently inside a long BOLD sentence
        # that wrapped across multiple physical PDF lines (e.g. Apple's
        # Risk Factors style, where the entire first sentence of a risk
        # item is bolded as an inline "topic sentence"). Without this,
        # the short trailing fragment of such a sentence (e.g.
        # "local currencies.") gets scored as a standalone heading,
        # even though it's just the tail end of body text.
        bold_run_open = False

        for index, line in enumerate(lines):

            analysis = self._analyze_line(
                line,
                baseline_size,
            )

            analysis["line_index"] = index

            text = analysis["text"]
            is_bold = analysis["is_bold"]

            if bold_run_open:

                # We're continuing a previously-opened long bold
                # sentence -- this line is a continuation fragment,
                # NOT a new standalone heading, regardless of its
                # own score.
                if is_bold:

                    if analysis["is_heading"]:
                        analysis["is_heading"] = False
                        analysis["level"] = 0
                        analysis["reasons"].append(
                            "continuation_of_bold_sentence"
                        )

                    # The run closes once the sentence actually ends.
                    if text.endswith("."):
                        bold_run_open = False

                else:
                    # Bold styling stopped -> the run is over.
                    bold_run_open = False

            else:

                # Open a new bold-run if this line is bold, long
                # enough to have been rejected purely for length
                # ("too_long"), and doesn't already end the sentence.
                if (
                    is_bold
                    and not analysis["is_heading"]
                    and "too_long" in analysis["reasons"]
                    and not text.endswith(".")
                ):
                    bold_run_open = True

            heading_candidates.append(analysis)

        detected_page["heading_analysis"] = {
            "candidates": heading_candidates,
            "heading_count": sum(
                1 for item in heading_candidates if item["is_heading"]
            ),
        }

        return detected_page

    # =========================================================
    # LINE ANALYSIS
    # =========================================================

    def _analyze_line(self, line, baseline_size):

        text = (line.get("text") or "").strip()

        if not text:
            return self._empty_result(text)

        spans = line.get("spans", [])

        if not spans:
            return self._empty_result(text)

        # A line can technically mix spans of different styles
        # (e.g. a checkbox glyph + bold text). We take the
        # DOMINANT style -- the style of the span with the most
        # characters -- since that best represents the line.
        dominant_span = max(
            spans,
            key=lambda s: len((s.get("text") or "").strip()),
        )

        size = dominant_span.get("size")
        flags = dominant_span.get("flags", 0) or 0

        if size is None:
            return self._empty_result(text)

        is_bold = bool(flags & self.BOLD_FLAG)
        is_italic = bool(flags & self.ITALIC_FLAG) and not is_bold

        size = float(size)
        relative_size = size / baseline_size if baseline_size else 1.0

        score, reasons = self._score_line(
            text=text,
            size=size,
            relative_size=relative_size,
            is_bold=is_bold,
            is_italic=is_italic,
        )

        is_heading = score >= self.heading_score_threshold

        level = (
            self._estimate_level(relative_size, is_bold, is_italic)
            if is_heading
            else 0
        )

        return {
            "text": text,
            "is_heading": is_heading,
            "level": level,
            "score": score,
            "reasons": reasons,
            "size": round(size, 2),
            "relative_size": round(relative_size, 2),
            "is_bold": is_bold,
            "is_italic": is_italic,
        }

    # =========================================================
    # EMPTY / NON-TEXT LINE
    # =========================================================

    def _empty_result(self, text):
        return {
            "text": text,
            "is_heading": False,
            "level": 0,
            "score": 0,
            "reasons": [],
            "size": None,
            "relative_size": None,
            "is_bold": False,
            "is_italic": False,
        }

    # =========================================================
    # SCORING
    # =========================================================

    def _score_line(self, text, size, relative_size, is_bold, is_italic):

        score = 0
        reasons = []

        words = text.split()
        word_count = len(words)

        # ---------------------------------------------------
        # Hard rejects (checked BEFORE style scoring)
        # ---------------------------------------------------

        letters = [c for c in text if c.isalpha()]

        # A "heading" with zero alphabetic characters (e.g. "-",
        # "94-2404110", "(408) 996-1010") is never a real heading --
        # it's table/cover-page data that happens to be bold.
        # Seen on: Apple 2024 cover page securities-registration table.
        if not letters:
            return 0, ["no_letters"]

        # A single word ending in "." (e.g. "condition.") is almost
        # always the tail end of a wrapped sentence that picked up
        # stray bold styling -- not a real heading. Numbered headings
        # like "Item 1." are the one legitimate exception, so we only
        # allow a period here if the word before it is numeric/roman
        # (e.g. "1.", "IV.", "Item 1.").
        if word_count == 1 and text.endswith("."):
            core = text[:-1]
            looks_like_numbering = (
                core.isdigit()
                or re.fullmatch(r"[IVXLCDM]+", core, re.IGNORECASE)
                or re.fullmatch(r"[A-Za-z]*\s*\d+", core)
            )
            if not looks_like_numbering:
                return 0, ["dangling_sentence_fragment"]

        # ---------------------------------------------------
        # Style signals (strongest predictors, per real data)
        # ---------------------------------------------------

        if is_bold:
            score += 3
            reasons.append("bold")

        elif is_italic:
            score += 2
            reasons.append("italic")

        # ---------------------------------------------------
        # Size relative to THIS document's own body baseline
        # ---------------------------------------------------

        if relative_size >= 1.5:
            score += 3
            reasons.append("much_larger_than_body")

        elif relative_size >= 1.15:
            score += 2
            reasons.append("larger_than_body")

        elif relative_size >= 0.98:
            # Same size as body -- only counts alongside bold/italic
            # (this is exactly the "Products and Services Performance"
            # case: same size as body, bold is what makes it a heading)
            score += 1
            reasons.append("body_size_but_styled")

        else:
            # Smaller than body text -- footnote markers, page
            # numbers, etc. Actively penalize.
            score -= 2
            reasons.append("smaller_than_body")

        # ---------------------------------------------------
        # Length signals
        # ---------------------------------------------------

        if word_count == 0:
            return 0, ["empty"]

        if word_count <= self.max_heading_words:
            score += 2
            reasons.append("short")
        elif word_count <= self.max_heading_words * 2:
            score += 0  # neutral, borderline
        else:
            score -= 3
            reasons.append("too_long")

        # ---------------------------------------------------
        # Sentence-ending punctuation (headings rarely end in '.')
        # ---------------------------------------------------

        if text.endswith(".") and word_count > 3:
            score -= 2
            reasons.append("ends_like_sentence")

        # ---------------------------------------------------
        # All-caps bonus (still just a bonus, never the sole signal --
        # this avoids the earlier Title-Case-only bug that broke on
        # ALL CAPS headings from other companies)
        # ---------------------------------------------------

        # (letters already computed above in the hard-reject check)
        upper_ratio = sum(c.isupper() for c in letters) / len(letters)

        if upper_ratio >= 0.9:
            score += 1
            reasons.append("all_caps")

        return score, reasons

    # =========================================================
    # LEVEL ESTIMATION
    # =========================================================

    def _estimate_level(self, relative_size, is_bold, is_italic):
        """
        Level 1 = document/part title (e.g. "Apple Inc.", "FORM 10-K")
        Level 2 = major section heading (e.g. "Products and Services
                   Performance", "Item 7. Management's Discussion...")
        Level 3 = sub-heading (e.g. italic "iPhone", "Mac" inside MD&A)
        """

        if relative_size >= 1.5:
            return 1

        if is_italic and relative_size < 1.15:
            return 3

        if is_bold:
            return 2

        return 3


# =============================================================
# LOAD CLEANED JSON
# =============================================================

def load_cleaned_reports(input_dir):

    input_dir = Path(input_dir)

    reports = []

    if not input_dir.exists():
        print(f"Folder not found: {input_dir}")
        return reports

    for company_dir in sorted(input_dir.iterdir()):

        if not company_dir.is_dir():
            continue

        for json_file in sorted(company_dir.glob("*_cleaned.json")):

            with open(json_file, "r", encoding="utf-8") as f:
                reports.append(json.load(f))

    return reports


# =============================================================
# SAVE DETECTED JSON
# =============================================================

def save_detected_reports(
    detected_reports,
    output_dir="STAGE_1/heading_detection",
):

    output_dir = Path(output_dir)

    for report in detected_reports:

        company = report["company"]

        company_dir = output_dir / company

        company_dir.mkdir(parents=True, exist_ok=True)

        output_file = company_dir / (
            Path(report["file_name"]).stem + "_headings.json"
        )

        with open(output_file, "w", encoding="utf-8") as f:

            json.dump(
                report,
                f,
                indent=4,
                ensure_ascii=False,
                default=str,
            )

        print(f"Saved Heading Detection: {output_file}")


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":

    INPUT_DIR = "STAGE_1/cleaned"
    OUTPUT_DIR = "STAGE_1/heading_detection"

    print("\n====================================")
    print(" Heading Detector Started")
    print("====================================\n")

    cleaned_reports = load_cleaned_reports(INPUT_DIR)

    if not cleaned_reports:

        print("No cleaned JSON files found.")
        print("Run TextCleaner first.")

    else:

        detector = HeadingDetector()

        detected_reports = detector.detect(cleaned_reports)

        save_detected_reports(detected_reports, OUTPUT_DIR)

        total_headings = 0

        for report in detected_reports:
            for page in report["pages"]:
                total_headings += page["heading_analysis"]["heading_count"]

        print("\n====================================")
        print(" Heading Detection Completed")
        print("====================================")
        print(f"Reports Processed : {len(detected_reports)}")
        print(f"Total Headings    : {total_headings}")
        print("\nOutput:")
        print(OUTPUT_DIR)