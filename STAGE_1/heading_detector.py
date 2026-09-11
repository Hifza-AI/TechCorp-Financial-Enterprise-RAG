import json
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path


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

    def _line_y(self, line):

        bbox = line.get("bbox")

        if not bbox or len(bbox) < 2:
            return None

        try:
            return float(bbox[1])
        except (TypeError, ValueError):
            return None

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

        # NEW: buffers (index, analysis, text) for every line inside
        # the current run, so the FULL sentence can be reconstructed
        # once it closes, AND so we can decide -- only once we know
        # the outcome -- whether the earlier fragment lines should be
        # marked for downstream skipping. Confirmed on Apple 2016's
        # Item 1A: the vast majority of individual risk-factor headers
        # actually wrap across 2 (sometimes 3) physical PDF lines --
        # e.g. "Global markets for the Company's products and services
        # are highly competitive and subject to rapid technological
        # change," / "and the Company may be unable to compete
        # effectively in these markets." Buffering (instead of eagerly
        # tagging each line as we go) means: if the combined run does
        # NOT end up qualifying as a heading, every buffered line is
        # left completely untouched -- it falls back to being read as
        # normal paragraph text by downstream consumers, so nothing is
        # ever silently lost even in an edge case the merge logic
        # doesn't recognize.
        bold_run_buffer = []

        # NEW (Netflix 2025, confirmed via real cleaned.json output):
        # holds a bare Note-number marker ("1.", "7.", "14.") that was
        # just seen sitting completely alone on its own line-object,
        # waiting to be prepended onto the NEXT line's text before
        # that next line is analyzed. See the docstring on
        # _looks_like_orphaned_number_marker() below for the full
        # root-cause explanation.
        pending_number_prefix = None

        for index, line in enumerate(lines):

            raw_text = (line.get("text") or "").strip()

            # NEW: detect a bare Note-number marker sitting alone on
            # its own line, immediately followed by its real title
            # text at the SAME visual row (matching y-coordinate) but
            # a different line-object due to a wide tab/gap in the
            # original PDF between the number and the title.
            #
            # Confirmed on Netflix 2025: "1." (bbox x=34) and
            # "Organization and Summary of Significant Accounting
            # Policies" (bbox x=55, IDENTICAL y=105.02 top/bottom) are
            # two SEPARATE line-objects, not two physically stacked
            # lines. The same pattern recurred for "7." + "Debt", "9."
            # + "Commitments and Contingencies", "10." + "Stockholders'
            # Equity", and others -- 9 of Netflix's 14 numbered Notes
            # in total. The one Note that happened to survive with its
            # number intact, "6. Acquisitions", did so only because
            # ITS number and title were already a single combined
            # line-object ("6. Acquisitions") in the raw extraction --
            # confirming the bug is about this specific PDF-extraction
            # inconsistency, not about the Note-numbering convention
            # itself (which heading_detector already otherwise
            # handles fine, per the Chipotle bare-number fix above).
            #
            # Without this fix, a standalone "1." has ZERO alphabetic
            # characters, so it hits the "no_letters" hard-reject in
            # _score_line() below and is NEVER classified as a
            # heading at all -- it falls through as an ordinary
            # paragraph, where fix_paragraphs.py's page-footer-number
            # cleanup (built for Google's "53." style footer
            # artifacts) then silently deletes it entirely. Meanwhile
            # the title text on its own ("Organization and Summary of
            # Significant Accounting Policies", "Debt", "Stockholders'
            # Equity") DOES become its own heading -- but with no
            # number prefix, is_note_marker() can never recognize it,
            # so the Note loses its container protection completely:
            # real output showed "Debt" (Note 7), "Commitments and
            # Contingencies" (Note 9), and "Stockholders' Equity"
            # (Note 10) all collapsing as CHILDREN of whichever
            # earlier Note-marker happened to still be open ("6.
            # Acquisitions"), instead of being correct top-level
            # siblings.
            #
            # Fix: when this exact pattern is detected, DON'T analyze
            # the bare-number line as its own candidate at all --
            # instead remember its text, and prepend it onto the very
            # next line's text before that line is analyzed. This
            # keeps every line's own `line_index` perfectly aligned
            # with the untouched original `lines` list (paragraph_
            # parser.py and fix_paragraphs.py need no changes at all),
            # while letting every existing downstream signal (bold,
            # size, is_note_marker, the whole scoring pipeline) see
            # the complete, correct "N. Title" text exactly as if it
            # had always been one line.
            if pending_number_prefix is None and re.fullmatch(r"\d{1,3}\.", raw_text):

                next_line = lines[index + 1] if index + 1 < len(lines) else None

                if next_line is not None:

                    next_text = (next_line.get("text") or "").strip()
                    bbox = line.get("bbox")
                    next_bbox = next_line.get("bbox")

                    if (
                        bbox and next_bbox and next_text
                        and any(c.isalpha() for c in next_text)
                        and abs(bbox[1] - next_bbox[1]) <= 2.0
                    ):

                        pending_number_prefix = raw_text

                        inert_analysis = self._empty_result(raw_text)
                        inert_analysis["line_index"] = index
                        inert_analysis["in_bold_run"] = False
                        heading_candidates.append(inert_analysis)

                        continue

            effective_line = line

            if pending_number_prefix is not None:

                effective_line = dict(line)
                effective_line["text"] = (
                    f"{pending_number_prefix} {raw_text}"
                )

                pending_number_prefix = None

            analysis = self._analyze_line(
                effective_line,
                baseline_size,
            )

            analysis["line_index"] = index

            # NEW: always present (default False) so downstream
            # consumers (e.g. paragraph_parser.py) can reliably check
            # this on EVERY candidate without a .get() fallback. Only
            # ever set True on the EARLIER fragment lines of a run
            # that successfully merged into a heading on its closing
            # line -- meaning "this line's content is already fully
            # captured in a later heading; skip it, don't treat it as
            # paragraph text."
            analysis["in_bold_run"] = False

            text = analysis["text"]
            is_bold = analysis["is_bold"]
            word_count = len(text.split())

            if bold_run_open:

                # We're continuing a previously-opened long bold
                # sentence -- this line is a continuation fragment,
                # NOT a new standalone heading, regardless of its
                # own score.
                if is_bold:

                    bold_run_buffer.append((index, analysis, text))

                    if analysis["is_heading"]:
                        analysis["is_heading"] = False
                        analysis["level"] = 0
                        analysis["reasons"].append(
                            "continuation_of_bold_sentence"
                        )

                    # The run closes once the sentence actually ends.
                    # Re-score the FULL merged sentence and, if it
                    # earns heading status, attach it here (on the
                    # closing line) with the complete combined text --
                    # and mark every EARLIER buffered line as safe to
                    # skip downstream, since its content now lives
                    # entirely in this closing line's merged text.
                    if text.endswith("."):

                        combined_text = " ".join(
                            t for _, _, t in bold_run_buffer
                        )

                        combined_score, combined_reasons = self._score_line(
                            text=combined_text,
                            size=analysis["size"],
                            relative_size=analysis["relative_size"],
                            is_bold=True,
                            is_italic=analysis["is_italic"],
                        )

                        if combined_score >= self.heading_score_threshold:

                            for _, earlier_analysis, _ in bold_run_buffer[:-1]:
                                earlier_analysis["in_bold_run"] = True

                            analysis["is_heading"] = True
                            analysis["text"] = combined_text
                            analysis["score"] = combined_score
                            analysis["reasons"] = combined_reasons + [
                                "merged_wrapped_bold_sentence"
                            ]
                            analysis["level"] = self._estimate_level(
                                combined_text,
                                analysis["relative_size"],
                                True,
                                False,
                            )
                            analysis["is_note_marker"] = self._is_note_marker(
                                combined_text
                            )
                            analysis["is_top_level_marker"] = (
                                self._is_top_level_marker(combined_text)
                            )
                            analysis["is_prominent_boundary"] = (
                                "much_larger_than_body" in combined_reasons
                                and "all_caps" in combined_reasons
                            )

                        # If combined_score didn't qualify, we
                        # deliberately do NOT touch any buffered
                        # line's in_bold_run/is_heading -- they stay
                        # exactly as their own individual analysis
                        # produced, so their text still surfaces
                        # normally as paragraph content.

                        bold_run_open = False
                        bold_run_buffer = []

                else:
                    # Bold styling stopped -> the run is abandoned
                    # (no combined heading is created from a partial,
                    # never-closed run; buffered lines are left
                    # untouched, same safety fallback as above).
                    bold_run_open = False
                    bold_run_buffer = []

            else:

                # Open a new bold-run if this line is bold, long
                # enough that it's clearly a WRAPPED sentence (not a
                # short standalone label/heading), and doesn't
                # already end the sentence.
                #
                # FIX: this used to check `"too_long" in reasons`,
                # but the length-scoring below has a MIDDLE "neutral"
                # bucket (word_count between max_heading_words and
                # 2x that) which never appends any reason string at
                # all. A bold sentence-opener landing in that neutral
                # zone (confirmed on Apple 2016 page 12: "To remain
                # competitive and stimulate customer demand, the
                # Company must successfully manage frequent product"
                # -- 14 words, `reasons=['bold','body_size_but_styled']`,
                # no "too_long" tag) silently failed to open the run,
                # so its trailing fragment on the next line
                # ("introductions and transitions.") was scored as
                # its own standalone heading instead of being
                # recognized as a continuation.
                #
                # Checking word_count directly (instead of depending
                # on a specific reason string existing) catches both
                # the "too_long" bucket AND this in-between "neutral"
                # bucket, since both exceed max_heading_words.
                #
                # NEW (Starbucks 2025, confirmed via real cleaned.json
                # output): a long bold CAPTION line -- e.g. "Fiscal
                # Years ended September 28, 2025, September 29, 2024,
                # and October 1, 2023" (13 words, no ending period,
                # sitting right at the top of the Notes section, once,
                # right before Note 1) -- can ALSO cross this same
                # word-count threshold, even though it is structurally
                # nothing like an unfinished wrapping sentence. It's a
                # complete, self-contained date-range caption that
                # just happens to be long and lacks terminal
                # punctuation (dates and commas, not a sentence).
                #
                # Confirmed real-world impact: this incorrectly opened
                # a bold-run right before "Note 1: Summary of
                # Significant Accounting Policies and Estimates" --
                # so Note 1's own title, AND its very first sub-topic
                # "Description of Business", were both swallowed as
                # if they were trailing FRAGMENTS of the fiscal-year
                # caption's sentence, rather than being recognized as
                # their own real headings. Because this caption only
                # ever appears ONCE, right before Note 1 specifically
                # (Notes 2-19 don't have it immediately above them),
                # only Note 1 lost its heading status this way --
                # every other Note title on the page was unaffected.
                #
                # Fix: recognize this specific "Fiscal Year(s) ended
                # <dates>" caption shape and exclude it from ever
                # opening a bold-run, the same way the existing
                # units-disclaimer caption ("(in millions)") is
                # already excluded elsewhere from being mistaken for
                # real heading/sentence content. This is narrowly
                # scoped to the fiscal-year-caption phrasing
                # specifically, so it cannot affect any genuine long
                # bold sentence-opener elsewhere (which is exactly
                # what this mechanism still needs to keep catching).
                is_fiscal_year_caption = bool(
                    re.match(
                        r"^Fiscal\s+Years?\s+Ended\b",
                        text.strip(),
                        re.IGNORECASE,
                    )
                )

                if (
                    is_bold
                    and not analysis["is_heading"]
                    and word_count > self.max_heading_words
                    and not text.endswith(".")
                    and not is_fiscal_year_caption
                ):
                    bold_run_open = True
                    bold_run_buffer = [(index, analysis, text)]

            heading_candidates.append(analysis)

        # NEW (ServiceNow 2016, confirmed via real cleaned.json
        # output): a genuine grouped-column table header -- e.g. a
        # Stockholders' Equity statement's "Common Stock" / "Treasury
        # Stock" / "Additional Paid-In Capital" / "Accumulated Other
        # Comprehensive Loss" / "Total Stockholders' Equity" group
        # titles, wrapped across several physical lines and packed
        # into a tight vertical band -- is made of MANY short, bold
        # text fragments that individually satisfy every signal
        # _score_line() looks for (bold, short, often all-caps or
        # title-case), so each one independently crosses the heading
        # threshold on its own.
        #
        # This didn't matter for the ORIGINAL case this heading-over-
        # table priority rule was built for (Apple 2016's "iPhone" /
        # "Mac" / "Services" section titles, which sit ALONE, with
        # normal paragraph text before and after -- never surrounded
        # by a cluster of other short-bold fragments). But confirmed
        # here on ServiceNow 2016: 13 separate short fragments ("Common
        # Stock", "Additional", "Paid-in", "Capital", "Accumulated",
        # "Deficit", "Accumulated", "Other", "Comprehensive", "Loss",
        # "Total", "Stockholders'", "Equity", "Shares", "Amount") all
        # independently scored as genuine headings, packed into a
        # ~26pt-tall vertical band across 6 distinct x-positions --
        # the unmistakable shape of a multi-row, multi-column table
        # header, not a sequence of real section titles.
        #
        # Because paragraph_parser.py checks is_heading_line BEFORE
        # is_table_line (deliberately, for the Apple 2016 fix above),
        # every one of these words got pulled out as its own spurious
        # heading -- leaving the table with NONE of its real header
        # words to build clean column names from. table_parser then
        # had nothing to work with except the first data row, which
        # is why the table's own real title ("CONSOLIDATED STATEMENTS
        # OF STOCKHOLDERS' EQUITY" -- which DOES still correctly
        # become its own heading, unaffected by this bug) ended up
        # with a lone, raw, unlabeled pipe-separated table under it
        # instead of a properly parsed one.
        #
        # Fix: a SHORT (<= 3 words) heading candidate that is NOT
        # itself a known structural marker (Item/PART/Note/
        # CONSOLIDATED-title/etc -- those are never demoted, no
        # matter how densely packed the surrounding page is) gets
        # DEMOTED back to a non-heading if it has at least 3 OTHER
        # short heading-candidates within a tight +/-30pt vertical
        # band. A genuine standalone short heading (Apple's "iPhone")
        # never has this many short-bold neighbors immediately
        # surrounding it -- normal paragraph text (long, non-bold)
        # sits before and after it instead. Only candidates that
        # would otherwise become headings are ever considered here,
        # so this can only ever correct a false-positive heading back
        # into (correctly) falling through to table detection --  it
        # never removes a heading that had no realistic table-header
        # explanation to begin with.
        DEMOTE_MAX_WORDS = 3
        DEMOTE_Y_WINDOW = 30
        DEMOTE_MIN_NEIGHBORS = 3

        short_heading_ys = [
            (item["line_index"], self._line_y(lines[item["line_index"]]))
            for item in heading_candidates
            if item["is_heading"]
            and len(item["text"].split()) <= DEMOTE_MAX_WORDS
            and not item.get("is_note_marker")
            and not item.get("is_top_level_marker")
        ]

        for item in heading_candidates:

            if not item["is_heading"]:
                continue

            if len(item["text"].split()) > DEMOTE_MAX_WORDS:
                continue

            if item.get("is_note_marker") or item.get("is_top_level_marker"):
                continue

            this_y = self._line_y(lines[item["line_index"]])

            if this_y is None:
                continue

            neighbor_count = sum(
                1
                for other_index, other_y in short_heading_ys
                if other_index != item["line_index"]
                and other_y is not None
                and abs(other_y - this_y) <= DEMOTE_Y_WINDOW
            )

            if neighbor_count >= DEMOTE_MIN_NEIGHBORS:
                item["is_heading"] = False
                item["reasons"] = item.get("reasons", []) + [
                    "demoted_table_header_zone"
                ]

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
            self._estimate_level(text, relative_size, is_bold, is_italic)
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
            "is_note_marker": is_heading and self._is_note_marker(text),
            "is_top_level_marker": is_heading and self._is_top_level_marker(text),
            # NEW: a heading that is BOTH strongly-oversized relative
            # to body text ("much_larger_than_body") AND fully
            # ALL-CAPS is a distinctly MORE prominent styling tier
            # than an ordinary bold sub-heading -- confirmed on Intel
            # 2019, where every genuine Note-level section boundary
            # ("BORROWINGS", "DERIVATIVE FINANCIAL INSTRUMENTS",
            # "RETIREMENT BENEFIT PLANS", etc.) used this exact,
            # consistent combination, while that same document's
            # ordinary Note sub-topics did not. This matters because
            # Intel's 2019 filing -- unlike its own 2025 filing, or
            # any of Apple/Google/Nvidia/Meta's -- doesn't put a
            # "Note N" prefix on the ACTUAL section-starting heading
            # at all (the "Note N: Title" text only appears in the
            # Index/cross-reference listing and in inline mentions
            # elsewhere, never on the real heading line itself), so
            # is_note_marker can never match it no matter which
            # punctuation variant it looks for. Without recognizing
            # this some other way, "CONSOLIDATED STATEMENTS OF
            # STOCKHOLDERS' EQUITY" (a genuine note-marker via the
            # CONSOLIDATED-title pattern) never gets closed out, and
            # ends up swallowing EVERY subsequent Note and even the
            # Exhibits section at the very end of the filing as its
            # descendants.
            "is_prominent_boundary": (
                is_heading
                and "much_larger_than_body" in reasons
                and "all_caps" in reasons
            ),
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
            "is_note_marker": False,
            "is_top_level_marker": False,
            "is_prominent_boundary": False,
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

        # A parenthetical units-disclaimer -- "(in millions)", "(in
        # thousands, except per share amounts)", "(dollars in
        # millions)" -- is a UNIVERSAL SEC-filing convention sitting
        # directly under nearly every financial statement's real
        # title. It's short and often bold/italic, so it otherwise
        # scores well enough to be misclassified as its own SEPARATE
        # heading (a sibling of the real title, not part of it).
        # Confirmed on Google/Alphabet 2025: "CONSOLIDATED BALANCE
        # SHEETS" and "(in millions, except par value per share
        # amounts)" became TWO separate same-level headings, so the
        # actual data table ended up attached to the meaningless
        # caption instead of the real statement title -- across all
        # 4 core financial statements (Balance Sheet, Income
        # Statement, Comprehensive Income, Cash Flows). This pattern
        # is not company-specific -- it's the standard way SEC
        # filings caption their statements -- so it will recur for
        # any company.
        #
        # NEW (Chipotle 2025, confirmed via real hierarchy_outline.txt
        # output): the ORIGINAL pattern only allowed a fixed lead-in
        # of "dollars "/"amounts " (or nothing) right before "in
        # millions/thousands/billions". Chipotle's caption --
        # "(dollar and share amounts in thousands, unless otherwise
        # specified)" -- has a longer, differently-worded lead-in
        # ("dollar and share amounts"), which didn't match either
        # fixed alternative, so it slipped through and became its own
        # spurious heading node (a sibling of "1. Description of
        # Business...", sitting directly under "NOTES TO CONSOLIDATED
        # FINANCIAL STATEMENTS").
        #
        # Since a real statement/section title is never itself
        # wrapped in parentheses, ANY parenthetical whose content
        # contains "in millions/thousands/billions" as a substring
        # -- regardless of what other words surround it inside the
        # parens -- is safe to treat as this same units-disclaimer
        # convention. Matching on that anchor phrase (rather than a
        # fixed, enumerated set of lead-in phrases) generalizes to
        # any company's own wording of this caption without needing
        # to special-case each new variant as it's discovered.
        if re.fullmatch(
            r"\(\s*(dollars\s+|amounts\s+)?in\s+(millions|thousands|billions)"
            r"(\s*,\s*[^)]*)?\)",
            text.strip(),
            re.IGNORECASE,
        ) or re.search(
            r"\bin\s+(millions|thousands|billions)\b",
            text.strip(),
            re.IGNORECASE,
        ) and text.strip().startswith("(") and text.strip().endswith(")"):
            return 0, ["units_disclaimer_caption"]

        # A bare column-header DATE ("Jan 25, 2026", "December 31,
        # 2025") or a period-range label ("Year Ended", "Quarter
        # Ended", "Three Months Ended") is another universal SEC-
        # filing convention: financial statement tables commonly
        # stack their column headers across MULTIPLE physical PDF
        # lines -- e.g. "Year Ended" on one line, then each date on
        # its own line below it, all sitting directly above the
        # actual numeric data. These are short and often bold, so
        # they otherwise score well enough to be misclassified as
        # SEPARATE, genuine section headings.
        #
        # Confirmed on Nvidia 2026: "Year Ended", "Jan 25, 2026",
        # "Jan 26, 2025", and "Jan 28, 2024" each became their own
        # heading node directly above EVERY core financial statement
        # (Income Statement, Comprehensive Income, Balance Sheet,
        # Shareholders' Equity, Cash Flows) and several MD&A tables.
        # Since table_analyzer never got a chance to treat these
        # lines as the table's own header-zone, table_parser fell
        # back to using the table's FIRST DATA ROW as if it were the
        # header -- e.g. a Revenue table's header showed "215,938,
        # 130,497, 60,922" (the actual Revenue figures) instead of
        # "Jan 25, 2026, Jan 26, 2025, Jan 28, 2024", corrupting
        # every downstream table's column meaning across the entire
        # filing.
        if re.fullmatch(
            r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
            r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|"
            r"Nov(?:ember)?|Dec(?:ember)?)\.?\s+\d{1,2},\s+\d{4}",
            text.strip(),
            re.IGNORECASE,
        ):
            return 0, ["table_column_header_date"]

        if re.fullmatch(
            r"(Year|Years|Quarter|Quarters|Month|Months|Week|Weeks|"
            r"Three\s+Months|Six\s+Months|Nine\s+Months)\s+Ended",
            text.strip(),
            re.IGNORECASE,
        ):
            return 0, ["table_column_header_date"]

        # NEW (PayPal 2025, confirmed via real hierarchy_outline.txt
        # output): some companies combine the period-label AND the
        # actual month/day directly onto ONE physical line -- "Year
        # Ended December 31," -- rather than as two fully separate
        # lines ("Year Ended" alone, then a bare date below it, as
        # already handled by the two exclusions just above). The
        # YEAR itself is often left off this combined caption
        # (appearing instead as each column's own repeated header
        # value further down, once per year-column), so it can't be
        # matched by the bare-date pattern above either.
        #
        # Confirmed real-world impact: "Year Ended December 31,"
        # became its own heading FOUR separate times -- directly
        # above the Income Statement, Comprehensive Income, and (twice)
        # the Cash Flow Statement -- each one swallowing that
        # statement's own short boilerplate closing line ("The
        # accompanying notes are an integral part of these
        # consolidated financial statements.") as if it were genuine,
        # unique heading content, and fragmenting what should be a
        # single clean statement into a statement-title node plus a
        # near-empty "Year Ended December 31," child node.
        # NEW (PayPal 2016, confirmed via real chunks.json output):
        # the Balance Sheet uses a POINT-IN-TIME caption ("As of
        # December 31,") rather than the PERIOD caption ("Year Ended
        # December 31,") the Income/Cash-Flow statements use. Same
        # underlying convention (a table column-header caption with
        # the year itself often omitted, appearing separately per
        # column below), same fix shape: recognize "As of <Month
        # Day>[,][<year>]" as an additional alternative alongside the
        # "<Period> Ended <date>" pattern already handled above.
        # NEW (Intuit 2026, confirmed via real hierarchy_outline.txt
        # output): Intuit's own point-in-time table caption uses "At
        # <Month Day, Year>:" -- a THIRD wording for the exact same
        # convention "As of <date>" and "Year Ended <date>" already
        # cover, just with "At" instead of "As of" and a trailing
        # colon. Confirmed real-world impact: "At July 31, 2026:" and
        # "At July 31, 2025:" each became their own empty spurious
        # heading directly above Note 6's intangible-asset
        # amortization sub-table.
        if re.fullmatch(
            r"(?:As\s+of\s+|At\s+|(?:Year|Years|Quarter|Quarters|Month|"
            r"Months|Week|Weeks|Three\s+Months|Six\s+Months|"
            r"Nine\s+Months)\s+Ended\s+)"
            r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
            r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|"
            r"Nov(?:ember)?|Dec(?:ember)?)\.?\s+\d{1,2},?\s*(\d{4})?:?",
            text.strip(),
            re.IGNORECASE,
        ):
            return 0, ["table_column_header_date"]

        # NEW (Intuit 2026, confirmed via real hierarchy_outline.txt
        # output): a bare "Fiscal year ending <Month Day>," caption
        # (the year itself omitted, since it repeats once per
        # year-column below) sitting alone on its own line, directly
        # analogous to the already-handled "Year Ended"/"As of"/"At"
        # captions but using yet another common SEC-filing wording.
        if re.fullmatch(
            r"Fiscal\s+years?\s+ending\s+"
            r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
            r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|"
            r"Nov(?:ember)?|Dec(?:ember)?)\.?\s+\d{1,2},?\s*(\d{4})?",
            text.strip(),
            re.IGNORECASE,
        ):
            return 0, ["table_column_header_date"]

        # NEW: some companies combine the period-label AND the units-
        # disclaimer onto ONE physical line -- "Years Ended (In
        # Millions)", "Years Ended ($ In Millions)", "Years Ended (In
        # Millions, Except Per Share Amounts)" -- rather than as two
        # separate lines. Neither of the two exclusions above catches
        # this combined form on its own (the first requires the line
        # to be JUST "Year(s) Ended" with nothing else; the units-
        # disclaimer check earlier in this function requires the line
        # to be JUST the parenthetical). Confirmed on Intel 2025: this
        # combined caption became its own heading 25 separate times
        # across the filing -- above the Income Statement, Cash Flow
        # Statement, Comprehensive Income Statement, and numerous Note
        # tables -- causing the same header/first-data-row corruption
        # already seen on Nvidia.
        if re.fullmatch(
            r"(Year|Years|Quarter|Quarters|Month|Months|Week|Weeks|"
            r"Three\s+Months|Six\s+Months|Nine\s+Months)\s+Ended\s*"
            r"\(\s*\$?\s*(dollars\s+|amounts\s+)?in\s+"
            r"(millions|thousands|billions)(\s*,\s*[^)]*)?\)",
            text.strip(),
            re.IGNORECASE,
        ):
            return 0, ["table_column_header_date"]

        # NEW (Microsoft 2026, confirmed via real hierarchy_outline.txt
        # + video-frame output): a standalone table-cell VALUE that
        # expresses a weighted-average duration -- "10 years", "5
        # years", "9 years", "0 years" -- appears bold (financial
        # tables commonly bold their numeric values) inside the
        # "Weighted Average Life" column of Note 9 - Intangible
        # Assets' second sub-table. Bold + short + doesn't end in a
        # period was enough to independently cross the heading
        # threshold, producing spurious sibling nodes "[SECTION L3]
        # 10 years $" and "[SECTION L3] 5 years $" instead of these
        # being read as ordinary table cell values under Note 9.
        #
        # A genuine section heading is never JUST a bare number
        # followed by the word "year"/"years" (with an optional
        # trailing "$" symbol that leaked in from an adjacent
        # column) -- this phrasing only ever occurs as a table VALUE
        # in these filings -- so it's safe to hard-reject regardless
        # of styling, the same way the date/units captions above are.
        if re.fullmatch(
            r"\d+(\.\d+)?\s+years?(\s*\$)?",
            text.strip(),
            re.IGNORECASE,
        ):
            return 0, ["table_duration_value"]

        # NEW (Microsoft 2026, confirmed via real hierarchy_outline.txt
        # output): a small, fixed set of SEC-standard financial-
        # statement SECTION-DIVIDER labels -- "Assets" and
        # "Liabilities and stockholders'/shareholders' equity" on the
        # Balance Sheet; "Operations", "Financing", "Investing" on the
        # Statement of Cash Flows; "Common stock and paid-in capital",
        # "Retained earnings", "Accumulated other comprehensive
        # (loss)/income" on the Statement of Stockholders' Equity --
        # are bold, short, title-case lines that independently score
        # as genuine headings, exactly like the ServiceNow 2016
        # table-header-zone words already handled above. Unlike that
        # case, these dividers are NOT densely clustered (each sits
        # alone, spaced out through its own statement), so the
        # existing neighbor-count demotion pass does not catch them.
        #
        # Confirmed real-world impact on Microsoft's Balance Sheet:
        # pulling "Assets" and "Liabilities and stockholders' equity"
        # out as their own standalone headings corrupted the
        # surrounding table's bbox enough that the ENTIRE 38-row
        # Balance Sheet table (Total assets: 758,376; Total
        # liabilities and stockholders' equity: 758,376) ended up
        # attached under the PRECEDING heading ("COMPREHENSIVE INCOME
        # STATEMENTS") instead of under "BALANCE SHEETS" -- a genuine
        # value-level retrieval failure, not just a cosmetic one.
        # Cash Flows' "Operations"/"Financing"/"Investing" similarly
        # became three empty orphan sibling nodes.
        #
        # NEW (AMD 2025, confirmed via real hierarchy_outline.txt
        # output): the exact-string set approach kept needing a new
        # entry added every time a new company's own wording variant
        # showed up (MSFT's bare "Assets"/"Liabilities and
        # stockholders' equity"; now AMD's "Current assets:", "Total
        # current assets", "Total assets", "Current liabilities:",
        # "Total current liabilities", "Stockholders' equity:",
        # "Total stockholders' equity", "Total liabilities and
        # stockholders' equity", "Capital stock", "Retained earnings
        # (accumulated deficit)" -- none of which matched the
        # narrower exact-string set at all, since it never accounted
        # for a "Current"/"Total"/"Total current" PREFIX or an
        # optional trailing colon).
        #
        # Confirmed real-world impact: ALL of AMD's own Balance Sheet
        # sub-dividers ("Current assets:", "Total current assets",
        # "Total assets", "Current liabilities:", "Total current
        # liabilities", "Stockholders' equity:", "Total stockholders'
        # equity", "Total liabilities and stockholders' equity") and
        # its own Stockholders' Equity statement dividers ("Capital
        # stock", "Retained earnings (accumulated deficit)") became
        # spurious, empty sibling headings instead of staying inside
        # their own tables -- the exact same failure mode as MSFT's
        # Balance Sheet bug, just with a wording convention the
        # earlier fix's exact-string set didn't anticipate.
        #
        # Rather than keep growing an exact-match set one company's
        # wording at a time, this is now a REGEX matching the known,
        # FIXED set of SEC-standard structural NOUN PHRASES (assets /
        # liabilities / stockholders'-or-shareholders' equity /
        # capital stock / retained earnings / accumulated other
        # comprehensive income-or-loss / operations / financing /
        # investing), with an OPTIONAL "Current"/"Total"/"Total
        # current" prefix and an OPTIONAL trailing colon -- covering
        # every variant found across companies so far in one general
        # rule, without risking a false match on unrelated content:
        # a heading like "Total Revenue" or "Total Compensation
        # Expense" still safely falls through, since "revenue" and
        # "compensation expense" aren't among these fixed structural
        # nouns.
        _financial_statement_divider_re = re.compile(
            r"^(Current\s+|Total\s+|Total\s+current\s+)?"
            r"("
            r"assets"
            r"|liabilities(\s+and\s+((stockholders'?|shareholders'?)\s+)?equity(\s*\(deficit\))?)?"
            r"|(stockholders'?|shareholders'?)\s+equity(\s*\(deficit\))?"
            r"|common\s+stock\s+and\s+paid-in\s+capital"
            r"|capital\s+stock"
            r"|retained\s+earnings(\s*\(accumulated\s+deficit\))?"
            r"|accumulated\s+other\s+comprehensive\s+(loss|income)"
            r"(\s*\((loss|income)\))?"
            r"|operations"
            r"|financing"
            r"|investing"
            r")"
            r"\s*:?\s*$",
            re.IGNORECASE,
        )

        if _financial_statement_divider_re.match(text.strip()):
            return 0, ["financial_statement_section_divider"]

        # NEW (PayPal 2016, confirmed via real chunks.json output): a
        # short, colon-ending phrase that merely REFERENCES a
        # consolidated financial statement -- "consolidated statement
        # of income:", "consolidated balance sheets:" -- is a
        # narrative cross-reference or table-intro caption, never a
        # genuine standalone heading, regardless of whether it uses
        # the singular or plural form of "statement(s)". This is
        # broader than (and independent of) the is_note_marker()
        # tightening above: even after that fix correctly stops this
        # exact phrase from being treated as a NOTE-level structural
        # marker, it was STILL independently scoring high enough via
        # ordinary bold+short-heading criteria (bold +3, body-size
        # +1, short +2 = 6) to become a heading on its own -- fixing
        # is_note_marker() alone stopped it from popping "NOTE 1"
        # off the stack entirely (the critical part of the bug), but
        # left this phrase cluttering Note 1's children as its own
        # spurious, content-free sub-heading.
        #
        # The colon is the key safety signal here (matching the same
        # convention already used elsewhere in this codebase): a
        # genuine reference to "the consolidated statement of
        # income" always continues into more sentence text when it's
        # part of a real title (which never ends in a colon) or
        # trails onward mid-sentence (which never sits ALONE on a
        # colon-terminated line). Requiring BOTH the "consolidated
        # ... statement(s)/balance sheet(s)" phrase AND a trailing
        # colon keeps this narrowly scoped to exactly this reference-
        # caption shape.
        if text.strip().endswith(":") and re.search(
            r"consolidated\s+(balance\s+sheets?|statements?\s+of\s+\w+)",
            text.strip(),
            re.IGNORECASE,
        ):
            return 0, ["statement_reference_caption"]

        # NEW (Costco 2016, confirmed via real chunks.json output): a
        # heading whose text ENDS with "(Continued)" (any case, with
        # or without a space before the parenthesis) is a page-
        # pagination artifact, NEVER a genuine new section -- it is
        # the SAME heading text repeated verbatim at the top of the
        # next physical page, for whatever multi-page section was
        # already open (Item 1-Business, Item 1A-Risk Factors, Note
        # 1, Note 4, Note 7, Note 8, Note 10, Note 12, and more --
        # confirmed via real hierarchy output on Costco's 2016 10-K,
        # where this fragmented AT LEAST 9 different sections/Notes
        # each into two disconnected sibling nodes instead of one
        # continuous section).
        #
        # This is a DIFFERENT root cause from the already-existing
        # frequency-based boilerplate removal in paragraph_parser.py
        # (_remove_repeated_boilerplate): that mechanism only catches
        # a string that repeats identically many times across the
        # WHOLE document -- but each section's own "(Continued)"
        # variant is DIFFERENT text ("Item 1-Business (Continued)"
        # vs "Note 4-Debt (Continued)"), so no single variant ever
        # repeats often enough to cross that frequency threshold.
        # Recognizing the "(Continued)" SUFFIX PATTERN directly, at
        # the individual-heading level, catches every variant at
        # once regardless of what precedes it or how many times its
        # own specific wording happens to recur.
        #
        # A genuine, substantive SEC-filing section title is never
        # itself named "... (Continued)" -- that phrase is exclusively
        # a PDF-pagination convention -- so hard-rejecting it here
        # (rather than letting it score normally and then trying to
        # suppress it later) is safe for every company: this can only
        # ever stop a page-continuation repeat from fragmenting an
        # already-open section, it can never demote or discard a
        # genuine, unique heading.
        if re.search(r"\(\s*continued\s*\)\s*$", text.strip(), re.IGNORECASE):
            return 0, ["continued_pagination_artifact"]

        # NEW (MSFT 2017 / Costco, confirmed via real chunks.json
        # output): a short, bold table-VALUE fragment that leaked
        # through as its own heading candidate -- e.g. "$ 17,072
        # (a)" (a Goodwill total with its footnote-reference marker),
        # "$ 16,667 (b) $" (an Intangible Assets total sitting
        # between two "$" column markers), "2,148 19 years" (a
        # weighted-average-life value glued to its own duration
        # figure). Each of these independently scored as a heading
        # (bold + short + body-size) purely because it happens to sit
        # alone on its own physical PDF line, exactly like a genuine
        # short title would -- but unlike a real title, its content
        # is almost ENTIRELY digits, currency symbols, commas, and
        # single-letter/single-digit footnote markers, with at most
        # one incidental unit-word ("years") and no real, substantive
        # wording of its own.
        #
        # This is the SAME underlying failure already fixed narrowly
        # for ONE specific recurring shape ("table_duration_value",
        # e.g. "10 years $") -- but confirmed on a DIFFERENT company/
        # year (MSFT 2017's own Goodwill and Intangible Assets notes,
        # with completely different dollar figures than MSFT 2026's
        # case), proving the narrow fix doesn't generalize: every
        # filing's own specific numbers are different, so a fix
        # tied to one exact numeric shape can never cover the next
        # company's own table-value leak.
        #
        # General, robust test: strip out every token that is PURELY
        # a currency symbol, a number (with commas/periods), or a
        # short parenthetical footnote marker ("(a)", "(b)", "(1)",
        # "(12)") from the text, then count what's left. A genuine
        # heading -- "iPhone", "PART II", "NOTE 1 - ACCOUNTING
        # POLICIES", "Debt" -- always has substantial REAL wording of
        # its own once this furniture is removed. A leaked table
        # value has, at most, ONE short generic unit-word left over
        # ("years", "months", etc.) and nothing else -- confirmed
        # this is true for every example found so far, on two
        # different companies' filings five years apart.
        #
        # Scoped to SHORT lines only (<=6 words) so this can never
        # touch a genuine longer sentence that merely happens to
        # contain some numbers (e.g. a real narrative line quoting a
        # dollar figure) -- those already have plenty of real words
        # of their own and would never reduce to zero or one leftover
        # word by this stripping process.
        if word_count <= 6:

            # NEW: operate per-TOKEN rather than stripping digits out
            # of the whole line -- a mixed alphanumeric token like
            # "3M" or "COVID-19" must be counted as a real word AS
            # A WHOLE, not have its digits stripped out from inside
            # it (which would wrongly reduce "3M" down to a bare "M"
            # and reject it as if it were numeric-only furniture).
            # Only a token that is PURELY a currency symbol, a pure
            # number (with commas/periods), or a short parenthetical
            # footnote marker ("(a)", "(12)") on its own is treated
            # as furniture to discard; anything else that contains
            # at least one letter counts as real wording, however
            # short.
            _real_words = []

            for _w in text.split():

                if re.fullmatch(r"\$", _w):
                    continue

                if re.fullmatch(r"(\(\s*[A-Za-z0-9]{1,3}\s*\))+", _w):
                    continue

                if re.fullmatch(r"[\d,\.]+", _w):
                    continue

                # NEW (Intuit 2026, confirmed via real
                # hierarchy_outline.txt output): a dollar sign FUSED
                # directly onto its number with no space ("$21.4",
                # "$8.6") -- unlike the bare "$" case above, which
                # only matches a currency symbol sitting entirely on
                # its own -- is just as much pure numeric furniture
                # as a bare number, so it needs its own explicit
                # strip here too.
                if re.fullmatch(r"\$[\d,]*\.?\d+", _w):
                    continue

                if re.search(r"[A-Za-z]", _w):
                    _real_words.append(_w)

            # NEW: this set is now built to be COMPREHENSIVE rather
            # than growing it one narrow shape at a time -- confirmed
            # necessary after finding FIVE different leaked-table-
            # value shapes across just two companies' filings ("10
            # years $", "$ 17,072 (a)", "2,148 19 years", "21,204
            # (b)(c)", "3,006 September 14, 2017"): a purely
            # POSITIONAL fix (e.g. "reject if this heading sits next
            # to numeric-heavy lines") was deliberately rejected as
            # an alternative here, because genuine headings like
            # "iPhone", "Debt", "Goodwill" ALSO always sit directly
            # above their own numeric table -- that adjacency is
            # exactly why the heading-over-table PRIORITY rule exists
            # in paragraph_parser.py in the first place, so a
            # proximity-based reject would risk breaking those
            # already-verified cases across all 17 companies.
            #
            # A content-word-based filler list is a BOUNDED problem
            # (English has a finite set of duration/calendar/scale/
            # connector words that can plausibly be the one leftover
            # "real word" in a numeric fragment), unlike a statistical
            # threshold, which could misfire in ways impossible to
            # predict without already having seen the data. This
            # covers every category found so far in one pass:
            #   - time/duration units (was already present)
            #   - calendar month names/abbreviations (was already
            #     present, added for the "3,006 September 14, 2017"
            #     Dividends-table leak)
            #   - numeric-scale words ("million"/"billion"/etc.) --
            #     covers a "$500 million"-shaped leak that hasn't
            #     been observed yet but follows the identical pattern
            #   - the SAME connector-word set table_parser.py's
            #     _ORPHAN_CONNECTOR_WORDS already trusts elsewhere in
            #     this codebase for the identical "is this word
            #     meaningful standing alone" judgment call -- reusing
            #     an already-vetted list here is safer than inventing
            #     a new one from scratch.
            _GENERIC_UNIT_WORDS = {
                "year", "years", "month", "months",
                "day", "days", "week", "weeks",
                "quarter", "quarters", "hour", "hours",
                "minute", "minutes",
                "january", "february", "march", "april", "may", "june",
                "july", "august", "september", "october", "november",
                "december", "jan", "feb", "mar", "apr", "jun", "jul",
                "aug", "sep", "sept", "oct", "nov", "dec",
                "million", "billion", "thousand", "trillion",
                # NEW (Intuit 2026, confirmed via real
                # hierarchy_outline.txt output): single-letter scale
                # ABBREVIATIONS ("$21.4 B", "$8.6 B" -- "B" for
                # billion) are just as common in a "Financial
                # Highlights" infographic-style callout as the full
                # word "billion" is in a table caption. Confirmed
                # real-world impact: "$21.4 B", "$12.9 B", "$8.6 B",
                # "$5.9 B", "$4.6 B", and "$8.8 B" all independently
                # scored as headings (at LEVEL 1, since these are
                # rendered in oversized callout-stat font), each one
                # fragmenting a real sentence into three pieces --
                # e.g. "Global Business Solutions revenue of" /
                # "$21.4 B" / "up 14% from fiscal 2025" ended up as
                # three disconnected blocks instead of one sentence.
                "b", "m", "k", "t",
                "and", "or", "of", "the", "in", "for", "to", "with", "&",
            }

            if not _real_words or (
                len(_real_words) == 1
                and _real_words[0].lower() in _GENERIC_UNIT_WORDS
            ):
                return 0, ["numeric_table_value_fragment"]

        # ---------------------------------------------------
        # Style signals (strongest predictors, per real data)
        # ---------------------------------------------------

        # NEW (Microsoft 2026, confirmed via real hierarchy_outline.txt
        # + video-frame output): computed HERE, ahead of the bold/
        # italic check right below, so that check can also grant a
        # style-equivalent bonus to a known structural marker that
        # ISN'T rendered bold or italic at all. See the new
        # "structural_marker_unstyled" branch immediately below for
        # the full rationale (Microsoft underlines its "NOTE N --
        # Title" headings and its "PART II"/"Item 8" running-header
        # caption instead of bolding them). This is the same
        # is_structural_marker value already used further down for
        # the smaller_than_body waiver and the length-bonus -- moving
        # its computation earlier changes nothing else about how it's
        # used, it's simply needed sooner now.
        is_structural_marker = (
            self._is_top_level_marker(text) or self._is_note_marker(text)
        )

        if is_bold:
            score += 3
            reasons.append("bold")

        elif is_italic:
            score += 2
            reasons.append("italic")

        elif is_structural_marker:
            # NEW (Microsoft 2026, confirmed via real video-frame
            # output): Microsoft renders its numbered Note titles
            # ("NOTE 1 -- ACCOUNTING POLICIES", "NOTE 9 -- INTANGIBLE
            # ASSETS", etc.) UNDERLINED, in REGULAR (non-bold) weight
            # -- a styling convention never seen in the first 15
            # verified companies, all of which bold their Note
            # titles. The same is true of Microsoft's small "PART
            # II" / running-header captions repeated at the top of
            # every page.
            #
            # PyMuPDF's span flags (BOLD_FLAG/ITALIC_FLAG) do not
            # expose underline decoration at all, so this can't be
            # detected via styling -- but the TEXT itself already,
            # independently, matches a known, universal SEC-filing
            # structural-marker pattern (is_note_marker /
            # is_top_level_marker), which is a completely reliable
            # signal on its own regardless of the font used to render
            # it.
            #
            # Confirmed real-world impact: without this, "NOTE 1 --
            # ACCOUNTING POLICIES" (not bold) scored only 4 points
            # (body_size_but_styled +1, short +2, all_caps +1) --
            # exactly one point under the heading_score_threshold of
            # 5 -- so it never became a heading at all, and its text
            # silently merged into the middle of the PRECEDING
            # paragraph ("Refer to accompanying notes. NOTE 1 --
            # ACCOUNTING POLICIES Our consolidated financial
            # statements..."). Every one of Note 1's real sub-topics
            # then flattened out as siblings of "NOTES TO FINANCIAL
            # STATEMENTS" instead of nesting under Note 1. The same
            # scoring gap affected the "PART II"/"Item 8" running
            # header, letting it leak into TABLE candidacy instead of
            # being excluded as a heading (see the Note 3 fix in
            # _is_top_level_marker for that side of this bug).
            #
            # Granting the SAME style-tier credit bold text gets (not
            # italic's lesser +2) -- but ONLY when the text
            # independently matches a known structural-marker pattern
            # -- fixes this without touching how any non-structural-
            # marker text is scored. The full +3 (matching "bold",
            # not "italic"'s +2) is needed, not just +2: Microsoft's
            # bare "Item 8" running-header caption is mixed-case (no
            # all_caps +1 bonus available, unlike "PART II" or
            # "NOTE 1 -- ..."), so with only +2 here it still lands
            # one point under threshold (structural +2, size-neutral
            # +0, short +2 = 4). +3 closes that gap (3+0+2=5) without
            # over-crediting any of the other, already-passing cases.
            score += 3
            reasons.append("structural_marker_unstyled")

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

        elif is_structural_marker:
            # NEW (Palo Alto Networks 2025, confirmed via real
            # cleaned.json output): a line whose TEXT independently
            # matches a known, SEC-mandated structural-marker pattern
            # (a core financial-statement title like "CONSOLIDATED
            # STATEMENTS OF OPERATIONS", a "Note N" marker, an "Item
            # N."/"PART N" marker, "Report of Independent Registered
            # Public Accounting Firm", the bare "NOTES TO CONSOLIDATED
            # FINANCIAL STATEMENTS" divider, etc.) is DEFINITELY meant
            # to be a heading by its content alone, regardless of how
            # small its specific font happens to render relative to
            # THIS document's own body-text baseline.
            #
            # Confirmed real-world impact: PANW's 4 core financial
            # statement titles are all bold, ALL-CAPS, and
            # unambiguously match the CONSOLIDATED-title is_note_marker
            # pattern -- yet render at size 7.99, SMALLER than this
            # filing's own body-paragraph baseline of 9.0
            # (relative_size=0.888). The unconditional -2
            # "smaller_than_body" penalty here was dragging their
            # total score (bold +3, smaller -2, short +2, all_caps +1
            # = 4) just under the heading_score_threshold of 5 -- so
            # NONE of "CONSOLIDATED STATEMENTS OF OPERATIONS",
            # "...COMPREHENSIVE INCOME", "...STOCKHOLDERS' EQUITY", or
            # "...CASH FLOWS" ever became a heading at all. Every one
            # of their tables then silently attached to whichever
            # heading was still open from earlier in the document
            # (here, "Liabilities and stockholders' equity" -- an
            # internal Balance-Sheet sub-item), completely losing
            # their real section identity for retrieval even though
            # the underlying numbers were still correct.
            #
            # Waiving the penalty (staying neutral, +0) ONLY when the
            # text independently matches a known structural-marker
            # pattern keeps this narrowly scoped and safe: ordinary
            # small text that ISN'T one of these mandated patterns
            # (footnote markers, stray small-font captions, page
            # numbers) is still penalized exactly as before.
            reasons.append("smaller_than_body_but_structural_marker")

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

        # NEW: "Item N.", "PART N", and "Note N" are SEC-mandated
        # structural markers -- they must NEVER be scored down just
        # because their descriptive title portion happens to be long.
        # Confirmed on Apple 2024: "Item 7. Management's Discussion
        # and Analysis of Financial Condition and Results of
        # Operations" (13 words) fell into the neutral "medium_length"
        # bucket (no bonus, since it's a TITLE and correctly doesn't
        # end in a period, so it also didn't qualify for the
        # bold-sentence bonus either) -- scoring only 4, just under
        # the heading threshold of 5. Meanwhile short titles like
        # "Item 6. [Reserved]" or "Item 7A." passed easily. This
        # silently dropped "Item 7." as a heading entirely, which
        # then made EVERY one of its real sub-sections (Segment
        # Operating Performance, Operating Expenses, Provision for
        # Income Taxes, etc.) incorrectly flatten under "Item 6.
        # [Reserved]" instead -- a systemic, high-impact bug, since
        # Item 7 (MD&A) is one of the most commonly retrieved
        # sections in any 10-K.
        #
        # (is_structural_marker itself is now computed earlier, up in
        # the size-scoring section above, so the exact same value is
        # simply reused here.)

        if word_count <= self.max_heading_words or is_structural_marker:
            score += 2
            reasons.append(
                "short"
                if word_count <= self.max_heading_words
                else "structural_marker"
            )
        elif is_bold and text.strip().endswith(".") and word_count <= self.max_heading_words * 4:
            # NEW: the bonus above only applies when the sentence is
            # actually COMPLETE (ends in a period) -- a bold line
            # that's still mid-sentence (e.g. ends in a comma because
            # it wraps onto the next physical PDF line, like "Global
            # markets for the Company's products and services are
            # highly competitive and subject to rapid technological
            # change,") must NOT get this bonus. Without the period
            # check, that incomplete first fragment could score high
            # enough to become a heading BY ITSELF, which blocks the
            # bold-run-continuation-tracking below from ever opening
            # (it only opens when the line does NOT already qualify
            # as its own heading) -- so the sentence's second half
            # ("and the Company may be unable to compete effectively
            # in these markets.") would then ALSO score as its own
            # separate heading, splitting one sentence into two false
            # headings instead of one correct one.
            #
            # The length cap is wider here (4x, not 2x) than the
            # ordinary "short" bucket above: this branch only ever
            # fires for a COMPLETE bold sentence, and once bold-run
            # merging reconstructs a full multi-line risk-factor
            # header, its combined length can genuinely run to
            # 30-40+ words (confirmed on Apple 2016's Item 1A, e.g.
            # "There may be breaches of the Company's information
            # technology systems that materially damage business
            # partner and customer relationships..." -- 41 words).
            # Real 10-K risk-factor topic sentences don't run much
            # beyond that, so this still guards against a genuinely
            # mistaken full BODY PARAGRAPH picking up stray bold
            # styling.
            score += 2
            reasons.append("short_bold_sentence")
        elif word_count <= self.max_heading_words * 2:
            score += 0  # neutral, borderline
            reasons.append("medium_length")
        else:
            score -= 3
            reasons.append("too_long")

        # ---------------------------------------------------
        # Sentence-ending punctuation (headings rarely end in '.')
        #
        # FIX: this penalty used to apply unconditionally, which
        # worked against a very common, legitimate SEC 10-K
        # convention -- Risk Factors items, where EVERY item's
        # heading is one complete BOLD sentence ending in a period
        # (e.g. "The Company depends on the performance of
        # distributors, carriers and other resellers."). Confirmed
        # on Apple 2016 page 12: this exact line scored 4 (just under
        # the threshold of 5) purely because of this -2 penalty, and
        # was rejected as a heading even though it's a genuine,
        # correctly-formatted risk-factor header.
        #
        # A bold, complete, reasonably short sentence ending in a
        # period is far more likely to be this SEC heading
        # convention than an accidental stray bold sentence sitting
        # mid-paragraph (bold in 10-Ks is reserved for structural
        # elements -- titles, table headers, risk-factor headers --
        # not incidental emphasis). So we only apply this penalty to
        # NON-bold lines; a bold sentence never loses points just for
        # ending in a period.
        # ---------------------------------------------------

        if text.endswith(".") and word_count > 3 and not is_bold:
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

    def _estimate_level(self, text, relative_size, is_bold, is_italic):
        """
        Level 1 = document/cover title (huge font, e.g. "Apple Inc.")
        Level 2 = top-level outline markers ("PART I", "Item 1.",
                   "Item 7A.") -- SEC 10-Ks are consistently numbered
                   this way, so matching that pattern is a much more
                   reliable depth signal than font size alone.
        Level 3 = generic bold subsection heading (e.g. "Products and
                   Services Performance", "CONSOLIDATED BALANCE SHEETS")
        Level 4 = italic sub-heading (e.g. "iPhone", "Mac" inside MD&A)

        Without the PART/Item pattern check, nearly every bold heading
        scored as "level 2" regardless of its real depth, which
        flattened the whole document tree (top-level Items and their
        subsections all became siblings instead of parent/child).
        """

        if relative_size >= 1.5:
            return 1

        if self._is_top_level_marker(text):
            return 2

        if is_italic and relative_size < 1.15:
            return 4

        # NEW (Microsoft 2026, confirmed via real hierarchy_outline.txt
        # output): a known structural marker (is_note_marker /
        # is_top_level_marker) must get the SAME level-3 treatment a
        # bold heading gets, even when it isn't actually bold or
        # italic. Microsoft underlines its "NOTE N -- Title" headings
        # instead of bolding them (see the "structural_marker_unstyled"
        # scoring fix), so without this, is_bold is False for every
        # Note title and _estimate_level() fell all the way through to
        # the final "return 4" default -- the SAME level as (or deeper
        # than) that Note's own bold sub-topics ("Accounting
        # Principles", "Principles of Consolidation", level 3).
        #
        # Confirmed real-world impact: because a Note title (level 4)
        # was NOT shallower than its own sub-topics (level 3), the
        # hierarchy_builder stack-popping loop (`stack[-1]["level"] >=
        # level`) never found a reason to pop back up to sibling depth
        # when the NEXT Note arrived -- 3 >= 4 is false, so NOTE 2
        # attached as a CHILD of whatever sub-topic NOTE 1 had left on
        # top of the stack, NOTE 3 nested even deeper under NOTE 2's
        # own last sub-topic, and so on -- 18 Notes cascading into one
        # continuously deepening chain instead of 18 siblings under
        # "NOTES TO FINANCIAL STATEMENTS".
        #
        # Granting level 3 here (the same depth a bold Note title
        # already gets in every other verified company) fixes this:
        # a Note is now guaranteed to be shallower than or equal to
        # its own sub-topics, so the existing is_note_marker
        # exclusivity exception in hierarchy_builder (which already
        # correctly keeps a Note open against same-level, non-marker
        # sub-topics, but lets a same-level INCOMING Note marker pop
        # it) can do its job correctly.
        is_structural_marker = (
            self._is_top_level_marker(text) or self._is_note_marker(text)
        )

        if is_bold or is_structural_marker:

            # NEW: a bold heading that's a COMPLETE SENTENCE (ends in
            # a period, reasonably long) is far more likely to be a
            # specific topic-sentence sub-item -- e.g. individual SEC
            # Risk-Factor headers like "The Company depends on the
            # performance of distributors, carriers and other
            # resellers." -- than a section-umbrella title.
            #
            # Genuine section titles ("Risk Factors", "Segment
            # Operating Performance", "CONSOLIDATED BALANCE SHEETS")
            # are consistently short NOUN PHRASES that never end in a
            # period. Demoting only the sentence-shaped headings to
            # level 4 lets them nest as CHILDREN of their section's
            # real title instead of becoming false siblings of it.
            #
            # Confirmed on Apple 2016's Item 1A "Risk Factors": before
            # this, every individual risk-factor sentence was
            # flattened to the SAME level as "Risk Factors" itself,
            # turning "Risk Factors" into an empty pass-through node
            # instead of the parent of all the actual risk items.
            if text.strip().endswith(".") and len(text.split()) > 3:
                return 4

            return 3

        return 4

    # =========================================================
    # TOP-LEVEL OUTLINE MARKER (PART / ITEM)
    # =========================================================

    def _is_top_level_marker(self, text):
        """
        Matches SEC 10-K style top-level outline headings:
          "PART I", "PART II"
          "Item 1.", "Item 1A.", "Item 7. Management's Discussion..."
        This pattern is essentially universal across SEC filings
        (it's mandated by the SEC's filing format), so it generalizes
        across companies -- unlike matching specific section titles.
        """

        stripped = text.strip()

        if re.match(r"^PART\s+[IVXLCDM]+\b", stripped, re.IGNORECASE):
            return True

        if re.match(r"^Item\s+\d+[A-Za-z]?\.", stripped, re.IGNORECASE):
            return True

        # NEW (Intuit 2026, confirmed via real hierarchy_outline.txt
        # output): some companies punctuate their top-level Item
        # markers with a DASH instead of a period -- "ITEM 7 -
        # MANAGEMENT'S DISCUSSION AND ANALYSIS...", "ITEM 1A - RISK
        # FACTORS" -- rather than the period-punctuated "Item 7."
        # convention every other verified company uses. The existing
        # pattern above requires a literal period immediately after
        # the item number/letter, so NONE of Intuit's own Item
        # markers (every single one in the filing, Item 1 through
        # Item 15) ever matched it at all.
        #
        # Confirmed real-world impact: every Item boundary in this
        # filing scored as a generic Level-3 heading instead of the
        # correct Level-2 top-level marker, and lost the
        # is_top_level_marker=True protection that lets a genuine
        # Item/Part boundary force-close an already-open Note or
        # core-statement container (the SAME exclusivity mechanism
        # already relied on for every other verified company) --
        # this happened to not visibly corrupt Intuit's own Notes
        # nesting only because no Note/statement was still open at
        # each Item boundary here, but the underlying protection was
        # silently missing regardless, and could misfire on a
        # differently-laid-out filing using this same convention.
        #
        # Matching a dash (plain hyphen, en-dash, or em-dash) here
        # mirrors is_note_marker()'s own established handling of
        # exactly this kind of punctuation variance for "Note N".
        if re.match(r"^Item\s+\d+[A-Za-z]?\s*[-\u2013\u2014]", stripped, re.IGNORECASE):
            return True

        # NEW (Microsoft 2026, confirmed via real video-frame output):
        # every content page of Item 8 repeats a small, two-line
        # running header at the very top -- "PART II" (matches the
        # PART pattern above) immediately followed by a BARE "Item 8"
        # on its own physical line, with NO trailing period. This is
        # a plain running-header CAPTION, not the actual "Item 8.
        # Financial Statements and Supplementary Data" section title
        # (which appears once, elsewhere, WITH its period and full
        # title text, and already matches the pattern above just
        # fine).
        #
        # The existing "Item N." pattern requires a period, so this
        # bare caption never matched it -- and since it also isn't
        # bold in this filing, it scored too low to become a heading
        # on style/size grounds alone either (see the new
        # "structural_marker_unstyled" branch in _score_line, which
        # this fix works together with). Left undetected, it was
        # free to leak into TABLE candidacy on every single page of
        # Item 8, corrupting table bboxes and column names --
        # confirmed real damage: "PART II Item 8" became a literal
        # bogus COLUMN NAME in at least two different Note tables, a
        # bogus "section_title" on another, and -- most severely --
        # dragged the entire 38-row Balance Sheet table's bbox up
        # far enough that it mis-attached to the PRECEDING heading
        # ("COMPREHENSIVE INCOME STATEMENTS") instead of "BALANCE
        # SHEETS".
        #
        # Matching is deliberately narrow: the ENTIRE line (not just
        # its start) must be nothing but "Item N[Letter]" with an
        # OPTIONAL trailing period -- this can never accidentally
        # match a real sentence that happens to START with "Item 8"
        # followed by more words, since re.fullmatch requires the
        # whole line to be just this short caption.
        if re.fullmatch(r"Item\s+\d+[A-Za-z]?\.?", stripped, re.IGNORECASE):
            return True

        return False

    def _is_note_marker(self, text):
        """
        Matches "Note N - Title" / "Note N – Title" / "Note N. Title"
        -- different companies punctuate their numbered notes
        differently (Apple uses a hyphen/en-dash: "Note 1 - Summary
        of Significant Accounting Policies"; Google/Alphabet uses a
        period: "Note 1. Summary of Significant Accounting Policies",
        "Note 10. Commitments and Contingencies"). Like Item/Part,
        this numbering convention is universal across virtually every
        company's "Notes to Financial Statements" -- so recognizing
        it generalizes the same way _is_top_level_marker() does, as
        long as the punctuation variants are all covered.

        This does NOT change the heading's own "level" number (it
        still scores as a normal bold Level-3 heading, same as its
        own sub-topics) -- it's used separately by hierarchy_builder
        to decide when a Note should be closed out. See
        hierarchy_builder.py for the full rationale: a "Note N"
        heading and its own sub-topic headings ("Basis of
        Presentation", "Cash Equivalents", etc.) are styled
        IDENTICALLY in the PDF, so heading_detector has no way to
        tell "this is a container" from "this is one of its topics"
        by styling alone -- confirmed on Apple 2024, where 7 of 13
        numbered notes had their own sub-topics flattening out as
        SIBLINGS of the Note instead of nesting under it, because a
        same-level heading was closing the Note out prematurely.
        """

        stripped = text.strip()

        # Matches "Note N - Title" (Apple, hyphen), "Note N. Title"
        # (Google/Nvidia/Meta, period), and "Note N : Title" (Intel,
        # colon -- sometimes with a space before the colon, "Note 1
        # :"). Confirmed on Intel 2025: this colon variant caused
        # Notes 1-9 specifically to have their number-marker line
        # ("Note 3 :") physically separated (as its own PDF line)
        # from their real title line ("Operating Segments") -- unlike
        # Notes 10-19, where both fit on one combined physical line.
        # Recognizing "Note N :" as a note-marker on its own is
        # enough to fix this correctly even when split: the existing
        # note-marker exclusivity rule in hierarchy_builder means the
        # marker line stays open (isn't closed by a same-level
        # generic heading), so the very next heading ("Operating
        # Segments") correctly nests as ITS CHILD rather than
        # becoming an unrelated sibling -- not a single combined
        # title, but the content ends up correctly grouped under the
        # right Note either way.
        #
        # NEW (Microsoft 2026, confirmed via real video-frame output):
        # added re.IGNORECASE. Microsoft renders this marker
        # completely in ALL CAPS -- "NOTE 1 -- ACCOUNTING POLICIES",
        # using an em-dash separator -- which the character class
        # here already covered, but the literal word "Note" in the
        # pattern was being matched case-SENSITIVELY, so "NOTE" (all
        # caps) never matched at all. Confirmed real-world impact:
        # EVERY numbered Note in this filing (1, 6, 7, 8, 9, 10, 11,
        # 12, 13, 14, 15, 17, 18) failed is_note_marker, which also
        # meant none of them qualified for the "smaller_than_body_
        # but_structural_marker" size-penalty waiver or the new
        # "structural_marker_unstyled" style credit below -- see
        # those two branches in _score_line for how this combined
        # with Microsoft's underlined (non-bold) Note-title styling
        # to keep every single Note title's score one point under
        # threshold, so none of them became headings at all.
        if re.match(r"^Note\s+\d+\s*[-.:\u2013\u2014]", stripped, re.IGNORECASE):
            return True

        # "Report of Independent Registered Public Accounting Firm" is
        # standard, PCAOB-mandated wording that appears verbatim in
        # every 10-K's auditor opinion section. It behaves exactly
        # like a Note for nesting purposes -- it must NEVER become a
        # "container" that swallows unrelated sections as its
        # children, regardless of whether it happens to appear BEFORE
        # or AFTER the numbered Notes in a given company's layout.
        #
        # This was ORIGINALLY classified as a top-level (Item/PART-
        # style, level 2) marker instead, which worked correctly for
        # Apple (where the audit report appears AFTER the last
        # numbered Note: the shallower level-2 marker correctly
        # closed out the deeper level-3 Note). But confirmed on
        # Google/Alphabet 2025, whose audit report appears BEFORE the
        # Notes section: being level-2 (shallower) meant it never got
        # closed out by the level-3 Notes that followed -- instead it
        # incorrectly became their PARENT CONTAINER, since a shallower
        # heading always opens a container for any deeper heading
        # that follows. "Note 12. Net Income Per Share" ended up
        # nested under "Report of Independent Registered Public
        # Accounting Firm" instead of being its sibling.
        #
        # Reclassifying it as a Note-style marker (level 3, same
        # exclusivity rules as Notes -- only closed by ANOTHER Note-
        # style marker or a genuine Item/PART boundary, never by its
        # own subsections) makes it symmetric: it correctly stays a
        # sibling of the Notes regardless of which side of them it
        # appears on.
        if re.match(
            r"^Reports?\s+of\s+Independent\s+Registered\s+Public\s+Accounting\s+Firm",
            stripped,
            re.IGNORECASE,
        ):
            return True

        # "CONSOLIDATED BALANCE SHEETS", "CONSOLIDATED STATEMENTS OF
        # INCOME/OPERATIONS/CASH FLOWS/COMPREHENSIVE INCOME/
        # STOCKHOLDERS' EQUITY" -- the 4-5 core financial statement
        # titles -- are ALSO universal, ALL-CAPS SEC-filing boundary
        # markers, same as Notes and the audit report. Without
        # treating them the same way, they're just plain generic
        # headings, so they can get swallowed as CHILDREN of whatever
        # note-style container happens to still be open before them.
        # Confirmed on Google/Alphabet 2025: the audit opinion on
        # Internal Control ("Report of Independent...") sits directly
        # before "CONSOLIDATED BALANCE SHEETS" in the document, and
        # since the statement title wasn't recognized as its own
        # boundary marker, it (and by extension the Income Statement,
        # Comprehensive Income, Stockholders' Equity, and Cash Flow
        # statements that follow it on the same page-run) all nested
        # as children of that audit-opinion section instead of being
        # its siblings.
        # NEW (PayPal 2025, confirmed via real cleaned.json +
        # heading_detection JSON output): this regex has no END
        # anchor, so it only checks that the text STARTS WITH
        # "CONSOLIDATED BALANCE SHEETS" or "CONSOLIDATED STATEMENTS
        # OF" -- it was never a problem for the first 17 verified
        # companies because none of them happened to have a random
        # NARRATIVE SENTENCE that starts with those exact words. But
        # PayPal's Cash Flow statement has a genuine, wrapped,
        # non-bold 3-line sentence -- "The table below reconciles
        # cash, cash equivalents, and restricted cash as reported in
        # the / consolidated balance sheets to the total of the same
        # amounts shown in the consolidated / statements of cash
        # flows:" -- whose SECOND physical line happens to begin with
        # "consolidated balance sheets to the total...", which this
        # regex matched purely from its opening words, ignoring the
        # 14-word narrative tail that follows.
        #
        # Confirmed real-world impact: with is_note_marker() firing,
        # is_structural_marker became True for this ordinary sentence
        # fragment, which (via the "structural_marker_unstyled" and
        # "smaller_than_body_but_structural_marker" credits added for
        # Microsoft's underlined Note titles) pushed its score to
        # exactly the heading_score_threshold of 5 -- turning a plain
        # narrative sentence fragment into its own spurious heading.
        #
        # A genuine core-statement title is always SHORT (the longest
        # real variant, "CONSOLIDATED STATEMENTS OF COMPREHENSIVE
        # INCOME", is 5 words) -- so requiring the overall line to be
        # no longer than 8 words (a safety margin above that) before
        # even trying this specific regex keeps it working exactly as
        # before for every genuine statement title, while refusing to
        # match a long sentence that merely happens to start with the
        # same opening words.
        # NEW (PayPal 2016, confirmed via real chunks.json output):
        # tightened further after finding a SHORT, lowercase,
        # colon-ending fragment -- "consolidated statement of
        # income:" -- that still slipped through the word-count cap
        # above (it's only 4 words, well under the cap of 8) because
        # the regex allowed the SINGULAR "STATEMENT" (via the "?" on
        # "STATEMENTS?") and didn't care about trailing punctuation.
        #
        # Confirmed real-world impact: this fragment is a genuine
        # PayPal table/reconciliation caption sitting INSIDE Note 1's
        # own text (introducing an "As Reported / Adjustments /
        # Revised" recast table), not a real statement title -- but
        # because it matched is_note_marker(), hierarchy_builder's
        # same-Note-marker-pops-Note-marker rule closed out the
        # currently-open "NOTE 1-Overview and Summary of Significant
        # Accounting Policies" the instant this fragment arrived,
        # and every one of Note 1's real remaining sub-topics (Use of
        # estimates, Cash and cash equivalents, Investments, Loans
        # and interest receivable, Customer accounts, Property and
        # equipment, Goodwill, Revenue recognition, Income taxes,
        # Net income per share, Recent Accounting Pronouncements, and
        # more -- essentially ALL of Note 1's real content) ended up
        # nested under this meaningless fragment instead of under
        # Note 1 itself.
        #
        # Two independent, safe signals distinguish a genuine
        # statement title from this kind of caption: (1) every real
        # SEC statement title uses the PLURAL "STATEMENTS OF ..."
        # (never singular "STATEMENT OF") -- confirmed true across
        # every verified company so far; (2) a real title is never
        # followed by a colon -- that punctuation always marks an
        # introductory reference ("...as reported in the consolidated
        # statement of income:") or table caption, never the title
        # itself. Requiring the plural form AND excluding colon-
        # terminated text keeps this safe without narrowing the
        # word-count cap further (which could risk excluding a
        # genuine longer title phrase instead).
        if (
            len(stripped.split()) <= 8
            and not stripped.endswith(":")
            and re.match(
                r"^CONSOLIDATED\s+(BALANCE\s+SHEETS?|STATEMENTS\s+OF\s+)",
                stripped,
                re.IGNORECASE,
            )
        ):
            return True

        # NEW (Microsoft 2026, confirmed via real video-frame output):
        # Microsoft titles its 5 core financial statements WITHOUT
        # the word "CONSOLIDATED" and without "STATEMENTS OF" -- just
        # "INCOME STATEMENTS", "COMPREHENSIVE INCOME STATEMENTS",
        # "BALANCE SHEETS", "CASH FLOWS STATEMENTS", "STOCKHOLDERS'
        # EQUITY STATEMENTS". None of these match the CONSOLIDATED-
        # prefixed pattern immediately above, so none of them were
        # recognized as note-markers.
        #
        # Confirmed real-world impact: without note-marker status,
        # these titles have no protection from hierarchy_builder's
        # same-level-sibling exclusivity rule, so their own internal
        # sub-items -- "Assets" / "Liabilities and stockholders'
        # equity" under Balance Sheets; "Operations" / "Financing" /
        # "Investing" under Cash Flows; "Common stock and paid-in
        # capital" / "Retained earnings" / "Accumulated other
        # comprehensive loss" under Stockholders' Equity -- all
        # popped their parent statement off the stack and became its
        # SIBLINGS instead of its children the moment they arrived.
        #
        # Matching requires the ENTIRE line (case-insensitive,
        # optional leading "CONSOLIDATED") to be exactly one of these
        # 5 known statement-title phrases, so this cannot misfire on
        # unrelated text that merely mentions "balance sheet" or
        # "cash flow" in passing -- it only recognizes the standalone
        # title line itself.
        if re.match(
            r"^(CONSOLIDATED\s+)?"
            r"(BALANCE\s+SHEETS?|INCOME\s+STATEMENTS?|"
            r"COMPREHENSIVE\s+INCOME\s+STATEMENTS?|"
            r"CASH\s+FLOWS?\s+STATEMENTS?|"
            r"STOCKHOLDERS'?\s+EQUITY\s+STATEMENTS?)\s*$",
            stripped,
            re.IGNORECASE,
        ):
            return True

        # NEW (Adobe 2025, confirmed via real hierarchy_outline.txt +
        # chunks.json output): the bare divider heading "NOTES TO
        # CONSOLIDATED FINANCIAL STATEMENTS" -- the section title
        # that introduces the whole Notes section, sitting directly
        # after the last core financial statement (Cash Flows) and
        # directly before "NOTE 1." -- carries NO "Note N" number
        # itself, so none of the regexes above ever match it. It also
        # isn't a PART/Item marker.
        #
        # Without recognizing it, it fails every check in the
        # exclusivity rule (not note_marker, not top_level_marker,
        # not prominent_boundary), so the previously-open note-marker
        # container right before it -- "CONSOLIDATED STATEMENTS OF
        # CASH FLOWS" -- never gets closed. Confirmed on real output:
        # this single miss caused ALL of Notes 1 through 18 (the
        # entire Notes section of the filing) to nest as descendants
        # of "CONSOLIDATED STATEMENTS OF CASH FLOWS" instead of being
        # its siblings, e.g. real section_path values coming out as
        # "...CASH FLOWS > NOTE 8. GOODWILL AND OTHER INTANGIBLES"
        # instead of "...NOTES TO CONSOLIDATED FINANCIAL STATEMENTS >
        # NOTE 8. GOODWILL AND OTHER INTANGIBLES".
        #
        # The repeated "(Continued)" variant that appears at the top
        # of every subsequent Notes page is already correctly removed
        # by the boilerplate-repetition fix elsewhere in the pipeline
        # (confirmed: 0 chunks contain "Continued" in their
        # section_path), so it never reaches heading_detector as a
        # real candidate -- but the optional suffix is still matched
        # here defensively in case a filing's boilerplate threshold
        # doesn't happen to catch it (e.g. a very short filing where
        # the page-repetition count falls under the boilerplate
        # floor).
        if re.match(
            r"^NOTES?\s+TO\s+(CONSOLIDATED\s+)?FINANCIAL\s+STATEMENTS"
            r"\s*(\(Continued\))?\s*$",
            stripped,
            re.IGNORECASE,
        ):
            return True

        # NEW (Chipotle 2025, confirmed via real hierarchy_outline.txt
        # + chunks.json output): some companies number their Notes
        # WITHOUT the word "Note" on the actual heading line at all --
        # just a bare "N. Title", e.g. "1. Description of Business and
        # Summary of Significant Accounting Policies", "3. Revenue
        # Recognition", "9. Leases", "14. Segment Reporting". Chipotle
        # still says "Note 14" in its own inline cross-references
        # elsewhere in the document ('...included in Note 14.
        # "Segment Reporting"'), but the real section-starting heading
        # line itself never carries that word -- so none of the
        # "Note N ..." regex variants above can ever match it, no
        # matter which punctuation/case variant they check for.
        #
        # This is the SAME underlying failure mode already documented
        # for Intel 2019 (a Note heading with no recognizable "Note N"
        # text on the line itself) -- but there it was solved by
        # is_prominent_boundary, because Intel's real boundaries were
        # ALSO distinctly oversized/all-caps. Chipotle's Note headings
        # are NOT distinctly larger or all-caps -- they're styled
        # IDENTICALLY (same bold weight, same size, same non-italic)
        # to their own topic sub-headings ("Cash and Cash
        # Equivalents", "Fair Value Measurements", etc.), so
        # is_prominent_boundary can't distinguish them either.
        #
        # Confirmed real-world impact: every one of Note 1's 15
        # sub-topics (Principles of Consolidation, Management
        # Estimates, Cash and Cash Equivalents, ... Income Taxes)
        # flattened out as SIBLINGS of "1. Description of Business..."
        # -- and of "NOTES TO CONSOLIDATED FINANCIAL STATEMENTS"
        # itself -- instead of nesting under it, e.g. real
        # section_path values coming out as "...NOTES TO CONSOLIDATED
        # FINANCIAL STATEMENTS > Cash and Cash Equivalents" with no
        # trace that this was ever part of Note 1's accounting-policy
        # discussion. This is exactly the original Apple-2024 bug
        # (same-level sub-topic prematurely closing its own Note),
        # just resurfacing through a different root cause (a missing
        # "Note" word instead of a missing punctuation variant).
        #
        # Matching is deliberately narrow to minimize any risk of
        # false-triggering on unrelated bold numbered text elsewhere
        # in a filing: the number must be 1-2 digits (Notes are never
        # numbered in the hundreds), immediately followed by a literal
        # period and at least one space, then a real multi-letter
        # capitalized word (excludes bare fragments like "1. A" or
        # "1. I", and excludes anything already covered above by
        # "Item N." / "PART N" / "Note N", which all require their own
        # distinct leading word and are matched separately before this
        # point is ever reached).
        # NEW (Intuit 2017, confirmed via real hierarchy_outline.txt +
        # chunks.json output): this bare "N. Title" pattern had no
        # word-count limit at all, so it could ALSO match the START
        # of a long, unrelated NARRATIVE SENTENCE that happens to
        # begin with a digit, period, and capitalized word purely by
        # coincidence -- e.g. "7. Revenue in our Small Business
        # segment increased 9% due to 25% growth in Small Business
        # Online Ecosystem revenue and the impact of the..." (24
        # words) matched this exactly, since re.match only anchors
        # the START of the text, never requiring the whole line to
        # be short.
        #
        # Confirmed real-world impact: with is_note_marker() firing
        # on this sentence fragment, it independently picked up BOTH
        # the "structural_marker_unstyled" style credit AND the
        # length-cap BYPASS (the "word_count <= max_heading_words or
        # is_structural_marker" bonus) that this codebase reserves
        # for genuine short structural markers -- turning an ordinary
        # 24-word MD&A sentence into its own spurious heading, and
        # (per the same is_note_marker exclusivity mechanism) closing
        # out or nesting content around it incorrectly.
        #
        # A genuine bare-numbered Note title is always short -- even
        # Chipotle's longest real example, "1. Description of
        # Business and Summary of Significant Accounting Policies",
        # is only 10 words -- so capping this pattern at 12 words
        # (a small safety margin above that) keeps every genuine
        # verified Note title matching while rejecting narrative
        # sentences that merely happen to start the same way.
        if (
            len(stripped.split()) <= 12
            and re.match(r"^\d{1,2}\.\s+[A-Z][a-z]+", stripped)
        ):
            return True

        # NEW (ServiceNow 2025, confirmed via real hierarchy_outline.txt
        # output): a THIRD distinct Note-numbering convention -- the
        # number wrapped in PARENTHESES, with no "Note" word and no
        # trailing period at all, e.g. "(1) Description of the
        # Business", "(2) Summary of Significant Accounting Policies",
        # "(14) Stockholders' Equity". All 19 of ServiceNow's numbered
        # Notes use this exact style consistently.
        #
        # This is the SAME underlying failure mode already fixed twice
        # before (Chipotle's bare "N." and, before that, Adobe's
        # missing-"Note"-word divider) -- a Note heading whose own text
        # carries no recognizable marker for any of the regexes above,
        # so is_note_marker() could never protect it from being closed
        # by its own same-level, identically-styled sub-topics.
        # Confirmed real-world impact: EVERY one of "(2) Summary of
        # Significant Accounting Policies"'s sub-topics (Principles of
        # Consolidation, Common Stock Split, Use of Estimates, Foreign
        # Currency Translation and Transactions, Revenue Recognition,
        # and more) flattened out as siblings of the Note instead of
        # nesting under it -- real hierarchy_outline.txt showed all of
        # them at the identical indentation level as "(2) Summary..."
        # itself.
        #
        # Matching is scoped the same way as the bare "N." fix above:
        # the number must be 1-2 digits immediately inside the
        # parentheses, followed by a space and a real multi-letter
        # capitalized word -- so a bare, isolated footnote-reference
        # marker like "(1)" on its own (with nothing else on that
        # heading candidate's text) never matches, since there's no
        # trailing word for "\s+[A-Z][a-z]+" to find.
        if (
            len(stripped.split()) <= 12
            and re.match(r"^\(\d{1,2}\)\s+[A-Z][a-z]+", stripped)
        ):
            return True

        return False



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