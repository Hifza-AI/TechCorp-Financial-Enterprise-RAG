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

        for index, line in enumerate(lines):

            analysis = self._analyze_line(
                line,
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
                if (
                    is_bold
                    and not analysis["is_heading"]
                    and word_count > self.max_heading_words
                    and not text.endswith(".")
                ):
                    bold_run_open = True
                    bold_run_buffer = [(index, analysis, text)]

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
        if re.fullmatch(
            r"\(\s*(dollars\s+|amounts\s+)?in\s+(millions|thousands|billions)"
            r"(\s*,\s*[^)]*)?\)",
            text.strip(),
            re.IGNORECASE,
        ):
            return 0, ["units_disclaimer_caption"]

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
        is_structural_marker = (
            self._is_top_level_marker(text) or self._is_note_marker(text)
        )

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

        if is_bold:

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

        if re.match(r"^Note\s+\d+\s*[-.\u2013\u2014]", stripped):
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
        if re.match(
            r"^CONSOLIDATED\s+(BALANCE\s+SHEETS?|STATEMENTS?\s+OF\s+)",
            stripped,
            re.IGNORECASE,
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