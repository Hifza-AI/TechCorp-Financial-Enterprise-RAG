import json
import re
from pathlib import Path
print(">>> RUNNING FIXED-TABLE-PARSER-V4-SEGMENT-TITLE-FIX <<<")

class TableParser:

    # Standalone symbols that should NEVER be treated as part of a label.
    # These commonly sit just to the left of the first data column
    # (e.g. "$" before a number) and were leaking into labels before.
    NON_LABEL_TOKENS = {"$", "-", "--", "—", "%", "(", ")"}

    def __init__(
        self,
        y_tolerance=6.0,
        # x_tolerance was 15.0 -- but real observed data (Apple 2016's
        # stock-performance table) showed a CONSISTENT header-to-data
        # x-offset of ~16.1-16.3 across every column, just outside
        # the old tolerance. That caused EVERY value in that table to
        # come back null (no column matched). Bumped to 20.0 to give
        # a safety margin above the largest observed real offset,
        # without being so loose it starts matching the wrong
        # adjacent column on tightly-packed tables.
        x_tolerance=20.0,
    ):
        self.y_tolerance = y_tolerance
        self.x_tolerance = x_tolerance

    # =========================================================
    # MAIN
    # =========================================================

    def parse_report(self, report):

        parsed_report = {
            "company": report.get("company"),
            "file_name": report.get("file_name"),
            "file_path": report.get("file_path"),
            "total_pages": report.get("total_pages"),
            "tables": [],
        }

        for page in report.get("pages", []):

            page_tables = self._parse_page(page)

            parsed_report["tables"].extend(page_tables)

        return parsed_report

    # =========================================================
    # PAGE
    # =========================================================

    def _parse_page(self, page):

        table_analysis = page.get("table_analysis", {})

        candidates = table_analysis.get("candidate_lines", [])

        # Only lines marked as table candidates
        candidate_lines = [
            item for item in candidates
            if item.get("is_candidate") is True
        ]

        if not candidate_lines:
            return []

        # Get actual cleaned lines
        actual_lines = page.get("lines", [])

        selected_lines = []

        for candidate in candidate_lines:

            index = candidate.get("line_index")

            if index is None:
                continue

            if index < 0:
                continue

            if index >= len(actual_lines):
                continue

            selected_lines.append(actual_lines[index])

        if not selected_lines:
            return []

        # -----------------------------------------------------
        # Group lines into visual rows
        # -----------------------------------------------------

        rows = self._group_into_rows(selected_lines)

        if len(rows) < 2:
            return []

        # -----------------------------------------------------
        # Merge wrapped-continuation row-labels (see method
        # docstring) BEFORE splitting into table regions.
        # -----------------------------------------------------

        rows = self._merge_wrapped_continuation_labels(rows)

        # -----------------------------------------------------
        # Detect table regions
        # -----------------------------------------------------

        table_regions = self._split_table_regions(rows)

        parsed_tables = []

        for region in table_regions:

            if len(region) < 2:
                continue

            table = self._parse_table_region(page, region)

            if table is not None:
                parsed_tables.append(table)

        # -----------------------------------------------------
        # NEW (CVS-style business-segment title fix):
        #
        # Apple's geographic segments have their title sitting just
        # 1 line above the actual data table (close enough that the
        # WITHIN-region rescue in _resplit_on_repeated_headers, added
        # earlier, catches it). CVS's business-line segments
        # ("Health Care Benefits Segment", "Health Services Segment",
        # "Pharmacy & Consumer Wellness Segment", "Corporate/Other
        # Segment" -- confirmed on CVS 2025 10-K pages 75/77/79/81)
        # are different: the title is followed by a full intro
        # sentence ("The following table summarizes the ... segment's
        # performance for the respective periods:"), and together
        # these sit far enough (>100pt gap) above the real table that
        # they already form their OWN separate header-less region by
        # the time _split_table_regions is done -- so the
        # within-region rescue never even sees them.
        #
        # Fix: a lightweight, separate cross-table pass at the page
        # level -- see _attach_orphaned_segment_titles() -- that
        # forwards a title sitting in one header-less table onto the
        # very next header-detected table on the same page. This
        # does NOT touch any existing region/table logic above; it
        # only adds a "section_title" field afterward.
        # -----------------------------------------------------

        self._attach_orphaned_segment_titles(parsed_tables)

        return parsed_tables

    def _attach_orphaned_segment_titles(self, parsed_tables):
        """
        See the NEW fix note in _parse_page() above for the full
        rationale. Walks consecutive parsed tables on the same page;
        if a header-less table's first row is a short, numeric-free
        title line (e.g. "Health Care Benefits Segment") and it is
        immediately followed by a header-detected table with no
        section_title yet, that title is forwarded onto the
        following table. The header-less table itself is left
        completely untouched.
        """

        for index in range(len(parsed_tables) - 1):

            current_table = parsed_tables[index]
            next_table = parsed_tables[index + 1]

            if current_table.get("header_detected"):
                continue

            if not next_table.get("header_detected"):
                continue

            if next_table.get("section_title"):
                continue

            current_rows = current_table.get("rows", [])

            if not current_rows:
                continue

            first_row_cells = current_rows[0].get("cells", [])

            title_text = " ".join(
                cell["text"] for cell in first_row_cells
            ).strip()

            if not title_text:
                continue

            if any(char.isdigit() for char in title_text):
                continue  # a numeric-bearing line isn't a title

            if title_text.endswith(":") or title_text.endswith("."):
                continue  # an intro sentence, not the title itself

            if len(title_text.split()) > 6:
                continue  # titles are short; sentences aren't

            next_table["section_title"] = title_text

    # =========================================================
    # GROUP LINES INTO ROWS
    # =========================================================

    def _group_into_rows(self, lines):

        sorted_lines = sorted(
            lines,
            key=lambda line: self._get_y(line) or 0.0,
        )

        rows = []

        for line in sorted_lines:

            y = self._get_y(line)

            if y is None:
                continue

            placed = False

            for row in rows:

                if abs(y - row["y"]) <= self.y_tolerance:

                    row["lines"].append(line)

                    # Update running average Y
                    row["y"] = sum(
                        self._get_y(x) for x in row["lines"]
                    ) / len(row["lines"])

                    placed = True
                    break

            if not placed:

                rows.append({"y": y, "lines": [line]})

        # Sort every row left -> right
        for row in rows:
            row["lines"].sort(key=lambda line: self._get_x(line))

        rows.sort(key=lambda row: row["y"])

        return rows

    def _row_has_real_numeric(self, row):
        """
        NEW (page-72 header-order bug fix): like checking whether
        any cell in a row looks numeric, but ignores a standalone
        footnote/superscript marker -- "(1)", "(2)", etc. -- when it
        is rendered in a visibly SMALLER font than the rest of its
        row. See _is_undersized_footnote_marker() for why that
        distinction matters. Used only by the wrapped-continuation-
        label merge below; every other numeric check in this file
        (value-matching, table-candidate scoring, etc.) is completely
        untouched.

        Root cause this fixes: on CVS 2025 10-K page 72, the
        "Health Services⁽¹⁾" / "Intersegment Eliminations⁽²⁾"
        column headers each wrap across 2 lines, with a footnote
        marker sitting between them. `_looks_numeric_cell("(1)")`
        correctly strips the parens and sees a digit, so it reads as
        "numeric" -- which made this WHOLE header row look like a
        data row to the merge logic below, and it fused THREE
        separate header rows (330pt / 339pt / 348pt) into ONE.
        Once merged, column order was no longer top-to-bottom by
        line -- it became a left-to-right x-sort across all the
        combined cells, which silently swaps the reported order of
        any two stacked header words whose bottom line happens to
        start slightly further LEFT than its own top line (confirmed:
        "Services" sits at x=242 under "Health" at x=250) -- producing
        "Services Health" / "Eliminations Intersegment" instead of
        the correct "Health Services" / "Intersegment Eliminations".
        """

        span_heights = []
        span_bottoms = []

        for line in row["lines"]:
            spans = line.get("spans") or [line]
            for span in spans:
                bbox = span.get("bbox")
                if bbox and len(bbox) >= 4:
                    span_heights.append(bbox[3] - bbox[1])
                    span_bottoms.append(bbox[3])

        typical_height = (
            sorted(span_heights)[len(span_heights) // 2]
            if span_heights else None
        )

        typical_bottom = (
            sorted(span_bottoms)[len(span_bottoms) // 2]
            if span_bottoms else None
        )

        for line in row["lines"]:

            spans = line.get("spans") or [line]

            for span in spans:

                text = span.get("text", "").strip()

                if not text:
                    continue

                if self._is_undersized_footnote_marker(
                    span, text, typical_height, typical_bottom
                ):
                    continue

                if self._looks_numeric_cell(text):
                    return True

        return False

    def _is_undersized_footnote_marker(
        self, span, text, typical_height, typical_bottom=None
    ):
        """
        True for a standalone "(N)" or "(NN)" token that's genuinely
        rendered as a footnote/superscript reference, not a real
        value -- checked via TWO independent signals, since how
        subtly a company renders its superscripts varies:

          1. Height meaningfully smaller (<90%) than the row's
             typical span height.
          2. Baseline meaningfully RAISED (bottom edge sits above --
             i.e. numerically less than -- the row's typical bottom)
             -- the defining visual trait of a superscript, since it
             sits higher than the surrounding text regardless of how
             much smaller its font is.

        Either signal alone is enough. Confirmed necessary on real
        data: CVS's footnote markers were ~35% shorter than
        surrounding text (caught easily by the height check alone),
        but Apple's are only ~17% shorter (7.44pt vs 8.93pt) -- just
        under the ORIGINAL 80% cutoff, so the Share Repurchase
        table's whole multi-line, multi-column header was getting
        merged into one scrambled row. Apple's marker's baseline
        (bottom=178.41) sits clearly above its row's typical bottom
        (~180.19), so the raised-baseline signal catches it even
        where the height signal alone narrowly misses.

        A genuine negative dollar figure like "(1,687)" or "(5)" is
        never rendered smaller OR raised relative to the rest of its
        row, so neither signal can mistake a real value for a
        footnote marker.
        """

        if not re.fullmatch(r"\(\d{1,2}\)", text):
            return False

        bbox = span.get("bbox")

        if not bbox or len(bbox) < 4:
            return False

        height = bbox[3] - bbox[1]
        bottom = bbox[3]

        if typical_height is not None and height < typical_height * 0.9:
            return True

        if typical_bottom is not None and bottom < typical_bottom - 1.0:
            return True

        return False

    def _merge_wrapped_continuation_labels(self, rows):
        """
        Fixes: a row-label that wraps across MULTIPLE physical lines
        (2, 3, or more -- e.g. "Adjustment for net (gains)/losses
        realized and included in net" / "income, net of tax expense/
        (benefit) of $(104), $475 and $131," / "respectively") where
        the actual numeric values sit at the same y as only the LAST
        line. Without this, every preceding line forms its own
        numeric-free "row" that no rescue-pass catches, and is
        silently lost -- confirmed on Apple 2018's Comprehensive
        Income statement (a 3-line wrap lost its first TWO lines,
        not just one).

        Distinguishing signal: genuine standalone section-headers in
        these tables ("Current assets:", "Shareholders' equity:")
        consistently END WITH A COLON. A wrapped label-continuation
        does not. So: collect a RUN of consecutive numeric-free,
        non-colon-ending rows, and merge the WHOLE run into the next
        row that actually has numeric values (however many lines
        that run spans), instead of only looking one row back.
        """

        merged_rows = []
        index = 0

        while index < len(rows):

            run_rows = []
            run_index = index

            while run_index < len(rows):

                row = rows[run_index]

                cells = self._extract_cells(row)

                row_text = " ".join(cell["text"] for cell in cells).strip()

                has_numeric = self._row_has_real_numeric(row)

                if has_numeric:
                    break

                if not row_text or row_text.endswith(":"):
                    break  # genuine section-header or blank -- stop the run

                # Bold needs nuance: Costco bolds BOTH its section-
                # titles ("REVENUE") AND individual line-items whose
                # label wraps across 2 bold lines ("EFFECT OF
                # EXCHANGE RATE CHANGES ON CASH AND CASH" /
                # "EQUIVALENTS", where "EQUIVALENTS" itself carries
                # the real values) -- so "is this row bold" alone
                # can't tell them apart. The real signal: a genuine
                # standalone title is followed by DIFFERENTLY-styled
                # data (bold title -> non-bold data-row, like
                # "REVENUE" -> "Net sales"). A wrapped bold label
                # continues into ANOTHER bold row. So only treat a
                # bold row as a hard stop if the row right after it
                # is NOT also bold.
                is_bold_row = any(cell.get("bold") for cell in cells)

                if is_bold_row:

                    peek_index = run_index + 1

                    if peek_index >= len(rows):
                        break

                    peek_cells = self._extract_cells(rows[peek_index])

                    peek_is_bold = any(
                        cell.get("bold") for cell in peek_cells
                    )

                    if not peek_is_bold:
                        break  # bold title -> non-bold data: genuine standalone header

                run_rows.append(row)
                run_index += 1

            if run_rows and run_index < len(rows):

                next_row = rows[run_index]

                next_cells = self._extract_cells(next_row)

                next_has_numeric = self._row_has_real_numeric(next_row)

                # A standalone single-cell YEAR row (e.g. "2025"
                # marking a year-sub-section, same pattern already
                # treated as a hard boundary elsewhere in this file
                # for text-header detection) is a SECTION MARKER,
                # not a wrapped label's data row -- it must never be
                # used as a merge target here, even though it does
                # have real numeric content and often matches the
                # run's bold-ness. Without this guard, once the
                # footnote-marker fix above (correctly) stops a bold
                # header row from looking "numeric", the run just
                # keeps extending and swallows the NEXT real row
                # instead -- confirmed on CVS 2025 10-K page 72,
                # where the header rows were merging straight through
                # into the "2025" row that follows them.
                next_is_standalone_year = (
                    len(next_cells) == 1
                    and self._is_year(next_cells[0]["text"].strip())
                )

                # -------------------------------------------------
                # Bold-status must MATCH between the collected run
                # and the row carrying the values, not just "is the
                # run bold" in isolation. Two genuinely different
                # situations look structurally identical (bold,
                # numeric-free row followed by a numeric row) but
                # need OPPOSITE handling:
                #   - "REVENUE" (bold) -> "Net sales" (NORMAL weight,
                #     163,220...) -- a real section-divider directly
                #     above an unrelated line-item. Bold-status
                #     CHANGES here -- do NOT merge.
                #   - "EFFECT OF EXCHANGE RATE CHANGES ON CASH AND
                #     CASH" (bold) -> "EQUIVALENTS" (ALSO bold, 70/
                #     (15)/(37)) -- the same bold title simply wraps
                #     across 2 lines, with its value on the last
                #     line. Bold-status STAYS THE SAME -- DO merge.
                # Confirmed on Costco 2020's Cash Flow Statement,
                # where Costco bolds entire major line-items (not
                # just top-level section dividers), unlike Apple.
                # -------------------------------------------------

                run_is_bold = any(
                    cell.get("bold")
                    for r in run_rows
                    for cell in self._extract_cells(r)
                )

                next_is_bold = any(cell.get("bold") for cell in next_cells)

                if (
                    next_has_numeric
                    and not next_is_standalone_year
                    and run_is_bold == next_is_bold
                ):

                    combined_lines = []

                    for run_row in run_rows:
                        combined_lines.extend(run_row["lines"])

                    combined_lines.extend(next_row["lines"])

                    combined_lines.sort(key=lambda line: self._get_x(line))

                    merged_rows.append({
                        "y": next_row["y"],
                        "lines": combined_lines,
                    })

                    index = run_index + 1
                    continue

            # No valid merge target found -- emit the collected run
            # (if any) and the current row as-is, and move on.
            if run_rows:
                merged_rows.extend(run_rows)
                index = run_index
            else:
                merged_rows.append(rows[index])
                index += 1

        return merged_rows

    # =========================================================
    # SPLIT TABLE REGIONS
    # =========================================================

    def _split_table_regions(self, rows):

        if not rows:
            return []

        regions = []

        current_region = [rows[0]]

        for previous, current in zip(rows, rows[1:]):

            gap = current["y"] - previous["y"]

            # Large vertical gap = likely a new/different table.
            #
            # This threshold keeps growing because a single fixed
            # number can't serve two opposite failure-modes at once:
            #   - Too TIGHT: genuinely-one table with long, wrapped,
            #     footnote-heavy row-labels (confirmed: Apple 2017's
            #     Comprehensive Income statement, where labels like
            #     "net of tax benefit/(expense) of $(478), $(7) and
            #     $(441), respectively" create large gaps) gets
            #     fractured into multiple headerless orphan-regions.
            #   - Too LOOSE: genuinely-different tables stacked close
            #     together (confirmed: Apple 2022's PP&E / Other
            #     Non-Current Liabilities / Other Income, each with a
            #     DIFFERENT column-count) get wrongly merged.
            #
            # Resolution: lean on the GAP only to avoid one giant
            # region spanning unrelated content far apart on a page
            # (very generous threshold now), and rely on
            # _resplit_on_repeated_headers() -- which looks for an
            # actual repeated year-header, real evidence rather than
            # a distance-guess -- as the PRECISE splitter for
            # genuinely-different tables sitting close together.
            if gap <= 100:
                current_region.append(current)
            else:
                if current_region:
                    regions.append(current_region)
                current_region = [current]

        if current_region:
            regions.append(current_region)

        # -----------------------------------------------------
        # Refinement pass: a wider gap-threshold alone isn't
        # reliable -- confirmed on Apple 2022 page 45, where THREE
        # genuinely-separate small tables (Property Plant &
        # Equipment, Other Non-Current Liabilities, Other Income/
        # Expense -- each with a DIFFERENT number of year-columns!)
        # sat close enough together (~35-40pt gaps) to get merged
        # into one region by the gap-check alone. Once merged, the
        # FIRST sub-table's header/column-count got locked in and
        # applied to ALL rows -- corrupting the later sub-tables'
        # values (a 3-column table's data got jammed into a
        # 2-column structure, numbers landing in the wrong slots).
        #
        # Fix: within each gap-based region, also look for a
        # REPEATED header-like row (another row, not the first,
        # that itself has >=2 bold year-cells) -- that's a reliable
        # signal a NEW table's header appears here, regardless of
        # how small the y-gap before it was. Split there too.
        # -----------------------------------------------------

        refined_regions = []

        for region in regions:
            refined_regions.extend(self._resplit_on_repeated_headers(region))

        return refined_regions

    def _resplit_on_repeated_headers(self, region):

        if len(region) <= 1:
            return [region]

        split_points = []

        for index, row in enumerate(region):

            if index == 0:
                continue  # the region's own first row is expected to be a header

            cells = self._extract_cells(row)

            year_cells = [c for c in cells if self._is_year(c["text"])]

            if len(year_cells) >= 2:
                split_points.append(index)

        if not split_points:
            return [region]

        # -----------------------------------------------------
        # NEW (segment-title fix): a short, numeric-free "title"
        # row (e.g. "Europe", "Japan", "Greater China", "Rest of
        # Asia Pacific", "Americas") sitting DIRECTLY ABOVE a
        # repeated year-header is that NEXT segment's own section
        # title -- not a trailing row of the table above it.
        #
        # Confirmed on Apple 2016's "Segment Operating Performance"
        # section: each segment is laid out as
        #     <segment name>              <- short italic title
        #     "The following table presents ... (dollars in millions):"
        #     [2016 | Change | 2015 | Change | 2014]   <- year header
        #     Net sales ...
        #     Percentage of total net sales ...
        #
        # Without this adjustment, the split point lands exactly AT
        # the year-header row, so the title row one line above it
        # stays trapped in the PREVIOUS region and comes out as a
        # garbage row with all-null values in the WRONG table
        # (e.g. "Europe" showing up as a null row inside the
        # Americas table, while the Europe table itself has no name
        # at all). Pulling the split point back by one row moves the
        # title into the region it actually introduces, where
        # _parse_table_region() below turns it into a proper
        # "section_title" field instead of a fake data row.
        # -----------------------------------------------------

        adjusted_points = []
        previous_point = 0

        for point in split_points:

            candidate_point = point

            if (
                point - 1 >= previous_point
                and self._looks_like_title_row(region[point - 1])
            ):
                candidate_point = point - 1

            adjusted_points.append(candidate_point)
            previous_point = candidate_point

        split_points = adjusted_points

        sub_regions = []
        start = 0

        for point in split_points:
            if point <= start:
                continue
            sub_regions.append(region[start:point])
            start = point

        sub_regions.append(region[start:])

        return [sub for sub in sub_regions if sub]

    def _looks_like_title_row(self, row):
        """
        True for a short, purely-textual row with NO numeric content
        at all -- the signature of a segment/section title like
        "Europe" or "Americas" sitting just above its own mini-table.

        Guards against false positives:
          - any numeric cell at all -> not a title (real data rows,
            like "Net sales" or "Percentage of total net sales",
            always carry numbers)
          - ends with ":" -> a genuine section-header inside a
            financial statement ("Current assets:"), handled by the
            wrapped-continuation-label logic elsewhere, not this one
          - ends with "." -> a narrative sentence fragment
            ("...partially offset by a decline..."), not a title
          - more than 6 words -> titles are short; sentences aren't
        """

        cells = self._extract_cells(row)

        if not cells:
            return False

        text = " ".join(cell["text"] for cell in cells).strip()

        if not text:
            return False

        if any(self._looks_numeric_cell(cell["text"]) for cell in cells):
            return False

        if text.endswith(":") or text.endswith("."):
            return False

        word_count = len(text.split())

        return 1 <= word_count <= 6

    # =========================================================
    # GROUPED (2-LEVEL) YEAR HEADER
    #
    # NEW: some tables use a genuinely 2-LEVEL header -- a row of
    # YEAR cells that are actually GROUP headers, each spanning
    # several real data sub-columns declared on the very next row
    # (e.g. "Insured | ASC | Total", repeated once per year).
    # Confirmed on CVS 2025 10-K page 76's Medical Membership table:
    #
    #                    2025                       2024
    #   In thousands  Insured  ASC  Total   Insured  ASC  Total
    #   Commercial      3,447  15,350 18,797   4,691  14,160 18,851
    #
    # Without this, the plain single-level year-header path (below)
    # treats "2025"/"2024" as if each were ONE data column -- since
    # it still finds >=2 year cells and successfully extracts 2
    # columns, it "succeeds" and returns before ever noticing the
    # real structure is wrong. Symptoms confirmed on real output:
    #   - the row's first real value ("Insured") gets merged into
    #     the row LABEL instead of a value (e.g. "Commercial 3,447")
    #   - the "Total" sub-column is dropped completely (silent data
    #     loss, not just a labeling problem)
    #
    # This check runs BEFORE the plain year-header path and only
    # fires when the specific repeating-subheader pattern is
    # actually present; otherwise it returns nothing and every
    # existing code path below is completely unaffected.
    # =========================================================

    def _find_grouped_year_header(self, rows):

        search_window = min(len(rows) - 1, 10)

        for index in range(max(search_window, 0)):

            year_cells = [
                cell for cell in self._extract_cells(rows[index])
                if self._is_year(cell["text"])
            ]

            year_count = len(year_cells)

            if year_count < 2:
                continue

            sub_row_index = index + 1

            if sub_row_index >= len(rows):
                continue

            sub_cells_all = self._extract_cells(rows[sub_row_index])

            # The sub-header row may have a leading row-label-column
            # header of its own (e.g. "In thousands") before the
            # real repeating sub-columns start -- try dropping 0, 1,
            # then 2 leading cells until the remainder cleanly
            # divides into `year_count` equal, REPEATING groups.
            for drop_leading in (0, 1, 2):

                sub_cells = sub_cells_all[drop_leading:]

                if len(sub_cells) < year_count * 2:
                    continue

                if len(sub_cells) % year_count != 0:
                    continue

                group_size = len(sub_cells) // year_count

                sub_texts = [cell["text"].strip() for cell in sub_cells]

                first_group = sub_texts[:group_size]

                is_repeating = all(
                    sub_texts[g * group_size:(g + 1) * group_size] == first_group
                    for g in range(year_count)
                )

                if not is_repeating:
                    continue

                # Real sub-headers are short text labels ("Insured",
                # "ASC", "Total") -- if what repeated is numeric or
                # year-like, this row is a DATA row, not a header,
                # and we've matched by coincidence.
                if any(
                    self._looks_numeric_cell(t) or self._is_year(t)
                    for t in first_group
                ):
                    continue

                year_cells_sorted = sorted(year_cells, key=lambda c: c["x"])

                columns = []

                for group_index, year_cell in enumerate(year_cells_sorted):

                    group_cells = sub_cells[
                        group_index * group_size:(group_index + 1) * group_size
                    ]

                    for sub_cell in group_cells:

                        columns.append({
                            "name": f"{year_cell['text']} {sub_cell['text']}".strip(),
                            "x": sub_cell["x"],
                        })

                columns.sort(key=lambda item: item["x"])

                return {index, sub_row_index}, columns

        return set(), []

    # =========================================================
    # PARSE TABLE REGION
    # =========================================================

    def _parse_table_region(self, page, rows):

        # -----------------------------------------------------
        # Try the 2-level grouped year header FIRST (see method
        # above). Only returns non-empty when the specific
        # repeating-subheader pattern is genuinely present -- if
        # not, falls straight through to the existing single-level
        # logic below, completely unchanged.
        # -----------------------------------------------------

        grouped_header_rows, grouped_columns = self._find_grouped_year_header(rows)

        if grouped_columns:

            data_rows = []

            for index, row in enumerate(rows):

                if index in grouped_header_rows:
                    continue

                parsed_row = self._parse_data_row(row, grouped_columns)

                if parsed_row is not None:
                    data_rows.append(parsed_row)

            if data_rows:

                return {
                    "page_number": page.get("page_number"),
                    "bbox": self._get_table_bbox(rows),
                    "header": [column["name"] for column in grouped_columns],
                    "rows": data_rows,
                    "header_detected": True,
                    "column_type": "year_grouped",
                    "row_count": len(data_rows),
                    "column_count": len(grouped_columns),
                }

        header_index = self._find_header(rows)

        # -----------------------------------------------------
        # Segment-title extraction (see _looks_like_title_row /
        # _resplit_on_repeated_headers above): if the row directly
        # above the detected year-header is a short, numeric-free
        # title, pull it out as this table's own "section_title"
        # instead of letting it become a garbage null-value data
        # row. Only fires when the title sits IMMEDIATELY above the
        # header (header_index == 1) -- if there's anything else in
        # between, it's not the clean single-title pattern we've
        # confirmed on real data, so leave it alone rather than
        # guess.
        # -----------------------------------------------------

        section_title = None

        if (
            header_index is not None
            and header_index == 1
            and self._looks_like_title_row(rows[0])
        ):

            title_cells = self._extract_cells(rows[0])

            section_title = " ".join(
                cell["text"] for cell in title_cells
            ).strip()

            rows = rows[1:]
            header_index = 0

        # -----------------------------------------------------
        # Header found -> structured column mapping
        # -----------------------------------------------------

        if header_index is not None:

            header_row = rows[header_index]

            columns = self._extract_columns(header_row)

            if len(columns) >= 2:

                data_rows = []

                for index, row in enumerate(rows):

                    if index == header_index:
                        continue

                    parsed_row = self._parse_data_row(row, columns)

                    if parsed_row is not None:
                        data_rows.append(parsed_row)

                if data_rows:

                    table = {
                        "page_number": page.get("page_number"),
                        "bbox": self._get_table_bbox(rows),
                        "header": [column["name"] for column in columns],
                        "rows": data_rows,
                        "header_detected": True,
                        "column_type": "year",
                        "row_count": len(data_rows),
                        "column_count": len(columns),
                    }

                    if section_title:
                        table["section_title"] = section_title

                    return table

        # -----------------------------------------------------
        # No YEAR header -> try TEXT-LABEL header instead.
        #
        # Many financial tables (Share Repurchase Program, stock
        # option/RSU activity, etc.) use wrapped text-label column
        # headers like "Total Number of Shares Purchased" instead of
        # years. These wrap across MULTIPLE physical lines per
        # column (e.g. "Total" / "Number" / "of Shares" / "Purchased"
        # each on their own line, all near the same x-position). This
        # pattern recurs across most 10-K filings, so it's worth
        # detecting properly rather than always falling back to raw.
        # -----------------------------------------------------

        text_header_rows, text_columns = self._find_text_header(rows)

        if text_columns and len(text_columns) >= 2:

            data_rows = []

            for index, row in enumerate(rows):

                if index in text_header_rows:
                    continue

                parsed_row = self._parse_data_row(row, text_columns, value_tolerance=65.0)

                if parsed_row is not None:
                    data_rows.append(parsed_row)

            if data_rows:

                table = {
                    "page_number": page.get("page_number"),
                    "bbox": self._get_table_bbox(rows),
                    "header": [column["name"] for column in text_columns],
                    "rows": data_rows,
                    "header_detected": True,
                    "column_type": "text_label",
                    "row_count": len(data_rows),
                    "column_count": len(text_columns),
                }

                if section_title:
                    table["section_title"] = section_title

                return table

        # -----------------------------------------------------
        # No reliable header of either kind -> raw fallback
        # -----------------------------------------------------

        return self._parse_without_header(page, rows)

    # =========================================================
    # TEXT-LABEL HEADER DETECTION (fallback when no years found)
    # =========================================================

    def _row_numeric_fraction(self, row):
        """
        What fraction of this row's cells look like real numbers
        (not header-labels). A row is treated as DATA once this is
        high -- everything above it, in the region, is candidate
        header material.
        """

        cells = self._extract_cells(row)

        if not cells:
            return 0.0

        numeric_count = sum(
            1 for cell in cells
            if self._looks_numeric_cell(cell["text"])
        )

        return numeric_count / len(cells)

    def _looks_numeric_cell(self, text):

        cleaned = text.strip()

        if cleaned in ("$", "-", "--", "—", "%", "(", ")"):
            return False

        cleaned = re.sub(r"[\$,\.\-\+\(\)%\s]", "", cleaned)

        return bool(cleaned) and cleaned.isdigit()

    def _looks_like_value_cell(self, text):
        """
        NEW (Adobe 2025, confirmed via real chunks.json output):
        True for any cell that represents a genuine VALUE slot in a
        financial-statement row -- either a real number (see
        _looks_numeric_cell) OR an explicit dash/em-dash placeholder
        for zero/not-applicable ("-", "--", "—"). Financial tables
        use a bare dash as a real, meaningful VALUE (e.g. "Common
        Stock Amount: -"), not as label punctuation -- this is the
        exact same exception _parse_data_row's value-matching logic
        already makes for NON_LABEL_TOKENS (see the
        `text not in ("-", "--", "—")` check there).

        Used only by _consolidate_raw_label_fragments() below, to
        find where a headerless/raw row's LABEL ends and its VALUES
        begin. Every other numeric/value check in this file is
        completely untouched.
        """

        cleaned = text.strip()

        if cleaned in ("-", "--", "—"):
            return True

        return self._looks_numeric_cell(text)

    def _consolidate_raw_label_fragments(self, cells):
        """
        NEW (Adobe 2025, confirmed via real chunks.json output): a
        row-label that wraps across 2+ physical PDF lines (e.g.
        "Other comprehensive income" / "(loss), net of taxes",
        "Re-issuance of treasury stock" / "under stock compensation"
        / "plans", "Value of shares in deferred" / "compensation
        plan") reaches this HEADERLESS/raw fallback path as SEVERAL
        separate leading cells instead of one merged label.

        Root cause: _merge_wrapped_continuation_labels() (see above)
        already correctly merges the underlying PHYSICAL LINES of
        such a row together -- but it only guarantees those lines
        end up grouped into one ROW; it does not merge their TEXT
        into a single CELL. When _extract_cells() then re-splits
        that combined row back out cell-by-cell (one cell per
        physical line/span), the wrapped label re-fragments into N
        separate leading cells.

        For a table with a DETECTED header (year/text-label/grouped),
        this is already handled correctly downstream -- see
        _parse_data_row(), which joins every cell to the LEFT of the
        first real column position into one `label` string. But a
        table that falls all the way through to THIS raw fallback
        (e.g. because its header is a compound, multi-row, non-year
        grouped header -- like Adobe's Consolidated Statements of
        Stockholders' Equity, whose "Common Stock"/"Treasury Stock"
        group headers each span 2 sub-columns, a pattern none of the
        structured header-detectors above recognize yet) never went
        through that label-consolidation step at all.

        Confirmed real symptom on Adobe's Stockholders' Equity
        table: rows rendered downstream as
            "Other comprehensive income | (loss), net of taxes | - | ... | 8"
        instead of
            "Other comprehensive income (loss), net of taxes | - | ... | 8"
        -- silently shifting every subsequent value's apparent
        column position by one cell.

        Fix: merge every LEADING cell up to (but not including) the
        first cell that looks like a genuine value slot (per
        _looks_like_value_cell -- a real number, or an explicit
        "-"/"--"/"—" placeholder) into ONE cell. A standalone "$"
        immediately before that first value is dropped, matching the
        exact same convention already used everywhere else in this
        file (NON_LABEL_TOKENS).

        This never changes the return shape (still a list of
        {"text", "x"} cells, same as before) and never touches any
        row where label-wrapping doesn't apply (a row with no value
        cells at all, or one that already starts with a value, is
        returned completely untouched) -- so no other table's output
        is affected.
        """

        first_value_index = None

        for index, cell in enumerate(cells):

            text = cell["text"].strip()

            if text == "$":
                continue

            if self._looks_like_value_cell(text):
                first_value_index = index
                break

        if first_value_index is None or first_value_index == 0:
            # No value cells at all, or the row already starts with
            # a value (nothing to consolidate) -- leave untouched.
            return cells

        label_cells = cells[:first_value_index]
        value_cells = cells[first_value_index:]

        label_parts = [
            c["text"].strip() for c in label_cells
            if c["text"].strip() and c["text"].strip() != "$"
        ]

        if not label_parts:
            return cells

        merged_label_cell = {
            "text": " ".join(label_parts),
            "x": label_cells[0]["x"],
        }

        return [merged_label_cell] + value_cells

    def _find_text_header(self, rows, max_header_rows=14, x_cluster_tolerance=None):
        """
        Detects wrapped, multi-line TEXT-LABEL column headers (e.g.
        "Total Number of Shares Purchased" spread across 3-4 stacked
        short lines), as an alternative to the year-based header.

        Approach:
          1. Find where real DATA starts -- the first row whose cells
             are mostly numeric. Everything above that (up to
             max_header_rows) is the "header zone".
          2. Within the header zone, cluster cells by x-position --
             cells stacked at roughly the same x across multiple
             header-zone rows belong to the SAME column, and their
             text (in top-to-bottom order) concatenates into one
             combined column label.
          3. Return these as {"name": ..., "x": ...} columns, in the
             exact same shape _extract_columns() produces for years --
             so all the existing row-matching logic downstream just
             works, unchanged, regardless of which detector found the
             columns.

        Returns (header_row_indices, columns). columns is an empty
        list if nothing confident was found (caller falls back to
        the raw/unstructured parser in that case).
        """

        if x_cluster_tolerance is None:
            x_cluster_tolerance = self.x_tolerance

        # Step 1: find where data starts
        data_start_index = None

        for index, row in enumerate(rows[:max_header_rows + 1]):

            # A standalone bold YEAR row ("2020") marks a hard
            # boundary -- it's a year-sub-section label, not part of
            # the column-header text, and NOT a data-row itself
            # either. Confirmed on Costco's geographic-segment table
            # (US/Canadian/Other-International Operations): without
            # this check, the header-zone scan kept going past the
            # "2020" row and even absorbed the FIRST real data-row's
            # numbers into the column names themselves (e.g. "United
            # States Operations 122,142" as one garbled "column").
            row_cells = self._extract_cells(row)

            if (
                len(row_cells) == 1
                and self._is_year(row_cells[0]["text"])
            ):
                data_start_index = index
                break

            if self._row_numeric_fraction(row) >= 0.5:
                data_start_index = index
                break

        if data_start_index is None or data_start_index == 0:
            # No clear header zone (table starts with data immediately,
            # or nothing looked numeric within our search window).
            return set(), []

        header_row_indices = set(range(data_start_index))

        # -----------------------------------------------------
        # NEW: identify and exclude TABLE TITLE/CAPTION lines
        # sitting inside the header zone (e.g. "CONSOLIDATED
        # STATEMENTS OF SHAREHOLDERS' EQUITY" / "(In millions,
        # except number of shares which are reflected in
        # thousands)") before clustering.
        #
        # Confirmed on Apple 2016's Shareholders' Equity statement:
        # these two centered caption lines were being read as their
        # own column-header fragment, adding a 6th BOGUS column
        # whose "name" was the literal caption text (always null,
        # since no real value ever sits at that x) -- and because
        # the real column count/x-calibration was thrown off by
        # this extra phantom column, a real value (e.g. "Accumulated
        # Other Comprehensive Income/(Loss)") went unmatched while
        # its neighbour's value ("Retained Earnings") got duplicated
        # into both columns for every row.
        #
        # Distinguishing signal: a genuine per-column header
        # fragment (even a wide one spanning 2 sub-columns, like
        # "Common Stock and Additional Paid-In Capital") is still
        # meaningfully narrower than the FULL table -- a title/
        # caption line spans nearly the entire table width instead.
        # We use the first real DATA row (right after the header
        # zone, which always spans the true full column range) as
        # the width reference, and drop any single-cell header-zone
        # row whose own line is almost as wide as that.
        # -----------------------------------------------------

        reference_cells = self._extract_cells(rows[data_start_index])

        table_width = None

        if len(reference_cells) >= 2:
            table_width = (
                max(c["x"] for c in reference_cells)
                - min(c["x"] for c in reference_cells)
            )

        title_row_indices = set()

        # NEW (Amazon 2025, confirmed via real chunks.json output): a
        # SHORT units-disclaimer caption -- "(in millions)", "(in
        # thousands)" -- sitting alone on its own header-zone line
        # was NOT being excluded by the width-based check below, even
        # though it's the exact same kind of caption the width-check
        # was originally built to catch (Apple 2016's wider "(In
        # millions, except number of shares which are reflected in
        # thousands)"). The width heuristic only fires at >=40% of
        # the table's width -- "(in millions)" alone is short enough
        # to fall well under that on a wide multi-column statement
        # like Stockholders' Equity, so it slipped through and became
        # its own bogus column.
        #
        # Confirmed real symptom on Amazon's Consolidated Statements
        # of Stockholders' Equity: "(in millions)" became a phantom
        # column between "Common Stock Amount" and "Treasury Stock",
        # shifting every value one slot to the right from that point
        # on -- e.g. the real Treasury Stock figure ((7,837)) was
        # recorded under the bogus "(in millions)" key, and the real
        # Additional Paid-In Capital figure (75,066) was recorded
        # under the "Treasury Stock" key instead. This is a genuine
        # value-misattribution (not just a cosmetic label issue): a
        # query for Amazon's Treasury Stock would return the wrong
        # number.
        #
        # Fix: also recognize this specific caption pattern directly
        # by CONTENT, independent of width -- reusing the same
        # units-disclaimer wording heading_detector.py already
        # recognizes elsewhere in the pipeline for the identical
        # reason. A content match is actually a MORE reliable signal
        # than width for this exact case (a units caption is never
        # mistaken for a real column name), so this runs regardless
        # of table_width being available at all.
        _units_disclaimer_re = re.compile(
            r"^\(\s*(dollars\s+|amounts\s+)?in\s+(millions|thousands|billions)"
            r"(\s*,\s*[^)]*)?\)$",
            re.IGNORECASE,
        )

        for row_index in header_row_indices:

            row_cells = self._extract_cells(rows[row_index])

            if len(row_cells) != 1:
                continue  # a real grouped header fragment can share a row

            if _units_disclaimer_re.match(row_cells[0]["text"].strip()):
                title_row_indices.add(row_index)

        if table_width and table_width > 0:

            for row_index in header_row_indices:

                if row_index in title_row_indices:
                    continue  # already excluded by the content match above

                row_cells = self._extract_cells(rows[row_index])

                if len(row_cells) != 1:
                    continue  # a real grouped header fragment can share

                line = rows[row_index]["lines"][0]
                bbox = line.get("bbox")

                if not bbox or len(bbox) < 4:
                    continue

                cell_width = bbox[2] - bbox[0]

                if cell_width >= table_width * 0.4:
                    title_row_indices.add(row_index)

        # Step 2: collect all header-zone cells, in row (top-to-bottom)
        # order, and cluster by x-position.
        clusters = []  # each: {"x": representative_x, "parts": [text, ...]}

        for row_index in sorted(header_row_indices):

            if row_index in title_row_indices:
                continue  # table title/caption line -- not a real column

            for cell in self._extract_cells(rows[row_index]):

                text = cell["text"].strip()

                if not text or text in ("$", "-", "--", "—"):
                    continue

                placed = False

                for cluster in clusters:

                    if abs(cell["x"] - cluster["x"]) <= x_cluster_tolerance:
                        cluster["parts"].append(text)
                        placed = True
                        break

                if not placed:
                    clusters.append({"x": cell["x"], "parts": [text]})

        if len(clusters) < 2:
            return header_row_indices, []

        columns = [
            {
                "name": " ".join(cluster["parts"]).strip(),
                "x": cluster["x"],
            }
            for cluster in clusters
            if " ".join(cluster["parts"]).strip()
        ]

        # Drop spurious "columns" that are just a footnote marker
        # (e.g. "(1)") that happened to sit far enough from every
        # real column to form its own isolated cluster. Confirmed on
        # Apple 2016's Share Repurchase table -- a stray "(1)" at the
        # far-right edge became a fake 6th column otherwise.
        columns = [
            column for column in columns
            if not re.fullmatch(r"\(\d+\)", column["name"].strip())
        ]

        columns.sort(key=lambda item: item["x"])

        # The LEFTMOST cluster in these tables is consistently the
        # row-LABEL column's own header (e.g. "Periods"), not a real
        # value-column -- exactly like year-header tables, where the
        # row label ("Net sales") is never itself one of the
        # `columns`. If we kept it, _parse_data_row's label-boundary
        # check (x < first_column_x) would wrongly swallow real row
        # labels ("purchases", "Open market...") as if they were
        # this column's VALUE, since their x sits to the right of
        # this narrow label-header, not to the left of it.
        if len(columns) > 2 and columns[0]["x"] < 100:
            columns = columns[1:]

        return header_row_indices, columns

    # =========================================================
    # HEADER DETECTION
    # =========================================================

    def _find_header(self, rows):

        best_index = None
        best_score = 0

        # NOTE: this used to only search rows[:4] -- but the header-
        # zone-rescue pass added to TableAnalyzer (which pulls in
        # short text lines ABOVE a table's numeric data, to catch
        # wrapped text-column-headers) can now legitimately add MORE
        # rows before the real header on dense pages (e.g. a page
        # with a title + multi-line intro paragraph before a
        # 5-year financial summary table). That pushed the real
        # year-header past index 3, causing detection to fail
        # entirely on tables that used to parse correctly (confirmed:
        # Apple 2016's "Selected Financial Data" 5-year table).
        #
        # Fix: search a much wider window. This is safe -- the
        # scoring itself (multiple 4-digit years, bold) is specific
        # enough that a genuine data-row won't accidentally win; we
        # were just artificially limiting WHERE we looked.
        search_window = min(len(rows), 15)

        for index, row in enumerate(rows[:search_window]):

            cells = self._extract_cells(row)

            if len(cells) < 2:
                continue

            year_cells = [
                cell for cell in cells
                if self._is_year(cell["text"])
            ]

            if len(year_cells) < 2:
                continue

            score = 0

            # Multiple years = strong signal
            score += len(year_cells) * 2

            # Bold years = stronger signal
            bold_years = sum(1 for cell in year_cells if cell["bold"])
            score += bold_years * 2

            if index == 0:
                score += 2

            if score > best_score:
                best_score = score
                best_index = index

        return best_index

    # =========================================================
    # EXTRACT COLUMNS (from header row)
    # =========================================================

    def _extract_columns(self, header_row):

        cells = self._extract_cells(header_row)

        columns = []

        for cell in cells:

            text = cell["text"].strip()

            if not self._is_year(text):
                continue

            columns.append({"name": text, "x": cell["x"]})

        columns.sort(key=lambda item: item["x"])

        return columns

    # =========================================================
    # PARSE DATA ROW
    # =========================================================

    def _parse_data_row(self, row, columns, value_tolerance=None):

        if value_tolerance is None:
            value_tolerance = self.x_tolerance

        cells = self._extract_cells(row)

        if not cells:
            return None

        # -----------------------------------------------------
        # Identify label (everything left of the first data column,
        # excluding stray currency/symbol tokens like "$")
        # -----------------------------------------------------

        first_column_x = columns[0]["x"]

        label_parts = []

        for cell in cells:

            text = cell["text"].strip()

            # Skip standalone symbols -- these are NOT part of the label
            # even if they sit to the left of the first data column.
            # (Fixes: "Net sales" incorrectly becoming "Net sales $")
            if text in self.NON_LABEL_TOKENS:
                continue

            if cell["x"] < (first_column_x - value_tolerance):
                label_parts.append(text)

        label = " ".join(label_parts).strip()

        # -----------------------------------------------------
        # Map values to columns by nearest x-position
        #
        # value_tolerance is wider than the default x_tolerance when
        # matching TEXT-LABEL columns (see caller). Those columns
        # can be visually wide (e.g. "Total Number of Shares
        # Purchased as Part of Publicly Announced Plans or
        # Programs"), and the column's x-anchor is based on where
        # its LEFTMOST header-word starts -- but the actual numeric
        # data underneath is often right-aligned within that wide
        # column, sitting well to the right of that anchor. A tight
        # tolerance (fine for narrow year-columns) left most values
        # unmatched (null) for these wider text-label columns.
        # Confirmed on real data: Apple 2016's Share Repurchase
        # table had genuine offsets up to ~37 points.
        # -----------------------------------------------------

        values = {}
        claimed_cell_ids = set()

        for column in columns:

            target_x = column["x"]

            matching_cells = []

            for cell in cells:

                text = cell["text"].strip()

                # Symbols alone should never "win" a column match --
                # EXCEPT dash/em-dash, which in financial tables
                # represents an explicit zero/not-applicable VALUE
                # (e.g. "Cumulative effect of change in accounting
                # principle: — — (136)"). Treating it the same as a
                # stray "$" caused genuine zero-values to disappear
                # as null instead of being recorded as "—".
                if text in self.NON_LABEL_TOKENS and text not in ("-", "--", "—"):
                    continue

                # NEW: a cell already assigned to an earlier (closer)
                # column can never be reused by a later column. Each
                # column used to search ALL cells independently, so
                # two columns whose target x's both fell within
                # tolerance of the SAME cell would both claim it --
                # confirmed on Apple 2016's Shareholders' Equity
                # statement, where "Accumulated Other Comprehensive
                # Income/(Loss)" and "Retained Earnings" both matched
                # Retained Earnings' own value cell, silently
                # duplicating "39,510" into both columns for the
                # "Net income" row instead of AOCI correctly getting
                # "-". Columns are processed left-to-right (sorted by
                # x), so the closer/earlier column keeps first claim,
                # which is always the geometrically correct one.
                if id(cell) in claimed_cell_ids:
                    continue

                distance = abs(cell["x"] - target_x)

                if distance <= value_tolerance:
                    matching_cells.append((distance, cell))

            if matching_cells:

                matching_cells.sort(key=lambda item: item[0])

                best_cell = matching_cells[0][1]

                values[column["name"]] = best_cell["text"]
                claimed_cell_ids.add(id(best_cell))

            else:

                values[column["name"]] = None

        # -----------------------------------------------------
        # Positional fallback for still-unmatched columns.
        #
        # Recurring pattern (seen 3 times now: the "September"
        # date-header table, the Share Repurchase table, and now
        # Apple's EPS row): a row can have NARROW values (e.g. EPS
        # "6.11") sitting far from a column-anchor calibrated
        # against WIDE values (e.g. revenue "316,199") elsewhere in
        # the SAME table, because right-aligned numbers of very
        # different lengths land at different x-positions even
        # within "the same visual column". Strict x-tolerance
        # matching alone can't bridge that reliably.
        #
        # Fix: if some columns are still unmatched, and there are
        # exactly as many UNCLAIMED value-like cells left in this
        # row as there are unmatched columns, assume they correspond
        # in left-to-right order (both sorted by x) and assign them
        # positionally. This only fires when the counts match
        # exactly, which keeps it safe -- it won't guess when the
        # row's shape doesn't cleanly line up with the columns.
        # -----------------------------------------------------

        unmatched_columns = [
            column for column in columns
            if values[column["name"]] is None
        ]

        if unmatched_columns:

            unclaimed_cells = [
                cell for cell in cells
                if (
                    cell["text"].strip() not in self.NON_LABEL_TOKENS
                    or cell["text"].strip() in ("-", "--", "—")
                )
                and id(cell) not in claimed_cell_ids
                and cell["x"] >= (first_column_x - value_tolerance)
            ]

            if len(unclaimed_cells) == len(unmatched_columns):

                sorted_columns = sorted(unmatched_columns, key=lambda c: c["x"])
                sorted_cells = sorted(unclaimed_cells, key=lambda c: c["x"])

                for column, cell in zip(sorted_columns, sorted_cells):
                    values[column["name"]] = cell["text"]

        # Ignore completely empty rows
        if not label and not any(v is not None for v in values.values()):
            return None

        return {
            "label": label,
            "values": values,
        }

    # =========================================================
    # EXTRACT CELLS FROM SPANS
    # =========================================================

    def _extract_cells(self, row):

        cells = []

        for line in row["lines"]:

            spans = line.get("spans", [])

            # -------------------------------------------------
            # Normal case: spans available
            # -------------------------------------------------

            if spans:

                for span in spans:

                    text = span.get("text", "").strip()

                    if not text:
                        continue

                    bbox = span.get("bbox")

                    if bbox and len(bbox) >= 4:
                        x = float(bbox[0])
                    else:
                        x = self._get_x(line)

                    flags = span.get("flags", 0)
                    bold = bool(flags & 16)

                    cells.append({
                        "text": text,
                        "x": x,
                        "bold": bold,
                    })

            # -------------------------------------------------
            # Fallback: line itself (no spans)
            # -------------------------------------------------

            else:

                text = line.get("text", "").strip()

                if text:
                    cells.append({
                        "text": text,
                        "x": self._get_x(line),
                        "bold": False,
                    })

        cells.sort(key=lambda cell: cell["x"])

        return cells

    # =========================================================
    # TABLE WITHOUT HEADER
    # =========================================================

    def _parse_without_header(self, page, rows):

        parsed_rows = []

        for row in rows:

            cells = self._extract_cells(row)

            if not cells:
                continue

            # NEW: consolidate any wrapped-label fragments BEFORE
            # emitting this row -- see _consolidate_raw_label_fragments()
            # docstring for the full rationale (confirmed on Adobe
            # 2025's Consolidated Statements of Stockholders' Equity
            # table). Output shape is unchanged (still a list of
            # {"text", "x"} cells) -- only rows that actually had a
            # wrapped leading label are affected.
            consolidated_cells = self._consolidate_raw_label_fragments(cells)

            parsed_rows.append({
                "cells": [
                    {"text": cell["text"], "x": cell["x"]}
                    for cell in consolidated_cells
                ]
            })

        if not parsed_rows:
            return None

        return {
            "page_number": page.get("page_number"),
            "bbox": self._get_table_bbox(rows),
            "header": [],
            "rows": parsed_rows,
            "header_detected": False,
            "row_count": len(parsed_rows),
            "column_count": 0,
        }

    # =========================================================
    # YEAR CHECK
    # =========================================================

    def _is_year(self, text):

        text = text.strip()

        if len(text) != 4 or not text.isdigit():
            return False

        year = int(text)

        return 1900 <= year <= 2100

    # =========================================================
    # COORDINATES
    # =========================================================

    def _get_x(self, line):

        bbox = line.get("bbox")

        if not bbox or len(bbox) < 4:
            return 0.0

        try:
            return float(bbox[0])
        except (TypeError, ValueError):
            return 0.0

    def _get_y(self, line):

        bbox = line.get("bbox")

        if not bbox or len(bbox) < 4:
            return None

        try:
            return float(bbox[1])
        except (TypeError, ValueError):
            return None

    # =========================================================
    # TABLE BBOX
    # =========================================================

    def _get_table_bbox(self, rows):

        boxes = []

        for row in rows:
            for line in row["lines"]:
                bbox = line.get("bbox")
                if bbox and len(bbox) >= 4:
                    boxes.append(bbox)

        if not boxes:
            return None

        return [
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        ]

    # =========================================================
    # SAVE
    # =========================================================

    def save_parsed_reports(
        self,
        parsed_reports,
        output_dir="STAGE_1/parsed_tables",
    ):

        output_dir = Path(output_dir)

        for report in parsed_reports:

            company = report.get("company", "unknown")

            company_dir = output_dir / company

            company_dir.mkdir(parents=True, exist_ok=True)

            file_name = Path(
                report.get("file_name", "report.json")
            ).stem

            output_file = company_dir / (
                file_name + "_parsed_tables.json"
            )

            with open(output_file, "w", encoding="utf-8") as f:

                json.dump(
                    report,
                    f,
                    indent=4,
                    ensure_ascii=False,
                    default=str,
                )

            print(f"Saved Parsed Tables: {output_file}")


# =============================================================
# LOAD TABLE-ANALYZED JSON
# =============================================================

def load_analyzed_reports(input_dir):

    input_dir = Path(input_dir)

    reports = []

    if not input_dir.exists():
        print(f"Folder not found: {input_dir}")
        return reports

    for company_dir in sorted(input_dir.iterdir()):

        if not company_dir.is_dir():
            continue

        for json_file in sorted(
            company_dir.glob("*_table_analyzed.json")
        ):

            with open(json_file, "r", encoding="utf-8") as f:
                reports.append(json.load(f))

    return reports


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":

    INPUT_DIR = "STAGE_1/table_analysis"
    OUTPUT_DIR = "STAGE_1/parsed_tables"

    print("\n====================================")
    print(" Table Parser Started")
    print("====================================\n")

    analyzed_reports = load_analyzed_reports(INPUT_DIR)

    if not analyzed_reports:

        print("No table-analyzed JSON files found.")
        print("Run TableAnalyzer first.")

    else:

        parser = TableParser()

        parsed_reports = []

        for report in analyzed_reports:

            print(f"Parsing: {report.get('file_name')}")

            parsed_report = parser.parse_report(report)

            parsed_reports.append(parsed_report)

        parser.save_parsed_reports(parsed_reports, OUTPUT_DIR)

        total_tables = sum(
            len(report["tables"]) for report in parsed_reports
        )

        with_header = sum(
            1
            for report in parsed_reports
            for table in report["tables"]
            if table.get("header_detected")
        )

        with_section_title = sum(
            1
            for report in parsed_reports
            for table in report["tables"]
            if table.get("section_title")
        )

        print("\n====================================")
        print(" Table Parsing Completed")
        print("====================================")
        print(f"Reports Parsed         : {len(parsed_reports)}")
        print(f"Tables Parsed           : {total_tables}")
        print(f"Tables With Header      : {with_header}")
        print(f"Tables Without Header   : {total_tables - with_header}")
        print(f"Tables With Section Title: {with_section_title}")
        print("\nOutput:")
        print(OUTPUT_DIR)