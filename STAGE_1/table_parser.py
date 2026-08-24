import json
import re
from pathlib import Path
print(">>> RUNNING FIXED-TABLE-PARSER-V3-COSTCO-YEAR-BOUNDARY-FIX <<<")

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

        return parsed_tables

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

                has_numeric = any(
                    self._looks_numeric_cell(cell["text"]) for cell in cells
                )

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

                next_has_numeric = any(
                    self._looks_numeric_cell(cell["text"]) for cell in next_cells
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

                if next_has_numeric and run_is_bold == next_is_bold:

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

        sub_regions = []
        start = 0

        for point in split_points:
            sub_regions.append(region[start:point])
            start = point

        sub_regions.append(region[start:])

        return [sub for sub in sub_regions if sub]

    # =========================================================
    # PARSE TABLE REGION
    # =========================================================

    def _parse_table_region(self, page, rows):

        header_index = self._find_header(rows)

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

                    return {
                        "page_number": page.get("page_number"),
                        "bbox": self._get_table_bbox(rows),
                        "header": [column["name"] for column in columns],
                        "rows": data_rows,
                        "header_detected": True,
                        "column_type": "year",
                        "row_count": len(data_rows),
                        "column_count": len(columns),
                    }

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

                parsed_row = self._parse_data_row(row, text_columns, value_tolerance=45.0)

                if parsed_row is not None:
                    data_rows.append(parsed_row)

            if data_rows:

                return {
                    "page_number": page.get("page_number"),
                    "bbox": self._get_table_bbox(rows),
                    "header": [column["name"] for column in text_columns],
                    "rows": data_rows,
                    "header_detected": True,
                    "column_type": "text_label",
                    "row_count": len(data_rows),
                    "column_count": len(text_columns),
                }

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

        # Step 2: collect all header-zone cells, in row (top-to-bottom)
        # order, and cluster by x-position.
        clusters = []  # each: {"x": representative_x, "parts": [text, ...]}

        for row_index in sorted(header_row_indices):

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

            parsed_rows.append({
                "cells": [
                    {"text": cell["text"], "x": cell["x"]}
                    for cell in cells
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

        print("\n====================================")
        print(" Table Parsing Completed")
        print("====================================")
        print(f"Reports Parsed        : {len(parsed_reports)}")
        print(f"Tables Parsed          : {total_tables}")
        print(f"Tables With Header     : {with_header}")
        print(f"Tables Without Header  : {total_tables - with_header}")
        print("\nOutput:")
        print(OUTPUT_DIR)