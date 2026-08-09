import json
from pathlib import Path


class TableParser:

    # Standalone symbols that should NEVER be treated as part of a label.
    # These commonly sit just to the left of the first data column
    # (e.g. "$" before a number) and were leaking into labels before.
    NON_LABEL_TOKENS = {"$", "-", "--", "—", "%", "(", ")"}

    def __init__(
        self,
        y_tolerance=6.0,
        x_tolerance=15.0,
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

            # Large vertical gap = likely a new/different table
            if gap <= 30:
                current_region.append(current)
            else:
                if current_region:
                    regions.append(current_region)
                current_region = [current]

        if current_region:
            regions.append(current_region)

        return regions

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
                        "row_count": len(data_rows),
                        "column_count": len(columns),
                    }

        # -----------------------------------------------------
        # No reliable header -> raw fallback
        # -----------------------------------------------------

        return self._parse_without_header(page, rows)

    # =========================================================
    # HEADER DETECTION
    # =========================================================

    def _find_header(self, rows):

        best_index = None
        best_score = 0

        # Table header is usually near the beginning of the region
        for index, row in enumerate(rows[:4]):

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

    def _parse_data_row(self, row, columns):

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

            if cell["x"] < (first_column_x - self.x_tolerance):
                label_parts.append(text)

        label = " ".join(label_parts).strip()

        # -----------------------------------------------------
        # Map values to columns by nearest x-position
        # -----------------------------------------------------

        values = {}

        for column in columns:

            target_x = column["x"]

            matching_cells = []

            for cell in cells:

                text = cell["text"].strip()

                # Symbols alone should never "win" a column match
                if text in self.NON_LABEL_TOKENS:
                    continue

                distance = abs(cell["x"] - target_x)

                if distance <= self.x_tolerance:
                    matching_cells.append((distance, cell))

            if matching_cells:

                matching_cells.sort(key=lambda item: item[0])

                best_cell = matching_cells[0][1]

                values[column["name"]] = best_cell["text"]

            else:

                values[column["name"]] = None

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