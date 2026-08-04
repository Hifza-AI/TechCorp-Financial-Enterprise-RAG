import re

from table_analyzer import TableRegion
from parsed_table import ParsedTable


class TableParser:

    def parse(
        self,
        table_regions,
    ):

        parsed_tables = []

        for region in table_regions:

            parsed = self._parse_region(
                region
            )

            parsed_tables.append(
                parsed
            )

        return parsed_tables

    # ----------------------------------------------------

    def _parse_region(
        self,
        region,
    ):

        table_name = self._extract_title(
            region.lines
        )

        rows = []

        line_items = []

        for line in region.lines:

            row = self._parse_row(line)

            if row is None:
                continue

            rows.append(row)

            line_items.append(
                row["label"]
            )

        return ParsedTable(

            table_id=region.table_id,

            table_name=table_name,

            page_number=region.page_number,

            top=region.top,

            bottom=region.bottom,

            left=region.left,

            right=region.right,

            rows=rows,

            line_items=line_items,

        )

    # ----------------------------------------------------

    def _extract_title(
        self,
        lines,
    ):

        if not lines:
            return "Unknown Table"

        first = lines[0].strip()

        if len(first.split()) <= 12:
            return first

        return "Financial Table"

    # ----------------------------------------------------

    def _parse_row(
        self,
        line,
    ):

        line = line.strip()

        if not line:
            return None

        m = re.match(

            r"^(.*?)(\(?-?\$?[\d,]+(?:\.\d+)?%?(?:\s+\(?-?\$?[\d,]+(?:\.\d+)?%?)*\)?)$",

            line

        )

        if not m:
            return None

        label = m.group(1).strip()

        values = re.findall(

            r"\(?-?\$?[\d,]+(?:\.\d+)?%?\)?",

            m.group(2)

        )

        return {

            "label": label,

            "values": values,

        }