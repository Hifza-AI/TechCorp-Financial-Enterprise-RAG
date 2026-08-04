import re
from dataclasses import dataclass
from typing import List


@dataclass
class TableRegion:

    table_id: int

    page_number: int

    top: float

    bottom: float

    left: float

    right: float

    lines: List[str]


class TableAnalyzer:

    def __init__(self):

        self.table_counter = 0

    # ---------------------------------------------------------

    def analyze(
        self,
        pages,
    ):

        table_regions = []

        for page in pages:

            page_tables = self._detect_tables_on_page(
                page
            )

            table_regions.extend(
                page_tables
            )

        return table_regions

    # ---------------------------------------------------------

    def _detect_tables_on_page(
        self,
        page,
    ):

        table_regions = []

        current_table = []

        start_top = None

        end_bottom = None

        left = None

        right = None

        page_number = page.page_number

        for block in page.blocks:

            text = block.text.strip()

            if not text:
                continue

            if self._is_table_line(text):

                if not current_table:

                    start_top = block.top

                    left = block.left

                    right = block.right

                current_table.append(text)

                end_bottom = block.bottom

                left = min(left, block.left)

                right = max(right, block.right)

            else:

                if current_table:

                    table_regions.append(

                        TableRegion(

                            table_id=self.table_counter,

                            page_number=page_number,

                            top=start_top,

                            bottom=end_bottom,

                            left=left,

                            right=right,

                            lines=current_table,

                        )

                    )

                    self.table_counter += 1

                    current_table = []

                    start_top = None

                    end_bottom = None

                    left = None

                    right = None

        # last table on page

        if current_table:

            table_regions.append(

                TableRegion(

                    table_id=self.table_counter,

                    page_number=page_number,

                    top=start_top,

                    bottom=end_bottom,

                    left=left,

                    right=right,

                    lines=current_table,

                )

            )

            self.table_counter += 1

        return table_regions

    # ---------------------------------------------------------

    def _is_table_line(
        self,
        text,
    ):

        text = text.strip()

        if not text:
            return False

        # ---------------------------------
        # Numbers
        # ---------------------------------

        numbers = re.findall(

            r"\(?-?\$?[\d,]+(?:\.\d+)?%?\)?",

            text

        )

        # ---------------------------------
        # Multiple spaces
        # ---------------------------------

        multi_spaces = len(

            re.findall(r"\s{2,}", text)

        )

        # ---------------------------------
        # Tabs
        # ---------------------------------

        tabs = text.count("\t")

        # ---------------------------------
        # Accounting values
        # ---------------------------------

        accounting = len(

            re.findall(

                r"\([\d,]+\)",

                text

            )

        )

        # ---------------------------------
        # Table score
        # ---------------------------------

        score = 0

        if len(numbers) >= 2:
            score += 2

        if multi_spaces >= 1:
            score += 1

        if tabs >= 1:
            score += 2

        if accounting >= 1:
            score += 2

        return score >= 3