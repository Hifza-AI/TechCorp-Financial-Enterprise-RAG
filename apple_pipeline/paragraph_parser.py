import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class Paragraph:

    id: int

    text: str

    page_number: int

    paragraph_index: int

    block_type: str

    top: float

    bottom: float

    left: float

    right: float

    table_id: Optional[int] = None

    section_hint: Optional[str] = None


class ParagraphParser:

    def __init__(self):

        self.footer_patterns = [
            r"^Apple Inc\.?$",
            r"^Page\s+\d+$",
            r"^\d+$",
            r"^See accompanying Notes",
            r"^<<<PAGE:.*>>>$",
        ]

    # ---------------------------------------------------------

    def is_footer(self, line):

        line = line.strip()

        for pattern in self.footer_patterns:

            if re.match(pattern, line, re.IGNORECASE):
                return True

        return False

    # ---------------------------------------------------------

    def is_table_line(self, line):

        words = line.split()

        if len(words) < 2:
            return False

        numbers = sum(1 for w in words if re.match(r"^[\$\(\)\d\.,%-]+$", w))

        return numbers >= len(words) * 0.5

    # ---------------------------------------------------------

    def parse(self, pages):

        paragraphs = []

        current = []

        current_top = None

        current_bottom = None

        current_left = None

        current_right = None

        current_page = 1

        paragraph_id = 0

        paragraph_index = 0

        table_counter = 0

        for page in pages:

            current_page = page.page_number

            for block in page.blocks:

                line = block.text.strip()

                if not line:
                    continue

                # keep page marker as its own paragraph

                if line.startswith("<<<PAGE:"):

                    if current:
                        paragraphs.append(
                            Paragraph(
                                id=paragraph_id,
                                text=" ".join(current),
                                page_number=current_page,
                                paragraph_index=paragraph_index,
                                block_type="paragraph",
                                top=current_top,
                                bottom=current_bottom,
                                left=current_left,
                                right=current_right,
                            )
                        )

                        paragraph_id += 1

                        paragraph_index += 1

                        current = []

                        current_top = None

                        current_bottom = None

                        current_left = None

                        current_right = None

                    m = re.search(r"PAGE:(\d+)", line)

                    if m:
                        current_page = int(m.group(1))

                    paragraphs.append(
                        Paragraph(
                            id=paragraph_id,
                            text=line,
                            page_number=current_page,
                            paragraph_index=paragraph_index,
                            block_type="page_marker",
                            top=block.top,
                            bottom=block.bottom,
                            left=block.left,
                            right=block.right,
                        )
                    )

                    paragraph_id += 1

                    paragraph_index += 1

                    continue

                if self.is_footer(line):
                    continue

                # keep company marker

                if line.startswith("<<<COMPANY"):

                    paragraphs.append(
                        Paragraph(
                            id=paragraph_id,
                            text=line,
                            page_number=current_page,
                            paragraph_index=paragraph_index,
                            block_type="company_marker",
                            top=block.top,
                            bottom=block.bottom,
                            left=block.left,
                            right=block.right,
                        )
                    )

                    paragraph_id += 1

                    paragraph_index += 1

                    continue

                if line.startswith("<<<YEAR"):

                    paragraphs.append(
                        Paragraph(
                            id=paragraph_id,
                            text=line,
                            page_number=current_page,
                            paragraph_index=paragraph_index,
                            block_type="year_marker",
                            top=block.top,
                            bottom=block.bottom,
                            left=block.left,
                            right=block.right,
                        )
                    )

                    paragraph_id += 1

                    paragraph_index += 1

                    continue

                # keep tables together

                if self.is_table_line(line):

                    if current:
                        paragraphs.append(
                            Paragraph(
                                id=paragraph_id,
                                text=" ".join(current),
                                page_number=current_page,
                                paragraph_index=paragraph_index,
                                block_type="paragraph",
                                top=current_top,
                                bottom=current_bottom,
                                left=current_left,
                                right=current_right,
                            )
                        )

                        paragraph_id += 1

                        paragraph_index += 1

                        current = []

                        current_top = None

                        current_bottom = None

                        current_left = None

                        current_right = None

                    paragraphs.append(
                        Paragraph(
                            id=paragraph_id,
                            text=line,
                            page_number=current_page,
                            paragraph_index=paragraph_index,
                            block_type="table",
                            top=block.top,
                            bottom=block.bottom,
                            left=block.left,
                            right=block.right,
                            table_id=table_counter,
                        )
                    )

                    table_counter += 1

                    paragraph_id += 1

                    paragraph_index += 1

                    continue

                # heading usually starts new paragraph

                if len(line.split()) <= 10 and line == line.title():

                    if current:
                        paragraphs.append(
                            Paragraph(
                                id=paragraph_id,
                                text=" ".join(current),
                                page_number=current_page,
                                paragraph_index=paragraph_index,
                                block_type="paragraph",
                                top=current_top,
                                bottom=current_bottom,
                                left=current_left,
                                right=current_right,
                            )
                        )

                        paragraph_id += 1

                        paragraph_index += 1

                        current = []

                        current_top = None

                        current_bottom = None

                        current_left = None

                        current_right = None

                    paragraphs.append(
                        Paragraph(
                            id=paragraph_id,
                            text=line,
                            page_number=current_page,
                            paragraph_index=paragraph_index,
                            block_type="heading",
                            top=block.top,
                            bottom=block.bottom,
                            left=block.left,
                            right=block.right,
                        )
                    )

                    paragraph_id += 1

                    paragraph_index += 1

                    continue

                current.append(line)

                if current_top is None:

                    current_top = block.top

                    current_left = block.left

                    current_right = block.right

                current_bottom = block.bottom

                current_left = min(current_left, block.left)

                current_right = max(current_right, block.right)

        if current:

            paragraphs.append(
                Paragraph(
                    id=paragraph_id,
                    text=" ".join(current),
                    page_number=current_page,
                    paragraph_index=paragraph_index,
                    block_type="paragraph",
                    top=current_top,
                    bottom=current_bottom,
                    left=current_left,
                    right=current_right,
                )
            )

        return paragraphs