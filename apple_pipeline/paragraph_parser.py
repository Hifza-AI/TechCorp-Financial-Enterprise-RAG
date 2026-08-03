import re


class ParagraphParser:
    """
    Enterprise Paragraph Parser

    Responsibilities:
    -----------------
    1. Remove headers/footers
    2. Merge broken OCR lines
    3. Split into logical paragraphs
    4. Ignore table blocks
    """


    def __init__(self):

        self.footer_patterns = [

            r"^Apple Inc\.?$",

            r"^See accompanying Notes",

            r"^Page\s+\d+$",

            r"^\d+$",

        ]


    # -------------------------------------
    # Footer / Header Detection
    # -------------------------------------

    def is_footer(self, line):

        line = line.strip()

        for pattern in self.footer_patterns:

            if re.search(pattern, line, re.IGNORECASE):

                return True

        return False


    # -------------------------------------
    # Broken Line Merge
    # -------------------------------------

    def normalize_lines(self, lines):

        merged = []

        buffer = ""

        for line in lines:

            line = line.strip()

            if not line:
                continue

            if self.is_footer(line):
                continue

            # Ignore table placeholders
            if line.startswith("<<<TABLE"):
                continue

            # Remove soft hyphenation
            if buffer.endswith("-"):

                buffer = buffer[:-1] + line

                continue

            # Sentence continues
            if buffer and not re.search(r"[.!?:]$", buffer):

                buffer += " " + line

            else:

                if buffer:

                    merged.append(buffer)

                buffer = line

        if buffer:

            merged.append(buffer)

        return merged


    # -------------------------------------
    # Paragraph Detection
    # -------------------------------------

    def parse(self, text):

        raw_lines = text.split("\n")

        normalized = self.normalize_lines(raw_lines)

        paragraphs = []

        current = []

        for line in normalized:

            if line == "":

                if current:

                    paragraphs.append(
                        " ".join(current).strip()
                    )

                    current = []

                continue

            current.append(line)

        if current:

            paragraphs.append(
                " ".join(current).strip()
            )

        return paragraphs