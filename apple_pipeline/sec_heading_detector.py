import re
from dataclasses import dataclass
from typing import List

from paragraph_parser import Paragraph


@dataclass
class HeadingCandidate:

    paragraph: Paragraph

    text: str

    is_heading: bool

    level: int

    score: float


class HeadingDetector:
    """Detect heading candidates using generic document features.

    This detector is intentionally domain-independent.
    It does NOT rely on words like:
        - Item
        - Note
        - Chapter
        - Financial
        etc.
    """

    def __init__(self):

        self.max_heading_length = 120

    def detect(self, paragraphs: List[Paragraph]) -> List[HeadingCandidate]:

        results = []

        for paragraph in paragraphs:

            if paragraph.block_type not in (
                "paragraph",
                "heading",
            ):
                continue

            text = paragraph.text

            score = self._calculate_score(text)

            is_heading = score >= 0.60

            level = self._estimate_level(text, score) if is_heading else 0

            results.append(
                HeadingCandidate(
                    paragraph=paragraph,
                    text=text,
                    is_heading=is_heading,
                    level=level,
                    score=round(score, 2),
                )
            )

        return results

    # -------------------------------------------------------------

    def _calculate_score(self, text: str) -> float:

        score = 0.0

        clean = text.strip()

        if not clean:
            return 0.0

        words = clean.split()

        # ---------------------------------------------------------
        # Short text
        # ---------------------------------------------------------

        if len(words) <= 10:
            score += 0.25

        elif len(words) <= 20:
            score += 0.10

        # ---------------------------------------------------------
        # Character length
        # ---------------------------------------------------------

        if len(clean) <= self.max_heading_length:
            score += 0.10

        # ---------------------------------------------------------
        # Uppercase ratio
        # ---------------------------------------------------------

        letters = [c for c in clean if c.isalpha()]

        if letters:

            upper = sum(c.isupper() for c in letters)

            ratio = upper / len(letters)

            if ratio >= 0.80:
                score += 0.25

            elif ratio >= 0.50:
                score += 0.15

        # ---------------------------------------------------------
        # Ends without full stop
        # ---------------------------------------------------------

        if not clean.endswith("."):
            score += 0.10

        # ---------------------------------------------------------
        # Mostly alphabetic
        # ---------------------------------------------------------

        alpha = sum(ch.isalpha() for ch in clean)

        if alpha / max(len(clean), 1) > 0.60:
            score += 0.10

        # ---------------------------------------------------------
        # Numbering pattern
        # Examples:
        # 1
        # 1.1
        # A.
        # IV.
        # ---------------------------------------------------------

        if re.match(r"^[A-Za-zIVXivx0-9]+[\.\-]?", clean):
            score += 0.10

        return min(score, 1.0)

    # -------------------------------------------------------------

    def _estimate_level(self, text: str, score: float) -> int:

        words = text.split()

        if text.isupper():
            return 1

        if len(words) <= 5:
            return 2

        return 3