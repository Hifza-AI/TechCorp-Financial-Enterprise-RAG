from dataclasses import dataclass, field
from typing import List

from sec_heading_detector import HeadingCandidate
from paragraph_parser import Paragraph


@dataclass
class SectionNode:

    title: str

    level: int

    page_start: int = 0

    page_end: int = 0

    paragraphs: List[Paragraph] = field(default_factory=list)

    tables: List = field(default_factory=list)

    children: List["SectionNode"] = field(default_factory=list)


class HierarchyBuilder:

    def build(self, candidates: List[HeadingCandidate]) -> List[SectionNode]:

        root = SectionNode(
            title="ROOT",
            level=0
        )

        stack = [root]

        for item in candidates:

            if item.is_heading:

                node = SectionNode(
                    title=item.text,
                    level=item.level,
                    page_start=item.paragraph.page_number
                )

                while stack and stack[-1].level >= node.level:
                    stack.pop()

                stack[-1].children.append(node)

                stack.append(node)

            else:

                if len(stack) == 1:

                    root.paragraphs.append(item.paragraph)

                else:

                    stack[-1].paragraphs.append(item.paragraph)

        return root.children