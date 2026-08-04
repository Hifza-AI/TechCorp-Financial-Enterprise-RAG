from dataclasses import dataclass, field
from typing import List


@dataclass
class TextBlock:

    text: str

    top: float

    bottom: float

    left: float

    right: float


@dataclass
class Page:

    page_number: int

    width: float

    height: float

    blocks: List[TextBlock] = field(default_factory=list)


class PageSplitter:

    def split(
        self,
        pdf_document,
    ):

        pages = []

        for page_index in range(len(pdf_document)):

            pdf_page = pdf_document.load_page(
                page_index
            )

            page = Page(

                page_number=page_index + 1,

                width=pdf_page.rect.width,

                height=pdf_page.rect.height,

            )

            blocks = pdf_page.get_text(
                "blocks"
            )

            for block in blocks:

                x0 = block[0]
                y0 = block[1]
                x1 = block[2]
                y1 = block[3]

                text = block[4].strip()

                if not text:
                    continue

                page.blocks.append(

                    TextBlock(

                        text=text,

                        top=y0,

                        bottom=y1,

                        left=x0,

                        right=x1,

                    )

                )

            pages.append(page)

        return pages