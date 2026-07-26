import pdfplumber
import re

pdf_path = r"C:\Users\riaze\Desktop\TechCorp-Financial-Enterprise-RAG\data\pdfs\Apple\Apple_2022_10K.pdf"

headings = []

with pdfplumber.open(pdf_path) as pdf:

    for page_no, page in enumerate(pdf.pages):

        text = page.extract_text()

        if text is None:
            continue

        lines = text.split("\n")

        for line in lines:

            line = line.strip()

            if len(line) < 4:
                continue

            # Item headings
            if re.match(r"Item\s+\d+[A-Z]?\.", line, re.IGNORECASE):
                headings.append((page_no + 1, line))

            # Capital headings
            elif (
                line.isupper()
                and len(line) < 80
                and len(line.split()) <= 8
            ):
                headings.append((page_no + 1, line))

print("=" * 80)
print("HEADINGS FOUND")
print("=" * 80)

for page, heading in headings:
    print(f"Page {page:3} : {heading}")