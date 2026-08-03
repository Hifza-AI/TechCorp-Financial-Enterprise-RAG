import os
import re
import pdfplumber

# ==========================================
# PATHS
# ==========================================

PDF_FOLDER = r"C:\Users\riaze\Desktop\TechCorp-Financial-Enterprise-RAG\data\pdfs\Apple"

OUTPUT_FOLDER = r"C:\Users\riaze\Desktop\TechCorp-Financial-Enterprise-RAG\apple_pipeline\headings"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

OUTPUT_FILE = os.path.join(
    OUTPUT_FOLDER,
    "all_headings.txt"
)

# ==========================================
# HEADING DETECTOR
# ==========================================

def is_heading(line):

    line = line.strip()

    if not line:
        return False

    if len(line) < 3:
        return False

    if len(line.split()) > 10:
        return False

    # Ignore numbers only
    if re.fullmatch(r"[\d\W]+", line):
        return False

    # Ignore ending full stop
    if line.endswith("."):
        return False

    # SEC Items
    if re.match(r"^Item\s+\d+[A-Z]?\.", line, re.IGNORECASE):
        return True

    # Notes
    if re.match(r"^Note\s+\d+", line, re.IGNORECASE):
        return True

    # PART I PART II
    if re.match(r"^PART\s+[IVX]+$", line, re.IGNORECASE):
        return True

    # ALL CAPS
    if line.isupper() and len(line.split()) <= 12:
        return True

    # Title Case
    words = line.split()

    capitalized = sum(
        1 for w in words
        if w[:1].isupper()
    )

    if capitalized >= len(words) * 0.9:
        return True

    return False


# ==========================================
# EXTRACT HEADINGS
# ==========================================

pdfs = sorted(
    [
        f for f in os.listdir(PDF_FOLDER)
        if f.endswith(".pdf")
    ]
)

print(f"\nFound {len(pdfs)} Apple Reports\n")

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as out:

    for pdf_name in pdfs:

        print(f"Processing {pdf_name}")

        out.write("\n")
        out.write("=" * 80 + "\n")
        out.write(pdf_name + "\n")
        out.write("=" * 80 + "\n\n")

        pdf_path = os.path.join(PDF_FOLDER, pdf_name)

        headings = []

        with pdfplumber.open(pdf_path) as pdf:

            for page_no, page in enumerate(pdf.pages, start=1):

                text = page.extract_text()

                if text is None:
                    continue

                for line in text.split("\n"):

                    line = line.strip()

                    if is_heading(line):

                        headings.append(
                            (page_no, line)
                        )

        # Remove duplicates while preserving order

        seen = set()

        unique = []

        for page, heading in headings:

            if heading not in seen:

                seen.add(heading)

                unique.append((page, heading))

        for page, heading in unique:

            out.write(
                f"Page {page:3} : {heading}\n"
            )

        out.write(
            f"\nTotal Headings : {len(unique)}\n\n"
        )

print("\n===================================")
print("HEADINGS EXTRACTION COMPLETE")
print("===================================")
print(f"Saved at:\n{OUTPUT_FILE}")