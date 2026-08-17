import fitz  # PyMuPDF

# Apne pdf file ka actual path check kar lein
pdf_path = "STAGE_1/data/pdfs/CVS/CVS_2018_10K.pdf" 

try:
    document = fitz.open(pdf_path)
    page = document.load_page(33)  # Physical Page 34 (0-indexed)

    raw_text = page.get_text("text")
    lines = raw_text.strip().split("\n")

    print(f"Total lines on this page: {len(lines)}")
    print("\n--- Last 8 lines (repr form) ---")

    for line in lines[-8:]:
        print(repr(line))

    document.close()
except Exception as e:
    print(f"Error opening file: {e}")