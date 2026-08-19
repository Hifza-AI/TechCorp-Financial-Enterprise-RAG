"""
Pure PDF Trimmer (No Text Cleaning).
Trims front pages based on company config and creates clean, isolated PDFs.
"""

from pathlib import Path
import fitz  # PyMuPDF


def trim_pdf(input_path, output_path, start_page=1, end_page=None):
    src_doc = fitz.open(input_path)
    total_doc_pages = len(src_doc)

    if end_page is None or end_page == -1:
        end_page = total_doc_pages

    # Create a new blank PDF
    new_doc = fitz.open()

    # Simple page loop & insert (No cleaning/regex applied)
    for i in range(start_page - 1, end_page):
        new_doc.insert_pdf(src_doc, from_page=i, to_page=i)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    new_doc.save(str(output_path))
    new_doc.close()
    src_doc.close()

    print(f"✅ Trimmed & Saved: {input_path.name} -> {output_path}")


# =============================================================
# COMPANY START PAGE RULES
# =============================================================
COMPANY_CONFIG = {
    "Apple": {"start_page": 4},
    "JPM": {"start_page": 3},
    "COSTCO": {"start_page": 1},
    "CVS": {"start_page": 1},
    "Microsoft": {"start_page": 1},
}


if __name__ == "__main__":
    INPUT_DIR = Path("STAGE_1/data/pdfs")
    OUTPUT_DIR = Path("STAGE_1/data/pdfs_trimmed")

    if not INPUT_DIR.exists():
        print(f"❌ Input directory not found: {INPUT_DIR}")
        exit()

    # Dynamic folder iteration
    for company_folder in INPUT_DIR.iterdir():
        if company_folder.is_dir():
            company_name = company_folder.name

            # Get company start page rule (Default = 1)
            rule = COMPANY_CONFIG.get(company_name, {"start_page": 1})
            start_page = rule.get("start_page", 1)

            print(f"\n📂 Processing Folder: {company_name} (Start Page: {start_page})")

            pdf_files = list(company_folder.glob("*.pdf"))
            if not pdf_files:
                print(f"  ⚠️ No PDF files found in {company_name}")
                continue

            for pdf_path in pdf_files:
                output_path = OUTPUT_DIR / company_name / pdf_path.name
                try:
                    trim_pdf(pdf_path, output_path, start_page=start_page)
                except Exception as e:
                    print(f"  ⚠️ FAILED: {pdf_path.name} -- {e}")

    print("\n🎉 Done! Pure trimming complete.")