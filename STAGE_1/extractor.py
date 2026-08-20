import json
from pathlib import Path
import pymupdf as fitz


class PDFExtractor:
    """
    Extracts every page of every PDF using PyMuPDF's "dict" mode --
    this preserves font-size, bold/italic flags, and bbox
    (x/y-position) for every span of text. This rich metadata is
    what makes the downstream pipeline possible:
      - HeadingDetector uses font-size + bold/italic to tell headings
        apart from body text (company-agnostic, works regardless of
        casing/wording differences between companies).
      - TableAnalyzer/TableParser use bbox (x, y positions) to group
        numbers into rows and columns.

    page_number is simply the PDF's own physical, sequential
    position (1, 2, 3... N) -- guaranteed unique and consistent.
    We deliberately do NOT try to read/match the printed footer
    page-number (e.g. "Page 59" at the bottom of a report): browser-
    print-generated PDFs can have that printed number drift out of
    sync with physical position (content can spill across a page
    boundary), making it unreliable as an internal key. Since
    citations will always point back to OUR OWN extracted PDF at
    this same physical page_number, there's no mismatch risk --
    verification always happens against the same file, at the same
    index, every time.
    """

    def __init__(self):
        pass

    # ---------------------------------------------------------
    # Removes any raw image/bytes data PyMuPDF embeds in "dict"
    # blocks -- not needed for text/heading/table analysis, and
    # bytes objects aren't JSON-serializable.
    # ---------------------------------------------------------

    def _strip_bytes(self, obj):

        if isinstance(obj, dict):
            cleaned = {}
            for key, value in obj.items():
                if isinstance(value, bytes):
                    continue
                cleaned[key] = self._strip_bytes(value)
            return cleaned

        elif isinstance(obj, list):
            return [self._strip_bytes(item) for item in obj]

        else:
            return obj

    # ---------------------------------------------------------

    def extract(self, data_folder):

        extracted_reports = []
        failed_files = []
        data_folder = Path(data_folder)

        for company_folder in sorted(data_folder.iterdir()):

            if not company_folder.is_dir():
                continue

            company_name = company_folder.name

            for pdf_file in sorted(company_folder.glob("*.pdf")):

                report = self._extract_pdf(company_name, pdf_file)

                if report is not None:
                    self._save_report(report)
                    extracted_reports.append(report)
                else:
                    failed_files.append(f"{company_name}/{pdf_file.name}")

        return extracted_reports, failed_files

    # ---------------------------------------------------------

    def _extract_pdf(self, company_name, pdf_path):

        try:
            document = fitz.open(pdf_path)

            pages = []

            for page_index in range(len(document)):

                page = document.load_page(page_index)

                # "dict" mode -- gives font size, bold/italic flags,
                # and bbox for every span. This is the rich metadata
                # everything downstream (heading detection, table
                # parsing) depends on.
                page_dict = page.get_text("dict")

                cleaned_blocks = self._strip_bytes(page_dict["blocks"])

                pages.append({
                    "page_number": page_index + 1,  # physical, sequential, always reliable
                    "width": page.rect.width,
                    "height": page.rect.height,
                    "blocks": cleaned_blocks,
                })

            document.close()

            return {
                "company": company_name,
                "file_name": pdf_path.name,
                "file_path": str(pdf_path),
                "total_pages": len(pages),
                "pages": pages,
            }

        except Exception as e:
            print(f"FAILED: {company_name}/{pdf_path.name} — {e}")
            return None

    # ---------------------------------------------------------

    def _save_report(self, report):

        output_folder = Path("STAGE_1/extracted") / report["company"]
        output_folder.mkdir(parents=True, exist_ok=True)

        output_file = output_folder / (Path(report["file_name"]).stem + ".json")

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4, ensure_ascii=False)


# =============================================================
# Main
# =============================================================

if __name__ == "__main__":

    DATA_FOLDER = "STAGE_1/data/pdfs"

    extractor = PDFExtractor()
    reports, failed_files = extractor.extract(DATA_FOLDER)

    print("\n====================================")
    print(" PDF Extraction Completed")
    print("====================================")
    print(f"\nTotal Reports  : {len(reports)}")
    print(f"Failed Reports : {len(failed_files)}")

    total_pages = sum(report["total_pages"] for report in reports)
    print(f"Total Pages    : {total_pages}")

    print("\nSaved JSON Files:")
    for report in reports:
        print(f"  {report['company']} -> {report['file_name']} ({report['total_pages']} pages)")

    if failed_files:
        print("\nFailed Files:")
        for failed in failed_files:
            print(f"  - {failed}")

    print("\nDone.")