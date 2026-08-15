import json
from pathlib import Path
import pymupdf as fitz


class PDFExtractor:

    def __init__(self):

        pass

    # ---------------------------------------------------------

    def _strip_bytes(self, obj):
        """
        Recursively removes any bytes-type values from a nested dict/list
        structure. PyMuPDF's dict-mode output can contain bytes in
        different keys depending on the PDF's internal image/mask
        encoding (not always just "image") -- this handles ALL cases,
        not just the specific key we happened to see in Apple's PDFs.
        """
        if isinstance(obj, dict):
            cleaned = {}
            for key, value in obj.items():
                if isinstance(value, bytes):
                    continue  # drop this key entirely
                cleaned[key] = self._strip_bytes(value)
            return cleaned

        elif isinstance(obj, list):
            return [self._strip_bytes(item) for item in obj]

        else:
            return obj

    # ---------------------------------------------------------

    def extract(
        self,
        data_folder,
    ):

        extracted_reports = []
        failed_files = []

        data_folder = Path(data_folder)

        # -------------------------------------
        # Company Folders
        # -------------------------------------

        for company_folder in sorted(data_folder.iterdir()):

            if not company_folder.is_dir():
                continue

            company_name = company_folder.name

            # ---------------------------------
            # PDF Files
            # ---------------------------------

            for pdf_file in sorted(company_folder.glob("*.pdf")):

                report = self._extract_pdf(

                    company_name,

                    pdf_file,

                )

                if report is not None:

                    self._save_report(report)

                    extracted_reports.append(report)

                else:

                    failed_files.append(
                        f"{company_name}/{pdf_file.name}"
                    )

        return extracted_reports, failed_files

    # ---------------------------------------------------------

    def _extract_pdf(

        self,

        company_name,

        pdf_path,

    ):

        try:

            document = fitz.open(pdf_path)

            pages = []

            for page_index in range(len(document)):

                page = document.load_page(page_index)

                page_dict = page.get_text("dict")

                # ---------------------------------
                # Remove image bytes (Recursively)
                # ---------------------------------

                cleaned_blocks = self._strip_bytes(page_dict["blocks"])

                pages.append(

                    {

                        "page_number": page_index + 1,

                        "width": page.rect.width,

                        "height": page.rect.height,

                        "blocks": cleaned_blocks,

                    }

                )

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

    def _save_report(
        self,
        report,
    ):

        output_folder = Path(

            "STAGE_1/extracted"

        ) / report["company"]

        output_folder.mkdir(

            parents=True,

            exist_ok=True,

        )

        output_file = output_folder / (

            Path(report["file_name"]).stem + ".json"

        )

        with open(

            output_file,

            "w",

            encoding="utf-8",

        ) as f:

            json.dump(

                report,

                f,

                indent=4,

                ensure_ascii=False,

            )


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    DATA_FOLDER = "STAGE_1/data/pdfs"

    extractor = PDFExtractor()

    reports, failed_files = extractor.extract(DATA_FOLDER)

    print("\n====================================")
    print(" PDF Extraction Completed")
    print("====================================")

    print(f"\nTotal Reports : {len(reports)}")
    print(f"Failed Reports : {len(failed_files)}")

    total_pages = sum(

        report["total_pages"]

        for report in reports

    )

    print(f"Total Pages   : {total_pages}")

    print("\nSaved JSON Files:")

    for report in reports:

        print(

            f"{report['company']}"

            f" -> "

            f"{report['file_name']}"

        )

    if failed_files:

        print("\nFailed Files:")

        for failed in failed_files:

            print(f" - {failed}")

    print("\nDone.")