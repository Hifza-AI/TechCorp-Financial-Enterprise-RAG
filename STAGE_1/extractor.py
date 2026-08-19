import json
import re
from pathlib import Path
import pymupdf as fitz


def extract_printed_page_number(page_dict):
    """
    Page ki sabse neeche (bottom-most) short/simple text-line se
    printed page number nikaalta hai. Y-position (bbox) use karta hai,
    raw text ke line-order pe depend nahi karta -- isliye table ke
    andar koi standalone number (jaise '252') galti se match nahi
    hoga, chahe wo raw text stream mein footer se pehle aaye.
    """

    candidates = []

    for block in page_dict.get("blocks", []):
        for line in block.get("lines", []):

            bbox = line.get("bbox")
            if not bbox:
                continue

            y1 = bbox[3]

            spans = line.get("spans", [])
            text = "".join(s.get("text", "") for s in spans).strip()

            if not text:
                continue

            # Table rows/paragraphs lambi hoti hain -- footer chhota
            # hota hai. Length-check margin-agnostic hai, har company
            # ke layout pe kaam karega.
            if len(text) > 60:
                continue

            candidates.append((y1, text))

    if not candidates:
        return None

    candidates.sort(key=lambda c: -c[0])

    for y1, text in candidates[:6]:

        match = re.fullmatch(r"page\s+(\d{1,4})", text, re.IGNORECASE)
        if match:
            num = int(match.group(1))
            if num >= 1:
                return num

        if "|" in text or "–" in text:
            match = re.search(r"(?:\||–)\s*(\d{1,4})\s*$", text)
            if match:
                num = int(match.group(1))
                if num >= 1:
                    return num

        if re.fullmatch(r"\d{1,3}", text):
            num = int(text)
            if num >= 1:
                return num

    return None


def is_signatures_page(page_dict):
    """
    Detect karta hai ke is page pe "SIGNATURES" heading hai --
    ye SEC 10-K body ka official end-marker hai. Iske baad exhibits/
    attachments hote hain jinka page-numbering unreliable hota hai.
    """
    for block in page_dict.get("blocks", []):
        for line in block.get("lines", []):

            text = "".join(
                s.get("text", "") for s in line.get("spans", [])
            ).strip()

            if re.fullmatch(r"signatures?", text, re.IGNORECASE):
                return True

    return False


class PDFExtractor:

    def __init__(self):
        pass

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

    def _extract_pdf(self, company_name, pdf_path):

        try:
            document = fitz.open(pdf_path)
            pages = []

            found_signatures = False

            for page_index in range(len(document)):

                page = document.load_page(page_index)

                page_dict = page.get_text("dict")
                cleaned_blocks = self._strip_bytes(page_dict["blocks"])

                if not found_signatures and is_signatures_page(page_dict):
                    found_signatures = True

                printed_num = extract_printed_page_number(page_dict)

                pages.append({
                    # -----------------------------------------------
                    # "page_index" = PHYSICAL position in the PDF
                    # (1, 2, 3... N). Always unique, always sequential.
                    # Use this for internal joining/matching between
                    # pipeline stages.
                    #
                    # "page_number" = what's actually PRINTED in the
                    # report's footer. Used for citations. Can be None
                    # (unnumbered front-matter) or duplicate (if
                    # numbering restarts, e.g. after exhibits).
                    #
                    # "is_post_signatures" = True once we've passed the
                    # "SIGNATURES" heading -- everything after this is
                    # exhibits/attachments, not the official 10-K body.
                    # Downstream (chunk_builder) should skip these pages.
                    # -----------------------------------------------
                    "page_index": page_index + 1,
                    "page_number": printed_num,
                    "is_post_signatures": found_signatures,

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

    def _save_report(self, report):

        output_folder = Path("STAGE_1/extracted") / report["company"]
        output_folder.mkdir(parents=True, exist_ok=True)

        output_file = output_folder / (Path(report["file_name"]).stem + ".json")

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4, ensure_ascii=False)


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
        post_sig = sum(1 for p in report["pages"] if p["is_post_signatures"])
        print(f"{report['company']} -> {report['file_name']} (post-signatures pages: {post_sig})")

    if failed_files:
        print("\nFailed Files:")
        for failed in failed_files:
            print(f"  - {failed}")

    print("\nDone.")