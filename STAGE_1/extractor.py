import json
import re
from pathlib import Path
import pymupdf as fitz


def extract_printed_page_number(page, page_dict):
    """
    Footer ko position (bottom-most y-coordinate) se dhoondta hai,
    raw text line-order pe depend nahi karta -- isliye table ke
    andar ka koi standalone number galti se match nahi hoga.
    """
    height = page.rect.height
    footer_zone_top = height - 60

    candidates = []

    for block in page_dict.get("blocks", []):
        for line in block.get("lines", []):

            line_bbox = line.get("bbox")
            if not line_bbox:
                continue

            y0 = line_bbox[1]

            if y0 < footer_zone_top:
                continue

            text = "".join(
                span.get("text", "") for span in line.get("spans", [])
            ).strip()

            if text:
                candidates.append((y0, text))

    candidates.sort(key=lambda c: -c[0])

    for y, text in candidates:

        match = re.fullmatch(r"page\s+(\d{1,4})", text, re.IGNORECASE)
        if match:
            return int(match.group(1))

        if "|" in text or "–" in text:
            match = re.search(r"(?:\||–)\s*(\d{1,4})\s*$", text)
            if match:
                return int(match.group(1))

        if re.fullmatch(r"\d{1,3}", text):
            return int(text)

    return None


def is_signatures_page(page_dict):
    """
    Detect karta hai ke is page pe "SIGNATURES" heading hai ya nahi --
    ye SEC 10-K body ka official end-marker hai. Iske baad jo bhi
    aata hai (exhibits, certifications, attachments) unreliable
    page-numbering wala hota hai aur retrieval ke liye low-value hai.
    """
    for block in page_dict.get("blocks", []):
        for line in block.get("lines", []):

            text = "".join(
                span.get("text", "") for span in line.get("spans", [])
            ).strip()

            # Poori line "SIGNATURES" honi chahiye (case-insensitive)
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

            found_signatures = False  # ek baar True ho jaye to sab aage ke pages flag ho jayenge

            for page_index in range(len(document)):

                page = document.load_page(page_index)

                page_dict = page.get_text("dict")
                cleaned_blocks = self._strip_bytes(page_dict["blocks"])

                # ---------------------------------------------
                # Body-boundary check (SIGNATURES heading)
                # ---------------------------------------------
                if not found_signatures and is_signatures_page(page_dict):
                    found_signatures = True

                printed_num = extract_printed_page_number(page, page_dict)

                pages.append({
                    "physical_page_index": page_index + 1,
                    "printed_page_number": printed_num,          # informational only — display/citation ke liye
                    "page_number": page_index + 1,                # HAMESHA physical index — unique key, kabhi collide nahi hoga
                    "is_post_signatures": found_signatures,      # True is page se aage sab exhibits/attachments hain
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

    print(f"\nTotal Reports : {len(reports)}")
    print(f"Failed Reports : {len(failed_files)}")

    total_pages = sum(report["total_pages"] for report in reports)
    print(f"Total Pages   : {total_pages}")

    for report in reports:
        post_sig_count = sum(1 for p in report["pages"] if p["is_post_signatures"])
        print(f"{report['company']} -> {report['file_name']} (post-signatures pages: {post_sig_count})")

    if failed_files:
        print("\nFailed Files:")
        for failed in failed_files:
            print(f" - {failed}")

    print("\nDone.")