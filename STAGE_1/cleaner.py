import json
from copy import deepcopy
from pathlib import Path

from cleaning_rules import CleaningRules


class TextCleaner:

    def __init__(self):
        pass

    # ---------------------------------------------------------
    def clean(self, extracted_reports):
        cleaned_reports = []

        for report in extracted_reports:
            cleaned_report = self._clean_report(report)
            cleaned_reports.append(cleaned_report)

        return cleaned_reports

    # ---------------------------------------------------------
    def _clean_report(self, report):
        cleaned_pages = []

        for page in report["pages"]:
            cleaned_page = self._clean_page(page)
            cleaned_pages.append(cleaned_page)

        cleaned_report = deepcopy(report)
        cleaned_report["pages"] = cleaned_pages

        # Requested Line Addition
        cleaned_report = CleaningRules.clean_dict_structure(cleaned_report)

        return cleaned_report

    # ---------------------------------------------------------
    def _clean_page(self, page):
        cleaned_lines = []

        for block in page.get("blocks", []):
            # Only Text Blocks (Type 0)
            if block.get("type") != 0:
                continue

            for line in block.get("lines", []):
                parsed_line = self._parse_line(line)

                if parsed_line is not None:
                    cleaned_lines.append(parsed_line)

        return {
            "page_number": page["page_number"],
            "width": page["width"],
            "height": page["height"],
            "lines": cleaned_lines,
        }

    # ---------------------------------------------------------
    def _parse_line(self, line):
        spans = line.get("spans", [])

        if not spans:
            return None

        line_text = []
        span_metadata = []

        for span in spans:
            raw_text = span.get("text", "")

            text = raw_text.strip()

            if not text:
                continue

            line_text.append(text)

            span_metadata.append(
                {
                    "text": text,
                    "font": span.get("font"),
                    "size": span.get("size"),
                    "flags": span.get("flags"),
                    "bbox": span.get("bbox"),
                    "origin": span.get("origin"),
                }
            )

        if not line_text:
            return None

        full_text = " ".join(line_text).strip()

        if not full_text:
            return None

        return {
            "text": full_text,
            "bbox": line.get("bbox"),
            "direction": line.get("dir"),
            "spans": span_metadata,
        }

    # ---------------------------------------------------------
    def save_cleaned_reports(self, cleaned_reports, output_base_dir="STAGE_1/cleaned"):
        for report in cleaned_reports:
            output_folder = Path(output_base_dir) / report["company"]
            output_folder.mkdir(parents=True, exist_ok=True)

            output_file = output_folder / (
                Path(report["file_name"]).stem + "_cleaned.json"
            )

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(
                    report,
                    f,
                    indent=4,
                    ensure_ascii=False,
                    default=str
                )

            print(f"Saved Cleaned JSON: {output_file}")


# =========================================================
# Execution / Driver
# =========================================================
if __name__ == "__main__":

    extracted_dir = Path("STAGE_1/extracted")
    extracted_reports = []

    # Pehle stage ki extracted JSON files load karna
    if extracted_dir.exists():
        for company_dir in extracted_dir.iterdir():
            if company_dir.is_dir():
                for json_file in company_dir.glob("*.json"):
                    with open(json_file, "r", encoding="utf-8") as f:
                        extracted_reports.append(json.load(f))

    if extracted_reports:
        cleaner = TextCleaner()
        cleaned_reports = cleaner.clean(extracted_reports)
        cleaner.save_cleaned_reports(cleaned_reports)
        print("\nStructured Line-Parsing completed successfully!")
    else:
        print("STAGE_1/extracted folder mein koi files nahi milin. Pehle Extractor run karein.")