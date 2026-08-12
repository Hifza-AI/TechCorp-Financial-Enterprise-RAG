import json
from pathlib import Path


class RawExporter:

    def export(self, reports):

        for report in reports:

            self._export_report(report)

    # ------------------------------------------------------

    def _export_report(self, report):

        output_folder = (
            Path("STAGE_1/readable_raw")
            / report["company"]
        )

        output_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_file = output_folder / (
            Path(report["file_name"]).stem
            + "_raw.txt"
        )

        with open(
            output_file,
            "w",
            encoding="utf-8",
        ) as f:

            for page in report["pages"]:

                f.write(
                    f"\n{'='*70}\n"
                )

                f.write(
                    f"PAGE {page['page_number']}\n"
                )

                f.write(
                    f"{'='*70}\n\n"
                )

                for block in page.get("blocks", []):

                    if block.get("type") != 0:
                        continue

                    for line in block.get("lines", []):

                        spans = line.get("spans", [])

                        text = " ".join(
                            span.get("text", "")
                            for span in spans
                        ).strip()

                        if text:
                            f.write(text + "\n")

        print(f"Saved {output_file}")


# ======================================================

if __name__ == "__main__":

    extracted_reports = []

    extracted_dir = Path("STAGE_1/extracted")

    for company in extracted_dir.iterdir():

        if company.is_dir():

            for file in company.glob("*.json"):

                with open(file, "r", encoding="utf-8") as f:

                    extracted_reports.append(json.load(f))

    RawExporter().export(extracted_reports)

    print("\nRaw Export Finished")