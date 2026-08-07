import json
from pathlib import Path


class CleanExporter:

    def export(self, reports):

        for report in reports:

            self._export_report(report)

    # ------------------------------------------------------

    def _export_report(self, report):

        output_folder = (
            Path("STAGE_1/readable_clean")
            / report["company"]
        )

        output_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_file = output_folder / (
            Path(report["file_name"]).stem
            + "_clean.txt"
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

                for line in page["lines"]:

                    text = line["text"].strip()

                    if text:
                        f.write(text + "\n")

        print(f"Saved {output_file}")


# ======================================================

if __name__ == "__main__":

    cleaned_reports = []

    cleaned_dir = Path("STAGE_1/cleaned")

    for company in cleaned_dir.iterdir():

        if company.is_dir():

            for file in company.glob("*.json"):

                with open(file, "r", encoding="utf-8") as f:

                    cleaned_reports.append(json.load(f))

    CleanExporter().export(cleaned_reports)

    print("\nClean Export Finished")