"""
print_headings_outline.py  (BATCH VERSION)

Scans an ENTIRE heading_detection folder (all companies, all years) and
produces a readable, indented OUTLINE .txt file for every *_headings.json
file it finds -- so you never have to open a single raw JSON file to see
"which headings got detected". Output is organized into one subfolder per
company, mirroring the input folder structure.

USAGE:
    python print_headings_outline.py [input_heading_detection_dir] [output_dir]
"""

import json
import sys
from pathlib import Path


def build_outline(headings_json_path):

    with open(headings_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pages = data.get("pages", [])

    lines_out = []

    total_lines = 0
    total_headings = 0
    level_counts = {}

    for page in pages:

        page_number = page.get("page_number")

        heading_analysis = page.get("heading_analysis", {})

        candidates = heading_analysis.get("candidates", [])

        total_lines += len(candidates)

        for candidate in candidates:

            if not candidate.get("is_heading"):
                continue

            level = candidate.get("level", 1)

            total_headings += 1
            level_counts[level] = level_counts.get(level, 0) + 1

            indent = "  " * (level - 1)

            text = candidate.get("text", "").strip()

            lines_out.append(
                f"p{page_number:>4}  L{level}  {indent}{text}"
            )

    summary = []
    summary.append("")
    summary.append("=" * 60)
    summary.append("SUMMARY")
    summary.append("=" * 60)
    summary.append(f"File                : {Path(headings_json_path).name}")
    summary.append(f"Total lines scanned : {total_lines}")
    summary.append(f"Total headings found: {total_headings}")

    if total_lines:
        ratio = round(total_headings / total_lines * 100, 2)
        summary.append(f"Heading ratio       : {ratio}%")

    for level in sorted(level_counts):
        summary.append(f"  Level {level}: {level_counts[level]} headings")

    return lines_out + summary


def process_single_file(input_path, output_path):

    outline_lines = build_outline(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(outline_lines))

    print(f"  Saved: {output_path}")


def process_folder(input_dir, output_dir):

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        print(f"Input folder not found: {input_dir}")
        sys.exit(1)

    total_files = 0

    for company_dir in sorted(input_dir.iterdir()):

        if not company_dir.is_dir():
            continue

        company_name = company_dir.name

        json_files = sorted(company_dir.glob("*_headings.json"))

        if not json_files:
            continue

        print(f"\n{company_name}:")

        out_company_dir = output_dir / company_name

        for json_file in json_files:

            output_file = out_company_dir / (
                json_file.stem + "_outline.txt"
            )

            process_single_file(json_file, output_file)

            total_files += 1

    print(f"\n{'=' * 60}")
    print(f"Done. {total_files} outline file(s) written to: {output_dir}")
    print(f"{'=' * 60}")


def main():
    
    # Default paths hardcoded
    default_input_path = r"C:\Users\riaze\Desktop\TechCorp-Financial-Enterprise-RAG\STAGE_1\heading_detection"
    default_output_path = r"C:\Users\riaze\Desktop\TechCorp-Financial-Enterprise-RAG\STAGE_1\headings_outlines"

    if len(sys.argv) >= 2:
        input_path = Path(sys.argv[1])
    else:
        input_path = Path(default_input_path)

    if not input_path.exists():
        print(f"Path not found: {input_path}")
        sys.exit(1)

    if input_path.is_dir():

        output_dir = (
            Path(sys.argv[2])
            if len(sys.argv) >= 3
            else Path(default_output_path)
        )

        process_folder(input_path, output_dir)

    else:

        output_path = (
            Path(sys.argv[2])
            if len(sys.argv) >= 3
            else Path(input_path.stem + "_outline.txt")
        )

        process_single_file(input_path, output_path)


if __name__ == "__main__":
    main()