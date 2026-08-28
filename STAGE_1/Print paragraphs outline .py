"""
print_paragraphs_outline.py

Reads a *_paragraphs.json file (output of paragraph_parser.py) and
produces a READABLE, page-by-page outline .txt -- headings and
paragraphs shown in reading order, exactly as they'd appear in the
PDF -- so you can open the PDF side-by-side and visually confirm
headings/paragraphs are being captured correctly, without ever
scrolling through raw JSON.

USAGE (single file):
    python print_paragraphs_outline.py [path_to_one_paragraphs_json] [output_txt_path]

USAGE (batch -- whole folder, one subfolder per company):
    python print_paragraphs_outline.py [input_dir] [output_dir]
"""

import json
import sys
from pathlib import Path

# Default paths hardcoded for Apple folder
DEFAULT_INPUT_PATH = r"C:\Users\riaze\Desktop\TechCorp-Financial-Enterprise-RAG\STAGE_1\paragraphs\Apple"
DEFAULT_OUTPUT_PATH = r"C:\Users\riaze\Desktop\TechCorp-Financial-Enterprise-RAG\STAGE_1\paragraphs_outlines\Apple"


def build_outline(paragraphs_json_path, max_paragraph_chars=400):

    with open(paragraphs_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pages = data.get("pages", [])

    lines_out = []

    total_headings = 0
    total_paragraphs = 0
    total_chars = 0

    lines_out.append(f"FILE: {data.get('file_name', Path(paragraphs_json_path).name)}")
    lines_out.append(f"COMPANY: {data.get('company', '?')}")
    lines_out.append("=" * 70)
    lines_out.append("")

    for page in pages:

        page_number = page.get("page_number")

        blocks = page.get("blocks", [])

        if not blocks:
            continue

        lines_out.append(f"\n----- PAGE {page_number} " + "-" * 40)

        for block in blocks:

            block_type = block.get("block_type")

            text = block.get("text", "").strip()

            if not text:
                continue

            total_chars += len(text)

            if block_type == "heading":

                total_headings += 1

                level = block.get("level", "?")

                # Ensure level is an integer for calculation
                try:
                    level_num = int(level)
                except (ValueError, TypeError):
                    level_num = 1

                indent = "  " * (max(level_num, 1) - 1)

                lines_out.append(f"\n[H L{level}] {indent}{text}")

            else:

                total_paragraphs += 1

                display_text = text

                if len(display_text) > max_paragraph_chars:
                    display_text = display_text[:max_paragraph_chars] + " [...]"

                lines_out.append(f"    {display_text}")

    summary = []
    summary.append("")
    summary.append("")
    summary.append("=" * 70)
    summary.append("SUMMARY")
    summary.append("=" * 70)
    summary.append(f"Total pages with content : {sum(1 for p in pages if p.get('blocks'))}")
    summary.append(f"Total heading blocks     : {total_headings}")
    summary.append(f"Total paragraph blocks   : {total_paragraphs}")
    summary.append(f"Total characters (all text): {total_chars}")

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

    # If the input directory directly contains json files
    json_files = sorted(input_dir.glob("*_paragraphs.json"))

    if json_files:
        print(f"\nProcessing {input_dir.name}:")
        for json_file in json_files:
            output_file = output_dir / (json_file.stem + "_outline.txt")
            process_single_file(json_file, output_file)
        print(f"\n{'=' * 60}")
        print(f"Done. {len(json_files)} outline file(s) written to: {output_dir}")
        print(f"{'=' * 60}")
        return

    # If input directory has subdirectories per company
    total_files = 0
    for company_dir in sorted(input_dir.iterdir()):

        if not company_dir.is_dir():
            continue

        company_name = company_dir.name
        company_json_files = sorted(company_dir.glob("*_paragraphs.json"))

        if not company_json_files:
            continue

        print(f"\n{company_name}:")
        out_company_dir = output_dir / company_name

        for json_file in company_json_files:
            output_file = out_company_dir / (json_file.stem + "_outline.txt")
            process_single_file(json_file, output_file)
            total_files += 1

    print(f"\n{'=' * 60}")
    print(f"Done. {total_files} outline file(s) written to: {output_dir}")
    print(f"{'=' * 60}")


def main():

    if len(sys.argv) >= 2:
        input_path = Path(sys.argv[1])
    else:
        input_path = Path(DEFAULT_INPUT_PATH)

    if not input_path.exists():
        print(f"Path not found: {input_path}")
        sys.exit(1)

    if input_path.is_dir():

        output_dir = (
            Path(sys.argv[2])
            if len(sys.argv) >= 3
            else Path(DEFAULT_OUTPUT_PATH)
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