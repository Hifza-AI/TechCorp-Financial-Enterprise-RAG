"""
print_hierarchy_outline.py

Reads *_hierarchy.json files from STAGE_1/hierarchy (the output of
hierarchy_builder.py) and produces a READABLE, indented outline .txt
of the NESTED TREE -- section titles, their tables (header + row
count), and their paragraphs, all indented by nesting level -- so
you can see at a glance whether headings nested correctly and
whether each table landed under the RIGHT section, without ever
opening the raw (deeply nested) JSON.

USAGE (single file):
    python print_hierarchy_outline.py <path_to_one_hierarchy_json> [output_txt_path]

USAGE (batch -- whole folder, one subfolder per company):
    python print_hierarchy_outline.py <STAGE_1/hierarchy> <STAGE_1/hierarchy_outlines>

If run with NO arguments at all, it uses the DEFAULT_INPUT /
DEFAULT_OUTPUT paths hardcoded below -- edit those if your project
folder is somewhere else.

Example:
    python print_hierarchy_outline.py STAGE_1/hierarchy/Apple/Apple_2016_10K_hierarchy.json
    python print_hierarchy_outline.py STAGE_1/hierarchy STAGE_1/hierarchy_outlines
"""

import json
import sys
from pathlib import Path


def build_outline(hierarchy_json_path, max_paragraph_chars=300):

    with open(hierarchy_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sections = data.get("sections", [])

    lines_out = []

    stats = {"sections": 0, "paragraphs": 0, "tables": 0}

    lines_out.append(f"FILE: {data.get('file_name', Path(hierarchy_json_path).name)}")
    lines_out.append(f"COMPANY: {data.get('company', '?')}")
    lines_out.append("=" * 70)

    def walk(nodes, depth):

        for node in nodes:

            stats["sections"] += 1

            level = node.get("level", "?")

            page_start = node.get("page_start")
            page_end = node.get("page_end")

            indent = "  " * depth

            page_range = (
                f"p{page_start}"
                if page_start == page_end
                else f"p{page_start}-{page_end}"
            )

            lines_out.append(
                f"\n{indent}[SECTION L{level}] {node.get('title', '')}  ({page_range})"
            )

            for table in node.get("tables", []):

                stats["tables"] += 1

                header = table.get("header", [])
                row_count = table.get("row_count", len(table.get("rows", [])))
                col_type = table.get("column_type", "raw" if not table.get("header_detected") else "?")
                section_title = table.get("section_title")

                title_note = f" [section_title={section_title!r}]" if section_title else ""

                lines_out.append(
                    f"{indent}    TABLE ({col_type}, {row_count} rows){title_note}: {header}"
                )

            for para in node.get("paragraphs", []):

                stats["paragraphs"] += 1

                text = (para.get("text") or "").strip()

                if not text:
                    continue

                if len(text) > max_paragraph_chars:
                    text = text[:max_paragraph_chars] + " [...]"

                lines_out.append(f"{indent}    - {text}")

            walk(node.get("children", []), depth + 1)

    walk(sections, 0)

    summary = []
    summary.append("")
    summary.append("")
    summary.append("=" * 70)
    summary.append("SUMMARY")
    summary.append("=" * 70)
    summary.append(f"Total sections   : {stats['sections']}")
    summary.append(f"Total paragraphs : {stats['paragraphs']}")
    summary.append(f"Total tables     : {stats['tables']}")

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

        json_files = sorted(company_dir.glob("*_hierarchy.json"))

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

    # Defaults so the script runs directly (e.g. VS Code's Run button,
    # or plain "python print_hierarchy_outline.py") without needing
    # any terminal arguments. Still supports passing your own paths
    # as arguments if you want a different folder/file.
    DEFAULT_INPUT = r"C:\Users\riaze\Desktop\TechCorp-Financial-Enterprise-RAG\STAGE_1\hierarchy"
    DEFAULT_OUTPUT = r"C:\Users\riaze\Desktop\TechCorp-Financial-Enterprise-RAG\STAGE_1\hierarchy_outlines"

    if len(sys.argv) < 2:

        input_path = Path(DEFAULT_INPUT)
        output_dir = Path(DEFAULT_OUTPUT)

        if not input_path.exists():
            print(f"Path not found: {input_path}")
            print(
                "\n(No arguments were given, so this used the default "
                "path hardcoded at the top of main(). Edit DEFAULT_INPUT "
                "in this file if your folder is somewhere else, or pass "
                "a path as an argument instead.)"
            )
            sys.exit(1)

        process_folder(input_path, output_dir)
        return

    input_path = Path(sys.argv[1])

    if not input_path.exists():
        print(f"Path not found: {input_path}")
        sys.exit(1)

    if input_path.is_dir():

        output_dir = (
            Path(sys.argv[2])
            if len(sys.argv) >= 3
            else Path("STAGE_1/hierarchy_outlines")
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