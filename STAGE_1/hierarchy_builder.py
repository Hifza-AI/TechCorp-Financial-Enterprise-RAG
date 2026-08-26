import json
from pathlib import Path


class HierarchyBuilder:
    """
    Takes the flat, ordered sequence of blocks from ParagraphParser
    (heading/paragraph blocks, in document order) PLUS the parsed
    tables from TableParser (matched by page_number), and builds a
    nested tree: each heading becomes a node containing its own
    paragraphs, its own tables, and any sub-heading children.

    Uses a level-based stack (level 1 = top, level 3 = deepest) --
    the same standard technique used for turning a flat outline into
    a nested tree, generalized so it works for ANY heading depth
    without hardcoding section names.
    """

    def build(self, paragraph_report, table_report):

        root = {
            "title": "ROOT",
            "level": 0,
            "paragraphs": [],
            "tables": [],
            "children": [],
        }

        stack = [root]

        # Index tables by page number so we can attach them to
        # whichever heading is "open" (current top of stack) when
        # we reach that page in the paragraph sequence.
        tables_by_page = self._index_tables_by_page(table_report)

        attached_table_ids = set()

        for page in paragraph_report["pages"]:

            page_number = page["page_number"]

            for block in page["blocks"]:

                if block["block_type"] == "heading":

                    level = block.get("level", 2)

                    # A level of 0 can occasionally slip through if a
                    # block was downgraded (e.g. by fix_paragraphs.py).
                    # Treat it as a normal paragraph instead of trying
                    # to open a heading node with an invalid level.
                    if level <= 0:
                        stack[-1]["paragraphs"].append({
                            "text": block["text"],
                            "page_number": page_number,
                            "bbox": block.get("bbox"),
                        })
                        continue

                    node = {
                        "title": block["text"],
                        "level": level,
                        "page_start": page_number,
                        "page_end": page_number,
                        "paragraphs": [],
                        "tables": [],
                        "children": [],
                    }

                    # Pop back to the correct parent: anything on the
                    # stack with a level >= this heading's level is
                    # NOT an ancestor of this heading, so close it out.
                    while len(stack) > 1 and stack[-1]["level"] >= level:
                        stack.pop()

                    stack[-1]["children"].append(node)
                    stack.append(node)

                else:  # paragraph

                    stack[-1]["paragraphs"].append({
                        "text": block["text"],
                        "page_number": page_number,
                        "bbox": block.get("bbox"),
                    })

                stack[-1]["page_end"] = page_number

            # -------------------------------------------------
            # Attach this page's tables AFTER processing this
            # page's own blocks (not before).
            #
            # BUG (confirmed on real data -- Microsoft 2022's
            # segment-revenue table appeared 4 separate times,
            # each attached to a different WRONG heading: "Credit",
            # "Uncertain Tax Positions", "More Personal Computing"
            # -- none of which are genuinely related): SEC filings
            # very commonly put a heading and its table on the SAME
            # page ("The following table shows segment revenue:"
            # immediately followed by the table). Attaching tables
            # BEFORE processing this page's blocks meant the table
            # always attached to whatever heading was left open from
            # a PREVIOUS page, never to a heading that opens on this
            # SAME page -- which is the common case. Attaching AFTER
            # this page's own headings have had a chance to open
            # fixes the common (heading-then-table-on-one-page) case.
            # -------------------------------------------------

            for table in tables_by_page.get(page_number, []):

                table_id = id(table)

                if table_id in attached_table_ids:
                    continue

                stack[-1]["tables"].append(table)
                attached_table_ids.add(table_id)

        return root["children"]

    # =========================================================
    # INDEX TABLES BY PAGE
    # =========================================================

    def _index_tables_by_page(self, table_report):

        index = {}

        for table in table_report.get("tables", []):

            page_number = table.get("page_number")

            if page_number is None:
                continue

            index.setdefault(page_number, []).append(table)

        return index


# =============================================================
# LOADERS
# =============================================================

def load_paragraph_report(company, stem, base_dir="STAGE_1/paragraphs_fixed"):

    path = Path(base_dir) / company / f"{stem}_paragraphs.json"

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_table_report(company, stem, base_dir="STAGE_1/parsed_tables"):

    path = Path(base_dir) / company / f"{stem}_parsed_tables.json"

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def discover_stems(paragraphs_dir="STAGE_1/paragraphs_fixed"):

    paragraphs_dir = Path(paragraphs_dir)

    stems = []

    for company_dir in sorted(paragraphs_dir.iterdir()):

        if not company_dir.is_dir():
            continue

        for json_file in sorted(company_dir.glob("*_paragraphs.json")):

            stem = json_file.stem.replace("_paragraphs", "")

            stems.append((company_dir.name, stem))

    return stems


# =============================================================
# SAVE
# =============================================================

def save_hierarchy(company, stem, tree, output_dir="STAGE_1/hierarchy"):

    company_dir = Path(output_dir) / company

    company_dir.mkdir(parents=True, exist_ok=True)

    output_file = company_dir / f"{stem}_hierarchy.json"

    with open(output_file, "w", encoding="utf-8") as f:

        json.dump(
            {"company": company, "file_name": stem, "sections": tree},
            f,
            indent=4,
            ensure_ascii=False,
            default=str,
        )

    print(f"Saved Hierarchy: {output_file}")


# =============================================================
# STATS (for a quick sanity check after building)
# =============================================================

def count_tree(nodes):

    section_count = 0
    paragraph_count = 0
    table_count = 0

    for node in nodes:

        section_count += 1
        paragraph_count += len(node["paragraphs"])
        table_count += len(node["tables"])

        child_sections, child_paragraphs, child_tables = count_tree(
            node["children"]
        )

        section_count += child_sections
        paragraph_count += child_paragraphs
        table_count += child_tables

    return section_count, paragraph_count, table_count


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":

    print("\n====================================")
    print(" Hierarchy Builder Started")
    print("====================================\n")

    stems = discover_stems()

    if not stems:

        print("No fixed paragraph files found.")
        print("Run fix_paragraphs.py first.")

    else:

        builder = HierarchyBuilder()

        for company, stem in stems:

            print(f"Building: {company}/{stem}")

            paragraph_report = load_paragraph_report(company, stem)
            table_report = load_table_report(company, stem)

            tree = builder.build(paragraph_report, table_report)

            save_hierarchy(company, stem, tree)

            sections, paragraphs, tables = count_tree(tree)

            print(
                f"   Sections: {sections} | "
                f"Paragraphs: {paragraphs} | "
                f"Tables: {tables}"
            )

        print("\n====================================")
        print(" Hierarchy Building Completed")
        print("====================================")
        print("\nOutput:")
        print("STAGE_1/hierarchy")