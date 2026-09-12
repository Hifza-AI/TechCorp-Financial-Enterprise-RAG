import json
import re
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

            # -------------------------------------------------
            # Interleave this page's blocks AND tables into ONE
            # Y-sorted (top-to-bottom) sequence before processing,
            # instead of processing all blocks first and only
            # attaching tables afterward.
            #
            # BUG (confirmed on Apple 2016 page 21): that page has
            # THREE headings in sequence -- "Price Range of Common
            # Stock" -> "Holders" -> "Dividends" -- but only ONE
            # table (the price-range table), which visually sits
            # right after the FIRST heading. Attaching tables only
            # after finishing all of a page's blocks meant the table
            # always landed under whichever heading was open LAST on
            # that page ("Dividends") -- completely unrelated to what
            # the table is actually about. This is the same root
            # problem as the earlier documented Microsoft bug (table
            # landing under the wrong heading), just showing up from
            # the other direction: previously tables jumped to a
            # PREVIOUS page's leftover heading; now, without this
            # fix, they'd jump to the LAST heading on their own page
            # instead of the one they actually sit under.
            #
            # Sorting by each item's own bbox y-position (top of the
            # page = smallest y) and processing everything in that
            # true visual order means a table attaches to whichever
            # heading was open AT THE MOMENT it appears, matching how
            # a human reading the page top-to-bottom would naturally
            # associate it. Verified: "Price Range of Common Stock"
            # now correctly gets its own table; "Segment Operating
            # Performance"'s 5 child headings (Americas, Europe,
            # Greater China, Japan, Rest of Asia Pacific) each get
            # their own correct table, not a mix-up.
            # -------------------------------------------------

            sequenced_items = []

            for block in page["blocks"]:

                bbox = block.get("bbox")

                y = bbox[1] if bbox else float("inf")

                sequenced_items.append((y, "block", block))

            for table in tables_by_page.get(page_number, []):

                table_id = id(table)

                if table_id in attached_table_ids:
                    continue

                bbox = table.get("bbox")

                y = bbox[1] if bbox else float("inf")

                sequenced_items.append((y, "table", table))

            # Stable sort: ties (identical y, or both missing bbox)
            # keep their original relative order -- blocks were
            # appended before tables above, so a genuine tie falls
            # back to "heading/paragraph first", a reasonable default.
            sequenced_items.sort(key=lambda entry: entry[0])

            for y, kind, item in sequenced_items:

                if kind == "table":

                    stack[-1]["tables"].append(item)
                    attached_table_ids.add(id(item))

                    continue

                block = item

                if block["block_type"] == "heading":

                    level = block.get("level", 2)

                    is_note_marker = block.get("is_note_marker", False)

                    is_top_level_marker = block.get("is_top_level_marker", False)

                    is_prominent_boundary = block.get(
                        "is_prominent_boundary", False
                    )

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
                        stack[-1]["page_end"] = page_number
                        continue

                    # Pop back to the correct parent: anything on the
                    # stack with a level >= this heading's level is
                    # NOT an ancestor of this heading, so close it out.
                    #
                    # NEW EXCEPTION: a currently-open "Note N" heading
                    # (e.g. "Note 1 - Summary of Significant Accounting
                    # Policies") is NEVER popped just because the
                    # incoming heading happens to share the same raw
                    # level number. A Note and its own topic
                    # sub-headings ("Basis of Presentation", "Cash
                    # Equivalents", "Revenue", etc.) are styled
                    # IDENTICALLY in the PDF, so heading_detector.py has
                    # no way to tell "this is the Note's own container
                    # title" from "this is one of its topics" by
                    # styling alone -- both score as ordinary Level-3
                    # bold headings.
                    #
                    # Confirmed on Apple 2024: 7 of Apple's 13
                    # numbered notes (1, 4, 6, 7, 9, 10, 11, 12) had
                    # their own sub-topics flattening out as SIBLINGS
                    # of the Note (both landing directly under "Item
                    # 8.") instead of nesting under it, because the
                    # stack popped the Note the instant its first
                    # sub-topic heading arrived. A "Note N" heading is
                    # only closed out by ANOTHER Note-level marker
                    # (the next Note, or an Item/Part boundary) --
                    # never by a same-level GENERIC heading, which
                    # instead becomes its child.
                    # NEW EXCEPTION TO THE EXCEPTION: if the INCOMING
                    # heading is itself a prominent, ALL-CAPS,
                    # oversized standalone title -- the same styling
                    # tier as genuine Note/statement boundaries, just
                    # without any recognizable "Note N" text-pattern
                    # on it -- it should ALSO be allowed to close out
                    # a currently-open note-marker container, exactly
                    # like a real is_note_marker would. Confirmed on
                    # Intel 2019: unlike Intel's own 2025 filing, the
                    # ACTUAL section-starting heading for each Note
                    # has NO "Note N" prefix at all ("BORROWINGS",
                    # "DERIVATIVE FINANCIAL INSTRUMENTS", etc. appear
                    # bare -- the "Note N: Title" text only shows up
                    # in the Index and in inline cross-references
                    # elsewhere, never on the real heading line).
                    # Without this, "CONSOLIDATED STATEMENTS OF
                    # STOCKHOLDERS' EQUITY" never closes, and swallows
                    # every subsequent Note plus the Exhibits section
                    # at the very end of the filing as its children.
                    # Ordinary Note sub-topics ("Cash Equivalents",
                    # "Basis of Presentation") do NOT share this
                    # specific much-larger+all-caps combination, so
                    # this stays safe for the ORIGINAL exclusivity
                    # rule's purpose.
                    # NEW (Intuit 2026, confirmed via real
                    # hierarchy_outline.txt + chunks.json output): a
                    # genuine Item/PART marker (is_top_level_marker)
                    # must ALWAYS be able to pop back to the document
                    # ROOT, regardless of what level number happens to
                    # currently sit on top of the stack -- SEC filings
                    # never nest an Item or Part boundary under
                    # anything else, by definition.
                    #
                    # The existing level-comparison popping below only
                    # fires while `stack[-1]["level"] >= level` -- but
                    # a mid-document heading can occasionally render
                    # in OVERSIZED font (relative_size >= 1.5, our
                    # Level-1 threshold) purely for VISUAL emphasis,
                    # not because it's a genuine document-level title.
                    # Confirmed real-world impact: Intuit's own
                    # "Financial Highlights" infographic renders the
                    # segment name "Global Business Solutions" in
                    # large callout font (Level 1, page 53-54) --
                    # every subsequent Item boundary (Item 6, 7, 7A,
                    # 8, 9, 9A, 9B, 9C, all Level 2) then fails to
                    # pop it at all, since `1 >= 2` is false, so NONE
                    # of them can close it out through ordinary level
                    # comparison. Only the NEXT Level-1 heading
                    # ("PART III", page 125) was ever able to pop it
                    # -- meaning EVERY Note (1 through 15), all four
                    # core financial statements, and Item 9's Controls
                    # and Procedures section all ended up incorrectly
                    # nested as descendants of "Global Business
                    # Solutions" instead of their correct top-level
                    # positions, corrupting the section_path of
                    # dozens of real, financially-important chunks.
                    #
                    # This is checked and applied BEFORE the ordinary
                    # level-based popping loop below, and unconditionally
                    # clears the ENTIRE stack back to root when the
                    # incoming heading is a genuine top-level marker --
                    # there is no legitimate SEC-filing scenario where
                    # an Item/Part boundary should remain nested under
                    # a prior heading, so this can never incorrectly
                    # flatten a genuine parent-child relationship.
                    if is_top_level_marker:

                        while len(stack) > 1:
                            stack.pop()

                    while len(stack) > 1 and stack[-1]["level"] >= level:

                        if (
                            stack[-1].get("is_note_marker")
                            and not is_note_marker
                            and not is_top_level_marker
                            and not is_prominent_boundary
                        ):
                            break

                        stack.pop()

                    # NEW (AMD 2025, confirmed via real
                    # hierarchy_outline.txt output): a genuine core-
                    # statement title (a note_marker, per
                    # is_note_marker's CONSOLIDATED-title match) can
                    # repeat VERBATIM -- with NO "(Continued)" suffix
                    # at all -- at the top of a later physical page,
                    # when that ONE statement's table is long enough
                    # to span multiple pages (confirmed: AMD's
                    # Consolidated Statements of Cash Flows repeats
                    # its own exact title once for its main
                    # Operating/Investing/Financing section, pages
                    # 74-76, and AGAIN for its own "Supplemental cash
                    # flow information" section, pages 76-77, with
                    # nothing distinguishing the two headings' text at
                    # all).
                    #
                    # heading_detector.py already has a hard-reject
                    # for the "(Continued)" SUFFIX variant of this
                    # same pagination pattern (Costco 2016, PayPal
                    # 2025) -- but a bare repeat with no suffix at all
                    # is NOT a spurious/fragment heading the way
                    # "(Continued)" is; it independently scores as a
                    # perfectly legitimate heading each time, so it
                    # can't be rejected at the heading_detector stage
                    # without risking rejecting a company's ONLY
                    # (non-repeated) occurrence of that same title
                    # elsewhere. The distinguishing signal has to be
                    # STRUCTURAL instead: is this EXACT title already
                    # the most recently-closed child of the section
                    # we're about to attach into?
                    #
                    # Confirmed real-world impact: without this,
                    # "Consolidated Statements of Cash Flows" became
                    # TWO separate sibling nodes instead of one
                    # continuous section -- the main activities table
                    # (47 rows) under the first node, the supplemental
                    # disclosures table (14 rows) stranded under an
                    # unrelated second node -- structurally
                    # fragmenting one statement into two, even though
                    # every individual VALUE inside both tables
                    # remained correct.
                    #
                    # Fix: if this incoming heading is itself a
                    # note_marker, and the LAST child already appended
                    # to the section we're about to attach into has
                    # the EXACT SAME title text and is ALSO a
                    # note_marker, REOPEN that existing node (push it
                    # back onto the stack) instead of creating a brand
                    # new sibling -- so the new page's table/paragraph
                    # content continues to accumulate inside the SAME
                    # node the first occurrence already opened.
                    #
                    # Scoped narrowly to "the most recently-appended
                    # child specifically" (not a search across ALL
                    # earlier siblings) so this can never accidentally
                    # merge two genuinely-unrelated, far-apart mentions
                    # of the same title (e.g. AMD's own Item 15 Index/
                    # Exhibit-list page, which mentions "Consolidated
                    # Statements of Cash Flows" again much later purely
                    # as a cross-reference listing -- by that point,
                    # many OTHER headings have already been appended
                    # in between, so it is never the "last child" and
                    # correctly opens as its own separate node instead).
                    reopened_node = None

                    # NEW (Costco 2016, confirmed via real
                    # hierarchy_outline.txt + chunks.json output):
                    # this reopen-check originally only covered
                    # is_note_marker matches (the AMD Cash-Flows
                    # case) -- but the SAME "verbatim running-header
                    # repeats once per page" pattern also happens for
                    # genuine Item/PART markers. Confirmed on Costco:
                    # "Item 7-Management's Discussion and Analysis of
                    # Financial Condition and Results of Operations
                    # (amounts in..." repeats VERBATIM at the top of
                    # all 12 pages of its own MD&A section (pages
                    # 20-31) -- and since the dash-format fix now
                    # correctly recognizes this as is_top_level_marker
                    # =True, each repeat also now correctly triggers
                    # the "pop fully back to root" rule above, but
                    # without this extension, each one still opened
                    # as its OWN brand-new root-level sibling rather
                    # than continuing the same one, leaving 12
                    # redundant identically-titled top-level nodes
                    # instead of one clean, continuous MD&A section
                    # (a structural/cosmetic redundancy -- every real
                    # sub-topic's OWN content, e.g. "Comparable
                    # Sales", "Gross Margin", "Dividends", still
                    # correctly nests under whichever of the 12
                    # duplicate parents it physically sits under, so
                    # no data is actually lost, but the section is
                    # needlessly fragmented into 12 parallel copies
                    # instead of being unified).
                    #
                    # NEW (Google/Intel 2016, confirmed via real
                    # hierarchy_outline.txt output): this reopen-check
                    # was previously restricted to is_note_marker and
                    # is_top_level_marker headings only -- but the
                    # SAME underlying pattern (a heading's own exact
                    # title repeating verbatim, with no distinguishing
                    # "(Continued)" suffix, purely because a page-break
                    # falls in the middle of that ONE section's own
                    # content) is confirmed to ALSO happen on ordinary,
                    # non-marker bold sub-headings: Google's "Google
                    # segment" (an MD&A revenue-discussion sub-heading)
                    # and Intel's "Effective tax rate" / "Net deferred
                    # tax assets (liabilities)" (both regular Note
                    # sub-topics) each repeat verbatim once a table or
                    # page-break interrupts their own discussion.
                    #
                    # Confirmed real-world impact: each repeat split
                    # what should be ONE continuous sub-section's
                    # narrative (e.g. Intel's own explanation of ITS
                    # effective-tax-rate table, split before vs. after
                    # the table itself) into two disconnected sibling
                    # nodes -- structurally fragmenting the discussion
                    # even though every individual sentence and value
                    # remains present and correct somewhere in the
                    # tree.
                    #
                    # Removing the is_note_marker/is_top_level_marker
                    # restriction and checking ANY heading is safe
                    # specifically BECAUSE the scope stays exactly as
                    # narrow as before: only the LITERAL immediately-
                    # preceding sibling (never a search further back)
                    # with the EXACT same title text can ever trigger
                    # a reopen. Two genuinely different, intentionally
                    # short-titled sections being immediately adjacent
                    # siblings under the same parent AND sharing the
                    # exact same title is not a realistic scenario in
                    # a properly-organized SEC filing -- so this can
                    # only ever correctly reunite a genuine page-break
                    # split, never incorrectly merge two unrelated
                    # sections.
                    # NEW (Apple/Google 2016, confirmed via real
                    # hierarchy_outline.txt output): generalizing the
                    # reopen-check to ANY heading (see above) surfaced
                    # a genuine, important EXCEPTION -- a company's
                    # audit-report heading, "Report of [Firm Name],
                    # Independent Registered Public Accounting Firm",
                    # legitimately appears TWICE, back-to-back, with
                    # the EXACT same title: once for the opinion on
                    # the financial statements themselves, and again
                    # for the SEPARATE opinion on internal control
                    # over financial reporting (ICFR) -- both are
                    # complete, self-contained, independently-signed
                    # legal documents (each ending in its own "/s/
                    # [Firm]" signature and date), not a single report
                    # split across a page-break.
                    #
                    # Confirmed real-world impact if left unguarded:
                    # Apple's and Google's two genuinely-separate
                    # audit opinions would be incorrectly MERGED into
                    # one node, mixing the financial-statement
                    # opinion's text with the unrelated ICFR opinion's
                    # text as if they were one continuous document --
                    # a meaningful legal/compliance-content corruption,
                    # not just a cosmetic nesting issue.
                    #
                    # This is a narrow, explicit exclusion (matched by
                    # the same "Report of ... Accounting Firm" wording
                    # this codebase already recognizes elsewhere for
                    # is_note_marker), checked BEFORE the general
                    # reopen logic -- every other heading, marked or
                    # not, still gets the reopen treatment as normal.
                    _is_audit_report_title = bool(
                        re.search(
                            r"Report\s+of\s+.*Registered\s+Public\s+Accounting\s+Firm",
                            block["text"],
                            re.IGNORECASE,
                        )
                    )

                    existing_children = stack[-1]["children"]

                    if (
                        not _is_audit_report_title
                        and existing_children
                        and existing_children[-1]["title"] == block["text"]
                    ):
                        reopened_node = existing_children[-1]

                    if reopened_node is not None:

                        stack.append(reopened_node)

                    else:

                        node = {
                            "title": block["text"],
                            "level": level,
                            "is_note_marker": is_note_marker,
                            "is_top_level_marker": is_top_level_marker,
                            "page_start": page_number,
                            "page_end": page_number,
                            "paragraphs": [],
                            "tables": [],
                            "children": [],
                        }

                        stack[-1]["children"].append(node)
                        stack.append(node)

                else:  # paragraph

                    stack[-1]["paragraphs"].append({
                        "text": block["text"],
                        "page_number": page_number,
                        "bbox": block.get("bbox"),
                    })

                stack[-1]["page_end"] = page_number

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