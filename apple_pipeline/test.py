import re
from dataclasses import dataclass
from typing import List, Optional

# Pre-built modules import karein
from paragraph_parser import ParagraphParser
from section_builder import SectionBuilder


# --- File Data Models ---
@dataclass
class TextBlock:
    text: str
    top: float      # Represented as Line Start
    bottom: float   # Represented as Line End
    left: float = 0.0
    right: float = 100.0

@dataclass
class TextPage:
    page_number: int
    blocks: List[TextBlock]

@dataclass
class TableMock:
    table_id: int
    page_number: int
    top: float
    bottom: float
    data: List[List[str]]

@dataclass
class SectionNode:
    title: str
    paragraphs: List = None
    tables: List = None
    children: List = None
    page_start: int = 0
    page_end: int = 0

    def __post_init__(self):
        if self.paragraphs is None:
            self.paragraphs = []
        if self.tables is None:
            self.tables = []
        if self.children is None:
            self.children = []


# --- File Ingestion Engine ---
def load_and_parse_text_file(file_path: str):
    pages: List[TextPage] = []
    current_page_num = 1
    current_blocks: List[TextBlock] = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    line_pointer = 1

    for line in lines:
        cleaned = line.strip()
        
        if not cleaned:
            line_pointer += 1
            continue

        # Detect Page Delimiters: <<<PAGE:X>>>
        page_match = re.search(r"<<<PAGE:(\d+)>>>", cleaned)
        if page_match:
            if current_blocks:
                pages.append(TextPage(page_number=current_page_num, blocks=current_blocks))
                current_blocks = []
            current_page_num = int(page_match.group(1))
            line_pointer += 1
            continue

        # Wrap text lines into spatial block (Line Index as Bounding Box)
        block = TextBlock(
            text=cleaned,
            top=float(line_pointer),
            bottom=float(line_pointer + 1)
        )
        current_blocks.append(block)
        line_pointer += 1

    # Push last page
    if current_blocks:
        pages.append(TextPage(page_number=current_page_num, blocks=current_blocks))

    return pages


# --- Main Pipeline Execution ---
def execute_apple_pipeline():
    file_path = r"C:\Users\riaze\Desktop\TechCorp-Financial-Enterprise-RAG\apple_pipeline\extracted_text\Apple_2021_10K.txt"

    print("==================================================")
    print("STEP 1: Reading File & Simulating Page Boundaries...")
    print("==================================================")
    pages = load_and_parse_text_file(file_path)
    print(f"✅ Extracted {len(pages)} Pages with Spatial Line Indices.")

    # Step 2: Execute Paragraph Parser
    print("\n==================================================")
    print("STEP 2: Executing Paragraph Parser...")
    print("==================================================")
    parser = ParagraphParser()
    paragraphs = parser.parse(pages)
    print(f"✅ Generated {len(paragraphs)} Paragraph Objects.")

    # Step 3: Setup Hierarchy Structure Candidates
    # (Mock Section Headings mapped from 10-K Document Structure)
    sections_hierarchy = [
        SectionNode(title="FORM 10-K"),
        SectionNode(
            title="Item 7. Management's Discussion and Analysis",
            children=[
                SectionNode(title="Products Performance"),
                SectionNode(title="Services Performance")
            ]
        )
    ]

    # Step 4: Execute Section Builder
    print("\n==================================================")
    print("STEP 3: Executing Section Builder (Spatial Merging)...")
    print("==================================================")
    builder = SectionBuilder()
    
    # Passing empty tables list for plain text file parsing
    final_sections = builder.build(sections_hierarchy, paragraphs, parsed_tables=[])

    print("\n==================================================")
    print("SUMMARY OUTPUT OF PARSED SECTIONS:")
    print("==================================================")
    
    def print_tree(sections, depth=0):
        indent = "  " * depth
        for sec in sections:
            print(f"{indent}📁 [{sec.title}] | Page Span: {sec.page_start} -> {sec.page_end} | Paragraphs Count: {len(sec.paragraphs)}")
            if sec.children:
                print_tree(sec.children, depth + 1)

    print_tree(final_sections)


if __name__ == "__main__":
    execute_apple_pipeline()