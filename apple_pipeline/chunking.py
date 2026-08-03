import json
import os
import re
from table_analyzer import FinancialTableAnalyzer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==========================================
# PATHS
# ==========================================

INPUT_FOLDER = r"C:\Users\riaze\Desktop\TechCorp-Financial-Enterprise-RAG\apple_pipeline\clean_text"
OUTPUT_FOLDER = r"C:\Users\riaze\Desktop\TechCorp-Financial-Enterprise-RAG\apple_pipeline\chunks"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ==========================================
# CHUNK SETTINGS
# ==========================================

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


print("===================================")
print("APPLE CHUNKING PIPELINE READY")
print("===================================")

print(f"Input Folder : {INPUT_FOLDER}")
print(f"Output Folder: {OUTPUT_FOLDER}")
print(f"Chunk Size   : {CHUNK_SIZE}")
print(f"Overlap      : {CHUNK_OVERLAP}")


# =====================================================
# STATEMENT TYPE DETECTION
# =====================================================


def detect_statement_type(heading):
    """Detects standard SEC statement/financial section types from heading text."""
    h = heading.upper()

    if "BALANCE SHEET" in h:
        return "Balance Sheet"
    elif "STATEMENTS OF OPERATIONS" in h or "STATEMENT OF OPERATIONS" in h:
        return "Income Statement"
    elif "STATEMENTS OF CASH FLOWS" in h or "STATEMENT OF CASH FLOWS" in h:
        return "Cash Flow"
    elif (
        "SHAREHOLDERS" in h
        or "STOCKHOLDERS" in h
        or "SHAREOWNERS" in h
        or "EQUITY" in h
    ):
        return "Equity Statement"
    elif "RISK FACTORS" in h:
        return "Risk Factors"
    elif h.startswith("NOTE"):
        return "Notes"
    else:
        return "General"


def contains_table(text):

    lines = text.split("\n")
    score = 0

    for line in lines:

        line = line.strip()

        if not line:
            continue

        if len(re.findall(r"\d", line)) >= 4:
            score += 2

        if "$" in line:
            score += 2

        if re.search(r"\s{2,}", line):
            score += 1

        if re.search(r"\(?[\d,]+\)?", line):
            score += 1

    return score >= 5


def extract_table_name(section):
    return section.strip()


def extract_note_number(section):

    m = re.search(r"Note\s+(\d+)", section, re.I)

    if m:
        return int(m.group(1))

    return None


def extract_line_items(text):

    items = []

    for line in text.split("\n"):

        line = line.strip()

        if not line:
            continue

        m = re.match(
            r"^([A-Za-z&(),.'/\- ]{3,80}?)\s+\$?[-\d,(]",
            line
        )

        if m:

            item = m.group(1).strip()

            item = item.rstrip(":")

            if item not in items:
                items.append(item)

    return items


# =====================================================
# PHASE 1: DYNAMIC HEADING DETECTION
# =====================================================


def is_heading(line):
    line = line.strip()

    if not line:
        return False

    # Ignore page markers
    if line.startswith("<<<PAGE:"):
        return False

    # Ignore company/year markers
    if line.startswith("<<<COMPANY"):
        return False

    if line.startswith("<<<YEAR"):
        return False

    # Too short
    if len(line) < 3:
        return False

    # Too long
    if len(line.split()) > 10:
        return False

    # Ends with period -> sentence
    if line.endswith("."):
        return False

    # Pure numbers
    if re.fullmatch(r"[\d\W]+", line):
        return False

    # Reject heavy numeric/table rows
    if len(re.findall(r"\d", line)) >= 5:
        return False

    # Reject obvious table words
    reject_words = {
        "UNITED STATES",
        "SECURITIES AND EXCHANGE COMMISSION",
        "TABLE OF CONTENTS",
        "DOCUMENTS INCORPORATED BY REFERENCE",
        "YES",
        "NO",
        "YES NO",
        "SEPTEMBER",
        "OCTOBER",
        "DECEMBER",
        "NAME",
        "TITLE",
        "DATE",
        "AMOUNT",
        "AMOUNTS",
        "FEDERAL:",
        "STATE:",
        "FOREIGN:",
        "ASSETS:",
        "LIABILITIES AND SHAREHOLDERS’ EQUITY:",
        "NUMERATOR:",
        "DENOMINATOR:",
        "SAN JOSE, CALIFORNIA",
        "SIGNATURES",
        "EXHIBIT INDEX",
    }

    if line.upper() in reject_words:
        return False

    # Reject repeated words
    words = line.lower().split()

    if len(words) >= 2 and len(set(words)) <= len(words) / 2:
        return False

    # Reject colon labels
    if line.endswith(":") and len(line.split()) <= 2:
        return False

    # SEC Item
    if re.match(r"^Item\s+\d+[A-Z]?\.", line, re.IGNORECASE):
        return True

    # PART
    if re.match(r"^PART\s+[IVX]+$", line, re.IGNORECASE):
        return True

    # NOTE
    if re.match(r"^Note\s+\d+", line, re.IGNORECASE):
        return True

    # ALL CAPS headings
    if line.isupper():
        if 1 <= len(line.split()) <= 6:
            return True

    # Title Case headings
    words = line.split()

    capitalized = sum(1 for w in words if w[:1].isupper())

    if capitalized >= len(words) * 0.9:
        if len(words) <= 6:
            return True

    return False


# =====================================================
# PHASE 1: SECTION DETECTION
# =====================================================


def detect_sections(text):
    sections = []
    lines = text.split("\n")
    current_position = 0

    for line in lines:
        clean_line = line.strip()

        if is_heading(clean_line):
            sections.append(
                {"heading": clean_line, "position": current_position}
            )

        current_position += len(line) + 1

    return sections


# =====================================================
# REMOVE TABLE OF CONTENTS
# =====================================================


def remove_table_of_contents(text):
    lines = text.split("\n")
    cleaned = []

    for line in lines:
        line = line.rstrip()
        cleaned.append(line)

    return "\n".join(cleaned)


# =====================================================
# PHASE 2: SECTION + REAL PARAGRAPH DETECTION
# =====================================================

def looks_like_table_row(line):
    """
    Detects whether a line is likely part of a financial table.
    """

    line = line.strip()

    if not line:
        return False

    # Lots of numbers
    if len(re.findall(r"\d", line)) >= 4:
        return True

    # Dollar sign
    if "$" in line:
        return True

    # Multiple spaces (table alignment)
    if re.search(r"\s{2,}", line):
        return True

    # Numbers at end
    if re.search(r"\$?\(?[\d,]+\)?\s*$", line):
        return True

    return False

def build_sections(text):
    lines = text.split("\n")

    sections = []

    company = ""
    year = ""
    current_page = 1

    # Initial preface
    current_parent = "PREFACE"
    current_heading = "PREFACE"
    current_statement_type = "General"

    current_paragraphs = []
    paragraph_buffer = []

    cover_page = True
    inside_toc = False

    for raw_line in lines:
        line = raw_line.strip()

        # ---------------------------------
        # COMPANY
        # ---------------------------------
        if line.startswith("<<<COMPANY:"):
            company = (
                line.replace("<<<COMPANY:", "").replace(">>>", "").strip()
            )
            continue

        # ---------------------------------
        # YEAR
        # ---------------------------------
        if line.startswith("<<<YEAR:"):
            year = line.replace("<<<YEAR:", "").replace(">>>", "").strip()
            continue

        # ---------------------------------
        # PAGE
        # ---------------------------------
        if line.startswith("<<<PAGE:"):
            current_page = int(
                line.replace("<<<PAGE:", "").replace(">>>", "").strip()
            )
            continue

        # ---------------------------------
        # REMOVE COVER PAGE
        # ---------------------------------
        if cover_page:
            if "TABLE OF CONTENTS" in line.upper():
                cover_page = False
                inside_toc = True
                continue
            elif line.startswith("This Annual Report"):
                cover_page = False
            else:
                continue

        # ---------------------------------
        # REMOVE TABLE OF CONTENTS
        # ---------------------------------
        if inside_toc:
            if line.startswith("This Annual Report"):
                inside_toc = False
                continue
            continue

        # ---------------------------------
        # STOP AFTER SIGNATURES
        # ---------------------------------
        if line.upper() == "SIGNATURES":
            if paragraph_buffer:
                paragraph = " ".join(paragraph_buffer).strip()
                if paragraph:
                    current_paragraphs.append(paragraph)
                paragraph_buffer = []

            if current_paragraphs:
                sections.append(
                    {
                        "company": company,
                        "year": year,
                        "page": current_page,
                        "parent_section": current_parent,
                        "section": current_heading,
                        "statement_type": current_statement_type,
                        "paragraphs": current_paragraphs,
                    }
                )

            break

        # ---------------------------------
        # HEADING FOUND
        # ---------------------------------
        if is_heading(line):
            if paragraph_buffer:
                paragraph = ""
                if any("\n" in x for x in paragraph_buffer):
                    paragraph = "".join(paragraph_buffer).strip()
                else:
                    paragraph = " ".join(paragraph_buffer).strip()


                if paragraph:
                    current_paragraphs.append(paragraph)
                paragraph_buffer = []

            if current_paragraphs:
                sections.append(
                    {
                        "company": company,
                        "year": year,
                        "page": current_page,
                        "parent_section": current_parent,
                        "section": current_heading,
                        "statement_type": current_statement_type,
                        "paragraphs": current_paragraphs,
                    }
                )

            if re.match(r"^Item\s+\d", line, re.IGNORECASE):
                current_parent = line

            current_heading = line

            # Detect only if this heading starts a NEW statement
            new_statement = detect_statement_type(line)

            # Freeze / inherit statement type
            if new_statement != "General":
                current_statement_type = new_statement

            print(current_heading, " -----> ", current_statement_type)

            current_paragraphs = []

            continue

        # ---------------------------------
        # BLANK LINE = NEW PARAGRAPH
        # ---------------------------------
        if line == "":
            if paragraph_buffer:
                paragraph = ""
                if any("\n" in x for x in paragraph_buffer):
                    paragraph = "".join(paragraph_buffer).strip()
                else:
                    paragraph = " ".join(paragraph_buffer).strip()
                if paragraph:
                    current_paragraphs.append(paragraph)
                paragraph_buffer = []

            continue

        # ---------------------------------
        # NORMAL TEXT
        # ---------------------------------
        if looks_like_table_row(line):

            paragraph_buffer.append(line + "\n")
        else:
            paragraph_buffer.append(line)
    # ---------------------------------
    # LAST PARAGRAPH
    # ---------------------------------
    if paragraph_buffer:
        paragraph = ""
        if any("\n" in x for x in paragraph_buffer):
            paragraph = "".join(paragraph_buffer).strip()
    else:
        paragraph = " ".join(paragraph_buffer).strip()

        if paragraph:
            current_paragraphs.append(paragraph)

    # ---------------------------------
    # LAST SECTION
    # ---------------------------------
    if current_paragraphs:
        sections.append(
            {
                "company": company,
                "year": year,
                "page": current_page,
                "parent_section": current_parent,
                "section": current_heading,
                "statement_type": current_statement_type,
                "paragraphs": current_paragraphs,
            }
        )

    return sections


# =====================================================
# PHASE 3: CHUNK GENERATION
# =====================================================


def generate_chunks(sections):

    chunks = []
    chunk_counter = 1

    for sec in sections:

        full_text = "\n\n".join(sec["paragraphs"])

        if not full_text.strip():
            continue

        split_chunks = splitter.split_text(full_text)

        for idx, chunk in enumerate(split_chunks):

            # -------------------------
            # Table Detection
            # -------------------------

            analyzer = FinancialTableAnalyzer(
                sec["section"],
                chunk
)
            has_table = analyzer.detect_table()

            table_name = None
            line_items = []

            if has_table:
                table_name = extract_table_name(sec["section"])
                line_items = extract_line_items(chunk)

            # -------------------------
            # Note Number
            # -------------------------

            note_number = extract_note_number(sec["section"])

            # -------------------------
            # Paragraph Type
            # -------------------------

            paragraph_type = "table" if has_table else "text"

            # -------------------------
            # Content Type
            # -------------------------

            if has_table:
                content_type = "financial_table"

            elif sec["statement_type"] == "Notes":
                content_type = "financial_note"

            elif sec["statement_type"] == "Risk Factors":
                content_type = "risk_text"

            else:
                content_type = "financial_text"

            # -------------------------
            # Enterprise Chunk
            # -------------------------

            chunks.append({

                "chunk_id": chunk_counter,

                # Document Metadata
                "company": sec["company"],
                "fiscal_year": sec["year"],

                # Chunk Metadata
                "page": sec["page"],
                "parent_section": sec["parent_section"],
                "section": sec["section"],
                "statement_type": sec["statement_type"],

                "paragraph_type": paragraph_type,
                "content_type": content_type,

                "contains_table": has_table,
                "table_name": table_name,
                "note_number": note_number,
                "line_items": line_items,

                "chunk_index": idx,

                "text": chunk

            })

            chunk_counter += 1

    return chunks


# =====================================================
# SAVE JSON
# =====================================================


def save_chunks(filename, chunks):
    output_name = filename.replace(".txt", "_chunks.json")
    output_path = os.path.join(OUTPUT_FOLDER, output_name)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=4, ensure_ascii=False)

    print(f"\nSaved : {output_path}")
    print(f"Total Chunks : {len(chunks)}")


# =====================================================
# PROCESS ALL REPORTS
# =====================================================


def process_reports():
    files = sorted(os.listdir(INPUT_FOLDER))

    for file in files:
        if not file.endswith(".txt"):
            continue

        print("\nProcessing:", file)

        path = os.path.join(INPUT_FOLDER, file)

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        text = remove_table_of_contents(text)
        sections = build_sections(text)
        chunks = generate_chunks(sections)
        save_chunks(file, chunks)


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    process_reports()