import os
import re
import json

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
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]
)


print("===================================")
print("APPLE CHUNKING PIPELINE READY")
print("===================================")

print(f"Input Folder : {INPUT_FOLDER}")
print(f"Output Folder: {OUTPUT_FOLDER}")
print(f"Chunk Size   : {CHUNK_SIZE}")
print(f"Overlap      : {CHUNK_OVERLAP}")

# =====================================================
# PHASE 1
# DYNAMIC HEADING DETECTION
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

    capitalized = sum(
        1
        for w in words
        if w[:1].isupper()
    )

    if capitalized >= len(words) * 0.9:

        if len(words) <= 6:
            return True

    return False

# =====================================================
# PHASE 1
# SECTION DETECTION
# =====================================================

def detect_sections(text):

    sections = []

    lines = text.split("\n")

    current_position = 0

    for line in lines:

        clean_line = line.strip()

        if is_heading(clean_line):

            sections.append({

                "heading": clean_line,

                "position": current_position

            })

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
# PHASE 2
# SECTION + REAL PARAGRAPH DETECTION
# =====================================================

def build_sections(text):

    lines = text.split("\n")

    sections = []

    company = ""
    year = ""
    current_page = 1

    # Initial preface
    current_parent = "PREFACE"
    current_heading = "PREFACE"

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
            company = line.replace("<<<COMPANY:", "").replace(">>>", "").strip()
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

                sections.append({

                    "company": company,
                    "year": year,
                    "page": current_page,
                    "parent_section": current_parent,
                    "section": current_heading,
                    "paragraphs": current_paragraphs

                })

            break

        # ---------------------------------
        # HEADING FOUND
        # ---------------------------------
        if is_heading(line):

            if paragraph_buffer:

                paragraph = " ".join(paragraph_buffer).strip()

                if paragraph:
                    current_paragraphs.append(paragraph)

                paragraph_buffer = []

            if current_paragraphs:

                sections.append({

                    "company": company,
                    "year": year,
                    "page": current_page,
                    "parent_section": current_parent,
                    "section": current_heading,
                    "paragraphs": current_paragraphs

                })

            if re.match(r"^Item\s+\d", line, re.IGNORECASE):
                current_parent = line

            current_heading = line
            current_paragraphs = []

            continue

        # ---------------------------------
        # BLANK LINE = NEW PARAGRAPH
        # ---------------------------------
        if line == "":

            if paragraph_buffer:

                paragraph = " ".join(paragraph_buffer).strip()

                if paragraph:
                    current_paragraphs.append(paragraph)

                paragraph_buffer = []

            continue

        # ---------------------------------
        # NORMAL TEXT
        # ---------------------------------
        paragraph_buffer.append(line)

    # ---------------------------------
    # LAST PARAGRAPH
    # ---------------------------------
    if paragraph_buffer:

        paragraph = " ".join(paragraph_buffer).strip()

        if paragraph:
            current_paragraphs.append(paragraph)

    # ---------------------------------
    # LAST SECTION
    # ---------------------------------
    if current_paragraphs:

        sections.append({

            "company": company,
            "year": year,
            "page": current_page,
            "parent_section": current_parent,
            "section": current_heading,
            "paragraphs": current_paragraphs

        })

    return sections

# =====================================================
# PHASE 3
# CHUNK GENERATION
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

            chunks.append({

                "chunk_id": chunk_counter,

                "company": sec["company"],

                "year": sec["year"],

                "page": sec["page"],

                "parent_section": sec["parent_section"],

                "section": sec["section"],

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
# TEST PARAGRAPH DETECTION
# =====================================================

def test_section_detection():

    files = sorted(os.listdir(INPUT_FOLDER))

    if len(files) == 0:
        print("No files found.")
        return

    file_path = os.path.join(INPUT_FOLDER, files[0])

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    sections = build_sections(text)

    print("\n" + "=" * 80)
    print(files[0])
    print("=" * 80)

    for sec in sections[:10]:

        print("\n" + "-" * 70)

        print("Company :", sec["company"])
        print("Year :", sec["year"])
        print("Page :", sec["page"])
        print("Parent :", sec["parent_section"])
        print("Section :", sec["section"])

        print("Paragraph Count :", len(sec["paragraphs"]))

        print()

        for p in sec["paragraphs"][:3]:

            print("•", p[:200])

    print("\n" + "=" * 80)
    print("Total Sections :", len(sections))
    print("=" * 80)

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