import os
import re
import pdfplumber

# ===========================
# PATHS
# ===========================

PDF_FOLDER = r"C:\Users\riaze\Desktop\TechCorp-Financial-Enterprise-RAG\data\pdfs\Apple"

OUTPUT_FOLDER = r"C:\Users\riaze\Desktop\TechCorp-Financial-Enterprise-RAG\apple_pipeline\extracted_text"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ===========================
# PDF EXTRACTION
# ===========================

def extract_pdf(pdf_path):

    filename = os.path.basename(pdf_path)

    # Extract Year
    year_match = re.search(r"(20\d{2})", filename)

    year = year_match.group(1) if year_match else "UNKNOWN"

    company = "APPLE"

    output = []

    output.append(f"<<<COMPANY:{company}>>>")
    output.append(f"<<<YEAR:{year}>>>")
    output.append("")

    with pdfplumber.open(pdf_path) as pdf:

        for page_num, page in enumerate(pdf.pages, start=1):

            output.append(f"<<<PAGE:{page_num}>>>")
            output.append("")

            text = page.extract_text()

            if text:
                output.append(text)

            output.append("")

    txt_name = filename.replace(".pdf", ".txt")

    save_path = os.path.join(OUTPUT_FOLDER, txt_name)

    with open(save_path, "w", encoding="utf-8") as f:

        f.write("\n".join(output))

    print(f"✅ {txt_name} extracted")


# ===========================
# MAIN
# ===========================

def main():

    pdf_files = sorted(
        [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]
    )

    print(f"\nFound {len(pdf_files)} Apple reports\n")

    for pdf in pdf_files:

        extract_pdf(os.path.join(PDF_FOLDER, pdf))

    print("\n==============================")
    print("APPLE PDF EXTRACTION COMPLETE")
    print("==============================")


if __name__ == "__main__":

    main()