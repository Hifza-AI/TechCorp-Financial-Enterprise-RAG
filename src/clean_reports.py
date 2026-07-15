import os
import re

# =====================================================
# 1. Remove URLs
# =====================================================
def remove_urls(text):
    return re.sub(r"https?://\S+", "", text)

# =====================================================
# 2. Remove Timestamps
# =====================================================
def remove_timestamps(text):
    pattern = r"\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}\s?(AM|PM)"
    return re.sub(pattern, "", text)

# =====================================================
# 3. Remove Page Numbers
# =====================================================
def remove_page_numbers(text):

    patterns = [
        r"^\s*\d+\.?\s*$",
        r"\b\d+\s*/\s*\d+\b"
    ]

    for p in patterns:
        text = re.sub(p, "", text)

    return text

# =====================================================
# 4. Remove Checkboxes
# =====================================================
def remove_checkboxes(text):
    return re.sub(r"\[x\]|\[_\]|[☐☑☒□✓✔]", "", text, flags=re.IGNORECASE)

# =====================================================
# 5. Remove Report Headers
# =====================================================
def remove_report_headers(text):

    patterns = [

        r"Apple Inc\.\s*\|\s*\d{4}\s*Form\s*10-K\s*\|\s*\d+",
        r"Apple Inc\.",
        r"Form\s*10-K",
        r"For the Fiscal Year Ended.*",
        r"JPMorgan Chase\s*&\s*Co\./\d{4}\s*\d+",
        r"Bank of America\s+\d+",
        r"\d+\s+Bank of America",
        r"^\s*10-K\s*$",
        r"Pfizer Inc\.\s*\d{4}\s*\d+",
        r"Goldman Sachs\s+\d{4}\s+\d+",
        r"\d+\s+Goldman Sachs\s+\d{4}",
        r"Alphabet Inc\.",
        r"HP INC\. AND SUBSIDIARIES",
        r"\d+\s+MASTERCARD\s+\d{4}",
        r"MASTERCARD\s+\d{4}\s+\d+",
        r"McDonald's Corporation\s+\d{4}\s+Annual Report\s+\d+",
        r"Medtronic plc",
        r"NVIDIA CORPORATION AND SUBSIDIARIES",
        r"FY\s+\d{4}\s+\d+",
        r"PayPal Holdings,\s*Inc\.",
        r"\d{4}\s+Annual Report\s+\d+",
        r"TARGET CORPORATION\s+\d{4}\s+\d+",
        r"UNITED PARCEL SERVICE,\s*INC\.\s*AND\s*SUBSIDIARIES",
        r"VISA INC\.",

    ]

    for p in patterns:
        text = re.sub(p, "", text, flags=re.IGNORECASE)

    return text

# =====================================================
# 6. Remove Browser Artifacts
# =====================================================
def remove_browser_artifacts(text):

    patterns = [

        r"[A-Za-z0-9_-]+\.htm",
        r"[A-Za-z0-9_-]+\.html",
        r"^\s*Document\d*\s*$",
        r"\d+\s+JPMorgan Chase\s*&\s*Co\./\d{4}",
        r"\b[a-z]{1,6}-\d{8}\b",
        r"\d{8}\s*10K\s*FY_Taxonomy\d{4}",
        r"(https?://)?sec\.gov/Archives/edgar/data/\S*",


    ]

    for p in patterns:
        text = re.sub(p, "", text, flags=re.MULTILINE | re.IGNORECASE)

    return text

# =====================================================
# 7. Remove Table of Contents
# =====================================================
def remove_table_of_contents(text):

    return re.sub(
        r"Table\s+of\s+Contents",
        "",
        text,
        flags=re.IGNORECASE,
    )

# ======================================================
# 8. Normalize Spaces
# ======================================================
def normalize_spaces(text):
    return re.sub(r"[ \t]{2,}", " ", text)

# ======================================================
# 9. Normalize Blank Lines
# ======================================================
def normalize_blank_lines(text):
    return re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

# =====================================================
# 10. Normalize Unicode
# =====================================================
def normalize_unicode(text):

    replacements = {

        "•": "-",
        "–": "-",
        "—": "-",
        "…": "...",
        "\u00A0": " "

    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text

# =====================================================
# 11. Normalize Quotes
# =====================================================
def normalize_quotes(text):

    text = text.replace("“", '"')
    text = text.replace("”", '"')
    text = text.replace("‘", "'")
    text = text.replace("’", "'")

    return text

# =====================================================
# 12. Normalize Tabs
# =====================================================
def normalize_tabs(text):
    return text.replace("\t", " ")

# =====================================================
# 13. Remove Lonely Page Numbers
# =====================================================
def remove_lonely_page_numbers(text):
    return re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)

# =====================================================
# MASTER PIPELINE
# =====================================================
def clean_text(text):

    text = remove_urls(text)
    text = remove_timestamps(text)
    text = remove_page_numbers(text)
    text = remove_checkboxes(text)
    text = remove_report_headers(text)
    text = remove_browser_artifacts(text)
    text = remove_table_of_contents(text)
    text = normalize_spaces(text)
    text = normalize_blank_lines(text)
    text = normalize_unicode(text)
    text = normalize_quotes(text)
    text = normalize_tabs(text)
    text = remove_lonely_page_numbers(text)

    return text

# =====================================================
# CLEAN ALL REPORTS
# =====================================================
def clean_all_reports(input_folder, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    total_files = 0

    for company in os.listdir(input_folder):

        company_input = os.path.join(input_folder, company)

        if not os.path.isdir(company_input):
            continue

        company_output = os.path.join(output_folder, company)
        os.makedirs(company_output, exist_ok=True)

        print(f"\n📁 Cleaning {company}")

        for file in os.listdir(company_input):

            if not file.endswith(".txt"):
                continue

            input_path = os.path.join(company_input, file)

            output_path = os.path.join(company_output, file)

            with open(input_path, "r", encoding="utf-8") as f:
                text = f.read()

            cleaned = clean_text(text)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(cleaned)

            total_files += 1

            print(f"✅ Cleaned: {file}")

    print("\n==============================")
    print("Cleaning Completed Successfully")
    print("==============================")
    print(f"📄 Total Reports Cleaned : {total_files}")

# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":

    input_folder = r"C:\Users\riaze\Desktop\TechCorp-Financial-Enterprise-RAG\data\extracted_text"

    output_folder = r"C:\Users\riaze\Desktop\TechCorp-Financial-Enterprise-RAG\data\cleaned_text"

    clean_all_reports(input_folder, output_folder)