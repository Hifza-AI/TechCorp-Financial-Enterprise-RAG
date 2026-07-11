import re

# =====================================================
# 1. Remove URLs
# =====================================================
def remove_urls(text):
    return re.sub(r"https?://\S+", "", text)

# =====================================================
# 2. Remove Timestamps
# Example: 5/16/26, 9:58 AM
# =====================================================
def remove_timestamps(text):
    pattern = r"\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}\s?(AM|PM)"
    return re.sub(pattern, "", text)

# =====================================================
# 3. Remove Page Numbers
# Example: 13/67
# =====================================================
def remove_page_numbers(text):
    return re.sub(r"\b\d+\s*/\s*\d+\b", "", text)

# =====================================================
# 4. Remove Checkboxes
# =====================================================
def remove_checkboxes(text):
    return re.sub(r"[☐☑☒□✓✔]", "", text)

# =====================================================
# 5. Remove Apple Headers
# =====================================================
def remove_report_headers(text):

    patterns = [

    r"Apple Inc\.\s*\|\s*\d{4}\s*Form\s*10-K\s*\|\s*\d+",

    r"Apple Inc\.",

    r"Form\s*10-K",

    r"For the Fiscal Year Ended.*"

    r"JPMorgan Chase\s*&\s*Co\./\d{4}\s*\d+",

]

    for p in patterns:
        text = re.sub(p, "", text, flags=re.IGNORECASE)

    return text

# =====================================================
# 6. Remove Browser Artifacts
# =====================================================
def remove_browser_artifacts(text):

    patterns = [

        r"aapl-\d+",

        r"[A-Za-z0-9_-]+\.htm",

        r"[A-Za-z0-9_-]+\.html",

        r"Document\d*",

        r"nvda-\d+",

        r"jpm-\d+",

        r"\d+\s+JPMorgan Chase\s*&\s*Co\./\d{4}",
        

    ]

    for p in patterns:
        text = re.sub(p, "", text, flags=re.IGNORECASE)

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

# =====================================================
# 8. Normalize Multiple Spaces
# =====================================================
def normalize_spaces(text):
    return re.sub(r"[ ]{2,}", " ", text)

# =====================================================
# 9. Normalize Blank Lines
# =====================================================
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
#========================================================
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
# READ FILE
# =====================================================

input_file = r"C:\Users\riaze\Desktop\TechCorp-Financial-Enterprise-RAG\data\extracted_text\JPMorgan Chase\JPMorgan Chase_2024_10K copy.txt"

with open(input_file, "r", encoding="utf-8") as f:
    text = f.read()

# =====================================================
# APPLY PIPELINE
# =====================================================

cleaned_text = clean_text(text)

# =====================================================
# SAVE FILE
# =====================================================

output_file = r"C:\Users\riaze\Desktop\TechCorp-Financial-Enterprise-RAG\data\cleaned_text\Apple\JPMorgan\JPMorgan_2022_10K CLEAN.txt"

with open(output_file, "w", encoding="utf-8") as f:
    f.write(cleaned_text)

print("JPMorgan Chase Cleaning Completed Successfully!")