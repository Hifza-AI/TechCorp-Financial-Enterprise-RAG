# =====================================
# Enterprise Cleaning Pipeline V1
# Apple Report Testing
# =====================================

import re


# -------------------------------------
# Function 1 - Remove URLs
# -------------------------------------
def remove_urls(text):
    return re.sub(r"https?://\S+", "", text)


# -------------------------------------
# Function 2 - Remove Timestamps
# Example:
# 5/16/26, 9:58 AM
# -------------------------------------
def remove_timestamps(text):
    pattern = r"\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}\s?(AM|PM)"
    return re.sub(pattern, "", text)


# -------------------------------------
# Function 3 - Remove Page Numbers
# Example:
# 13/67
# -------------------------------------
def remove_page_numbers(text):
    pattern = r"\b\d+\s*/\s*\d+\b"
    return re.sub(pattern, "", text)


# -------------------------------------
# Function 4 - Remove Checkboxes
# -------------------------------------
def remove_checkboxes(text):
    pattern = r"[□☑✓✔☒☐]"
    return re.sub(pattern, "", text)


# -------------------------------------
# Function 5 - Header Placeholder
# -------------------------------------
def remove_headers(text):
    return text


# -------------------------------------
# Function 6 - Footer Placeholder
# -------------------------------------
def remove_footers(text):
    return text


# -------------------------------------
# Function 7 - Normalize Spaces
# -------------------------------------
def normalize_spaces(text):
    return re.sub(r"[ ]{2,}", " ", text)


# -------------------------------------
# Function 8 - Normalize Blank Lines
# -------------------------------------
def normalize_blank_lines(text):
    return re.sub(r"\n{3,}", "\n\n", text)


# =====================================
# Cleaning Pipeline
# =====================================
def clean_text(text):

    text = remove_urls(text)

    text = remove_timestamps(text)

    text = remove_page_numbers(text)

    text = remove_checkboxes(text)

    text = remove_headers(text)

    text = remove_footers(text)

    text = normalize_spaces(text)

    text = normalize_blank_lines(text)

    return text


# =====================================
# Read Apple Report
# =====================================

with open(
    r"C:\Users\riaze\Desktop\TechCorp-Financial-Enterprise-RAG\data\extracted_text\Apple\Apple_2021_10K_TEST.txt",
    "r",
    encoding="utf-8"
) as f:

    text = f.read()


# =====================================
# Apply Cleaning Pipeline
# =====================================

cleaned_text = clean_text(text)


# =====================================
# Save Clean File
# =====================================

with open(
    r"C:\Users\riaze\Desktop\TechCorp-Financial-Enterprise-RAG\data\cleaned_text\Apple_2021_10K_CLEANED.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(cleaned_text)


# =====================================
# Success Message
# =====================================

print("✅ Apple Report Cleaned Successfully")