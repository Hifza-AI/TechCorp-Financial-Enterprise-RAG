import os
import pickle
import re

# ==========================================
# PATHS
# ==========================================

chunks_root = r"C:\Users\riaze\Desktop\TechCorp-Financial-Enterprise-RAG\data\chunks\Apple"

save_path = r"C:\Users\riaze\Desktop\TechCorp-Financial-Enterprise-RAG\data\vector_store\apple_metadata_v3.pkl"

# ==========================================
# SECTION PATTERNS
# ==========================================

# ===================================================
# Section Detector V2
# ===================================================

def detect_section(text):

    text = text.lower()

    patterns = {

        "Risk Factors": [
            "risk factors",
            "item 1a"
        ],

        "Business": [
            "item 1. business",
            "company background",
            "products",
            "markets and distribution"
        ],

        "MD&A": [
            "management's discussion",
            "item 7",
            "results of operations",
            "financial condition"
        ],

        "Revenue": [
            "net sales",
            "total net sales",
            "disaggregated revenue",
            "product performance",
            "services net sales",
            "iphone net sales"
        ],

        "Income Statement": [
            "statements of operations",
            "net income",
            "operating income",
            "gross margin",
            "earnings per share"
        ],

        "Balance Sheet": [
            "balance sheets",
            "total assets",
            "cash and cash equivalents",
            "total liabilities",
            "shareholders' equity"
        ],

        "Cash Flow": [
            "cash flows",
            "operating activities",
            "investing activities",
            "financing activities"
        ],

        "Financial Statements": [
            "notes to consolidated financial statements",
            "item 8",
            "summary of significant accounting policies"
        ],

        "Services": [
            "applecare",
            "apple music",
            "apple tv+",
            "icloud",
            "digital content",
            "advertising",
            "payment services"
        ]
    }

    for section, keywords in patterns.items():

        for keyword in keywords:

            if keyword in text:
                return section

    return "Unknown"

metadata = []

# ==========================================
# READ REPORTS
# ==========================================

for report_folder in os.listdir(chunks_root):

    report_path = os.path.join(chunks_root, report_folder)

    if not os.path.isdir(report_path):
        continue

    company = "Apple"

    year_match = re.search(r"\d{4}", report_folder)

    year = year_match.group() if year_match else "Unknown"

    # Industry trick
    current_section = "Unknown"

    files = sorted(
        os.listdir(report_path),
        key=lambda x: int(
            x.replace("chunk_", "").replace(".txt", "")
        )
    )

    for file in files:

        if not file.endswith(".txt"):
            continue

        filepath = os.path.join(report_path, file)

        with open(filepath, "r", encoding="utf-8") as f:

            text = f.read()

        lower_text = text.lower()

        # Detect only when new heading appears
        for section, keywords in patterns.items():

            if any(keyword in lower_text for keyword in keywords):

                current_section = section
                break

        metadata.append({

            "company": company,

            "year": year,

            "report": report_folder,

            "chunk_id": file.replace(".txt",""),

            "section": current_section,

            "text": text

        })

# ==========================================
# SAVE
# ==========================================

with open(save_path,"wb") as f:

    pickle.dump(metadata,f)

print("="*60)
print("Metadata V3 Created Successfully")
print("="*60)

print("Total Chunks :",len(metadata))

print("\nExample:\n")

print(metadata[100])