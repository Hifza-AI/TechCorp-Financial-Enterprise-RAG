import os
import re

INPUT_FOLDER = r"C:\Users\riaze\Desktop\TechCorp-Financial-Enterprise-RAG\apple_pipeline\extracted_text"
OUTPUT_FOLDER = r"C:\Users\riaze\Desktop\TechCorp-Financial-Enterprise-RAG\apple_pipeline\clean_text"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def clean_text(text):

    # ---------------------------------------------------
    # Remove browser timestamp
    # Example:
    # 5/16/26, 9:56 AM Document
    # ---------------------------------------------------
    text = re.sub(
        r"\d{1,2}/\d{1,2}/\d{2},\s+\d{1,2}:\d{2}\s+[AP]M\s+Document",
        "",
        text,
    )

    # Remove remaining timestamps
    # Example:
    # 5/16/26, 9:56 AM
    text = re.sub(
        r"\d{1,2}/\d{1,2}/\d{2},\s+\d{1,2}:\d{2}\s+[AP]M",
        "",
        text,
    )

    # ---------------------------------------------------
    # Remove SEC filename
    # Example:
    # 10-K 1 a201610-k9242016.htm 10-K
    # ---------------------------------------------------
    text = re.sub(
        r".*a\d{6,}-k\d+\.htm.*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # ---------------------------------------------------
    # Remove SEC document IDs
    # Example:
    # aapl-20230930
    # ---------------------------------------------------
    text = re.sub(
        r"\baapl-\d{8}\b",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # ---------------------------------------------------
    # Remove SEC URLs
    # ---------------------------------------------------
    text = re.sub(
        r"https?://www\.sec\.gov\S+",
        "",
        text,
    )

    # ---------------------------------------------------
    # Remove footer
    # Example:
    # Apple Inc. | 2023 Form 10-K | 2
    # ---------------------------------------------------
    text = re.sub(
        r"Apple\s+Inc\.\s*\|\s*\d{4}\s+Form\s+10-K\s*\|\s*\d+",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # ---------------------------------------------------
    # Remove page counters
    # Example:
    # 2/63
    # 10/94
    # ---------------------------------------------------
    text = re.sub(
        r"\b\d+\s*/\s*\d+\b",
        "",
        text,
    )

    # ---------------------------------------------------
    # Remove checkbox symbols
    # ---------------------------------------------------
    text = re.sub(
        r"[☒☐☑■□✓✔]",
        "",
        text,
    )

    # ---------------------------------------------------
    # Remove extra spaces
    # ---------------------------------------------------
    text = re.sub(r"[ \t]+", " ", text)

    # ---------------------------------------------------
    # Remove multiple blank lines
    # ---------------------------------------------------
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def process():

    files = sorted(os.listdir(INPUT_FOLDER))

    print(f"\nFound {len(files)} files\n")

    for file in files:

        if not file.endswith(".txt"):
            continue

        path = os.path.join(INPUT_FOLDER, file)

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        cleaned = clean_text(text)

        save_path = os.path.join(OUTPUT_FOLDER, file)

        with open(save_path, "w", encoding="utf-8") as f:
            f.write(cleaned)

        print(f"✅ Cleaned {file}")

    print("\n==============================")
    print("APPLE CLEANING COMPLETE")
    print("==============================")


if __name__ == "__main__":
    process()