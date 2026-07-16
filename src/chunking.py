from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ===========================
# Input & Output Folders
# ===========================

input_folder = Path(
    r"C:\Users\riaze\Desktop\TechCorp-Financial-Enterprise-RAG\data\cleaned_text"
)

output_folder = Path(
    r"C:\Users\riaze\Desktop\TechCorp-Financial-Enterprise-RAG\data\chunks"
)

# ==========================
# Chunk Splitter
# ==========================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]
)

total_files = 0
total_chunks = 0

# ==========================
# Loop Through Companies
# ==========================

for company_folder in input_folder.iterdir():

    if not company_folder.is_dir():
        continue

    company_output = output_folder / company_folder.name
    company_output.mkdir(parents=True, exist_ok=True)

    # Loop through reports
    for txt_file in company_folder.glob("*.txt"):

        total_files += 1

        text = txt_file.read_text(
            encoding="utf-8",
            errors="ignore"
        )

        chunks = splitter.split_text(text)

        report_folder = company_output / txt_file.stem
        report_folder.mkdir(parents=True, exist_ok=True)

        for i, chunk in enumerate(chunks, start=1):

            chunk_file = report_folder / f"chunk_{i:03}.txt"

            chunk_file.write_text(
                chunk,
                encoding="utf-8"
            )

        total_chunks += len(chunks)

        print(f"✅ {txt_file.name}  ---> {len(chunks)} chunks")

# ==========================
# Summary
# ==========================

print("\n" + "="*60)
print("Chunking Completed Successfully")
print("="*60)
print(f"Reports Processed : {total_files}")
print(f"Total Chunks      : {total_chunks}")
print(f"Saved In          : {output_folder}")
print("="*60)