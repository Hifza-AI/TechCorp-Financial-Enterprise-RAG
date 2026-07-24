from pathlib import Path

apple_folder = Path(r"C:\Users\riaze\Desktop\TechCorp-Financial-Enterprise-RAG\data\chunks\Apple")

all_chunks = []

for file in apple_folder.rglob("chunk_*.txt"):
    text = file.read_text(encoding="utf-8")
    all_chunks.append(text)

print("-"*40)
print("Total Chunks:", len(all_chunks))
print("-"*40)

print("First Chunk:\n")
print(all_chunks[0][:500])