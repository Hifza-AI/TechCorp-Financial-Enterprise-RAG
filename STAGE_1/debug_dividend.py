# import pickle
# from pathlib import Path

# store_dir = Path("STAGE_1/vector_store")

# with open(store_dir / "metadata.pkl", "rb") as f:
#     metadata = pickle.load(f)

# # Dhoondo kitne chunks mein "dividend" word hai
# dividend_chunks = [
#     c for c in metadata
#     if c.get("text") and "dividend" in c["text"].lower()
# ]

# print(f"Total chunks with 'dividend': {len(dividend_chunks)}")

# for c in dividend_chunks[:10]:
#     print(f"\nCompany: {c.get('company')} | Year: {c.get('year')}")
#     print(f"Section: {c.get('section_path')}")
#     print(f"Text preview: {c.get('text')[:300]}")


# from retriever import Retriever

# r = Retriever()

# results = r.search("What is Apple's total current assets?", top_k=3)

# for i, res in enumerate(results, 1):
#     print(f"\n--- Result {i} ---")
#     print(f"Company: {res['company']} {res['year']}")
#     print(f"FULL TEXT:\n{res['text']}")  # poora text, preview nahi
#     print("-" * 50)


import json
from pathlib import Path

# apna Apple 2019 (ya jo bhi year) ka extracted JSON dhoondo
json_path = Path(r"C:\Users\riaze\Desktop\TechCorp-Financial-Enterprise-RAG\STAGE_1\extracted\Apple\Apple_2019_10K.json")  # apna actual path use karo

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# "Current liabilities" wala page dhoondo
for page in data.get("pages", []):
    page_text = str(page)
    if "Current liabilities" in page_text:
        print(f"Page number: {page.get('page_number')}")
        print(json.dumps(page, indent=2)[:3000])
        break

# import json
# import re
# from pathlib import Path

# json_path = Path(r"C:\Users\riaze\Desktop\TechCorp-Financial-Enterprise-RAG\STAGE_1\extracted\Apple\Apple_2019_10K.json")  # apna actual path check kar lena

# with open(json_path, "r", encoding="utf-8") as f:
#     data = json.load(f)

# for page in data.get("pages", []):
#     if page.get("page_number") != 34:
#         continue

#     print(f"=== Page {page['page_number']} — total blocks: {len(page['blocks'])} ===\n")

#     for block in page["blocks"]:
#         for line in block.get("lines", []):
#             for span in line.get("spans", []):
#                 text = span.get("text", "").strip()
#                 if not text:
#                     continue
#                 # numeric-looking text (dollar figures usually have digits + commas)
#                 if re.search(r"\d", text):
#                     print(f"[y={span['origin'][1]:.1f}] '{text}'")
#     break



# import json
# import re
# from pathlib import Path

# json_path = Path(r"C:\Users\riaze\Desktop\TechCorp-Financial-Enterprise-RAG\STAGE_1\extracted\Apple\Apple_2019_10K.json")

# with open(json_path, "r", encoding="utf-8") as f:
#     data = json.load(f)

# for page in data.get("pages", []):
#     if page.get("page_number") != 34:
#         continue

#     for block in page["blocks"]:
#         for line in block.get("lines", []):
#             for span in line.get("spans", []):
#                 text = span.get("text", "").strip()
#                 if not text:
#                     continue
#                 # is baar SIRF non-numeric labels dhoondo (row names)
#                 if not re.search(r"\d", text):
#                     print(f"[y={span['origin'][1]:.1f}] '{text}'")
#     break


# import pickle
# from pathlib import Path

# with open("STAGE_1/vector_store/metadata.pkl", "rb") as f:
#     metadata = pickle.load(f)

# # Apple 2019 ke table-type chunks dhoondo jinme "162,819" ho (total current assets ka number)
# for c in metadata:
#     if c.get("company") == "Apple" and c.get("year") == 2019 and c.get("chunk_type") == "table":
#         if c.get("text") and "162,819" in c["text"]:
#             print("MIL GAYA! Chunk text:")
#             print(c["text"])
#             print("\nSection:", c.get("section_path"))


# import pickle

# with open("STAGE_1/vector_store/metadata.pkl", "rb") as f:
#     metadata = pickle.load(f)

# for c in metadata:
#     if c.get("company") == "Apple" and c.get("year") == 2019 and c.get("chunk_type") == "table":
#         if c.get("text") and "Total liabilities" in c["text"]:
#             print("Section:", c.get("section_path"))
#             print(c["text"])
#             print("---")



# from retriever import Retriever

# r = Retriever()
# query_vector = r.embedder.embed_query("What is Apple's total current assets?")

# # is specific chunk ka embedding nikaal ke similarity check karo
# import numpy as np
# for i, c in enumerate(r.metadata):
#     if c.get("company") == "Apple" and c.get("year") == 2019 and c.get("text") and "162,819" in c["text"]:
#         print(f"Ye chunk metadata index {i} par hai")
#         print("Section:", c.get("section_path"))
#         break