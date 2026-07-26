from sentence_transformers import SentenceTransformer
import faiss
import pickle
import re

# =====================================================
# Load Model
# =====================================================

print("Loading Sentence Transformer...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Model Loaded!\n")

# =====================================================
# Load FAISS
# =====================================================

index = faiss.read_index(
    r"C:\Users\riaze\Desktop\TechCorp-Financial-Enterprise-RAG\data\vector_store\apple_index.faiss"
)

print("FAISS Loaded!")

# =====================================================
# Load Metadata V3
# =====================================================

with open(
    r"C:\Users\riaze\Desktop\TechCorp-Financial-Enterprise-RAG\data\vector_store\apple_metadata_v3.pkl",
    "rb"
) as f:

    metadata = pickle.load(f)

print("Metadata Loaded!")

# =====================================================
# Detect Year
# =====================================================

def extract_year(query):

    match = re.search(r"\b20\d{2}\b", query)

    if match:
        return match.group()

    return None


# =====================================================
# Detect Company
# =====================================================

def extract_company(query):

    companies = [

        "Apple",
        "Microsoft",
        "Amazon",
        "Google",
        "Meta",
        "Tesla",
        "NVIDIA"

    ]

    query = query.lower()

    for company in companies:

        if company.lower() in query:
            return company

    return None


# =====================================================
# Metadata Filter
# =====================================================

def metadata_filter(metadata, company=None, year=None):

    filtered = []

    for item in metadata:

        if company is not None:

            if item["company"] != company:
                continue

        if year is not None:

            if item["year"] != year:
                continue

        filtered.append(item)

    return filtered


# =====================================================
# Query
# =====================================================

query = "What was Apple's total revenue in 2022?"

company = extract_company(query)
year = extract_year(query)

print("\n==============================")
print("Query")
print("==============================")

print(query)

print("\nDetected Company :", company)
print("Detected Year    :", year)

# =====================================================
# Filter Metadata
# =====================================================

filtered_chunks = metadata_filter(

    metadata,
    company=company,
    year=year

)

print("\nFiltered Chunks :", len(filtered_chunks))

# =====================================================
# Show Sample
# =====================================================

print("\n==============================")
print("First 5 Filtered Chunks")
print("==============================")

for chunk in filtered_chunks[:5]:

    print("\n---------------------------------------")
    print("Chunk ID :", chunk["chunk_id"])
    print("Section  :", chunk["section"])
    print("Year     :", chunk["year"])
    print()

    print(chunk["text"][:600])