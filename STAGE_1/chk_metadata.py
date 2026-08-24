import pickle
from pathlib import Path


METADATA_PATH = Path("STAGE_1/vector_store/metadata.pkl")


def search_metadata(company, terms, max_results=5):
    """
    Directly searches metadata.pkl.
    Does NOT use FAISS or the retriever.
    """

    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)

    print("\n" + "=" * 80)
    print(f"COMPANY: {company}")
    print(f"SEARCH TERMS: {terms}")
    print("=" * 80)

    matches = []

    for i, chunk in enumerate(metadata):

        chunk_company = str(chunk.get("company", ""))
        text = str(chunk.get("text", ""))
        section = str(chunk.get("section_path", ""))
        chunk_type = str(chunk.get("chunk_type", ""))
        year = chunk.get("year")

        # Company must match
        if company.lower() not in chunk_company.lower():
            continue

        # At least one search term must appear
        combined_text = (
            text + " " +
            section
        ).lower()

        matched_terms = [
            term for term in terms
            if term.lower() in combined_text
        ]

        if matched_terms:
            matches.append({
                "index": i,
                "year": year,
                "chunk_type": chunk_type,
                "section_path": section,
                "matched_terms": matched_terms,
                "text": text
            })

    print(f"\nTotal matching chunks: {len(matches)}")

    for number, item in enumerate(matches[:max_results], 1):

        print("\n" + "-" * 80)
        print(f"[{number}] metadata index = {item['index']}")
        print(f"Year: {item['year']}")
        print(f"Type: {item['chunk_type']}")
        print(f"Matched terms: {item['matched_terms']}")
        print(f"Section: {item['section_path']}")

        print("\nText:")
        print(item["text"][:700])

    if not matches:
        print("\n❌ No matching chunk found.")

    return matches


if __name__ == "__main__":

    tests = [
        (
            "Apple",
            ["total assets", "assets"]
        ),

        (
            "Microsoft",
            ["total liabilities", "liabilities"]
        ),

        (
            "CVS",
            ["earnings per share", "diluted earnings per share"]
        ),

        (
            "Costco",
            ["operating income"]
        ),

        (
            "Costco",
            ["geographic", "operating income"]
        ),
    ]

    for company, terms in tests:
        search_metadata(company, terms, max_results=5)