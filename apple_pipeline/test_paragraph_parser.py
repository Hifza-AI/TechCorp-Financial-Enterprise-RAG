from paragraph_parser import ParagraphParser

parser = ParagraphParser()

with open(
    r"C:\Users\riaze\Desktop\TechCorp-Financial-Enterprise-RAG\apple_pipeline\clean_text\Apple_2021_10K.txt",
    "r",
    encoding="utf-8"
) as f:
    text = f.read()

paragraphs = parser.parse(text)

print("Total Paragraphs:", len(paragraphs))

for i, p in enumerate(paragraphs[:10]):
    print("=" * 80)
    print(f"Paragraph {i+1}")
    print(p)