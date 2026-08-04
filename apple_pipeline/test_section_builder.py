import json

from section_builder import SectionBuilder


builder = SectionBuilder()

with open(
    r"C:\Users\riaze\Desktop\TechCorp-Financial-Enterprise-RAG\apple_pipeline\clean_text\Apple_2021_10K.txt",
    "r",
    encoding="utf-8"
) as f:
    text = f.read()


sections = builder.build(text)

result = builder.to_dict(sections)

print(json.dumps(result, indent=4))