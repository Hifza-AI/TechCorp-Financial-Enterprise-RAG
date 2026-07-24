import os

chunks_folder = r"C:\Users\riaze\Desktop\TechCorp-Financial-Enterprise-RAG\data\chunks\Apple"

found = False

for report in os.listdir(chunks_folder):
    report_path = os.path.join(chunks_folder, report)

    if os.path.isdir(report_path):

        for file in os.listdir(report_path):

            if file.endswith(".txt"):

                file_path = os.path.join(report_path, file)

                with open(file_path, "r", encoding="utf-8") as f:

                    text = f.read()

                    if "2022" in text and "383.3" in text:
                        print("=" * 80)
                        print(file_path)
                        print()
                        print(text[:1200])
                        found = True

if not found:
    print("2022 revenue chunk NOT FOUND")