import os
import gc
import pdfplumber


def extract_all_pdfs(main_folder, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    total_companies = 0
    total_pdfs = 0

    # Loop through all company folders
    for company in os.listdir(main_folder):

        company_path = os.path.join(main_folder, company)

        if not os.path.isdir(company_path):
            continue

        print(f"\n📁 Company: {company}")

        # Create company output folder
        company_output = os.path.join(output_folder, company)
        os.makedirs(company_output, exist_ok=True)

        company_pdf_count = 0

        # Loop through PDFs
        for file in os.listdir(company_path):

            if file.endswith(".pdf"):

                pdf_path = os.path.join(company_path, file)

                full_text = ""

                try:

                    with pdfplumber.open(pdf_path) as pdf:

                        for page in pdf.pages:

                            text = page.extract_text()

                            if text:
                                full_text += text + "\n"

                    # Save extracted text
                    txt_file = file.replace(".pdf", ".txt")

                    save_path = os.path.join(company_output, txt_file)

                    with open(save_path, "w", encoding="utf-8") as f:
                        f.write(full_text)

                    # Free RAM
                    del full_text
                    gc.collect()

                    company_pdf_count += 1
                    total_pdfs += 1

                    print(f"✅ Saved: {txt_file}")

                except Exception as e:

                    print(f"❌ Error: {file}")
                    print(e)

        total_companies += 1

        print(f"📄 PDFs Processed: {company_pdf_count}")

    print("\n==============================")
    print("Enterprise Data Ingestion Complete")
    print("==============================")
    print(f"🏢 Total Companies : {total_companies}")
    print(f"📑 Total PDFs      : {total_pdfs}")


if __name__ == "__main__":

    main_folder = r"C:\Users\riaze\Desktop\TechCorp-Financial-Enterprise-RAG\data\pdfs"

    output_folder = r"C:\Users\riaze\Desktop\TechCorp-Financial-Enterprise-RAG\data\extracted_text"

    extract_all_pdfs(main_folder, output_folder)