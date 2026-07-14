import os
import pdfplumber
import gc

# =====================================================
# PDF Folder
# =====================================================

main_folder = r"C:\Users\riaze\Desktop\TechCorp-Financial-Enterprise-RAG\data\pdfs"

# =====================================================
# Output Folder
# =====================================================

output_folder = r"C:\Users\riaze\Desktop\TechCorp-Financial-Enterprise-RAG\data\extracted_text"

os.makedirs(output_folder, exist_ok=True)

# =====================================================
# Reports to Re-Extract
# =====================================================

reports = [

    ("FordMotors", "FORD MOTOR_2016_10K.pdf"),
    ("FordMotors", "FORD MOTOR_2017_10K.pdf"),
    ("FordMotors", "FORD MOTOR_2021_10K.pdf"),

    ("Goldman Sachs", "Goldman Sachs_2022_10K.pdf"),
    ("Goldman Sachs", "Goldman Sachs_2023_10K.pdf"),

    ("Google", "Google_2018_10K.pdf"),

    ("IBM", "IBM_2025_10K.pdf"),

    ("Netflix", "Netflix_2018_10K.pdf"),

    ("Qualcomm", "Qualcomm_2016_10K.pdf")

]

# =====================================================
# Re-Extract Selected Reports
# =====================================================

for company, pdf_file in reports:

    pdf_path = os.path.join(main_folder, company, pdf_file)

    company_output = os.path.join(output_folder, company)

    os.makedirs(company_output, exist_ok=True)

    txt_file = pdf_file.replace(".pdf", ".txt")

    save_path = os.path.join(company_output, txt_file)

    full_text = ""

    try:

        with pdfplumber.open(pdf_path) as pdf:

            for page in pdf.pages:

                text = page.extract_text()

                if text:
                    full_text += text + "\n"

        with open(save_path, "w", encoding="utf-8") as f:
            f.write(full_text)

        del full_text
        gc.collect()

        print(f"✅ Re-Extracted: {pdf_file}")

    except Exception as e:

        print(f"❌ Error: {pdf_file}")
        print(e)

# =====================================================
# Re-Extract Complete Pfizer Folder
# =====================================================

pfizer_folder = os.path.join(main_folder, "PFIZER INC")

pfizer_output = os.path.join(output_folder, "PFIZER INC")

os.makedirs(pfizer_output, exist_ok=True)

for file in os.listdir(pfizer_folder):

    if file.endswith(".pdf"):

        pdf_path = os.path.join(pfizer_folder, file)

        txt_file = file.replace(".pdf", ".txt")

        save_path = os.path.join(pfizer_output, txt_file)

        full_text = ""

        try:

            with pdfplumber.open(pdf_path) as pdf:

                for page in pdf.pages:

                    text = page.extract_text()

                    if text:
                        full_text += text + "\n"

            with open(save_path, "w", encoding="utf-8") as f:
                f.write(full_text)

            del full_text
            gc.collect()

            print(f"✅ Pfizer: {file}")

        except Exception as e:

            print(f"❌ Pfizer Error: {file}")
            print(e)

print("\n==============================")
print("Re-Extraction Completed Successfully")
print("==============================")