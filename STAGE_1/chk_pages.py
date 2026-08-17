import json
from pathlib import Path

# Un sub companies ke JSON check karein jo extracted folder mein hain
extracted_dir = Path("STAGE_1/extracted")

for company_dir in sorted(extracted_dir.iterdir()):
    if not company_dir.is_dir():
        continue

    for json_file in company_dir.glob("*.json"):
        print(f"\n====================================")
        print(f" Checking: {json_file.name}")
        print(f"====================================")

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        mismatches = 0
        last_printed = 0
        anomalies = []

        for page in data["pages"]:
            physical = page["physical_page_index"]
            printed = page["printed_page_number"]

            if printed is None:
                mismatches += 1
                status = "MISSING"
            else:
                status = "OK"
                # Anomaly check: Agar page number previous se chhota ho jaye (e.g. 45 ke baad 7 aagaya)
                if printed <= last_printed and last_printed != 0:
                    anomalies.append((physical, printed, last_printed))
                last_printed = printed

            print(f"Physical: {physical:4} | Printed: {str(printed):6} | {status}")

        print(f"\nSummary for {json_file.name}:")
        print(f"- Total pages: {len(data['pages'])}")
        print(f"- Missing footers (None): {mismatches}")
        
        if anomalies:
            print(f"- ⚠️ WARNING! Anomalies (Sequence Breaking) detected at physical pages:")
            for phys, prnt, prev in anomalies:
                print(f"  * Physical Page {phys}: Printed page dropped from {prev} to {prnt}!")
        else:
            print("- ✅ Sequence Check Passed: Printed page numbers flow smoothly/monotonically!")