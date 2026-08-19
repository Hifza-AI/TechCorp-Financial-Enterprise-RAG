import json
from pathlib import Path

# Extracted directory path
extracted_dir = Path("STAGE_1/extracted")

for company_dir in sorted(extracted_dir.iterdir()):
    if not company_dir.is_dir():
        continue

    for json_file in sorted(company_dir.glob("*.json")):
        print(f"\n====================================")
        print(f" Checking: {json_file.name}")
        print(f"====================================")

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        mismatches = 0
        last_printed = 0
        anomalies = []

        for page in data.get("pages", []):
            # Updated keys according to extraction output
            physical = page.get("page_index")
            printed = page.get("page_number")
            is_post_sig = page.get("is_post_signatures", False)

            if printed is None:
                mismatches += 1
                status = "MISSING"
            else:
                status = "OK"
                # Anomaly check: Sequence drop verification
                if printed <= last_printed and last_printed != 0:
                    anomalies.append((physical, printed, last_printed))
                last_printed = printed

            post_sig_tag = " [EXHIBIT/POST-SIG]" if is_post_sig else ""
            print(
                f"Physical: {physical:4} | Printed: {str(printed):6} | {status}{post_sig_tag}"
            )

        print(f"\nSummary for {json_file.name}:")
        print(f"- Total pages: {len(data.get('pages', []))}")
        print(f"- Missing footers (None): {mismatches}")

        if anomalies:
            print(
                f"- ⚠️ WARNING! Anomalies (Sequence Breaking) detected at physical pages:"
            )
            for phys, prnt, prev in anomalies:
                print(
                    f"  * Physical Page {phys}: Printed page dropped from {prev} to {prnt}!"
                )
        else:
            print(
                "- ✅ Sequence Check Passed: Printed page numbers flow smoothly/monotonically!"
            )