import glob
import json

companies = ["Microsoft", "CVS", "COSTCO"]

for company in companies:
    # Path pattern dhoondne ke liye wildcard use kiya hai
    pattern = f"STAGE_1/heading_detection/{company}/*_headings.json"
    files = glob.glob(pattern)

    if not files:
        print(f"{company}: File nahi mili path par ({pattern})")
        continue

    file_path = files[0]

    try:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        pages = data.get("pages", []) if isinstance(data, dict) else data
        total_headings = 0
        total_lines = 0

        for p in pages:
            h_analysis = p.get("heading_analysis", {})
            total_headings += h_analysis.get("heading_count", 0)
            total_lines += len(h_analysis.get("candidates", []))

        ratio = (total_headings / total_lines * 100) if total_lines > 0 else 0
        print(
            f"{company}: {total_headings} headings / {total_lines} lines ({ratio:.1f}%)"
        )

    except Exception as e:
        print(f"Error checking {company}: {e}")