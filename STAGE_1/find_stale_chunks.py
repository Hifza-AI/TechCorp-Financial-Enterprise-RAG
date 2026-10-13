"""
find_stale_chunks.py
Kaun si chunk files purani table_parser se bani hain (duplicate column names wala bug)?

Chalane ka tareeqa (project root se):
    python STAGE_1/find_stale_chunks.py

Ye STAGE_1/chunks ki har *_chunks.json kholta hai aur dekhta hai:
  - kitni tables ke 'Columns:' mein same naam dobara aaya (jaise 2016, 2015, 2016, 2015)
  - agar aise tables hain, matlab file purani hai aur dobara chalani hai
Natija: stale_chunks_report.txt
"""
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("STAGE_1/chunks")
files = sorted(root.rglob("*_chunks.json"))
if not files:
    print("Koi *_chunks.json nahi mili:", root.resolve())
    sys.exit(1)

stale, ok = [], 0
for f in files:
    try:
        chunks = json.load(open(f, encoding="utf-8"))
    except Exception as e:
        print("Parh nahi saka:", f.name, e)
        continue
    dup_tables = 0
    colN = sum(len(re.findall(r"\(col \d+\)", c.get("text", ""))) for c in chunks)
    for c in chunks:
        if c.get("chunk_type") != "table":
            continue
        m = re.search(r"Columns:\s*(.+)", c.get("text", ""))
        if not m:
            continue
        cols = [x.strip() for x in m.group(1).split(",")]
        if len(cols) != len(set(cols)):
            dup_tables += 1
    # Purani = duplicate naam hain AUR "(col N)" ek bhi nahi (matlab naya fix kabhi laga hi nahi)
    if dup_tables and colN == 0:
        stale.append((f.name, dup_tables))
    else:
        ok += 1

lines = [f"Total files: {len(files)}", f"Theek (naya fix laga hua hai): {ok}",
         f"Purani (dobara chalani hain): {len(stale)}", ""]
for name, n in sorted(stale, key=lambda x: -x[1]):
    lines.append(f"{n:4d} tables  {name}")
Path("stale_chunks_report.txt").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines[:40]))
print("\nPoori list: stale_chunks_report.txt")