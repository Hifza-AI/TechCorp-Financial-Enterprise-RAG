"""
find_folders.py  -- sirf folders/files ke naam batata hai (kuch change nahi karta).
Chalane ka tareeqa: project folder (jahan STAGE_1 hai) mein PowerShell kholo aur:
    python find_folders.py
Phir jo 'folder_report.txt' bane wo mujhe bhej do (ya terminal ka output paste kar do).
"""
import os, sys
from pathlib import Path

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
out = []
def p(s=""):
    print(s); out.append(s)

p(f"Root folder: {root.resolve()}")
p("")

# 1) Top-level aur STAGE_1 ke andar ke folders, file counts ke saath
def count(d, pat):
    try: return sum(1 for _ in d.rglob(pat))
    except Exception: return 0

p("== Folders (do level tak) ==")
for lvl1 in sorted(x for x in root.iterdir() if x.is_dir() and not x.name.startswith(".")):
    p(f"{lvl1.name}/   json={count(lvl1,'*.json')}  pdf={count(lvl1,'*.pdf')}  txt={count(lvl1,'*.txt')}")
    for lvl2 in sorted(x for x in lvl1.iterdir() if x.is_dir())[:15]:
        p(f"    {lvl2.name}/   json={count(lvl2,'*.json')}  pdf={count(lvl2,'*.pdf')}  txt={count(lvl2,'*.txt')}")

# 2) Har tarah ki file ka ek-ek namuna
p("")
p("== Namune (har type ki 3 files, poora path) ==")
for pat in ["*_chunks.json", "*_hierarchy.json", "*_hierarchy_outline.txt", "*.pdf", "*_cleaned.json"]:
    found = []
    for f in root.rglob(pat):
        if ".git" in f.parts or "node_modules" in f.parts: continue
        found.append(f)
        if len(found) >= 3: break
    p(f"{pat}:")
    for f in found: p(f"    {f.relative_to(root)}")
    if not found: p("    (nahi mili)")

# 3) Kul chunks.json kitni aur kitne company folders
chunks = [f for f in root.rglob("*_chunks.json")]
p("")
p(f"Total *_chunks.json files: {len(chunks)}")
p(f"Chunks wale company folders: {len({f.parent for f in chunks})}")

Path("folder_report.txt").write_text("\n".join(out), encoding="utf-8")
p("")
p("Done. 'folder_report.txt' ban gayi hai.")