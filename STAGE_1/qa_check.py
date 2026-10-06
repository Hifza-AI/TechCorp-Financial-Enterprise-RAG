"""
qa_check.py -- saari 500 files ka automatic QA. Kuch change/delete nahi karta.

Chalane ka tareeqa (project folder mein, jahan STAGE_1 hai):
    pip install pymupdf        (sirf ek baar)
    python qa_check.py                  # sab files (PDF ke saath, kuch minute lagenge)
    python qa_check.py --no-pdf         # sirf tez wale checks (PDF ke baghair)
    python qa_check.py --only Woodward  # sirf wo company jiske naam mein ye ho

Nateeja: qa_report.csv  +  qa_summary.txt   (dono mujhe bhej dena)
"""
import argparse, csv, json, re, sys
from pathlib import Path

ROOT = Path.cwd()
CHUNKS = ROOT / "STAGE_1" / "chunks"
HIER = ROOT / "STAGE_1" / "hierarchy"
PDF_DIR = ROOT / "STAGE_1" / "data"

ap = argparse.ArgumentParser()
ap.add_argument("--no-pdf", action="store_true")
ap.add_argument("--only", default="")
args = ap.parse_args()

fitz = None
if not args.no_pdf:
    try:
        import pymupdf as fitz
    except ImportError:
        try:
            import fitz
        except ImportError:
            print("pymupdf nahi mila. Pehle chalao: pip install pymupdf\n(ya --no-pdf ke saath chalao)")
            sys.exit(1)

key = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
def norm(s):
    s = s.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    s = re.sub("[\u2013\u2014\u2212\u2010\u2011]", "-", s)
    return re.sub(r"\s+", " ", s).lower()

pdf_index = {}
if fitz:
    for f in PDF_DIR.rglob("*.pdf"):
        pdf_index[key(f.stem)] = f

REQUIRED_PRESENT = ["1", "1A", "2", "3", "5", "7", "7A", "8", "9", "9A", "10", "11", "12", "13", "14", "15"]
REQUIRED_CONTENT = ["1", "1A", "7", "8"]

def sp(c):
    s = c.get("section_path", "")
    return s if isinstance(s, str) else " | ".join(s)

def walk_nodes(nodes):
    for n in nodes or []:
        yield n
        yield from walk_nodes(n.get("children"))

def has_content(n):
    if n.get("paragraphs") or n.get("tables"):
        return True
    return any(has_content(c) for c in n.get("children") or [])

def check_items(chunks, hier_path):
    found = {}
    if hier_path and hier_path.exists():
        data = json.load(open(hier_path, encoding="utf-8"))
        for n in walk_nodes(data.get("sections")):
            m = re.match(r"^\s*item\s+(\d+[A-Ca-c]?)\b", n.get("title", ""), re.I)
            if m:
                found[m.group(1).upper()] = found.get(m.group(1).upper(), False) or has_content(n)
        src = "hier"
    else:
        for c in chunks:
            m = re.match(r"^\s*item\s+(\d+[A-Ca-c]?)\b", sp(c).split(">")[0].strip(), re.I)
            if m:
                found[m.group(1).upper()] = True
        src = "chunks"
    missing = [i for i in REQUIRED_PRESENT if i not in found]
    if src == "hier":
        # "Item 1A." jaisi khali heading ke baad naam alag node mein ho sakta hai
        pass
    kw = {"1": "business", "1A": "risk factors", "7": "management", "8": "financial statements"}
    paths = [sp(c).lower() for c in chunks]
    empty = [i for i in REQUIRED_CONTENT if i in found and not found[i]
             and not any(kw[i] in p for p in paths)]
    return missing, empty, src

def parse_rows(text):
    for line in text.split("\n"):
        m = re.match(r"^(.*?) -- (.*)$", line)
        if not m:
            continue
        vals = {}
        for part in re.finditer(r"([^,:]+): (\(?-?[\d,\.]+\)?|-)", m.group(2)):
            vals[part.group(1).strip()] = part.group(2)
        yield m.group(1).strip(), vals

def to_num(s):
    s = s.replace(",", "")
    neg = s.startswith("(")
    s = s.strip("()")
    try:
        v = float(s)
    except ValueError:
        return None
    return -v if neg else v

def check_balance(chunks):
    """Total assets == Total liabilities and equity (pehle common column mein)."""
    for c in chunks:
        if c["chunk_type"] != "table":
            continue
        ta = tle = None
        for label, vals in parse_rows(c["text"]):
            l = label.lower()
            if l.startswith("total assets") and ta is None and vals:
                ta = vals
            if re.match(r"total liabilities,? .*(equity|deficit)", l) and vals:
                tle = vals
        if ta and tle:
            for col, v in ta.items():
                if col in tle:
                    a, b = to_num(v), to_num(tle[col])
                    if a is None or b is None:
                        continue
                    return "PASS" if abs(a - b) <= max(1, 0.0005 * abs(a)) else f"FAIL({v} vs {tle[col]})"
    return "NOT_FOUND"

def check_pdf(pdf_path, chunks):
    alltext = " ".join(c["text"] for c in chunks)
    ntext = norm(alltext)
    doc = fitz.open(str(pdf_path))
    nums_total = nums_abs = 0
    bad_pages = {}
    bul_total = bul_miss = 0
    for i, page in enumerate(doc, 1):
        txt = page.get_text()
        nums = set(re.findall(r"\b\d{1,3}(?:,\d{3})+\b", txt))
        miss = [n for n in nums if n not in alltext]
        nums_total += len(nums)
        nums_abs += len(miss)
        if len(miss) >= 5:
            bad_pages[i] = len(miss)
        L = [l.strip() for l in txt.split("\n")]
        for j, l in enumerate(L[:-1]):
            if l in ("•", "●", "▪", "‣") and len(L[j + 1]) >= 20:
                bul_total += 1
                if norm(L[j + 1])[:30] not in ntext:
                    bul_miss += 1
    n_pages = len(doc)
    doc.close()
    return n_pages, nums_total, nums_abs, bad_pages, bul_total, bul_miss

def main():
    files = sorted(CHUNKS.rglob("*_chunks.json"))
    if args.only:
        files = [f for f in files if args.only.lower() in str(f).lower()]
    print(f"{len(files)} files check hongi...")
    rows = []
    for n, f in enumerate(files, 1):
        stem = f.stem.replace("_chunks", "")
        company = f.parent.name
        flags = []
        try:
            chunks = json.load(open(f, encoding="utf-8"))
        except Exception as e:
            rows.append(dict(company=company, file=stem, flags=f"LOAD_ERROR {e}", score=99))
            continue
        hp = HIER / company / f"{stem}_hierarchy.json"
        missing, empty, src = check_items(chunks, hp)
        if missing: flags.append("ITEMS_MISSING:" + "/".join(missing))
        if empty: flags.append("ITEMS_EMPTY:" + "/".join(empty))
        bal = check_balance(chunks)
        if bal.startswith("FAIL"): flags.append("BALANCE_" + bal)
        if bal == "NOT_FOUND": flags.append("BALANCE_NOT_FOUND")

        tables = [c for c in chunks if c["chunk_type"] == "table"]
        n_tab = len(tables)
        blank = sum(1 for c in tables for ln in c["text"].split("\n") if ln.startswith(" --"))
        rows_total = sum(c["text"].count(" -- ") for c in tables) or 1
        blank_pct = round(100 * blank / rows_total, 1)
        tiny = sum(1 for c in chunks if len(c["text"].strip()) < 30)
        long_title = sum(1 for c in chunks if len(sp(c).split(">")[-1]) > 150)
        if n_tab < 20: flags.append(f"FEW_TABLES:{n_tab}")
        if blank_pct > 5: flags.append(f"BLANK_LABELS:{blank_pct}%")
        if long_title > 10: flags.append(f"LONG_TITLES:{long_title}")

        row = dict(company=company, file=stem, chunks=len(chunks), tables=n_tab,
                   items_src=src, balance=bal, blank_label_pct=blank_pct, tiny_chunks=tiny,
                   pdf_pages="", nums_in_pdf="", nums_absent="", nums_absent_pct="",
                   worst_pages="", bullets="", bullets_missing="")
        if fitz:
            pdf = pdf_index.get(key(stem))
            if pdf is None:
                flags.append("PDF_NOT_FOUND")
            else:
                try:
                    pg, nt, na, bad, bt, bm = check_pdf(pdf, chunks)
                    pct = round(100 * na / nt, 1) if nt else 0
                    row.update(pdf_pages=pg, nums_in_pdf=nt, nums_absent=na, nums_absent_pct=pct,
                               worst_pages=" ".join(f"p{p}:{c}" for p, c in sorted(bad.items(), key=lambda x: -x[1])[:5]),
                               bullets=bt, bullets_missing=bm)
                    if pct > 3: flags.append(f"NUMBERS_ABSENT:{pct}%")
                    if bt and bm / bt > 0.1: flags.append(f"BULLETS_MISSING:{bm}/{bt}")
                except Exception as e:
                    flags.append(f"PDF_ERROR:{str(e)[:40]}")
        row["flags"] = "; ".join(flags)
        cosmetic = ("BLANK_LABELS", "LONG_TITLES", "FEW_TABLES", "BALANCE_NOT_FOUND")
        row["score"] = sum(1 if f.startswith(cosmetic) else 10 for f in flags)
        rows.append(row)
        if n % 25 == 0:
            print(f"  {n}/{len(files)} ho gayi")

    cols = ["company", "file", "score", "flags", "chunks", "tables", "items_src", "balance",
            "blank_label_pct", "tiny_chunks", "pdf_pages", "nums_in_pdf", "nums_absent",
            "nums_absent_pct", "worst_pages", "bullets", "bullets_missing"]
    rows.sort(key=lambda r: -r.get("score", 0))
    with open("qa_report.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    total = len(rows)
    clean = sum(1 for r in rows if r.get("score", 0) == 0)
    def cnt(prefix): return sum(1 for r in rows if prefix in r.get("flags", ""))
    lines = [f"Files checked: {total}", f"Bilkul saaf (koi flag nahi): {clean} ({100*clean//max(total,1)}%)", "",
             f"Items missing:        {cnt('ITEMS_MISSING')}",
             f"Items empty:          {cnt('ITEMS_EMPTY')}",
             f"Balance FAIL:         {cnt('BALANCE_FAIL')}",
             f"Balance not found:    {cnt('BALANCE_NOT_FOUND')}",
             f"Numbers absent >3%:   {cnt('NUMBERS_ABSENT')}",
             f"Bullets missing >10%: {cnt('BULLETS_MISSING')}",
             f"Blank labels >5%:     {cnt('BLANK_LABELS')}",
             f"PDF not found:        {cnt('PDF_NOT_FOUND')}", "",
             "Sabse kharab 25 files:"]
    for r in rows[:25]:
        lines.append(f"  {r['score']}  {r['file']}  {r.get('flags','')}")
    Path("qa_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))
    print("\nDone. qa_report.csv aur qa_summary.txt ban gayi hain.")

main()