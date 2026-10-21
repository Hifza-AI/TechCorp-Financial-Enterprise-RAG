"""
check_data.py  -  Saare chunk files ki 1-minute jaanch: company ke naam aur saal (year) theek hain ya nahi.

Chalao:   python STAGE_1/check_data.py
Report:   screen par + STAGE_1/eval/data_check.txt

Kyun: retriever company aur saal se filter karta hai. Agar ek hi company do naamon se ho
("KMT" aur "KENNAMETAL") ya kisi file ka saal galat label ho, to sawal ka jawab hote hue bhi
retriever "No data found" kahega -- ye retriever ki nahi, data ke label ki galti hogi.

PDF ka text chunks tak pahuncha ya nahi (kuch minute, PDFs parhta hai):
    python STAGE_1/check_data.py --bullets
    -> har PDF ki aam lines aur bullet items (•, ●, ·, ...) chunks ke text / section naam mein dhoondta hai.
       Jo nahi milte wo STAGE_1/eval/missing_text.csv mein. (ServiceNow 2025: competitors ki 6 bullets PDF
       mein thin, chunk mein "including:" ke baad seedha agla paragraph -- list ghayab.)
"""
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from collections import Counter

COLS = re.compile(r"^Columns:\s*(.+)$", re.M)
YEAR4 = re.compile(r"\b(?:19|20)\d{2}\b")

# Bullet ke nishan jo 10-K PDFs mein aate hain: • ● ▪ ■ ◦ ‣ ⁃ ➢ ► · aur Symbol/Wingdings font wale (private-use)
BULLET_GLYPHS = "\u2022\u25cf\u25aa\u25a0\u25e6\u2023\u2043\u27a2\u25ba\u00b7\uf0b7\uf0a7\uf06e\uf076\uf0d8\uf0fc"


def table_year(chunks):
    """Tables ke 'Columns:' mein sabse naya saal -- poori file mein sabse zyada kaunsa aata hai."""
    cnt = Counter()
    for c in chunks:
        if c.get("chunk_type") != "table":
            continue
        m = COLS.search(c.get("text") or "")
        ys = [int(y) for y in YEAR4.findall(m.group(1))] if m else []
        if ys:
            cnt[max(ys)] += 1
    if not cnt:
        return None, 0, 0
    y, n = cnt.most_common(1)[0]
    return y, n, sum(cnt.values())


FISCAL = re.compile(r"\b[Ff]iscal (?:[Yy]ear )?((?:19|20)\d{2})\b")


def data_year(chunks):
    """File asal mein kis saal ka data rakhti hai (company ke apne 'fiscal' naam ke hisaab se).
    Saboot 1: tables ke 'Columns:' mein sabse naya saal (zyada tar tables mein).
    Saboot 2: text mein 'fiscal 2025' jaise alfaaz -- wo saal jo kaafi baar aayen (>= 20 aur top ka 25%).
    Faisla: 1) tables ka saal text mein bhi kaafi baar ho -> wahi.
            2) tables saaf (60%+) -> tables, sirf Home Depot jaisi soorat chhod kar (text = tables - 1, 50+ baar).
            3) warna text ke kaafi-baar saalon mein sabse naya.  4) Text na ho to sirf tables."""
    fc = Counter(int(y) for c in chunks for y in FISCAL.findall(c.get("text") or ""))
    ty, n, tot = table_year(chunks)
    strong = []
    if fc:
        top = fc.most_common(1)[0][1]
        strong = [y for y, k in fc.items() if k >= max(20, 0.25 * top)]
    if strong and ty is not None and ty in strong:
        return ty, f"tables: {ty} ({n}/{tot}) + text mein 'fiscal {ty}' {fc[ty]} baar"
    if ty is not None and tot and n / tot >= 0.6:
        # Tables saaf keh rahe hain. Text sirf ek soorat mein jeetta hai: company saal ko us ke SHURU wale
        # saal se pukarti hai (Home Depot: Feb 2017 mein khatam saal = "fiscal 2016"), yaani text = tables - 1
        # aur bohat zyada baar. Warna "fiscal 2020 restructuring" jaise purane programs ke naam dhoka dete hain.
        if (ty - 1) in strong and fc[ty - 1] >= 50:
            return ty - 1, f"text mein 'fiscal {ty - 1}' {fc[ty - 1]} baar (company ka naam); tables: {ty} ({n}/{tot})"
        return ty, f"tables ke columns mein sabse naya saal {ty} ({n}/{tot} tables)"
    if strong:
        fy = max(strong)
        return fy, f"text mein 'fiscal {fy}' {fc[fy]} baar; tables: {ty} ({n}/{tot})"
    if ty is None:
        return None, "saal ka saboot nahi mila"
    return ty, f"tables ke columns mein sabse naya saal {ty} ({n}/{tot} tables)"


def _alnum(s):
    """Sirf huroof aur number (chhote) -- PDF aur chunk ke text ko milane ke liye (spaces / hyphen / quotes ka farq khatam)."""
    s = (s or "").replace("ﬁ", "fi").replace("ﬂ", "fl").lower()
    return re.sub(r"[^a-z0-9]+", "", s)


def pdf_text_items(pdf_path):
    """PDF se do list: bullet items ki pehli line [(page, text)] aur aam prose lines [(page, text)].
    Aam line = 8+ lafz. Cover (pehle 2 page), URL, fehrist ke dots aur 3+ pages par dohrane wali line
    (header / footer) nahi ginte -- unhein pipeline jaan boojh kar hatati hai."""
    try:
        import pymupdf as fitz                    # PyMuPDF -- wahi jis se cleaned JSON bani
    except ImportError:
        import fitz
    with fitz.open(pdf_path) as doc:
        pages = [[x.strip() for x in page.get_text("text").splitlines()] for page in doc]
    repeated = Counter(s for lines in pages for s in set(lines))
    bullets, prose = [], []
    for pno, lines in enumerate(pages, 1):
        i = 0
        while i < len(lines):
            s = lines[i]
            if s and s[0] in BULLET_GLYPHS:
                body = s[1:].strip()
                if not body and i + 1 < len(lines):    # nishan akela, text agli line par
                    i += 1
                    body = lines[i]
                if len(_alnum(body)) >= 12:          # "and" / "Inc." jaise tukde nahi
                    bullets.append((pno, body))
            elif (pno > 2 and len(re.findall(r"[A-Za-z]{2,}", s)) >= 8 and "http" not in s
                  and "...." not in s and ". . ." not in s and repeated[s] < 3):
                prose.append((pno, s))
            i += 1
    return bullets, prose


def text_check(pdf_dir, chunk_files, csv_path):
    """Har chunk file ki PDF dhoondo; PDF ki aam lines aur bullet items chunks (text + section naam) mein talash karo.
    Wapas: report ki lines. Jo nahi mila wo csv_path mein."""
    pdfs = {_alnum(p.stem): p for p in Path(pdf_dir).rglob("*.pdf")}
    if not pdfs:
        return [f"  PDFs nahi milin: {Path(pdf_dir).resolve()}  (--pdfs se sahi folder do)"]
    T = Counter()
    per_file, rows, unmatched = [], [], []
    for n, f in enumerate(chunk_files, 1):
        if n % 25 == 0:
            print(f"  text check: {n}/{len(chunk_files)} files ...", flush=True)
        try:
            ch = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue                                   # padhne ki ghalti upar section 3 mein aa chuki
        fname = str((ch[0].get("file_name") if ch else "") or re.sub(r"_chunks$", "", f.stem, flags=re.I))
        pdf = pdfs.get(_alnum(Path(fname).stem if fname.lower().endswith(".pdf") else fname))
        if pdf is None:
            unmatched.append(f.name)
            continue
        try:
            bullets, prose = pdf_text_items(pdf)
        except Exception as e:
            unmatched.append(f"{f.name} ({e})")
            continue
        text = _alnum(" ".join(c.get("text") or "" for c in ch))
        head = _alnum(" ".join(str(c.get("section_path") or "") for c in ch))
        mb = ml = 0
        for pg, t in bullets:
            k = _alnum(t)[:40]
            T["bullets"] += 1
            if k in text:
                T["b_text"] += 1
            elif k in head:
                T["b_head"] += 1                       # heading ban gaya (section ke naam mein) -- ghayab nahi
            else:
                mb += 1
                rows.append((f.name, pg, "bullet", t))
        for pg, t in prose:
            k = _alnum(t)[:45]
            T["lines"] += 1
            if k not in text and k not in head:
                ml += 1
                rows.append((f.name, pg, "line", t))
        T["b_miss"] += mb
        T["l_miss"] += ml
        if mb or ml:
            first = next(r for r in rows if r[0] == f.name)
            per_file.append((mb + ml, ml, len(prose), mb, len(bullets), f.name, first))

    out = Path(csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:          # -sig: Excel / Sheets theek khole
        w = csv.writer(fh)
        w.writerow(["chunk_file", "pdf_page", "kind", "text"])
        w.writerows(rows)

    def pct(x, y):
        return f"{100.0 * x / y:.1f}%" if y else "-"
    L = [f"  Files milayi: {len(chunk_files) - len(unmatched)}",
         f"  a) Aam lines (8+ lafz; cover, URL, header/footer chhod kar): {T['lines']} mein se "
         f"{T['l_miss']} ({pct(T['l_miss'], T['lines'])}) chunks mein NAHI",
         f"  b) Bullet items: {T['bullets']} -- text mein {T['b_text']}, sirf section ke naam (heading) mein "
         f"{T['b_head']} (ye ghayab nahi), kahin NAHI {T['b_miss']} ({pct(T['b_miss'], T['bullets'])})",
         f"  Files jin mein kuch ghayab: {len(per_file)}"]
    if per_file:
        L.append("  Sab se zyada ghayab (lines ghayab/kul, bullets ghayab/kul):")
        for _, ml, nl, mb, nb, name, first in sorted(per_file, reverse=True)[:25]:
            L.append(f"    {name:46s} lines {ml:3d}/{nl:<5d} bullets {mb:3d}/{nb:<4d} e.g. p{first[1]}: {first[3][:55]}")
    L.append(f"  Poori list: {out}")
    L.append("  (Ye andaza hai -- kuch lines pipeline jaan boojh kar hatati hai. Asli kami aksar tables ke aas paas hoti")
    L.append("   hai: table_analyzer ke rescue passes (2-4) table ke upar/neeche ki chhoti lines -- intro jumla, footnote,")
    L.append("   list -- table mein le lete hain, paragraph_parser table lines chhod deta hai, aur jo TableParser")
    L.append("   istemal na kare wo gum ho jati hain.)")
    if unmatched:
        L.append(f"  PDF nahi mili ({len(unmatched)}): {', '.join(unmatched[:8])}" + (" ..." if len(unmatched) > 8 else ""))
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="STAGE_1/chunks")
    ap.add_argument("--out", default="STAGE_1/eval/data_check.txt")
    ap.add_argument("--bullets", action="store_true", help="PDF ka text (aam lines + bullets) chunks tak pahuncha ya nahi (kuch minute)")
    ap.add_argument("--pdfs", default="STAGE_1/data/pdfs")
    ap.add_argument("--missing-csv", default="STAGE_1/eval/missing_text.csv")
    a = ap.parse_args()

    files = sorted(Path(a.chunks).rglob("*_chunks.json"))
    if not files:
        raise SystemExit(f"Koi *_chunks.json nahi mili: {Path(a.chunks).resolve()}")

    by_company = defaultdict(list)          # company -> [(year, file)]
    year_issues, field_issues, read_issues = [], [], []
    for f in files:
        try:
            ch = json.load(open(f, encoding="utf-8"))
        except Exception as e:
            read_issues.append(f"{f.name}: parh nahi saka ({e})")
            continue
        comps = {str(c.get("company", "")).strip() for c in ch}
        years = {str(c.get("year", "")).strip() for c in ch}
        if len(comps) != 1 or "" in comps or len(years) != 1 or "" in years:
            field_issues.append(f"{f.name}: company={sorted(comps)} year={sorted(years)}")
        comp, year = sorted(comps)[0], sorted(years)[0]
        by_company[comp].append((year, f.name))
        dy, why = data_year(ch)
        if dy is None:
            year_issues.append(f"{f.name}: label {year}, {why} -- nahi mila (khud dekh lo)")
        elif str(dy) != year:
            year_issues.append(f"{f.name}: label {year}, lekin data {dy} ka lagta hai  [{why}]")

    lines = []
    P = lines.append
    P(f"Files: {len(files)}   Companies (naam ke hisaab se): {len(by_company)}\n")
    P("=" * 78)
    P("1) HAR COMPANY KE SAAL  (10 se kam/zyada, ya ek saal do baar = dhyan do)")
    P("=" * 78)
    for comp in sorted(by_company, key=str.lower):
        ys = sorted(y for y, _ in by_company[comp])
        dup = sorted({y for y in ys if ys.count(y) > 1})
        flag = []
        if len(ys) != 10:
            flag.append(f"{len(ys)} files")
        if dup:
            flag.append(f"do baar: {dup}")
        if "_" in comp or "  " in comp:
            flag.append("naam mein _ ya double space")
        if re.fullmatch(r"[A-Z0-9&]{2,5}", comp):
            flag.append("ticker jaisa naam")
        P(f"  {comp:32s} {', '.join(ys)}" + (f"    <-- {'; '.join(flag)}" if flag else ""))
    P("\n  Tip: agar ek hi company do naamon se nazar aaye (jaise KMT aur KENNAMETAL), to dono ka ek naam hona chahiye.")
    P("  Spelling bhi dekh lo (jaise 'Rockwell Auomation'). Ye naam jawab ki citation mein jayega.\n")

    P("=" * 78)
    P("2) SAAL KA LABEL vs FILE KE ANDAR KA DATA ('fiscal 2025' text, warna tables ke columns)")
    P("=" * 78)
    mism = [x for x in year_issues if "lekin data" in x]
    notf = [x for x in year_issues if "nahi mila" in x]
    P(f"  Label aur andar ke data ka saal alag: {len(mism)}")
    for x in mism:
        P("   ! " + x)
    P("  (Matlab: is file ka 'saal' us saal ka data nahi rakhta jo label kehta hai. Sawal 'X 2025' ka hoga")
    P("   to retriever galat file kholega. Har company ke liye ek hi rule rakho: label = fiscal year ka data.)")
    P(f"  Saal ka saboot nahi mila: {len(notf)}")
    if mism:
        P("  Theek karne ke liye:  python STAGE_1/fix_years.py   (pehle sirf dikhata hai, --apply se badalta hai)")
    for x in notf[:40]:
        P("   ? " + x)
    if len(notf) > 40:
        P(f"   ... aur {len(notf) - 40}")

    P("\n" + "=" * 78)
    P("3) COMPANY / YEAR FIELD KHAALI YA EK FILE MEIN DO")
    P("=" * 78)
    for x in field_issues + read_issues or ["  koi masla nahi"]:
        P("   " + x)

    if a.bullets:
        P("\n" + "=" * 78)
        P("4) PDF KA TEXT CHUNKS TAK PAHUNCHA?  (aam lines + bullet items)")
        P("=" * 78)
        lines.extend(text_check(a.pdfs, files, a.missing_csv))

    text = "\n".join(lines)
    print(text)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"\nReport save hui: {out.resolve()}")


if __name__ == "__main__":
    main()