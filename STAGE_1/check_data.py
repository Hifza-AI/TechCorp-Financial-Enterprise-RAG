"""
check_data.py  -  Saare chunk files ki 1-minute jaanch: company ke naam aur saal (year) theek hain ya nahi.

Chalao:   python STAGE_1/check_data.py
Report:   screen par + STAGE_1/eval/data_check.txt

Kyun: retriever company aur saal se filter karta hai. Agar ek hi company do naamon se ho
("KMT" aur "KENNAMETAL") ya kisi file ka saal galat label ho, to sawal ka jawab hote hue bhi
retriever "No data found" kahega -- ye retriever ki nahi, data ke label ki galti hogi.
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from collections import Counter

COLS = re.compile(r"^Columns:\s*(.+)$", re.M)
YEAR4 = re.compile(r"\b(?:19|20)\d{2}\b")


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="STAGE_1/chunks")
    ap.add_argument("--out", default="STAGE_1/eval/data_check.txt")
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

    text = "\n".join(lines)
    print(text)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"\nReport save hui: {out.resolve()}")


if __name__ == "__main__":
    main()