"""
spotcheck.py  -  Serving layer (financials) ki INSAANI jaanch: random numbers ko asal PDF khol kar dekho.

    python STAGE_1/spotcheck.py make       (ek dafa: STAGE_1/eval/spotcheck.csv -- 60 random rows, FREEZE)
    python STAGE_1/spotcheck.py report     (tum CSV bhar do, phir: kitne sahi, 95% CI ke saath)

Kyun: eval_sql.py routing naapta hai (sawal -> sahi row). Magar gold aur serving dono SEC ka WAHI concept use
karte hain, is liye "kya ye concept asal mein wahi number hai jo 10-K ke statement mein likha hai?" sirf insaan
PDF dekh kar bata sakta hai. Companies ise "manual QA sample" kehti hain.

CSV mein har row: company, saal, metric, hamara number, aur kahan dekhna hai (file, page agar maloom, statement).
Tum 3 khaane bharo (Excel mein kholo):
    pdf_value   PDF mein jo number dikha (jaisa likha, misaal "51,217")
    verdict     Y = hamara number sahi (rounding ka farq chalega)
                N = ghalat (ya ghalat line item, jaise net sales ki jagah total revenues)
                ? = PDF mein dhoondh nahi saki / samajh nahi aaya
    comment     kuch bhi (zaroori nahi)

Har saal ki 10-K ke statement mein wahi saal ka number dekhna hai (as originally reported) -- restated nahi.
"""
import argparse
import csv
import math
import random
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from financials_db import METRIC_LABELS, fmt     # noqa: E402

WHERE = {
    "revenue": "Income statement (Consolidated Statements of Operations / Income) -- total revenue / net sales",
    "gross_profit": "Income statement -- gross profit / gross margin",
    "operating_income": "Income statement -- operating income",
    "net_income": "Income statement -- net income attributable to the company",
    "eps_diluted": "Income statement -- diluted earnings per share",
    "rnd": "Income statement -- research and development",
    "total_assets": "Balance sheet -- total assets",
    "total_liabilities": "Balance sheet -- total liabilities",
    "equity": "Balance sheet -- total shareholders' equity (company ka, noncontrolling ke baghair)",
    "cash": "Balance sheet -- cash and cash equivalents",
    "operating_cash_flow": "Cash flow statement -- net cash provided by operating activities",
    "capex": "Cash flow statement -- capital expenditures / purchases of property, plant and equipment",
}
# Kis status se kitne (zyada khatre wale hisson se zyada): total 60
PLAN = [("sec_verified", None, 20), ("sec_only", "standard", 10), ("sec_only", "mismatch", 10),
        ("sec_only", "fallback", 8), ("sec_only", "missed", 6), ("pdf", None, 6)]
FIELDS = ["id", "company", "fiscal_year", "metric", "our_value", "our_exact", "status", "group", "file_name", "page",
          "where_to_look", "sec_concept", "note", "pdf_value", "verdict", "comment"]


def group_of(status, note):
    note = note or ""
    if status == "sec_verified":
        return "sec_verified"
    if status.startswith("pdf"):
        return "pdf"
    if "standard" in note:
        return "standard"
    if "agli filing" in note:
        return "fallback"
    if "row nahi thi" in note:
        return "missed"
    return "mismatch"


def make(db, out, seed=13):
    if out.exists():
        raise SystemExit(f"{out} pehle se hai (freeze). Naya banana ho to pehle khud delete karo.")
    con = sqlite3.connect(db)
    rows = con.execute("SELECT company, fiscal_year, metric, value, status, file_name, page, xbrl_concept, note "
                       "FROM financials WHERE value IS NOT NULL AND status != 'pdf_suspect'").fetchall()
    pools = defaultdict(list)
    for r in rows:
        pools[group_of(r[4], r[8])].append(r)
    rng = random.Random(seed)
    picked = []
    for status, sub, n in PLAN:
        key = status if sub is None else sub
        pool = pools.get(key, [])
        picked += [(key, r) for r in rng.sample(pool, min(n, len(pool)))]
    rng.shuffle(picked)
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for i, (g, (c, fy, m, v, st, fn, pg, xc, note)) in enumerate(picked, 1):
            exact = f"{v:,.2f}" if m == "eps_diluted" else f"{v / 1e6:,.1f} million"
            w.writerow({"id": f"Q{i:02d}", "company": c, "fiscal_year": fy, "metric": METRIC_LABELS[m],
                        "our_value": fmt(v, m), "our_exact": exact, "status": st, "group": g,
                        "file_name": fn or "", "page": pg or "(statement dhoondo)", "where_to_look": WHERE[m],
                        "sec_concept": xc or "", "note": note or "", "pdf_value": "", "verdict": "", "comment": ""})
    print(f"{out}: {len(picked)} rows  ->  Excel mein kholo, pdf_value / verdict (Y/N/?) / comment bharo, CSV hi "
          f"save karo.")
    for status, sub, n in PLAN:
        key = status if sub is None else sub
        print(f"  {key:<14}{sum(1 for g, _ in picked if g == key):>4}")


def wilson(k, n, z=1.96):
    if not n:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def report(out):
    if not out.exists():
        raise SystemExit("Pehle:  python STAGE_1/spotcheck.py make")
    with out.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    done = [r for r in rows if r["verdict"].strip().upper() in ("Y", "N")]
    unsure = [r for r in rows if r["verdict"].strip() == "?"]
    empty = len(rows) - len(done) - len(unsure)
    print(f"\nINSAANI SPOT-CHECK -- {len(rows)} rows: faisla {len(done)}, '?' {len(unsure)}, khaali {empty}\n")
    by = defaultdict(list)
    for r in done:
        by[r["group"]].append(r["verdict"].strip().upper() == "Y")
    for g, v in by.items():
        print(f"  {g:<14}{sum(v):>4}/{len(v):<4}{100 * sum(v) / len(v):6.1f}%")
    k = sum(r["verdict"].strip().upper() == "Y" for r in done)
    if done:
        lo, hi = wilson(k, len(done))
        print(f"\n  KUL: {k}/{len(done)} = {100 * k / len(done):.1f}%   (95% CI {100 * lo:.1f}-{100 * hi:.1f})")
        print("  Note: groups barabar nahi (khatre wale hisse zyada liye) -- asal serving accuracy is se behtar hogi;")
        print("        sec_verified group (zyada-tar rows) ka number alag dekho.")
    bad = [r for r in done if r["verdict"].strip().upper() == "N"] + unsure
    if bad:
        print("\n  N / ? rows:")
        for r in bad:
            print(f"    {r['id']} {r['company']} FY{r['fiscal_year']} {r['metric']}: hamara {r['our_value']} | PDF "
                  f"{r['pdf_value'] or '-'} [{r['verdict']}] {r['comment']}")
    print()


def main():
    ap = argparse.ArgumentParser(description="Serving layer ka insaani spot-check")
    ap.add_argument("--db", default="STAGE_1/facts.db")
    ap.add_argument("--out", default="STAGE_1/eval/spotcheck.csv")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("make")
    sub.add_parser("report")
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    make(a.db, out) if a.cmd == "make" else report(out)


if __name__ == "__main__":
    main()