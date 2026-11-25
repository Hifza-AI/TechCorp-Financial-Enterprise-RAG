"""
financials_db.py  -  SQL track ki SERVING (gold) layer: har company + fiscal year + metric ka EK final number,
jo answer layer seedha use karegi.

Layers (medallion):
    raw       facts.db -> tables / facts       PDF ka har cell (~1M rows)
    curated   facts.db -> metrics / checks     PDF se 12 metrics + apni checks          (metrics_db.py)
    external  facts.db -> xbrl_facts           SEC ka official XBRL (~1.3M facts)        (xbrl_db.py)
    check     facts.db -> validation           PDF vs SEC milan                          (validate_xbrl.py)
    serving   facts.db -> financials           YEH FILE: reconcile kar ke ek final number

Banao:   python STAGE_1/financials_db.py build
Dekho:   python STAGE_1/financials_db.py show Nike revenue
         python STAGE_1/financials_db.py show Apple net_income 2023
Hisaab:  python STAGE_1/financials_db.py report

Reconcile ka qaida (industry mein "source of truth" chunna):
    PDF = SEC (match)      -> SEC ka number, PDF ka page citation ke saath        status sec_verified
    PDF != SEC (mismatch)  -> SEC ka number (PDF parser ne ghalat row chuni)       status sec_only
    sirf SEC (PDF ne miss) -> SEC ka number                                        status sec_only
    sirf PDF (SEC mein nahi)-> PDF ka number, curated status ke saath               status pdf_verified /
                                                                                    pdf_unchecked / pdf_conflict
Har number "as originally reported" hai -- us saal ke 10-K mein jo chhapa tha (restated nahi).
value: USD mein (EPS: USD per share).
"""
import argparse
import re
import sqlite3
import time
from pathlib import Path

METRIC_LABELS = {
    "revenue": "Revenue", "gross_profit": "Gross profit", "operating_income": "Operating income",
    "net_income": "Net income", "eps_diluted": "Diluted EPS", "rnd": "Research and development expense",
    "total_assets": "Total assets", "total_liabilities": "Total liabilities", "equity": "Shareholders' equity",
    "cash": "Cash and cash equivalents", "operating_cash_flow": "Net cash from operating activities",
    "capex": "Capital expenditures",
}
ALIASES = {
    "sales": "revenue", "revenues": "revenue", "net sales": "revenue", "total revenue": "revenue",
    "gross margin": "gross_profit", "operating profit": "operating_income", "net income": "net_income",
    "net earnings": "net_income", "profit": "net_income", "eps": "eps_diluted", "earnings per share": "eps_diluted",
    "r&d": "rnd", "research and development": "rnd", "assets": "total_assets", "liabilities": "total_liabilities",
    "shareholders equity": "equity", "stockholders equity": "equity", "cash and cash equivalents": "cash",
    "operating cash flow": "operating_cash_flow", "cash from operations": "operating_cash_flow",
    "capital expenditures": "capex", "capital expenditure": "capex",
}
SCALE = {"thousands": 1e3, "millions": 1e6, "billions": 1e9}
SEC_FILING = "https://www.sec.gov/Archives/edgar/data/{cik}/{accn}/"
SEC_STATUSES = ("sec_verified", "sec_only")

SCHEMA = """
DROP TABLE IF EXISTS financials;
CREATE TABLE financials(
    company TEXT, fiscal_year INTEGER, metric TEXT,
    value REAL,              -- USD (EPS: USD per share)
    status TEXT,             -- sec_verified | sec_only | pdf_verified | pdf_unchecked | pdf_conflict
    source TEXT,             -- pdf+sec | sec | pdf
    file_name TEXT, page TEXT, row_label TEXT,       -- PDF citation (jab PDF ka number sahi hai)
    pdf_value_text TEXT, pdf_unit TEXT,
    xbrl_concept TEXT, sec_url TEXT, note TEXT,
    PRIMARY KEY(company, fiscal_year, metric));
CREATE INDEX ix_fin ON financials(company, metric);
"""


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9%&]+", " ", (s or "").lower())).strip()


def sec_url(cik, accn):
    return SEC_FILING.format(cik=cik, accn=accn.replace("-", "")) if cik and accn else None


def build(db):
    t0 = time.time()
    con = sqlite3.connect(db)
    need = [t for t in ("metrics", "validation", "validation_missed", "xbrl_companies") if not con.execute(
        "SELECT name FROM sqlite_master WHERE name=?", (t,)).fetchone()]
    if need:
        raise SystemExit(f"Table nahi mili: {need}. Pehle metrics_db.py build, xbrl_db.py load, validate_xbrl.py run.")
    con.executescript(SCHEMA)
    cik = dict(con.execute("SELECT company, cik FROM xbrl_companies"))
    # sec_only rows ke liye: us saal ki PDF file (document citation)
    files = {(c, fy): f for c, fy, f in con.execute(
        "SELECT company, fiscal_year, MIN(file_name) FROM metrics GROUP BY company, fiscal_year")}

    rows = []
    for (comp, fy, metric, our_text, our_unit, result, xv, xc, accn, status,
         fname, page, label, our_val) in con.execute(
            "SELECT v.company, v.fiscal_year, v.metric, v.our_text, v.our_unit, v.result, v.xbrl_value, "
            "v.xbrl_concept, v.xbrl_accn, v.curated_status, m.file_name, m.page, m.row_label, m.value "
            "FROM validation v JOIN metrics m USING(company, fiscal_year, metric)"):
        url = sec_url(cik.get(comp), accn)
        if result == "match":
            rows.append((comp, fy, metric, xv, "sec_verified", "pdf+sec", fname, page, label, our_text, our_unit,
                         xc, url, None))
        elif result == "mismatch":
            rows.append((comp, fy, metric, xv, "sec_only", "sec", fname, None, None, our_text, our_unit, xc, url,
                         "PDF parser ka number SEC se nahi mila -- SEC ka number liya"))
        else:  # no_xbrl -> sirf PDF
            if metric == "eps_diluted":
                v = our_val
            elif our_unit in SCALE and our_val is not None:
                v = our_val * SCALE[our_unit]
            else:
                v = None
            st = {"verified": "pdf_verified", "conflict": "pdf_conflict"}.get(status, "pdf_unchecked")
            rows.append((comp, fy, metric, v, st, "pdf", fname, page, label, our_text, our_unit, None, None,
                         None if v is not None else "unit maloom nahi -- number dollars mein nahi badla"))

    # SEC mein hai, PDF curated ne miss kiya
    accn_of = {(c, fy): a for c, fy, a in con.execute(
        "SELECT company, fiscal_year, MIN(xbrl_accn) FROM validation WHERE xbrl_accn IS NOT NULL "
        "GROUP BY company, fiscal_year")}
    for comp, fy, metric, xv, xc in con.execute(
            "SELECT company, fiscal_year, metric, xbrl_value, xbrl_concept FROM validation_missed"):
        rows.append((comp, fy, metric, xv, "sec_only", "sec", files.get((comp, fy)), None, None, None, None, xc,
                     sec_url(cik.get(comp), accn_of.get((comp, fy))), "PDF curated layer mein row nahi thi"))

    con.executemany("INSERT OR REPLACE INTO financials VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    con.commit()
    con.close()
    print(f"Serving layer ban gayi ({time.time() - t0:.0f} sec): {len(rows):,} rows")
    report(db)


# ---------------------------------------------------------------- answer layer ke liye (import karo)

def resolve_company(con, name):
    comps = [r[0] for r in con.execute("SELECT DISTINCT company FROM financials")]
    n = norm(name)
    hit = [c for c in comps if n == norm(c)] or [c for c in comps if n and (n in norm(c) or norm(c) in n)]
    return hit[0] if hit else None


def resolve_metric(name):
    n = norm(name)
    if n.replace(" ", "_") in METRIC_LABELS:
        return n.replace(" ", "_")
    return ALIASES.get(n)


def lookup(con, company, metric, years=None):
    """[{company, fiscal_year, metric, label, value, status, source, file_name, page, sec_url, ...}] saal ki tarteeb
    mein. company/metric pehle resolve_company / resolve_metric se pakke karo."""
    q = "SELECT * FROM financials WHERE company=? AND metric=?"
    args = [company, metric]
    if years:
        q += f" AND fiscal_year IN ({','.join('?' * len(years))})"
        args += list(years)
    cur = con.execute(q + " ORDER BY fiscal_year", args)
    cols = [c[0] for c in cur.description]
    out = []
    for r in cur.fetchall():
        d = dict(zip(cols, r))
        d["label"] = METRIC_LABELS[metric]
        d["display"] = fmt(d["value"], metric)
        out.append(d)
    return out


def fmt(v, metric):
    """USD -> parhne layak: $394.33 billion / $812.4 million / $6.13 per share."""
    if v is None:
        return "n/a"
    if metric == "eps_diluted":
        return f"${v:,.2f} per share" if v >= 0 else f"-${-v:,.2f} per share"
    sign, a = ("-" if v < 0 else ""), abs(v)
    for div, word in ((1e9, "billion"), (1e6, "million"), (1e3, "thousand")):
        if a >= div:
            return f"{sign}${a / div:,.2f} {word}"
    return f"{sign}${a:,.0f}"


# ---------------------------------------------------------------- CLI

def show(db, company, metric, year=None):
    con = sqlite3.connect(db)
    comp, m = resolve_company(con, company), resolve_metric(metric)
    if not comp or not m:
        print(f"Nahi mila: company={company!r} ({comp}) metric={metric!r} ({m}). Metrics: {', '.join(METRIC_LABELS)}")
        return
    rows = lookup(con, comp, m, [year] if year else None)
    if not rows:
        print(f"{comp} {METRIC_LABELS[m]}: koi row nahi")
        return
    print(f"{comp} -- {METRIC_LABELS[m]}")
    for r in rows:
        mark = {"sec_verified": "✓✓", "sec_only": "S ", "pdf_verified": "✓ "}.get(r["status"], "? ")
        cite = f"p{r['page']}" if r["page"] else "-"
        print(f"  FY{r['fiscal_year']}  {r['display']:>22}  {mark} {r['status']:<13} {cite:<6} "
              f"{(r['file_name'] or '')[:40]}" + (f"  [{r['note']}]" if r["note"] else ""))


def report(db):
    con = sqlite3.connect(db)
    total, comps = con.execute("SELECT COUNT(*), COUNT(DISTINCT company) FROM financials").fetchone()
    print(f"\nSERVING LAYER -- {total:,} numbers, {comps} companies\n")
    print(f"  {'metric':<22}{'rows':>6}{'sec_verified':>14}{'sec_only':>10}{'pdf_verified':>14}{'pdf_other':>11}")
    for m in METRIC_LABELS:
        r = con.execute("SELECT COUNT(*), SUM(status='sec_verified'), SUM(status='sec_only'), "
                        "SUM(status='pdf_verified'), SUM(status IN ('pdf_unchecked','pdf_conflict')) "
                        "FROM financials WHERE metric=?", (m,)).fetchone()
        print(f"  {m:<22}{r[0]:>6}{r[1] or 0:>14}{r[2] or 0:>10}{r[3] or 0:>14}{r[4] or 0:>11}")
    st = dict(con.execute("SELECT status, COUNT(*) FROM financials GROUP BY status"))
    sec = sum(st.get(s, 0) for s in SEC_STATUSES)
    print(f"\n  SEC se tasdeeq-shuda (sec_verified + sec_only): {sec:,}/{total:,} = {100 * sec / total:.1f}%")
    print(f"  PDF page citation ke saath:                   {st.get('sec_verified', 0) + st.get('pdf_verified', 0):,}")
    print(f"  Kamzor (pdf_unchecked / pdf_conflict):        "
          f"{st.get('pdf_unchecked', 0)} / {st.get('pdf_conflict', 0)}  (answer mein ehtiyat ka note)\n")


def main():
    ap = argparse.ArgumentParser(description="Serving layer: PDF + SEC reconcile -> ek final number")
    ap.add_argument("--db", default="STAGE_1/facts.db")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    sub.add_parser("report")
    s = sub.add_parser("show")
    s.add_argument("company")
    s.add_argument("metric")
    s.add_argument("year", nargs="?", type=int)
    a = ap.parse_args()
    if not Path(a.db).exists():
        raise SystemExit(f"Database nahi mila: {a.db}")
    {"build": lambda: build(a.db), "report": lambda: report(a.db),
     "show": lambda: show(a.db, a.company, a.metric, a.year)}[a.cmd]()


if __name__ == "__main__":
    main()