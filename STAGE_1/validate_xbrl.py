"""
validate_xbrl.py  -  PDF se nikale curated metrics ko SEC ke official XBRL numbers se milao.

    python STAGE_1/validate_xbrl.py run         (facts.db: metrics + xbrl_facts -> validation table + report)
    python STAGE_1/validate_xbrl.py report      (sirf report)
    python STAGE_1/validate_xbrl.py show Nike revenue

Har curated row (company, fiscal year, metric) ke liye:
  1. XBRL mein wo 10-K dhoondo jis ka "current period" isi fiscal year mein khatam hota hai.
  2. Us filing ka is metric ka number (period sahi: income/cash flow ~1 saal, balance sheet saal ke aakhri din).
  3. Hamara number us se milao. XBRL dollars mein hota hai, hamara millions/thousands mein -- scale khud dhoondo
     (1, 1e3, 1e6, 1e9). Jo scale mile wahi asli unit hai -> hamara unit andaza bhi check ho jata hai.

Natija (result):
  match        hamara number = SEC ka asal (as originally reported) number
  mismatch     SEC mein is period ka number hai, magar hamara us se nahi milta (parsing ghalti, ya ghalat row /
               concept chuna -- haath se dekhna)
  no_xbrl      SEC mein is metric / period ka number nahi mila (company alag concept use karti hai)

Saath mein:
  * cross-year "conflict" rows: agli filing ka number SEC ke restated number se mile -> "restatement confirmed"
  * coverage: SEC mein number hai magar hamari curated layer mein row hi nahi -> PDF pipeline ne miss kiya
"""
import argparse
import re
import sqlite3
import sys
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

# metric -> us-gaap concepts (pehla behtar). Companies alag concept use karti hain, is liye list.
CONCEPTS = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet", "SalesRevenueGoodsNet",
                "RevenuesNetOfInterestExpense", "SalesRevenueServicesNet", "RevenueNet"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"],
    "eps_diluted": ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted"],
    "rnd": ["ResearchAndDevelopmentExpense", "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
            "ResearchAndDevelopmentExpenseSoftwareExcludingAcquiredInProcessCost"],
    "total_assets": ["Assets"],
    "total_liabilities": ["Liabilities"],
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    # "cash + restricted cash" alag cheez hai (hamara metric sirf cash & equivalents), is liye shamil nahi
    "cash": ["CashAndCashEquivalentsAtCarryingValue", "CashAndDueFromBanks", "Cash"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities",
                            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets",
              "PaymentsForCapitalImprovements", "PaymentsToAcquirePropertyPlantAndEquipmentAndIntangibleAssets"],
}
INSTANT = {"total_assets", "total_liabilities", "equity", "cash"}
SCALES = {1.0: None, 1e3: "thousands", 1e6: "millions", 1e9: "billions"}

SCHEMA = """
DROP TABLE IF EXISTS validation;
CREATE TABLE validation(
    company TEXT, fiscal_year INTEGER, metric TEXT, our_value REAL, our_text TEXT, our_unit TEXT, unit_guess INTEGER,
    result TEXT, xbrl_value REAL, xbrl_concept TEXT, xbrl_accn TEXT, scale_unit TEXT, unit_ok INTEGER,
    curated_status TEXT, restatement TEXT, note TEXT, PRIMARY KEY(company, fiscal_year, metric));
DROP TABLE IF EXISTS validation_missed;
DROP TABLE IF EXISTS fy_offset;
CREATE TABLE fy_offset(company TEXT PRIMARY KEY, off INTEGER, matches_off0 INTEGER, matches_off1 INTEGER);
CREATE TABLE validation_missed(company TEXT, fiscal_year INTEGER, metric TEXT, xbrl_value REAL, xbrl_concept TEXT);
"""


def d(s):
    try:
        return date.fromisoformat(s) if s else None
    except ValueError:
        return None


def matches(ours, xv, metric):
    """Hamara number (kisi bhi scale mein) = XBRL dollars? -> scale (1/1e3/1e6/1e9) ya None. Sign nahi dekhte
    (capex kahin (695) kahin 695 likha hota hai)."""
    if ours is None or xv is None:
        return None
    a, b = abs(ours), abs(xv)
    if metric == "eps_diluted":
        return 1.0 if abs(a - b) <= 0.006 else None
    for s in SCALES:
        v = a * s
        if v == b or abs(v - b) <= max(0.6 * s, 0.001 * max(v, b)):  # display rounding ya 0.1%
            return s
    return None


def accession_map(con):
    """company -> list of (accn, form, filed, current_end) har 10-K ka; current_end = us ka sab se naya period."""
    out = defaultdict(list)
    # Sirf us-gaap ke ~1 saal wale periods: cover page (dei) ki tareekhein -- jaise "shares outstanding as of
    # Feb 2022" -- filing ka saal ghalat bata deti thin.
    for comp, accn, form, filed, end in con.execute(
            "SELECT company, accn, MIN(form), MIN(filed), MAX(end) FROM xbrl_facts "
            "WHERE form LIKE '10-K%' AND fp = 'FY' AND taxonomy = 'us-gaap' AND start IS NOT NULL "
            "AND julianday(end) - julianday(start) BETWEEN 300 AND 380 GROUP BY company, accn"):
        e = d(end)
        if e:
            out[comp].append((accn, form, filed, e))
    return out


def pick_accn(accns, fy, off=0):
    """Fiscal year -> us ka 10-K: period ka saal = fy + off (off har company ka ek hi -- company_offsets).
    Asal 10-K behtar 10-K/A se, pehle file hua behtar. Is saal ki filing SEC data mein na ho -> None.
    (Pehle yahan "na mile to agle saal ki Jan/Feb wali" ka rule tha -- Dell FY2019 ki filing na milne par wo
    chupke se FY2020 ki filing utha leta tha. Insaani spot-check ne pakra.)"""
    c = [a for a in accns if a[3].year == fy + off]
    return sorted(c, key=lambda a: (a[1] != "10-K", a[2] or ""))[0] if c else None


def company_offsets(con, acc):
    """Har company ka EK qaida: hamara fiscal year = SEC period ka saal (off 0) ya saal - 1 (off 1, jaise koi
    company "fiscal 2017" kahe aur saal Jan 2018 mein khatam ho). Jo qaida PDF numbers se zyada milta hai wahi.
    -> {company: (off, matches_off0, matches_off1)}"""
    out = {}
    rows = defaultdict(list)
    for comp, fy, metric, val in con.execute("SELECT company, fiscal_year, metric, value FROM metrics "
                                             "WHERE metric IN ('total_assets','revenue','net_income')"):
        rows[comp].append((fy, metric, val))
    for comp, rs in rows.items():
        n = []
        for off in (0, 1):
            k = 0
            for fy, metric, val in rs:
                a = pick_accn(acc.get(comp, []), fy, off)
                if a and any(matches(val, v, metric) for _, v in xbrl_values(con, comp, a[0], a[3], metric)):
                    k += 1
            n.append(k)
        out[comp] = (1 if n[1] > n[0] else 0, n[0], n[1])
    return out


def load_offsets(con):
    """validate run ke baad: {company: off}. Table na ho to sab 0."""
    try:
        return dict(con.execute("SELECT company, off FROM fy_offset"))
    except sqlite3.Error:
        return {}


def xbrl_values(con, comp, accn, end, metric):
    """Ek filing mein is metric ke values, sahi period ke saath: [(concept, value)] concept-preference ki tarteeb mein."""
    cs = CONCEPTS[metric]
    unit_like = "USD/shares" if metric == "eps_diluted" else "USD"
    rows = con.execute(
        f"SELECT concept, value, start, end FROM xbrl_facts WHERE company=? AND accn=? AND unit=? AND end=? "
        f"AND concept IN ({','.join('?' * len(cs))})", (comp, accn, unit_like, end.isoformat(), *cs)).fetchall()
    out = []
    for concept, value, start, e in rows:
        if metric in INSTANT:
            if start:
                continue
        else:
            s = d(start)
            if not s or not (300 <= (end - s).days <= 380):
                continue
        out.append((cs.index(concept), concept, value))
    out.sort()
    seen, res = set(), []
    for _, concept, value in out:
        if (concept, value) not in seen:
            seen.add((concept, value))
            res.append((concept, value))
    return res


def run(db):
    t0 = time.time()
    con = sqlite3.connect(db)
    need = [t for t in ("metrics", "xbrl_facts") if not con.execute(
        "SELECT name FROM sqlite_master WHERE name=?", (t,)).fetchone()]
    if need:
        raise SystemExit(f"Table nahi mili: {need}. Pehle metrics_db.py build / xbrl_db.py load.")
    con.executescript(SCHEMA)
    acc = accession_map(con)
    offs = company_offsets(con, acc)
    con.executemany("INSERT INTO fy_offset VALUES (?,?,?,?)", [(c, *v) for c, v in offs.items()])
    off_of = {c: v[0] for c, v in offs.items()}
    have = {r[0] for r in con.execute("SELECT DISTINCT company FROM xbrl_facts")}
    nxt = {(c, fy, m): (v, t) for c, fy, m, v, t in con.execute(
        "SELECT company, fiscal_year, metric, NULL, detail FROM checks WHERE check_name='cross_year' AND ok=0")}

    rows = con.execute("SELECT company, fiscal_year, metric, value, value_text, unit, unit_guess, status "
                       "FROM metrics").fetchall()
    out = []
    for comp, fy, metric, val, vt, unit, ug, status in rows:
        rec = dict(company=comp, fiscal_year=fy, metric=metric, our_value=val, our_text=vt, our_unit=unit,
                   unit_guess=ug, result="no_xbrl", xbrl_value=None, xbrl_concept=None, xbrl_accn=None,
                   scale_unit=None, unit_ok=None, curated_status=status, restatement=None, note=None)
        a = pick_accn(acc.get(comp, []), fy, off_of.get(comp, 0)) if comp in have else None
        if not a:
            rec["note"] = "XBRL mein is saal ka 10-K nahi" if comp in have else "company XBRL mein nahi"
            out.append(rec)
            continue
        rec["xbrl_accn"] = a[0]
        vals = xbrl_values(con, comp, a[0], a[3], metric)
        if not vals:
            out.append(rec)
            continue
        hit = next(((c, v, s) for c, v in vals for s in [matches(val, v, metric)] if s), None)
        if hit:
            c, v, s = hit
            rec.update(result="match", xbrl_value=v, xbrl_concept=c, scale_unit=SCALES[s])
            if metric != "eps_diluted" and unit:
                rec["unit_ok"] = int(SCALES[s] == unit)
        else:
            c, v = vals[0]
            rec.update(result="mismatch", xbrl_value=v, xbrl_concept=c,
                       note="; ".join((f"{cc}={vv:,.2f}" if metric == "eps_diluted" else f"{cc}={vv:,.0f}")
                                      for cc, vv in vals[:3]))
        # cross-year conflict: agli filing ka number SEC ke restated number se milta hai?
        if (comp, fy, metric) in nxt:
            a2 = pick_accn(acc.get(comp, []), fy + 1, off_of.get(comp, 0))
            later_val = None
            m = re.search(r"vs ([\(\)\d,.\-]+) \(filing", nxt[(comp, fy, metric)][1] or "")
            if m:
                txt = m.group(1)
                later_val = float(txt.replace(",", "").strip("()")) * (-1 if txt.startswith("(") else 1) \
                    if re.search(r"\d", txt) else None
            restated = xbrl_values(con, comp, a2[0], a[3], metric) if a2 else []
            if later_val is not None and any(matches(later_val, v, metric) for _, v in restated):
                rec["restatement"] = "confirmed" if rec["result"] == "match" else "later_value_matches_sec"
            elif restated:
                rec["restatement"] = "not_confirmed"
        out.append(rec)

    cols = ["company", "fiscal_year", "metric", "our_value", "our_text", "our_unit", "unit_guess", "result",
            "xbrl_value", "xbrl_concept", "xbrl_accn", "scale_unit", "unit_ok", "curated_status", "restatement", "note"]
    con.executemany(f"INSERT INTO validation VALUES ({','.join('?' * len(cols))})",
                    [tuple(r[c] for c in cols) for r in out])

    # coverage: SEC mein number hai, curated mein row nahi (sirf un filings ke liye jo hamare paas hain)
    ours = {(r["company"], r["fiscal_year"], r["metric"]) for r in out}
    fys = defaultdict(set)
    for r in out:
        fys[r["company"]].add(r["fiscal_year"])
    missed = []
    for comp, years in fys.items():
        for fy in years:
            a = pick_accn(acc.get(comp, []), fy, off_of.get(comp, 0))
            if not a:
                continue
            for metric in CONCEPTS:
                if (comp, fy, metric) in ours:
                    continue
                vals = xbrl_values(con, comp, a[0], a[3], metric)
                if vals:
                    missed.append((comp, fy, metric, vals[0][1], vals[0][0]))
    con.executemany("INSERT INTO validation_missed VALUES (?,?,?,?,?)", missed)
    con.commit()
    con.close()
    print(f"Validation ({time.time() - t0:.0f} sec): {len(out):,} curated rows, {len(missed):,} SEC values jo "
          f"curated mein nahi")
    odd = {c: v for c, v in offs.items() if v[0] == 1}
    print(f"Fiscal year qaida: {len(offs) - len(odd)} companies off 0" +
          (f", off 1: " + ", ".join(f"{c} ({v[1]} vs {v[2]})" for c, v in odd.items()) if odd else ""))
    no_file = sum(1 for r in out if r["note"] == "XBRL mein is saal ka 10-K nahi")
    print(f"Jin saalon ki apni 10-K SEC data mein nahi (ab agle saal ki filing NAHI uthate): {no_file} rows")
    report(db)


def report(db):
    con = sqlite3.connect(db)
    print(f"\nPDF CURATED METRICS vs SEC XBRL\n")
    print(f"  {'metric':<22}{'rows':>6}{'XBRL hai':>10}{'match':>8}{'match %':>9}{'mismatch':>10}{'SEC hai,PDF nahi':>18}")
    tot = [0, 0, 0, 0, 0]
    for m in CONCEPTS:
        n, x, ok, bad = con.execute(
            "SELECT COUNT(*), SUM(result!='no_xbrl'), SUM(result='match'), SUM(result='mismatch') FROM validation "
            "WHERE metric=?", (m,)).fetchone()
        x, ok, bad = x or 0, ok or 0, bad or 0
        miss = con.execute("SELECT COUNT(*) FROM validation_missed WHERE metric=?", (m,)).fetchone()[0]
        pct = f"{100 * ok / x:.1f}%" if x else "-"
        print(f"  {m:<22}{n:>6}{x:>10}{ok:>8}{pct:>9}{bad:>10}{miss:>18}")
        for i, v in enumerate((n, x, ok, bad, miss)):
            tot[i] += v
    print(f"  {'KUL':<22}{tot[0]:>6}{tot[1]:>10}{tot[2]:>8}"
          f"{(f'{100 * tot[2] / tot[1]:.1f}%' if tot[1] else '-'):>9}{tot[3]:>10}{tot[4]:>18}")
    if tot[1]:
        print(f"\n  Accuracy (jahan SEC number hai):  {tot[2]}/{tot[1]} = {100 * tot[2] / tot[1]:.1f}%")
    if tot[0] + tot[4]:
        print(f"  Recall (SEC ke numbers mein se PDF curated ne kitne pakre, match): "
              f"{tot[2]}/{tot[1] + tot[4]} = {100 * tot[2] / (tot[1] + tot[4]):.1f}%")
    u = con.execute("SELECT COUNT(*), SUM(unit_ok), SUM(unit_guess=1), SUM(unit_guess=1 AND unit_ok=1) "
                    "FROM validation WHERE unit_ok IS NOT NULL").fetchone()
    if u[0]:
        print(f"  Unit sahi (millions/thousands): {u[1]}/{u[0]} = {100 * u[1] / u[0]:.1f}%"
              + (f"   (andaze wale: {u[3]}/{u[2]})" if u[2] else ""))
    r = con.execute("SELECT restatement, COUNT(*) FROM validation WHERE restatement IS NOT NULL "
                    "GROUP BY restatement").fetchall()
    if r:
        print("  Cross-year conflicts: " + ", ".join(f"{k} {v}" for k, v in r))
    c = con.execute("SELECT curated_status, result, COUNT(*) FROM validation GROUP BY 1, 2").fetchall()
    if c:
        print("  Curated status vs SEC: " + "; ".join(f"{a}/{b} {n}" for a, b, n in c))
    bad = con.execute("SELECT company, fiscal_year, metric, our_text, our_unit, note FROM validation "
                      "WHERE result='mismatch' ORDER BY company, fiscal_year LIMIT 15").fetchall()
    if bad:
        print("\n  Mismatch misaalen (pehle 15):")
        for b in bad:
            print(f"    {b[0]} FY{b[1]} {b[2]}: hamara {b[3]} {b[4] or ''} | SEC: {b[5]}")
    print()


def show(db, company, metric):
    con = sqlite3.connect(db)
    rows = con.execute("SELECT fiscal_year, our_text, our_unit, result, xbrl_value, xbrl_concept, scale_unit, "
                       "restatement, note FROM validation WHERE lower(company) LIKE ? AND metric=? "
                       "ORDER BY fiscal_year", (f"%{company.lower()}%", metric)).fetchall()
    if not rows:
        print("Kuch nahi mila (company ka naam / metric check karo).")
    for fy, vt, unit, res, xv, xc, su, rs, note in rows:
        x = f"{xv:,.0f} ({xc}, scale {su})" if xv is not None and metric != "eps_diluted" else (
            f"{xv} ({xc})" if xv is not None else "-")
        print(f"  FY{fy}  hamara {vt:>12} {unit or '':<10} {res:<9} SEC {x}"
              + (f"  restatement:{rs}" if rs else "") + (f"  [{note}]" if note and res != "match" else ""))


def main():
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser(description="PDF curated metrics vs SEC XBRL")
    ap.add_argument("--db", default="STAGE_1/facts.db")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run")
    sub.add_parser("report")
    s = sub.add_parser("show")
    s.add_argument("company")
    s.add_argument("metric")
    a = ap.parse_args()
    if not Path(a.db).exists():
        raise SystemExit(f"Database nahi mila: {a.db}")
    {"run": lambda: run(a.db), "report": lambda: report(a.db),
     "show": lambda: show(a.db, a.company, a.metric)}[a.cmd]()


if __name__ == "__main__":
    main()