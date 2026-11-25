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
    us filing ke XBRL mein nahi -> SEC ki doosri 10-K (agle saal ka comparative column) se, upar wala qaida
    sirf PDF (SEC mein kahin nahi)-> PDF ka number, curated status ke saath         status pdf_verified /
                                                                                    pdf_unchecked / pdf_conflict
    sirf PDF, magar company ke SEC numbers se 5x door -> pdf_suspect (answer mein use nahi hota)
Har number "as originally reported" hai -- us saal ke 10-K mein jo chhapa tha (restated nahi).
value: USD mein (EPS: USD per share).
"""
import argparse
import re
import sqlite3
import time
from pathlib import Path

from validate_xbrl import CONCEPTS, INSTANT, accession_map, d, matches, pick_accn, xbrl_values

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
    status TEXT,             -- sec_verified | sec_only | pdf_verified | pdf_unchecked | pdf_conflict | pdf_suspect
    source TEXT,             -- pdf+sec | sec | pdf
    file_name TEXT, page TEXT, row_label TEXT,       -- PDF citation (jab PDF ka number sahi hai)
    pdf_value_text TEXT, pdf_unit TEXT,
    xbrl_concept TEXT, sec_url TEXT, note TEXT,
    accn TEXT,               -- SEC filing (accession) jis se number liya
    prior_value REAL,        -- pichle saal ka number ISI filing mein (restated ho to naya) -> % badlao isi se
    PRIMARY KEY(company, fiscal_year, metric));
CREATE INDEX ix_fin ON financials(company, metric);
"""


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9%&]+", " ", (s or "").lower())).strip()


def sec_url(cik, accn):
    return SEC_FILING.format(cik=cik, accn=accn.replace("-", "")) if cik and accn else None


def sec_fallback(con, comp, fy, metric):
    """Jab validation ko is saal ki filing ka number na mile: SEC ke KISI bhi 10-K mein is fiscal year ka number
    (aksar agle saal ki filing ka "pichla saal" column). -> (concept, value, accn) ya None.
    Saal ka qaida validate_xbrl.pick_accn jaisa: pehle period ka saal = fy, phir fy+1 (Jan/Feb mein khatam)."""
    cs = CONCEPTS[metric]
    unit = "USD/shares" if metric == "eps_diluted" else "USD"
    rows = con.execute(
        f"SELECT concept, value, start, end, filed, accn FROM xbrl_facts WHERE company=? AND unit=? "
        f"AND form LIKE '10-K%' AND concept IN ({','.join('?' * len(cs))}) "
        f"AND substr(end, 1, 4) IN (?, ?)", (comp, unit, *cs, str(fy), str(fy + 1))).fetchall()
    good = []
    for concept, value, start, e, filed, accn in rows:
        e = d(e)
        if not e:
            continue
        if metric in INSTANT:
            if start:
                continue
        else:
            st = d(start)
            if not st or not (300 <= (e - st).days <= 380):
                continue
        good.append((concept, value, e, filed or "", accn))
    for want in ((lambda e: e.year == fy), (lambda e: e.year == fy + 1 and e.month <= 2)):
        c = [g for g in good if want(g[2])]
        if c:
            # concept-preference, phir saal ka aakhri din (period end), phir sab se pehle file hua (as reported)
            c.sort(key=lambda g: (cs.index(g[0]), -g[2].toordinal(), g[3]))
            return c[0][0], c[0][1], c[0][4], c[0][2]
    return None


def prior_in_filing(con, comp, accn, concept, metric, end=None):
    """Isi filing (accn) mein pichle saal ka number: wahi concept, period ~1 saal pehle khatam. Company ne
    accounting badli ho (jaise ASC 606) to ye restated number hota hai -- saal-ba-saal badlao isi se sahi hai."""
    unit = "USD/shares" if metric == "eps_diluted" else "USD"
    by_end = {}
    for value, start, e in con.execute("SELECT value, start, end FROM xbrl_facts WHERE company=? AND concept=? "
                                       "AND accn=? AND unit=?", (comp, concept, accn, unit)):
        e = d(e)
        if not e:
            continue
        if metric in INSTANT:
            if start:
                continue
        else:
            st = d(start)
            if not st or not (300 <= (e - st).days <= 380):
                continue
        by_end.setdefault(e, value)
    ends = sorted(by_end, reverse=True)
    ref = end or (ends[0] if ends else None)       # end: fallback number ka apna period (agli filing se aaya ho)
    for e in ends:
        if ref and 330 <= (ref - e).days <= 400:
            return by_end[e]
    return None


def build(db):
    t0 = time.time()
    con = sqlite3.connect(db)
    need = [t for t in ("metrics", "validation", "validation_missed", "xbrl_companies", "xbrl_facts")
            if not con.execute("SELECT name FROM sqlite_master WHERE name=?", (t,)).fetchone()]
    if need:
        raise SystemExit(f"Table nahi mili: {need}. Pehle metrics_db.py build, xbrl_db.py load, validate_xbrl.py run.")
    con.executescript(SCHEMA)
    cik = dict(con.execute("SELECT company, cik FROM xbrl_companies"))
    # sec rows ke liye: us saal ki PDF file (document citation)
    files = {(c, fy): f for c, fy, f in con.execute(
        "SELECT company, fiscal_year, MIN(file_name) FROM metrics GROUP BY company, fiscal_year")}

    rows = {}
    own_accn = {}          # key -> accn jis se number liya (prior_value isi filing se)
    any_accn = {}
    fb_end = {}            # fallback rows: number ka period end (agli filing mein ye "current" period nahi hota)
    fallback_n = 0
    standard_differs = 0
    acc = accession_map(con)
    for (comp, fy, metric, our_text, our_unit, result, xv, xc, accn, status,
         fname, page, label, our_val) in con.execute(
            "SELECT v.company, v.fiscal_year, v.metric, v.our_text, v.our_unit, v.result, v.xbrl_value, "
            "v.xbrl_concept, v.xbrl_accn, v.curated_status, m.file_name, m.page, m.row_label, m.value "
            "FROM validation v JOIN metrics m USING(company, fiscal_year, metric)"):
        key = (comp, fy, metric)
        url = sec_url(cik.get(comp), accn)
        if result in ("match", "mismatch") and accn:
            own_accn[key] = any_accn[key] = accn
            # Ek metric = ek tareef: hamesha pehla (standard) concept -- jaise net income = NetIncomeLoss (company
            # ke shareholders ka), revenue = Revenues (total). PDF ne doosri line pakri ho (net sales, ya
            # noncontrolling ke saath net income) to bhi number standard wala; PDF page tabhi jab woh isi se mile.
            a = pick_accn(acc.get(comp, []), fy)
            vals = xbrl_values(con, comp, a[0], a[3], metric) if a and a[0] == accn else []
            pc, pv = vals[0] if vals else (xc, xv)
            if matches(our_val, pv, metric):
                rows[key] = (pv, "sec_verified", "pdf+sec", fname, page, label, our_text, our_unit, pc, url, None)
            elif result == "match":
                standard_differs += 1
                rows[key] = (pv, "sec_only", "sec", fname, None, None, our_text, our_unit, pc, url,
                             f"PDF ki row doosri line item thi ({xc}); standard {pc} liya")
            else:
                rows[key] = (pv, "sec_only", "sec", fname, None, None, our_text, our_unit, pc, url,
                             "PDF parser ka number SEC se nahi mila -- SEC ka number liya")
            continue
        # no_xbrl: pehle SEC ki doosri filings mein dhoondo
        fb = sec_fallback(con, comp, fy, metric)
        if fb:
            fallback_n += 1
            c, v, a, e = fb
            any_accn[key] = a
            own_accn[key], fb_end[key] = a, e
            if matches(our_val, v, metric):
                rows[key] = (v, "sec_verified", "pdf+sec", fname, page, label, our_text, our_unit, c,
                             sec_url(cik.get(comp), a), "SEC number agli filing se")
            else:
                rows[key] = (v, "sec_only", "sec", fname, None, None, our_text, our_unit, c, sec_url(cik.get(comp), a),
                             "SEC number agli filing se; PDF parser ka number nahi mila")
            continue
        if metric == "eps_diluted":
            v = our_val
        elif our_unit in SCALE and our_val is not None:
            v = our_val * SCALE[our_unit]
        else:
            v = None
        st = {"verified": "pdf_verified", "conflict": "pdf_conflict"}.get(status, "pdf_unchecked")
        rows[key] = (v, st, "pdf", fname, page, label, our_text, our_unit, None, None,
                     None if v is not None else "unit maloom nahi -- number dollars mein nahi badla")

    # SEC mein hai, PDF curated ne miss kiya
    accn_of = {(c, fy): a for c, fy, a in con.execute(
        "SELECT company, fiscal_year, MIN(xbrl_accn) FROM validation WHERE xbrl_accn IS NOT NULL "
        "GROUP BY company, fiscal_year")}
    for comp, fy, metric, xv, xc in con.execute(
            "SELECT company, fiscal_year, metric, xbrl_value, xbrl_concept FROM validation_missed"):
        if (comp, fy, metric) not in rows and accn_of.get((comp, fy)):
            own_accn[(comp, fy, metric)] = any_accn[(comp, fy, metric)] = accn_of[(comp, fy)]
        rows.setdefault((comp, fy, metric), (xv, "sec_only", "sec", files.get((comp, fy)), None, None, None, None,
                                             xc, sec_url(cik.get(comp), accn_of.get((comp, fy))),
                                             "PDF curated layer mein row nahi thi"))
    # hamare har filing-saal ke liye: jo metric abhi bhi khaali hai, SEC ki doosri filings se
    for comp, fy in files:
        for metric in METRIC_LABELS:
            if (comp, fy, metric) in rows:
                continue
            fb = sec_fallback(con, comp, fy, metric)
            if fb:
                fallback_n += 1
                c, v, a, e = fb
                any_accn[(comp, fy, metric)] = a
                own_accn[(comp, fy, metric)], fb_end[(comp, fy, metric)] = a, e
                rows[(comp, fy, metric)] = (v, "sec_only", "sec", files[(comp, fy)], None, None, None, None, c,
                                            sec_url(cik.get(comp), a), "PDF mein row nahi thi; SEC number agli filing se")

    # sanity guard: sirf-PDF number agar company ke SEC numbers (isi metric) se 5x se zyada door ho -> pdf_suspect
    # (misaal: Boeing ka "total assets" kisi segment ki chhoti balance sheet se -- apni check pass kar leta hai)
    suspect = 0
    sec_vals = {}
    for (comp, fy, metric), r in rows.items():
        if r[2] != "pdf" and r[0]:
            sec_vals.setdefault((comp, metric), []).append(abs(r[0]))
    for key, r in list(rows.items()):
        comp, fy, metric = key
        ref = sorted(sec_vals.get((comp, metric), []))
        if r[2] != "pdf" or not r[0] or not ref or metric == "eps_diluted":
            continue
        med = ref[len(ref) // 2]
        if not (med / 5 <= abs(r[0]) <= med * 5):
            suspect += 1
            rows[key] = (r[0], "pdf_suspect") + r[2:10] + (
                f"SEC ke baqi saalon ({fmt(med, metric)} ke aas paas) se bohat door -- answer mein use nahi",)

    # pichle saal ka number usi filing ke comparative column se (restatement ke baad wala, "same basis")
    priors = 0
    out = []
    for k, v in rows.items():
        pv = (prior_in_filing(con, k[0], own_accn[k], v[8], k[2], fb_end.get(k))
              if k in own_accn and v[8] and v[2] != "pdf" else None)
        priors += pv is not None
        out.append(k + v + (any_accn.get(k), pv))
    con.executemany("INSERT INTO financials VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", out)
    con.commit()
    con.close()
    print(f"Serving layer ban gayi ({time.time() - t0:.0f} sec): {len(rows):,} rows "
          f"(SEC fallback {fallback_n}, pdf_suspect {suspect}, pichle saal ka comparative {priors}, "
          f"PDF ki row standard concept se alag {standard_differs})")
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


# ---------------------------------------------------------------- answer layer: sawal -> structured source

# Sawal mein metric ke alfaaz (pehle khaas, phir aam).  Sirf "kitna / kya tha" jaise number sawalon ke liye.
METRIC_PATTERNS = [
    ("eps_diluted", r"\b(?:diluted )?(?:eps|earnings per (?:diluted )?share)\b"),
    ("gross_profit", r"\bgross (?:profit|margin)\b"),
    ("operating_income", r"\b(?:operating (?:income|profit|loss)|income from operations)\b"),
    # "profit" akela = net income ("gross profit" / "operating profit" nahi)
    ("net_income", r"\b(?:net (?:income|earnings|profit|loss)|(?<!gross )(?<!operating )profits?)\b"),
    ("rnd", r"\b(?:r ?& ?d|r and d|research and development)\b"),
    # "assets" / "liabilities" akele = total (current / intangible / lease wagaira nahi)
    ("total_assets", r"\b(?:total assets|(?<!current )(?<!intangible )(?<!net )(?<!other )(?<!tax )assets)\b"),
    ("total_liabilities", r"\b(?:total liabilities|(?<!current )(?<!contingent )(?<!lease )(?<!other )(?<!tax )"
                          r"liabilities)\b"),
    ("equity", r"\b(?:shareholders'?|stockholders'?|total|shareowners'?) equity\b"),
    ("operating_cash_flow", r"\b(?:operating cash flows?|cash flows? (?:from|provided by) operati\w*|"
                            r"cash (?:provided|generated|from|flow from) (?:by |from )?operati\w*|"
                            r"net cash from operati\w*)\b"),
    # "cash" akela = cash and cash equivalents (cash flow / free cash / cash dividends nahi)
    ("cash", r"\b(?:cash and cash equivalents|cash on hand|cash balance|"
             r"(?<!free )(?<!net )cash(?! (?:flows?|from|provided|generated|used|dividends?|paid|taxes|"
             r"conversion|returned|and equivalents)))\b|\bcash and equivalents\b"),
    ("capex", r"\b(?:capital expenditures?|capex|capital spending|capital investments?|purchases? of property)\b"),
    ("revenue", r"\b(?:revenues?|net sales|total sales|sales|(?<!inventory )(?<!asset )turnover)\b"),
]
# Bayani sawal (policy / kyun / kaise) -> structured number ki zaroorat nahi
DESCRIPTIVE = re.compile(r"\b(?:recogni[sz]\w*|polic\w*|why|strategy|risks?|describe|explain|segment\w*|"
                         r"drivers?|factors?|guidance|outlook)\b", re.I)
TREND = re.compile(r"\b(?:trend|over the (?:years|past|last)|history|historical|since|each year|every year|"
                   r"saalon|har saal)\b", re.I)
SERVE = ("sec_verified", "sec_only", "pdf_verified", "pdf_unchecked", "pdf_conflict")
STATUS_TEXT = {
    "sec_verified": "matches SEC XBRL and the filing's own table",
    "sec_only": "from SEC XBRL",
    "pdf_verified": "from the filing's table (internally consistent; SEC has no value)",
    "pdf_unchecked": "from the filing's table (not cross-checked)",
    "pdf_conflict": "from the filing's table (a later filing reports a different value)",
}


def detect_metrics(question):
    q = " " + norm(question.replace("'", "")) + " "
    return [m for m, rx in METRIC_PATTERNS if re.search(rx, q)][:3]


# Saalon ki range: "2019-2022", "2019 through 2022", "from 2019 until 2022", ya "har saal / each year ... 2019 to 2022"
# -> beech ke saal bhi.  "change from 2019 to 2022" / "2019 vs 2022" -> sirf dono kinare.
RANGE = re.compile(r"((?:19|20)\d{2})\s*(?:-|–|—|through|thru|until|till)\s*(?:fy\s*)?((?:19|20)\d{2})", re.I)
SPAN = re.compile(r"((?:19|20)\d{2})\s*(?:to|and)\s*(?:fy\s*)?((?:19|20)\d{2})", re.I)
EVERY = re.compile(r"\b(?:each|every|per) year\b|\bannual(?:ly)?\b|\byearly\b|\byear by year\b|\bhar saal\b|"
                   r"\bsaal ba saal\b|\blist\b|\bbreakdown\b", re.I)


def question_years(question, years):
    years = set(years or ())
    for rx, ok in ((RANGE, True), (SPAN, bool(EVERY.search(question)))):
        if not ok:
            continue
        for a, b in rx.findall(question):
            a, b = int(a), int(b)
            if a < b and b - a <= 15:
                years |= set(range(a, b + 1))
    return years


class FinancialData:
    """Answer layer ke liye: sawal -> serving layer ke numbers ek 'source' ki shakal mein ([D1]).
    Database na ho to available=False aur sab kuch pehle jaisa chalta hai."""

    def __init__(self, path):
        self.path = Path(path)
        self.available = False
        try:
            if self.path.exists():
                self.con = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, check_same_thread=False)
                self.available = bool(self.con.execute(
                    "SELECT name FROM sqlite_master WHERE name='financials'").fetchone())
        except sqlite3.Error:
            self.available = False

    def company(self, name):
        return resolve_company(self.con, name) if self.available else None

    def source(self, question, companies, years, tag="D1"):
        """-> source dict (build_sources jaisa) ya None."""
        if not self.available or not companies or DESCRIPTIVE.search(question):
            return None
        years = question_years(question, years)
        metrics = detect_metrics(question)
        if not metrics:
            return None
        comps = [c for c in dict.fromkeys(self.company(x) for x in sorted(companies)) if c][:4]
        if not comps:
            return None
        lines, rows_used, first = [], [], None
        for comp in comps:
            for m in metrics:
                rows = [r for r in lookup(self.con, comp, m) if r["status"] in SERVE and r["value"] is not None]
                if not rows:
                    continue
                by_year = {r["fiscal_year"]: r for r in rows}
                if years:
                    want = set(years) | {y - 1 for y in years if len(years) == 1}   # ek saal -> pichla bhi (badlao)
                    pick = [r for r in rows if r["fiscal_year"] in want]
                elif TREND.search(question):
                    pick = rows
                else:
                    pick = rows[-3:]                                               # saal nahi diya -> aakhri 3
                if not pick:
                    continue
                lines.append(f"{comp} -- {METRIC_LABELS[m]}" + (" (USD per share)" if m == "eps_diluted" else ""))
                for r in pick:
                    y = r["fiscal_year"]
                    prev = by_year.get(y - 1)
                    base = r.get("prior_value")
                    chg = ""
                    if base is None and prev:
                        base = prev["value"]
                    if base:
                        if m == "eps_diluted":
                            chg = f"; vs FY{y - 1} ({fmt(base, m)}): {r['value'] - base:+.2f} per share"
                        else:
                            pct = 100 * (r["value"] - base) / abs(base)
                            chg = (f"; change vs FY{y - 1}: {fmt(r['value'] - base, m)} ({pct:+.1f}%)"
                                   + (" on the same basis as the FY%d 10-K" % y if r.get("prior_value") else ""))
                        if (r.get("prior_value") is not None and prev and prev["value"]
                                and abs(prev["value"] - r["prior_value"]) > 0.005 * abs(prev["value"])):
                            chg += (f". NOTE: FY{y - 1} was restated in the FY{y} 10-K to "
                                    f"{fmt(r['prior_value'], m)} (originally reported {fmt(prev['value'], m)})")
                    cite = f"; filing table on PDF page {r['page']}" if r["page"] else ""
                    exact = "" if m == "eps_diluted" else f" (= {r['value']:,.0f} USD)"
                    lines.append(f"  FY{r['fiscal_year']}: {r['display']}{exact} -- {STATUS_TEXT[r['status']]}"
                                 f"{cite}{chg}")
                    rows_used.append(r)
                    first = first or r
        if len(comps) >= 2:                      # companies ka farq bhi Python mein
            for m in metrics:
                got = {}
                for r in rows_used:
                    if r["metric"] == m:
                        got.setdefault(r["fiscal_year"], {})[r["company"]] = r["value"]
                common = [y for y, cv in sorted(got.items()) if len(cv) >= 2]
                for y in (common if years else common[-1:]):
                    (ca, va), (cb, vb) = sorted(got[y].items(), key=lambda t: -t[1])[:2]
                    if vb:
                        diff = (f"{va - vb:+.2f} per share" if m == "eps_diluted"
                                else f"{fmt(va - vb, m)} ({100 * (va - vb) / abs(vb):+.1f}%)")
                        lines.append(f"Difference FY{y} {METRIC_LABELS[m]}: {ca} is higher than {cb} by {diff}")
        if not rows_used:
            return None
        years_used = sorted({r["fiscal_year"] for r in rows_used})
        pages = sorted({int(p) for r in rows_used for p in re.findall(r"\d+", str(r["page"] or ""))[:1]})
        text = ("Structured financial data from the project database: values in US dollars as originally reported "
                "in each year's 10-K, checked against the SEC's XBRL data. Prefer these figures over numbers in the "
                "passages. If a passage shows a different value for the same year, it is usually a restatement in a "
                "later filing: say so instead of mixing the two. All changes and differences are already computed "
                "below; use them and do not do your own arithmetic or write formulas.\n" + "\n".join(lines))
        return {
            "tag": tag, "company": ", ".join(comps), "company_key": first["company"],
            "year": years_used[-1], "section": "Financial data (SEC XBRL verified)", "pages": [],
            "pdf_pages": pages,
            "type": "structured", "text": text,
            "rows": [{k: r[k] for k in ("company", "fiscal_year", "metric", "value", "status", "page", "file_name",
                                        "sec_url")} for r in rows_used],
        }


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
        mark = {"sec_verified": "✓✓", "sec_only": "S ", "pdf_verified": "✓ ", "pdf_suspect": "✗ "}.get(r["status"], "? ")
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
                        "SUM(status='pdf_verified'), SUM(status IN ('pdf_unchecked','pdf_conflict','pdf_suspect')) "
                        "FROM financials WHERE metric=?", (m,)).fetchone()
        print(f"  {m:<22}{r[0]:>6}{r[1] or 0:>14}{r[2] or 0:>10}{r[3] or 0:>14}{r[4] or 0:>11}")
    st = dict(con.execute("SELECT status, COUNT(*) FROM financials GROUP BY status"))
    sec = sum(st.get(s, 0) for s in SEC_STATUSES)
    print(f"\n  SEC se tasdeeq-shuda (sec_verified + sec_only): {sec:,}/{total:,} = {100 * sec / total:.1f}%")
    print(f"  PDF page citation ke saath:                   {st.get('sec_verified', 0) + st.get('pdf_verified', 0):,}")
    print(f"  Kamzor (pdf_unchecked / pdf_conflict):        "
          f"{st.get('pdf_unchecked', 0)} / {st.get('pdf_conflict', 0)}  (answer mein ehtiyat ka note)")
    print(f"  pdf_suspect (answer mein use nahi):           {st.get('pdf_suspect', 0)}\n")


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