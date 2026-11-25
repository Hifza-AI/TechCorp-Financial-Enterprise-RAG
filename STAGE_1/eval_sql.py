"""
eval_sql.py  -  SQL routing ka imtihan: number wale sawal se sahi company, metric, saal aur sahi number tak.

    python STAGE_1/eval_sql.py make                 (ek dafa: STAGE_1/eval/sql_dev.csv + sql_test.csv banao, FREEZE)
    python STAGE_1/eval_sql.py run --set dev        (routing chalao + report; LLM nahi lagta, mufta)
    python STAGE_1/eval_sql.py run --set test       (sirf AAKHIR mein, ek dafa -- yahi report wala number)
    python STAGE_1/eval_sql.py show --set dev       (sirf fail sawal, tafseel ke saath)

Kya naapte hain (answer.py ka asal raasta: structured_source):
  trigger     number wale sawal par [D1] bana?  bayani / doosre sawal par NAHI bana?
  company     sahi company (ya companies)
  metric      sahi metric (revenue, net income ...)
  years       maange gaye saal maujood, aur faltu saal nahi
  value       har maange gaye (company, metric, saal) ka number = SEC ka number (gold)
  pass        sab sahi (end-to-end)

Gold: seedha SEC XBRL se (us saal ki apni 10-K, na ho to agli 10-K ka comparative column) -- PDF ya serving
layer se nahi. Gold CSV mein likh kar freeze hota hai: database baad mein badle to bhi imtihan wahi rehta hai.
Concept ka chunao (jaise Chevron revenue) gold aur serving mein ek jaisa hai -- us ki jaanch insaani spot-check
(spotcheck sheet) se hoti hai, is imtihan se nahi.

dev aur test: alag (company, metric, saal) aur alag jumlon ke saanche (templates). Ghaltiyan sirf dev par dekh
kar theek karo; test aakhir mein ek dafa -- warna wahi overfitting jo 100 sawalon wale imtihan mein roka tha.
"""
import argparse
import csv
import json
import math
import os
import random
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from validate_xbrl import accession_map, load_offsets, pick_accn, xbrl_values, matches   # noqa: E402
from financials_db import sec_fallback, METRIC_LABELS                       # noqa: E402

# Hamari company (folder naam, norm) -> log jo naam likhte hain
COMMON = {
    "3m": "3M", "amd": "AMD", "adobe": "Adobe", "amazon": "Amazon", "ametek": "Ametek", "apple": "Apple",
    "block": "Block", "boeing": "Boeing", "chevron": "Chevron", "carlisle": "Carlisle", "caterpillar": "Caterpillar",
    "chipotle mexican grill": "Chipotle", "colgate palmolive": "Colgate-Palmolive", "costco": "Costco",
    "dell": "Dell", "donaldson": "Donaldson", "ebay": "eBay", "emerson electric": "Emerson Electric",
    "franklin electric": "Franklin Electric", "google": "Google", "graco": "Graco", "home depot": "Home Depot",
    "intel": "Intel", "intuit": "Intuit", "kennametal": "Kennametal", "kimberly clark": "Kimberly-Clark",
    "lockheed martin": "Lockheed Martin", "meta": "Meta", "microsoft": "Microsoft", "nordson": "Nordson",
    "northrop": "Northrop Grumman", "netflix": "Netflix", "nike": "Nike", "nvidia": "Nvidia",
    "palo alto networks": "Palo Alto Networks", "parker hannifin": "Parker Hannifin", "paypal": "PayPal",
    "pentair plc": "Pentair", "rockwell auomation": "Rockwell Automation", "roper technologies": "Roper Technologies",
    "salesforce": "Salesforce", "servicenow": "ServiceNow", "southwest": "Southwest Airlines",
    "starbucks": "Starbucks", "union pacific": "Union Pacific", "watts water technologies": "Watts Water",
    "walmart": "Walmart", "woodward inc": "Woodward", "workday": "Workday", "xylem inc": "Xylem",
}
# Metric ke alfaaz jo log likhte hain (sirf hamari pasand ke nahi -- asli imtihan)
PHRASES = {
    "revenue": ["revenue", "total revenue", "net sales", "sales", "revenues"],
    "gross_profit": ["gross profit", "gross margin"],
    "operating_income": ["operating income", "income from operations", "operating profit"],
    "net_income": ["net income", "net earnings", "profit", "net profit"],
    "eps_diluted": ["diluted EPS", "earnings per share", "diluted earnings per share", "EPS"],
    "rnd": ["R&D spending", "research and development expense", "R&D expenses", "R&D"],
    "total_assets": ["total assets", "assets"],
    "total_liabilities": ["total liabilities", "liabilities"],
    "equity": ["shareholders' equity", "stockholders' equity", "total equity"],
    "cash": ["cash and cash equivalents", "cash balance", "cash on hand", "cash"],
    "operating_cash_flow": ["operating cash flow", "cash from operations", "cash flow from operating activities",
                            "net cash provided by operating activities"],
    "capex": ["capital expenditures", "capex", "capital spending", "purchases of property and equipment"],
}
# Saanche: (kind, template).  Har kind ke saanche dev / test mein baant diye jaate hain (alternate).
POS = {
    "single": ["What was {C}'s {M} in {Y}?", "What was {C}'s {M} in fiscal {Y}?", "How much {M} did {C} report for FY{Y}?",
               "{C} {M} {Y}", "Tell me {C}'s {M} for fiscal year {Y}.", "What were the {M} of {C} in {Y}?",
               "Give me the {M} figure for {C}, {Y}.", "{C}: {M} in {Y}?"],
    "single_ru": ["{C} ki {Y} mein {M} kitni thi?", "{Y} mein {C} ka {M} kya tha?", "{C} ka {Y} ka {M} batao",
                  "FY{Y} mein {C} ki {M} kitni thi?"],
    "no_year": ["What is {C}'s {M}?", "How much {M} does {C} have?", "{C} latest {M}", "What is the most recent {M} for {C}?"],
    "two_years": ["How did {C}'s {M} change from {Y1} to {Y2}?", "Compare {C}'s {M} in {Y1} and {Y2}.",
                  "{C} {M} {Y1} vs {Y2}", "Did {C}'s {M} go up or down between {Y1} and {Y2}?"],
    "range": ["Show {C}'s {M} for each year from {Y1} to {Y2}.", "List {C}'s {M} from {Y1} through {Y2}.",
              "{C} {M} {Y1}-{Y2}", "What was {C}'s {M} every year between {Y1} and {Y2}?"],
    "trend": ["How has {C}'s {M} changed over the years?", "Show the trend in {C}'s {M}.",
              "{C} ki {M} saalon mein kaise badli?", "What is the historical {M} of {C}?"],
    "compare": ["Compare {C1} and {C2} {M} in {Y}.", "Which had higher {M} in {Y}, {C1} or {C2}?",
                "{C1} vs {C2} {M} {Y}", "How does {C1}'s {M} compare with {C2}'s in fiscal {Y}?"],
    "compare_ru": ["{Y} mein {C1} aur {C2} mein se kis ka {M} zyada tha?", "{C1} aur {C2} ka {Y} ka {M} compare karo"],
    "multi_metric": ["What were {C}'s {M1} and {M2} in {Y}?", "{C} {M1} and {M2} for {Y}",
                     "Give {C}'s {M1} as well as its {M2} for fiscal {Y}.", "{C} ki {Y} mein {M1} aur {M2} kitni thi?"],
    "typo": ["What was {T}'s {M} in {Y}?", "{T} {M} {Y}", "How much {M} did {T} have in {Y}?",
             "{T} ka {Y} mein {M} kitna tha?"],
}
NEG = {
    "descriptive": ["How does {C} recognize revenue?", "Describe {C}'s revenue recognition policy.",
                    "What factors drive {C}'s revenue growth?", "What are the main risks to {C}'s net income?",
                    "Explain {C}'s accounting policy for cash and cash equivalents.",
                    "Which segments contribute most to {C}'s revenue?"],
    "no_metric": ["Who is the CEO of {C}?", "Where is {C} headquartered?", "How many employees does {C} have?",
                  "Who audits {C}'s financial statements?", "What does {C} do?", "{C} ke main competitors kaun hain?"],
    "unknown_company": ["What was Tesla's revenue in {Y}?", "Ford net income {Y}", "What was Pfizer's R&D expense in {Y}?",
                        "Coca-Cola total assets {Y}", "How much revenue did Oracle report in {Y}?",
                        "What was IBM's diluted EPS in fiscal {Y}?"],
    "unsupported_metric": ["What was {C}'s dividend per share in {Y}?", "What was {C}'s effective tax rate in {Y}?",
                           "How much did {C} spend on share buybacks in {Y}?", "What was {C}'s inventory turnover in {Y}?",
                           "How many shares did {C} have outstanding in {Y}?", "What was {C}'s backlog in {Y}?"],
    "no_data_year": ["What was {C}'s revenue in 2005?", "{C} net income 1999", "What were {C}'s total assets in 2008?"],
}
POS_COUNT = {"single": 55, "single_ru": 15, "no_year": 10, "two_years": 12, "range": 10, "trend": 8, "compare": 15,
             "compare_ru": 5, "multi_metric": 10, "typo": 10}
NEG_COUNT = {"descriptive": 15, "no_metric": 15, "unknown_company": 8, "unsupported_metric": 8, "no_data_year": 4}
FIELDS = ["id", "kind", "expect", "question", "companies", "metrics", "years_required", "years_allowed", "gold"]


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9&]+", " ", (s or "").lower())).strip()


def wilson(k, n, z=1.96):
    """95% confidence interval (Wilson) -- 'kitna pakka' number hai."""
    if not n:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


# ---------------------------------------------------------------- make

class Gold:
    """SEC XBRL se (company, fy, metric) ka number. Serving layer / PDF se nahi."""

    def __init__(self, con):
        self.con = con
        self.acc = accession_map(con)
        self.offs = load_offsets(con)
        self.memo = {}

    def value(self, comp, fy, metric):
        k = (comp, fy, metric)
        if k not in self.memo:
            v = None
            off = self.offs.get(comp, 0)
            a = pick_accn(self.acc.get(comp, []), fy, off)
            if a:
                vals = xbrl_values(self.con, comp, a[0], a[3], metric)
                if vals:
                    v = vals[0][1]
            if v is None:
                fb = sec_fallback(self.con, comp, fy, metric, off)
                v = fb[1] if fb else None
            self.memo[k] = v
        return self.memo[k]


def surface(rng, comp, ticker):
    """Company ka naam jaisa log likhte hain: aam naam (zyada), ticker, chhote huroof."""
    name = COMMON.get(norm(comp), comp.replace("_", " ").title())
    r = rng.random()
    if r < 0.2 and ticker:
        return ticker
    if r < 0.3:
        return name.lower()
    return name


def typo(rng, comp):
    name = COMMON.get(norm(comp), comp)
    words = name.split()
    i = max(range(len(words)), key=lambda j: len(words[j]))
    w = words[i]
    if len(w) < 6 or not w.isalpha():
        return None
    p = rng.randrange(2, len(w) - 2)
    words[i] = w[:p] + w[p + 1:]
    return " ".join(words)


def make(db, outdir, seed=7, fresh=None):
    """fresh=None: dev + test (pehli dafa).  fresh="test2": ek NAYA held-out set (test ke saanche), un
    (company, metric, saal) cells se jo dev / test mein kabhi nahi aaye -- jab purana test set dekh kar code
    badalna para ho, imaandar number ke liye naya set."""
    con = sqlite3.connect(db)
    gold = Gold(con)
    tick = dict(con.execute("SELECT company, ticker FROM xbrl_companies"))
    years = defaultdict(list)
    for c, fy in con.execute("SELECT DISTINCT company, fiscal_year FROM metrics ORDER BY 1, 2"):
        years[c].append(fy)
    comps = sorted(c for c in years if c in tick)
    rng = random.Random(seed)

    # (company, metric, saal) ko dev / test mein baanto -- dono mein kabhi ek hi cell nahi
    cells = [(c, m, y) for c in comps for m in PHRASES for y in years[c]]
    rng.shuffle(cells)
    if fresh:
        used = set()
        for old in outdir.glob("sql_*.csv"):
            with old.open(encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    for k in json.loads(r["gold"] or "{}"):
                        c, m, y = k.split("|")
                        used.add((c, m, int(y)))
        split = {cell: fresh for cell in cells if cell not in used}
        sets = [fresh]
    else:
        split = {cell: ("dev" if i % 2 == 0 else "test") for i, cell in enumerate(cells)}
        sets = ["dev", "test"]

    def tmpl(kind, which, table):
        ts = table[kind]
        own = [t for i, t in enumerate(ts) if (i % 2 == 0) == (which == "dev")] or ts   # naya set = test ke saanche
        return rng.choice(own)

    def gold_cells(c, m, ys):
        out = {}
        for y in ys:
            v = gold.value(c, y, m)
            if v is None:
                return None
            out[f"{c}|{m}|{y}"] = v
        return out

    for which in sets:
        rows = []
        mine = [x for x in cells if split.get(x) == which]

        def pick_cell(need_years=1):
            for _ in range(500):
                c, m, y = rng.choice(mine)
                if need_years == 1 or len(years[c]) >= need_years:
                    return c, m, y
            raise RuntimeError("cell nahi mila")

        def add(kind, expect, q, cs, ms, req, allowed, g):
            rows.append({"kind": kind, "expect": expect, "question": q, "companies": "|".join(cs),
                         "metrics": "|".join(ms), "years_required": "|".join(map(str, sorted(req))),
                         "years_allowed": "|".join(map(str, sorted(allowed))), "gold": json.dumps(g)})

        for kind, n in POS_COUNT.items():
            made, tries = 0, 0
            while made < n and tries < n * 50:
                tries += 1
                c, m, y = pick_cell()
                M = rng.choice(PHRASES[m])
                t = tmpl(kind, which, POS)
                C = surface(rng, c, tick.get(c))
                if kind in ("single", "single_ru"):
                    g = gold_cells(c, m, [y])
                    if g is None:
                        continue
                    add(kind, 1, t.format(C=C, M=M, Y=y), [c], [m], {y}, {y, y - 1}, g)
                elif kind == "typo":
                    T = typo(rng, c)
                    g = gold_cells(c, m, [y])
                    if not T or g is None:
                        continue
                    add(kind, 1, t.format(T=T, M=M, Y=y), [c], [m], {y}, {y, y - 1}, g)
                elif kind == "no_year":
                    have = [yy for yy in years[c] if gold.value(c, yy, m) is not None]
                    if len(have) < 3:
                        continue
                    g = gold_cells(c, m, [have[-1]])
                    add(kind, 1, t.format(C=C, M=M), [c], [m], {have[-1]}, set(have[-3:]), g)
                elif kind in ("two_years", "range"):
                    ys = [yy for yy in years[c] if gold.value(c, yy, m) is not None]
                    if len(ys) < 4:
                        continue
                    i = rng.randrange(0, len(ys) - 3)
                    y1, y2 = ys[i], ys[i + (1 if kind == "two_years" and rng.random() < 0.5 else 3)]
                    req = {y1, y2} if kind == "two_years" else {yy for yy in ys if y1 <= yy <= y2}
                    allowed = req | ({y1 - 1} if kind == "range" else set())
                    g = gold_cells(c, m, sorted(req))
                    if g is None:
                        continue
                    add(kind, 1, t.format(C=C, M=M, Y1=y1, Y2=y2), [c], [m], req, allowed, g)
                elif kind == "trend":
                    ys = [yy for yy in years[c] if gold.value(c, yy, m) is not None]
                    if len(ys) < 5:
                        continue
                    add(kind, 1, t.format(C=C, M=M), [c], [m], set(ys), set(years[c]), gold_cells(c, m, ys))
                elif kind in ("compare", "compare_ru"):
                    c2 = rng.choice([x for x in comps if x != c])
                    if y not in years[c2]:
                        continue
                    g1, g2 = gold_cells(c, m, [y]), gold_cells(c2, m, [y])
                    if g1 is None or g2 is None:
                        continue
                    C2 = surface(rng, c2, tick.get(c2))
                    add(kind, 1, t.format(C1=C, C2=C2, M=M, Y=y), [c, c2], [m], {y}, {y, y - 1}, {**g1, **g2})
                elif kind == "multi_metric":
                    m2 = rng.choice([x for x in PHRASES if x != m])
                    g1, g2 = gold_cells(c, m, [y]), gold_cells(c, m2, [y])
                    if g1 is None or g2 is None:
                        continue
                    M2 = rng.choice(PHRASES[m2])
                    add(kind, 1, t.format(C=C, M1=M, M2=M2, Y=y), [c], [m, m2], {y}, {y, y - 1}, {**g1, **g2})
                made += 1

        for kind, n in NEG_COUNT.items():
            for _ in range(n):
                c, m, y = pick_cell()
                t = tmpl(kind, which, NEG)
                add(kind, 0, t.format(C=surface(rng, c, tick.get(c)), Y=y), [c], [], set(), set(), {})

        seen, uniq = set(), []
        for r in rows:
            if r["question"].lower() not in seen:
                seen.add(r["question"].lower())
                uniq.append(r)
        for i, r in enumerate(uniq, 1):
            r["id"] = f"{'D' if which == 'dev' else 'X' if which == 'test' else 'N'}{i:03d}"
        path = outdir / f"sql_{which}.csv"
        if path.exists():
            raise SystemExit(f"{path} pehle se hai (freeze). Naya banana ho to pehle khud delete karo.")
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(uniq)
        kinds = Counter(r["kind"] for r in uniq)
        print(f"{path}: {len(uniq)} sawal  ({sum(r['expect'] for r in uniq)} number, "
              f"{sum(1 - r['expect'] for r in uniq)} negative)  " + ", ".join(f"{k} {v}" for k, v in kinds.items()))


# ---------------------------------------------------------------- run

def load_set(outdir, which):
    path = outdir / f"sql_{which}.csv"
    if not path.exists():
        raise SystemExit(f"{path} nahi mila. Pehle:  python STAGE_1/eval_sql.py make")
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def grade(row, src):
    """Ek sawal -> dict(trigger, company, metric, years, value, pass, why)."""
    expect = row["expect"] == "1"
    got = src is not None
    res = {"id": row["id"], "kind": row["kind"], "question": row["question"], "expect": expect, "triggered": got}
    if not expect:
        res["pass"] = not got
        res["why"] = "" if not got else "[D1] bana jab nahi banna chahiye tha: " + ", ".join(
            sorted({f"{r['company']}/{r['metric']}" for r in src["rows"]}))
        return res
    if not got:
        res.update({"company": False, "metric": False, "years": False, "value": False, "pass": False,
                    "why": "[D1] nahi bana"})
        return res
    want_c = set(row["companies"].split("|"))
    want_m = set(row["metrics"].split("|"))
    req = {int(y) for y in row["years_required"].split("|") if y}
    allowed = {int(y) for y in row["years_allowed"].split("|") if y}
    gold = json.loads(row["gold"])
    rows = src["rows"]
    got_c = {r["company"] for r in rows}
    got_m = {r["metric"] for r in rows}
    cell = {(r["company"], r["metric"], r["fiscal_year"]): r for r in rows}
    why = []
    res["company"] = got_c == want_c
    if not res["company"]:
        why.append(f"company {sorted(got_c)} (chahiye {sorted(want_c)})")
    res["metric"] = got_m == want_m
    if not res["metric"]:
        why.append(f"metric {sorted(got_m)} (chahiye {sorted(want_m)})")
    yok = True
    for c in want_c:
        for m in want_m:
            ys = {y for (cc, mm, y) in cell if cc == c and mm == m}
            if not req <= ys or not ys <= allowed:
                yok = False
                why.append(f"{c}/{m} saal {sorted(ys)} (chahiye {sorted(req)}, ijazat {sorted(allowed)})")
    res["years"] = yok
    vok = True
    for k, gv in gold.items():
        c, m, y = k.split("|")
        r = cell.get((c, m, int(y)))
        if r is None:
            vok = False
            continue
        if not matches(r["value"], gv, m) or (r["value"] < 0) != (gv < 0) and m != "capex":
            vok = False
            why.append(f"{c}/{m}/FY{y}: hamara {r['value']:,.2f} vs SEC {gv:,.2f} ({r['status']})")
    res["value"] = vok
    res["pass"] = all(res[x] for x in ("company", "metric", "years", "value"))
    res["why"] = "; ".join(why)
    return res


def run(db, store, outdir, which, quiet=False):
    import answer as A
    from retriever import Retriever
    os.environ["STRUCTURED_DATA"] = "1"
    os.environ["FIN_DB"] = str(db)
    A._FIN = None
    rows = load_set(outdir, which)
    t0 = time.time()
    retriever = Retriever(store)
    results = []
    for row in rows:
        try:
            src = A.structured_source(retriever, row["question"])
        except Exception as e:                                         # noqa: BLE001
            src = None
            print(f"  {row['id']} error: {e}")
        results.append(grade(row, src))
    out = outdir / f"sql_{which}_results.json"
    out.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n{len(rows)} sawal, {time.time() - t0:.0f} sec  ->  {out}")
    report(results, which)
    return results


def pct(k, n):
    lo, hi = wilson(k, n)
    return f"{k:>4}/{n:<4} {100 * k / n:5.1f}%  (95% CI {100 * lo:.1f}-{100 * hi:.1f})" if n else "   -"


def report(results, which):
    pos = [r for r in results if r["expect"]]
    neg = [r for r in results if not r["expect"]]
    print(f"\nSQL ROUTING -- {which.upper()} set\n")
    print(f"  Number sawal par [D1] bana (recall):        {pct(sum(r['triggered'] for r in pos), len(pos))}")
    print(f"  Doosre sawal par [D1] nahi bana:            {pct(sum(not r['triggered'] for r in neg), len(neg))}")
    trig = [r for r in pos if r["triggered"]]
    for f, label in (("company", "Company sahi"), ("metric", "Metric sahi"), ("years", "Saal sahi"),
                     ("value", "Number = SEC")):
        print(f"  {label + ' (jahan [D1] bana):':<44}{pct(sum(r[f] for r in trig), len(trig))}")
    allp = sum(r["pass"] for r in results)
    print(f"\n  END-TO-END (number sawal):                  {pct(sum(r['pass'] for r in pos), len(pos))}")
    print(f"  END-TO-END (sab sawal):                     {pct(allp, len(results))}")
    print("\n  Qism ke hisaab se:")
    by = defaultdict(list)
    for r in results:
        by[r["kind"]].append(r)
    for k, rs in by.items():
        p = sum(r["pass"] for r in rs)
        print(f"    {k:<20}{p:>4}/{len(rs):<4}{100 * p / len(rs):6.1f}%")
    fails = [r for r in results if not r["pass"]]
    if fails:
        print(f"\n  Fail ({len(fails)}), pehle 25:")
        for r in fails[:25]:
            print(f"    {r['id']} [{r['kind']}] {r['question'][:60]:<60}  {r['why'][:110]}")
    print()


def regold(db, outdir, which):
    """Gold ko dobara SEC se nikalo (sawal wahi, sirf sahi jawab). Sirf tab jab gold banane ki logic mein bug
    saabit ho -- jaise insaani spot-check ne Dell FY2019 mein agle saal ka number pakra. Purani file .v1 ban kar
    rehti hai aur har badla hua number print hota hai (changelog)."""
    path = outdir / f"sql_{which}.csv"
    rows = load_set(outdir, which)
    con = sqlite3.connect(db)
    gold = Gold(con)
    changed = []
    for r in rows:
        g = json.loads(r["gold"] or "{}")
        for k, old in g.items():
            c, m, y = k.split("|")
            new = gold.value(c, int(y), m)
            if new is None:
                changed.append((r["id"], k, old, None))
                continue
            if not matches(old, new, m):
                changed.append((r["id"], k, old, new))
                g[k] = new
        r["gold"] = json.dumps(g)
    backup = outdir / f"sql_{which}.v1.csv"
    if not backup.exists():
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"{path}: {len(changed)} gold numbers badle (purani file: {backup.name})")
    for i, k, old, new in changed:
        print(f"  {i}  {k}: {old:,.2f} -> " + (f"{new:,.2f}" if new is not None else "SEC mein ab nahi (purana rakha)"))


def show(outdir, which):
    path = outdir / f"sql_{which}_results.json"
    if not path.exists():
        raise SystemExit("Pehle run chalao.")
    for r in json.loads(path.read_text(encoding="utf-8")):
        if not r["pass"]:
            print(f"{r['id']} [{r['kind']}] {r['question']}\n    -> {r['why']}")


def main():
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser(description="SQL routing evaluation (dev / test)")
    ap.add_argument("--db", default="STAGE_1/facts.db")
    ap.add_argument("--store", default="STAGE_1/vector_store")
    ap.add_argument("--dir", default="STAGE_1/eval")
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make")
    mk.add_argument("--fresh", help="naya held-out set ka naam, jaise test2 (dev/test se alag cells)")
    mk.add_argument("--seed", type=int, default=7)
    rg = sub.add_parser("regold")
    rg.add_argument("--set", required=True)
    for name in ("run", "show"):
        s = sub.add_parser(name)
        s.add_argument("--set", default="dev", help="dev, test, ya naya set (test2 ...)")
    a = ap.parse_args()
    outdir = Path(a.dir)
    outdir.mkdir(parents=True, exist_ok=True)
    if a.cmd == "make":
        make(a.db, outdir, a.seed, a.fresh)
    elif a.cmd == "regold":
        regold(a.db, outdir, a.set)
    elif a.cmd == "run":
        if a.set != "dev":
            print("TEST set: ye aakhri, ek dafa ka imtihan hai. Is ki ghaltiyan dekh kar code mat badlo.\n")
        run(a.db, a.store, outdir, a.set)
    else:
        show(outdir, a.set)


if __name__ == "__main__":
    main()