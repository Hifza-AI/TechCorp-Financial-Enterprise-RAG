"""
xbrl_db.py  -  SEC ka official structured data (XBRL "companyfacts") hamari 50 companies ke liye, facts.db mein.

Kyun: PDF se nikale numbers ko SEC ke official numbers se milana (validation), aur number wale sawalon ke liye
ek doosra, pakka source. XBRL mein text nahi hota (risk factors, MD&A...) aur page number bhi nahi -- woh PDF
pipeline se hi aata hai. Ye dono sources milte hain, ek doosre ki jagah nahi lete.

Pehle .env mein (SEC har request ke saath naam + email maangta hai, warna request rok deta hai):
    SEC_USER_AGENT=Your Name your.email@example.com

Download:   python STAGE_1/xbrl_db.py download         (50 files -> STAGE_1/xbrl/, jo pehle aa chuki wo dobara nahi)
Database:   python STAGE_1/xbrl_db.py load             (STAGE_1/xbrl/*.json -> facts.db table xbrl_facts)
Dekho:      python STAGE_1/xbrl_db.py companies        (kaun si company kis ticker / CIK se mili)

SEC ki shart: 10 requests/second se kam -- yahan har request ke beech 0.3 sec.
"""
import argparse
import json
import os
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

# Hamare folder ke naam -> ticker. (Naam badle hon to bhi CIK wahi rehta hai: Square -> Block, Facebook -> Meta.)
# Ek company ke liye kai ticker diye hain: pehla jo SEC ki list mein mile wahi.
TICKERS = {
    "3m": ["MMM"], "adobe": ["ADBE"], "amazon": ["AMZN"], "amd": ["AMD"], "advanced micro devices": ["AMD"],
    "ametek": ["AME"], "apple": ["AAPL"], "block": ["XYZ", "SQ"], "square": ["XYZ", "SQ"], "boeing": ["BA"],
    "carlisle": ["CSL"], "caterpillar": ["CAT"], "chevron": ["CVX"], "chipotle": ["CMG"],
    "colgate": ["CL"], "costco": ["COST"], "dell": ["DELL"], "donaldson": ["DCI"], "ebay": ["EBAY"],
    "emerson": ["EMR"], "franklin electric": ["FELE"], "google": ["GOOGL", "GOOG"], "alphabet": ["GOOGL", "GOOG"],
    "graco": ["GGG"], "home depot": ["HD"], "intel": ["INTC"], "intuit": ["INTU"], "kennametal": ["KMT"],
    "kimberly": ["KMB"], "lockheed": ["LMT"], "meta": ["META"], "facebook": ["META"], "microsoft": ["MSFT"],
    "netflix": ["NFLX"], "nike": ["NKE"], "nordson": ["NDSN"], "northrop": ["NOC"], "nvidia": ["NVDA"],
    "palo alto": ["PANW"], "parker": ["PH"], "paypal": ["PYPL"], "pentair": ["PNR"], "rockwell": ["ROK"],
    "roper": ["ROP"], "salesforce": ["CRM"], "servicenow": ["NOW"], "southwest": ["LUV"], "starbucks": ["SBUX"],
    "union pacific": ["UNP"], "walmart": ["WMT"], "watts": ["WTS"], "woodward": ["WWD"], "workday": ["WDAY"],
    "xylem": ["XYL"],
}

SCHEMA = """
DROP TABLE IF EXISTS xbrl_facts;
DROP TABLE IF EXISTS xbrl_companies;
CREATE TABLE xbrl_companies(company TEXT PRIMARY KEY, ticker TEXT, cik INTEGER, sec_name TEXT);
CREATE TABLE xbrl_facts(
    company TEXT, taxonomy TEXT, concept TEXT, label TEXT, unit TEXT, value REAL,
    start TEXT, end TEXT, fy INTEGER, fp TEXT, form TEXT, filed TEXT, accn TEXT, frame TEXT);
CREATE INDEX ix_x ON xbrl_facts(company, concept);
CREATE INDEX ix_xf ON xbrl_facts(company, form, fy);
"""


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9&]+", " ", (s or "").lower())).strip()


def load_env():
    for p in (HERE.parent / ".env", HERE / ".env"):
        if p.exists():
            for line in p.read_text(encoding="utf-8-sig").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def user_agent():
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if not ua or "@" not in ua:
        raise SystemExit("SEC naam + email maangta hai. .env mein likho:  SEC_USER_AGENT=Your Name your.email@example.com")
    return ua


def get(url, ua, tries=3):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": ua, "Accept-Encoding": "identity"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code in (403, 429) and k + 1 < tries:
                time.sleep(5 * (k + 1))
                continue
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if k + 1 < tries:
                time.sleep(3)
                continue
            raise
    return None


def our_companies(db):
    con = sqlite3.connect(db)
    names = [r[0] for r in con.execute("SELECT DISTINCT company FROM tables ORDER BY company")]
    con.close()
    return names


def ticker_for(name):
    n = norm(name)
    for key in sorted(TICKERS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(key)}\b", n):
            return TICKERS[key]
    return None


def resolve(db, ua, xdir):
    """Hamari companies -> (ticker, cik, SEC ka naam). SEC ki ticker list ek dafa download hoti hai."""
    path = Path(xdir) / "company_tickers.json"
    if not path.exists():
        data = get(TICKERS_URL, ua)
        if not data:
            raise SystemExit("SEC ki ticker list nahi mili.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    sec = {v["ticker"].upper(): (int(v["cik_str"]), v["title"]) for v in json.loads(path.read_text()).values()}
    out, missing = {}, []
    for name in our_companies(db):
        hit = next(((t, *sec[t]) for t in (ticker_for(name) or []) if t in sec), None)
        if hit:
            out[name] = hit
        else:
            missing.append(name)
    return out, missing


def cmd_companies(db, ua, xdir):
    found, missing = resolve(db, ua, xdir)
    for name, (t, cik, title) in sorted(found.items()):
        print(f"  {name:<32} {t:<6} CIK {cik:<10} {title}")
    print(f"\n{len(found)} mil gayi" + (f", {len(missing)} nahi mili: {', '.join(missing)}" if missing else ""))


def cmd_download(db, ua, xdir):
    found, missing = resolve(db, ua, xdir)
    Path(xdir).mkdir(parents=True, exist_ok=True)
    t0, n_new = time.time(), 0
    for i, (name, (t, cik, title)) in enumerate(sorted(found.items()), 1):
        out = Path(xdir) / f"{t}.json"
        if out.exists() and out.stat().st_size > 1000:
            print(f"[{i:>2}/{len(found)}] {name:<30} pehle se hai")
            continue
        data = get(FACTS_URL.format(cik=cik), ua)
        if not data:
            print(f"[{i:>2}/{len(found)}] {name:<30} SEC par companyfacts nahi mila (CIK {cik})")
            continue
        out.write_bytes(data)
        n_new += 1
        print(f"[{i:>2}/{len(found)}] {name:<30} {t:<6} {len(data) / 1e6:5.1f} MB")
        time.sleep(0.3)
    print(f"\n{n_new} nayi files ({time.time() - t0:.0f} sec) -> {Path(xdir).resolve()}")
    if missing:
        print(f"Ticker nahi mila: {', '.join(missing)}  (TICKERS mein naam daal do)")


def cmd_load(db, ua, xdir):
    found, missing = resolve(db, ua, xdir)
    con = sqlite3.connect(db)
    con.executescript(SCHEMA)
    t0, total, tenk = time.time(), 0, 0
    for name, (t, cik, title) in sorted(found.items()):
        path = Path(xdir) / f"{t}.json"
        if not path.exists():
            print(f"  {name}: file nahi (pehle download)")
            continue
        con.execute("INSERT INTO xbrl_companies VALUES (?,?,?,?)", (name, t, cik, title))
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = []
        for tax, concepts in (data.get("facts") or {}).items():
            for concept, body in concepts.items():
                label = body.get("label")
                for unit, items in (body.get("units") or {}).items():
                    for it in items:
                        v = it.get("val")
                        try:                                  # REAL: bohat bara number bhi (SQLite INTEGER 64-bit tak)
                            v = float(v) if v is not None else None
                        except (TypeError, ValueError):
                            v = None
                        rows.append((name, tax, concept, label, unit, v, it.get("start"), it.get("end"),
                                     it.get("fy"), it.get("fp"), it.get("form"), it.get("filed"), it.get("accn"),
                                     it.get("frame")))
        con.executemany("INSERT INTO xbrl_facts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        n10 = sum(1 for r in rows if (r[10] or "").startswith("10-K"))
        total += len(rows)
        tenk += n10
        print(f"  {name:<30} {len(rows):>9,} facts  (10-K: {n10:,})")
    con.commit()
    con.close()
    print(f"\nXBRL load ({time.time() - t0:.0f} sec): {total:,} facts, jin mein 10-K se {tenk:,}")


def main():
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser(description="SEC XBRL companyfacts -> facts.db")
    ap.add_argument("cmd", choices=["companies", "download", "load"])
    ap.add_argument("--db", default="STAGE_1/facts.db")
    ap.add_argument("--dir", default="STAGE_1/xbrl")
    a = ap.parse_args()
    load_env()
    if not Path(a.db).exists():
        raise SystemExit(f"Database nahi mila: {a.db}. Pehle:  python STAGE_1/facts_db.py build")
    ua = user_agent()
    {"companies": cmd_companies, "download": cmd_download, "load": cmd_load}[a.cmd](a.db, ua, a.dir)


if __name__ == "__main__":
    main()