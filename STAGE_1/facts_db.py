"""
facts_db.py  -  SQL track ka pehla qadam: table chunks se ek SQLite database, har number ek row ("fact").

Banao:    python STAGE_1/facts_db.py build                      (STAGE_1/chunks -> STAGE_1/facts.db)
Dhoondo:  python STAGE_1/facts_db.py find Apple 2023 "total net sales"
          python STAGE_1/facts_db.py find Carlisle 2025 revenue --all      (saari milti rows, sirf top nahi)

Kyun: retriever ko "Apple ki 2023 ki revenue?" par sahi table chunk mil jata hai, lekin number us chunk ke andar
dhoondna parta hai. Yahan har table ki har row/column alag fact hai, is liye number seedha nikalta hai -- page aur
table ke naam ke saath (citation). PDF dobara nahi parhi jaati: wahi chunks jo retriever use karta hai.

Table chunk ka text (chunk_builder) do shaklon mein hota hai:
    The following table shows <title> data for <Company> <Year>:
    Columns: 2023, 2022, 2021
    Total net sales -- 2023: 383,285, 2022: 394,328, 2021: 365,817
ya header ke baghair:
    Cash paid | 276,188
Har (row, column) ek fact: company, filing ka saal, column ka saal, row ka naam, value, unit, page, section.
"""
import argparse
import difflib
import json
import re
import sqlite3
import time
from collections import Counter
from pathlib import Path

HEAD = re.compile(r"^The following table shows (.*) data for (.+?) (\d{4}):\s*$")
UNIT = re.compile(r"\bin (millions|thousands|billions)\b", re.I)
YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
DASHES = {"-", "—", "–", "-%", "—%", "–%"}
STATEMENT = re.compile(r"statements? of (?:consolidated )?(?:operations|income|earnings|cash flows?|comprehensive)|"
                       r"balance sheets?|statements? of financial position|income statements?|"
                       r"consolidated results of operations", re.I)

# Aam sawal ke alfaaz -> 10-K ki rows ke naam (sab se pehle wala sab se behtar). find mein kaam aata hai.
ITEM_SYNONYMS = {
    "revenue": ["total net sales", "net sales", "total revenues", "total revenue", "revenues", "revenue", "net revenues",
                "total net revenues", "sales and revenues", "total sales and revenues", "net sales and revenues", "sales"],
    "net income": ["net income", "net earnings", "net income attributable", "net earnings attributable"],
    "research and development": ["research and development", "research, development", "research and development expense"],
    "operating income": ["operating income", "income from operations", "operating profit", "total operating income"],
    "total assets": ["total assets"],
    "cash": ["cash and cash equivalents", "cash and equivalents"],
}
INCOME_ITEMS = {"revenue", "net income", "research and development", "operating income"}
BALANCE_ITEMS = {"total assets", "cash"}
INCOME_ST = re.compile(r"statements? of (?:consolidated )?(?:operations|income|earnings)|income statements?|"
                       r"consolidated results of operations", re.I)
BALANCE_ST = re.compile(r"balance sheets?|statements? of financial position", re.I)
ITEM_ALIASES = {"revenues": "revenue", "sales": "revenue", "net sales": "revenue", "profit": "net income",
                "net earnings": "net income", "r&d": "research and development", "rd": "research and development"}


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9%&]+", " ", (s or "").lower())).strip()


def split_columns(s):
    """'December 31, 2016, December 31, 2015' -> ['December 31, 2016', 'December 31, 2015'];  '2023, 2022' -> 2 cols.
    Qaida: jis tukde mein saal nahi aur agla tukda saal se shuru hota hai, dono ek hi column hain."""
    out = []
    for t in (x.strip() for x in s.split(", ")):
        if out and re.match(r"^(?:19|20)\d{2}\b", t) and not YEAR.search(out[-1]):
            out[-1] = f"{out[-1]}, {t}"
        elif t:
            out.append(t)
    return out


def row_pairs(rest, cols):
    """'2023: 383,285, 2022: 394,328' -> [('2023', '383,285'), ('2022', '394,328')] (sirf maloom column naam)."""
    uniq = list(dict.fromkeys(cols))
    if not uniq:
        return []
    alt = "|".join(re.escape(c) for c in sorted(uniq, key=len, reverse=True))
    marks = list(re.finditer(r"(?:^|, )(" + alt + r"): ", rest))
    out = []
    for k, m in enumerate(marks):
        end = marks[k + 1].start() if k + 1 < len(marks) else len(rest)
        out.append((m.group(1), rest[m.end():end].strip()))
    return out


def to_num(v):
    """'(1,234)' -> -1234.0 ;  '12.5%' -> 12.5 (pct) ;  '-' / '—' -> 0.0 ;  text -> None."""
    s = (v or "").strip().replace("$", "").replace(" ", "")
    if not s:
        return None, False
    if s in DASHES:
        return 0.0, s.endswith("%")
    pct = s.endswith("%") or s.endswith("%)")
    neg = s.startswith("(") or s.startswith("-")
    core = s.strip("()%-").replace(",", "")
    if not re.fullmatch(r"\d+(?:\.\d+)?|\.\d+", core):
        return None, False
    x = float(core)
    return (-x if neg else x), pct


def col_year(label):
    ys = YEAR.findall(label or "")
    return int(ys[-1]) if ys else None


def table_facts(c, prev_text, doc_unit):
    """Ek table chunk -> facts ki list (dicts). Kachra table (sirf page numbers / exhibit list) -> []."""
    lines = (c.get("text") or "").splitlines()
    if not lines:
        return []
    m = HEAD.match(lines[0])
    title = m.group(1).strip() if m else ""
    body = lines[1:] if m else lines
    u = UNIT.search(c.get("text") or "")
    unit_src = "table"
    if not u:
        u = UNIT.search((prev_text or "")[-400:])
        unit_src = "paas ka text"
    unit = u.group(1).lower() if u else doc_unit
    if not u:
        unit_src = "document ka aam unit" if doc_unit else ""
    cols = []
    out = []
    for ln in body:
        if ln.startswith("Columns:"):
            cols = split_columns(ln[len("Columns:"):].strip())
            continue
        if " -- " in ln and cols:
            label, rest = ln.split(" -- ", 1)
            pairs = row_pairs(rest, cols)
        elif " | " in ln:
            parts = [p.strip() for p in ln.split(" | ")]
            label, vals = parts[0], parts[1:]
            pairs = [(f"c{i + 1}" if len(vals) > 1 else "", v) for i, v in enumerate(vals)]
        else:
            continue
        label = label.strip()
        for col, val in pairs:
            num, pct = to_num(val)
            out.append({"row_label": label, "col_label": col, "col_year": col_year(col), "value_text": val,
                        "value_num": num, "pct": int(pct)})
    if not any(f["value_num"] is not None for f in out):
        return []
    for f in out:
        f.update(title=title, unit=None if any(x["pct"] for x in [f]) else unit, unit_src=unit_src)
    return out


# Do tables: har table ki maloomat ek dafa (tables), us ke numbers alag (facts) -- database chhota rehta hai.
SCHEMA = """
DROP VIEW IF EXISTS v_facts;
DROP TABLE IF EXISTS facts;
DROP TABLE IF EXISTS tables;
CREATE TABLE tables(
    id INTEGER PRIMARY KEY, company TEXT, year INTEGER, file_name TEXT, chunk_no INTEGER, pages TEXT,
    section TEXT, title TEXT, statement INTEGER, unit TEXT, unit_src TEXT);
CREATE TABLE facts(
    table_id INTEGER REFERENCES tables(id), row_label TEXT, row_norm TEXT, col_label TEXT, col_year INTEGER,
    value_text TEXT, value_num REAL, pct INTEGER);
CREATE INDEX ix_cy ON tables(company, year);
CREATE INDEX ix_fact ON facts(table_id);
CREATE INDEX ix_year ON facts(col_year);
CREATE VIEW v_facts AS SELECT t.company, t.year, t.file_name, t.chunk_no, t.pages, t.section, t.title, t.statement,
    t.unit, t.unit_src, f.row_label, f.row_norm, f.col_label, f.col_year, f.value_text, f.value_num, f.pct
    FROM facts f JOIN tables t ON t.id = f.table_id;
"""


def build(chunks_dir, db_path):
    files = sorted(Path(chunks_dir).rglob("*_chunks.json"))
    if not files:
        raise SystemExit(f"Koi *_chunks.json nahi mili: {Path(chunks_dir).resolve()}")
    t0 = time.time()
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA)
    seen, stats = set(), Counter()
    for f in files:
        ch = json.load(open(f, encoding="utf-8"))
        if not ch:
            continue
        key = (str(ch[0].get("company")), str(ch[0].get("year")), str(ch[0].get("file_name")))
        if key in seen:                                   # ek hi 10-K do baar (copy) -> ek dafa
            stats["dobara file chhodi"] += 1
            continue
        seen.add(key)
        units = Counter(m.lower() for c in ch for m in UNIT.findall(c.get("text") or ""))
        doc_unit = units.most_common(1)[0][0] if units else None
        n_facts = 0
        for i, c in enumerate(ch):
            if c.get("chunk_type") != "table":
                continue
            stats["tables"] += 1
            facts = table_facts(c, ch[i - 1].get("text") if i else "", doc_unit)
            if not facts:
                stats["kachra / bina number tables"] += 1
                continue
            sec = c.get("section_path") or ""
            sec = " > ".join(sec) if isinstance(sec, list) else str(sec)
            pages = ",".join(str(p) for p in (c.get("page_numbers") or []))
            stmt = int(bool(STATEMENT.search(sec + " " + facts[0]["title"])))
            unit = next((fa["unit"] for fa in facts if fa["unit"]), None)
            cur = con.execute("INSERT INTO tables(company, year, file_name, chunk_no, pages, section, title, statement, "
                              "unit, unit_src) VALUES (?,?,?,?,?,?,?,?,?,?)",
                              (c.get("company"), int(c.get("year") or 0), c.get("file_name"), i, pages, sec,
                               facts[0]["title"], stmt, unit, facts[0]["unit_src"] if unit else ""))
            con.executemany("INSERT INTO facts VALUES (?,?,?,?,?,?,?,?)",
                            [(cur.lastrowid, fa["row_label"], norm(fa["row_label"]), fa["col_label"], fa["col_year"],
                              fa["value_text"], fa["value_num"], fa["pct"]) for fa in facts])
            n_facts += len(facts)
        stats["facts"] += n_facts
        stats["files"] += 1
    con.commit()
    n_num = con.execute("SELECT COUNT(*) FROM facts WHERE value_num IS NOT NULL").fetchone()[0]
    n_unit = con.execute("SELECT COUNT(*) FROM v_facts WHERE unit IS NOT NULL AND pct = 0").fetchone()[0]
    con.close()
    print(f"Ban gaya ({time.time() - t0:.0f} sec): {Path(db_path).resolve()}")
    print(f"  files {stats['files']}, tables {stats['tables']} (kachra / bina number: "
          f"{stats['kachra / bina number tables']}), facts {stats['facts']:,} (number wale {n_num:,}, "
          f"unit pata {n_unit:,})" + (f", dobara file chhodi {stats['dobara file chhodi']}" if stats["dobara file chhodi"] else ""))


def fmt(x):
    return f"{x:,.0f}" if x is not None and abs(x - round(x)) < 1e-9 else (f"{x:,.2f}" if x is not None else "-")


def find(db_path, company, year, item, show_all=False, limit=8):
    con = sqlite3.connect(db_path)
    comps = [r[0] for r in con.execute("SELECT DISTINCT company FROM tables")]
    hit = [c for c in comps if norm(company) in norm(c)] or difflib.get_close_matches(company, comps, n=1, cutoff=0.6)
    if not hit:
        print(f"Company '{company}' database mein nahi. Maujood: {', '.join(sorted(comps)[:60])}")
        return
    q = norm(item)
    canon = ITEM_ALIASES.get(q, q)
    names = ITEM_SYNONYMS.get(canon, [q])
    pref_rx = INCOME_ST if canon in INCOME_ITEMS else BALANCE_ST if canon in BALANCE_ITEMS else STATEMENT
    custom = canon not in ITEM_SYNONYMS
    marks = ",".join("?" * len(hit))
    rows = con.execute(f"SELECT company, year, pages, title, statement, row_label, row_norm, col_label, col_year, "
                       f"value_text, value_num, unit, unit_src, section FROM v_facts WHERE company IN ({marks}) "
                       f"AND (col_year = ? OR (col_year IS NULL AND year = ?))", (*hit, year, year)).fetchall()
    scored = []
    for r in rows:
        rn = r[6]
        best = None
        for k, nm in enumerate(names):
            if rn == nm:
                best = (0, k)
            elif rn.startswith(nm + " "):                    # "net income loss", "net income attributable to ..."
                best = best or (1, k)
            elif custom and f" {nm} " in f" {rn} ":          # apna likha naam: andar kahin bhi
                best = best or (2, k)
            if best and best[0] == 0:
                break
        if best:
            # behtar: exact naam > milta naam;  usi saal ki filing > baad wali;  sahi statement (revenue -> income
            # statement, assets -> balance sheet) > koi aur statement > note / MD&A;  asal number > khaali "-"
            pref = int(bool(pref_rx.search(f"{r[3]} {r[13]}")))
            dash = int(r[9].strip() in DASHES or r[10] is None)
            side = int(bool(re.search(r"noncontrolling|non controlling|minority", rn)))   # 3M: "attributable to 3M" behtar
            scored.append(((r[1] != year, dash, -pref, side, best[0], -r[4], best[1]), r))
    if not scored:
        print(f"'{item}' ki koi row nahi mili ({', '.join(hit)} {year}). Doosra naam try karo, jaise 'total revenues'.")
        return
    scored.sort(key=lambda x: x[0])
    show = scored if show_all else scored[:1]
    for _, r in show[:limit if show_all else 1]:
        comp, fy, pages, title, stmt, label, _, col, cy, vt, vn, unit, usrc, _sec = r
        u = f" ({unit}{', andaza' if usrc == 'document ka aam unit' else ''})" if unit and not vt.strip().endswith("%") else ""
        print(f"{comp} {year} | {label} = {vt}{u} | PDF page {pages} | {title} | filing {fy}"
              + (" | financial statement" if stmt else ""))
    if not show_all and len(scored) > 1:
        print(f"   (aur {len(scored) - 1} milti rows; sab dekhne ke liye --all)")


def main():
    ap = argparse.ArgumentParser(description="Table chunks -> SQLite facts database")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="chunks se database banao")
    b.add_argument("--chunks", default="STAGE_1/chunks")
    b.add_argument("--db", default="STAGE_1/facts.db")
    f = sub.add_parser("find", help="company, saal, line item -> number + page")
    f.add_argument("company")
    f.add_argument("year", type=int)
    f.add_argument("item")
    f.add_argument("--db", default="STAGE_1/facts.db")
    f.add_argument("--all", action="store_true", help="saari milti rows dikhao")
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.chunks, a.db)
    else:
        if not Path(a.db).exists():
            raise SystemExit(f"Database nahi mila: {a.db}. Pehle chalao: python STAGE_1/facts_db.py build")
        find(a.db, a.company, a.year, a.item, a.all)


if __name__ == "__main__":
    main()