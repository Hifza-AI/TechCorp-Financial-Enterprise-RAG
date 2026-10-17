"""
make_eval_set.py  -  Retriever ko jaanchne ke liye ~450 sawal KHUD banata hai, har sawal ke sahi jawab ke pate ke saath.

Chalao:   python STAGE_1/make_eval_set.py              (default 450 sawal)
          python STAGE_1/make_eval_set.py --n 500
Banata:   STAGE_1/eval/questions.jsonl   (har line ek sawal + "gold" = sahi jawab kahan hai)

Har sawal tumhare apne chunks se banta hai, isliye pehle se pata hai ke sahi jawab kis company/saal/
section/number mein hai. eval_retriever.py phir khud check karta hai ke retriever ne wo top 5 mein diya ya nahi.

Sawalon ki 5 qismein:
  table_number   (~40%)  "What was Woodward's total assets in fiscal 2025?"  -> sahi = wo number (327,431...)
  section_topic  (~25%)  "What does Nordson's 2020 10-K say about Competitive Conditions?" -> sahi = wo section
  risk_factor    (~12%)  risk factors ke unwaan se
  natural_topic  (~15%)  aam insaan jaise: "How many employees does Graco have according to its 2025 10-K?"
  out_of_scope   (~8%)   jo data mein hai hi nahi (Tesla, Sony, 2031, mausam) -> sahi = "kuch nahi mila"

Note: table/section sawalon mein chunk ke apne alfaaz aate hain, isliye ye asli users se thore aasaan hain.
natural_topic wale asli users ke qareeb hain. Report mein dono alag dikhte hain.

v2 (v1 ke 441 sawalon ke 66 fail haath se dekhe -- 9 sawal khud galat the; unhein paida karne wale rule theek kiye):
  * Number sirf ASLI statement se, aur sahi statement se: balance sheet wali cheez (total assets, inventories,
    goodwill...) sirf balance sheet se, income wali sirf income statement se, D&A / capex cash flow se.
    (v1: "inventories" cash flow ki "inventories mein tabdeeli" se bana; Emerson ka "gross profit" ek doosri
    company ki table se.) Toota hua header ("Columns: 2021 733,117, ...") wali table chhod di jati hai.
  * Section sawal: heading mein saal ho ("2018 compared to 2017"), "%" ho, ya poora bada Item ho (MD&A) -> nahi.
    Fehrist / exhibit tukde (retriever.clutter_reason) kabhi sahi jawab nahi bante.
  * "suppliers" sawal ab "Supplemental ..." heading ko sahi nahi maanta; "environment" ki jagah "environmental".
  * Anjaan companies mein naye naam (Sony, Tencent, Adidas, ...) jo retriever ke code mein kahin nahi likhe --
    taake guard ka asli imtihan ho, rate kiya hua nahi.
"""
import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from build_embeddings import junk_reason, section_str
from retriever import DISPLAY_NAMES, EXTRA_ALIASES, clutter_reason, statement_type

SET_NAME = "v2"

COLS = re.compile(r"^Columns:\s*(.+)$", re.M)
TITLE = re.compile(r"^The following table shows (.+?) data for ", re.M)
NUM = r"\(?\$?\s?-?\d[\d,]*(?:\.\d+)?\s?\)?%?"
PAIR = re.compile(r"([^:]+?):\s*(" + NUM + r"|-|—)\s*(?:,\s*|$)")

# (row ka naam, kis statement mein hona chahiye: 1 = balance sheet, 2 = income statement, 3 = cash flow)
KEY_ROWS = [
    (r"total assets", {1}), (r"total liabilities", {1}), (r"total current assets", {1}),
    (r"total current liabilities", {1}), (r"cash and cash equivalents", {1}),
    (r"total (?:stockholders|shareholders|shareowners)'? equity", {1}), (r"total equity", {1}),
    (r"long-term debt", {1}), (r"inventories(?:, net| - net)?", {1}), (r"goodwill", {1}),
    (r"net (?:income|earnings)(?: attributable to [a-z .,&'-]{3,60})?", {2}), (r"net sales", {2}),
    (r"(?:total )?net revenues?", {2}), (r"total revenues?", {2}), (r"revenues?", {2}), (r"sales", {2}),
    (r"total sales(?: and revenues)?", {2}), (r"operating (?:income|profit)", {2}), (r"gross (?:profit|margin)", {2}),
    (r"research(?:,)? and development(?: expenses?| costs?)?", {2}), (r"provision for income taxes|income tax expense", {2}),
    (r"cost of (?:sales|goods sold|revenues?)", {2}), (r"selling, general and administrative(?: expenses)?", {2}),
    (r"total operating expenses", {2}), (r"interest expense(?:, net)?", {2}),
    (r"depreciation and amortization", {2, 3}), (r"capital expenditures", {3}),
]
KEY_RES = [(re.compile(r"^(?:" + rx + r")$"), types) for rx, types in KEY_ROWS]

GENERIC_LEAF = {"overview", "general", "introduction", "background", "other", "none", "summary", "business",
                "table of contents", "continued", "(continued)", "notes", "description", "index"}
BAD_LEAF = re.compile(r"\bended\b|in (?:millions|thousands)|\$|^\(?[a-z0-9]\)\s*$|^(?:item|part|note)\s*[\divx]+[a-z]?\.?$|"
                      r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d", re.I)
# Poora bada Item (sawal itna khula ke har chunk "sahi" ho jaye) -> sawal nahi banta
BIG_ITEM_LEAF = re.compile(r"^(?:management.s discussion and analysis\b.*|financial statements and supplementary data\b.*|"
                           r"exhibits?\b.*schedules?\b.*|notes to (?:the )?(?:consolidated )?financial statements\b.*|"
                           r"signatures?|part\s+[iv]+\b.*|form 10-k\b.*|annual report\b.*)$", re.I)
NUMBER_PREFIX = re.compile(r"^(?:(?:item|note)\s*\d+[a-z]?\s*[.:-]?\s*|\(?[a-z0-9]{1,3}\)\s*|\d+[.)]\s*)", re.I)

NATURAL = [
    ("Who are {co}'s main competitors and how does it compete", r"competi"),
    ("How many employees does {co} have", r"employees|human capital|workforce|personnel"),
    ("What properties or facilities does {co} own or lease", r"propert"),
    ("What legal proceedings or lawsuits is {co} involved in", r"legal proceedings|litigation|contingenc"),
    ("How does {co} manage cybersecurity risks", r"cybersecurity"),
    ("Who audits {co}'s financial statements", r"independent registered public accounting firm"),
    ("Did {co} buy back any of its own shares", r"repurchase|purchases of equity securities"),
    ("What dividends does {co} pay to shareholders", r"dividend"),
    ("What are {co}'s business segments", r"segment"),
    ("Which raw materials or suppliers is {co} dependent on",
     r"raw material|supplier|supply chain|\bsuppl(?:y|ies)\b|sources? of (?:raw )?materials"),
    ("What environmental rules affect {co}", r"environmental"),
    ("What is {co}'s order backlog", r"backlog"),
    ("How does {co} use derivatives to hedge risk", r"derivative|hedg"),
    ("What are {co}'s critical accounting estimates", r"critical accounting"),
    ("What does {co} disclose about income taxes", r"income tax"),
]

# Pehli line retriever ki "unknown company" list mein hai. Baqi naam retriever ke code mein KAHIN nahi likhe --
# ye asli imtihan hai ke anjaan company par "nahi mila" kehta hai ya kisi aur company ka data de deta hai.
OOS_COMPANIES = ["Tesla", "Netflix", "Nvidia", "Starbucks", "Walmart", "Ford", "Pfizer", "Disney", "Coca-Cola",
                 "JPMorgan", "Amazon", "Alphabet", "Meta Platforms", "Intel", "IBM", "Oracle", "Exxon",
                 "Siemens", "Toyota", "Samsung", "Unilever", "Nestle", "Alibaba",
                 "Sony", "Tencent", "Volkswagen", "Adidas", "Infosys", "HSBC", "BYD", "Aramco", "Shopify", "Spotify"]
OOS_TEMPLATES = ["What was {p} revenue in 2024?", "What risks does {c} face?", "How many employees does {c} have?",
                 "Who audits {p} financial statements?", "{c} 2024 total assets", "Does {c} pay dividends to shareholders?",
                 "What does {c} say about competition?"]
OOS_IRRELEVANT = ["What is the weather in Lahore today?", "Give me a recipe for chicken biryani.",
                  "Who won the last cricket world cup?", "What is the capital of France?",
                  "Write a short poem about the sea.", "How do I learn to swim?"]


def display_name(name):
    if name in DISPLAY_NAMES:                      # "Northrop Grumman", "Kimberly-Clark" ... (asli users jaise)
        return DISPLAY_NAMES[name].replace("Alphabet (Google)", "Google")
    n = name.replace("_", " ").strip()
    n = re.sub(r"\s+", " ", n)
    if n.isupper() and len(n) > 4:
        n = n.title()
    return n


def poss(co):
    return co + ("'" if co.endswith("s") else "'s")


def clean_label(lbl):
    l = lbl.replace("’", "'").strip().rstrip(":").strip()
    l = re.sub(r"\s+", " ", l)
    return l


def norm(t):
    return re.sub(r"\s+", " ", (t or "").replace("’", "'")).strip().lower()


def table_candidates(c, year):
    """Sirf asli statement (balance sheet / income / cash flow) ki table, aur row usi statement ki ho."""
    out = []
    stype = statement_type(c)
    if stype not in (1, 2, 3):
        return out
    t = c.get("text") or ""
    m = COLS.search(t)
    if not m or re.search(r"\d,\d{3}", m.group(1)):        # "Columns: 2021 733,117, ..." = toota hua header
        return out
    title = (TITLE.search(t).group(1) if TITLE.search(t) else "")
    for line in t.splitlines():
        if " -- " not in line:
            continue
        label, rest = line.split(" -- ", 1)
        lab = clean_label(label)
        types = next((ty for rx, ty in KEY_RES if rx.match(lab.lower())), None)
        if not types or stype not in types:
            continue
        for pm in PAIR.finditer(rest):
            col, val = pm.group(1).strip(" ,"), pm.group(2).strip()
            if "(col" in col:
                continue
            ys = re.findall(r"\b(?:19|20)\d{2}\b", col)
            if not ys or ys[-1] != str(year):
                continue
            mnum = re.search(r"\d[\d,]*(?:\.\d+)?", val)
            digits = mnum.group(0).rstrip(",") if mnum else ""
            if "," not in digits:              # sirf >= 1,000 wale numbers (pehchaan mein aasaan)
                continue
            out.append(dict(label=lab, value=digits, stmt=stype, title=title))
            break
    return out


def good_leaf(leaf, sec):
    words = re.findall(r"[A-Za-z]{2,}", leaf)
    return (2 <= len(words) <= 14 and leaf.lower() not in GENERIC_LEAF and not BAD_LEAF.search(leaf)
            and "|" not in leaf and "%" not in leaf and re.match(r"[A-Za-z]", leaf)
            and not re.search(r"\b(?:19|20)\d{2}\b", leaf) and not BIG_ITEM_LEAF.match(leaf)
            and not re.search(r"risk factors", sec, re.I))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="STAGE_1/chunks")
    ap.add_argument("--out", default="STAGE_1/eval/questions.jsonl")
    ap.add_argument("--n", type=int, default=450)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    rnd = random.Random(a.seed)

    files = sorted(Path(a.chunks).rglob("*_chunks.json"))
    if not files:
        raise SystemExit(f"Koi *_chunks.json nahi mili: {Path(a.chunks).resolve()}")
    print(f"{len(files)} files parh raha hoon...")

    cand = defaultdict(lambda: defaultdict(list))      # category -> company -> [items]
    years_by_co = defaultdict(set)
    for fi, f in enumerate(files, 1):
        ch = json.load(open(f, encoding="utf-8"))
        if not ch:
            continue
        co_raw = str(ch[0].get("company", "")).strip()
        yr = int(str(ch[0].get("year", "0")).strip() or 0)
        years_by_co[co_raw].add(yr)
        tabs, secs, risks = [], [], []
        sections_here = set()
        for c in ch:
            if junk_reason(c) or clutter_reason(c):
                continue
            sec = section_str(c)
            sections_here.add(sec)
            src = dict(file=f.name, page=c.get("page_numbers") or [], section=sec,
                       preview=(c.get("text") or "")[:240])
            if c.get("chunk_type") == "table":
                for row in table_candidates(c, yr):
                    tabs.append(dict(row, src=src))
                continue
            text = c.get("text") or ""
            if len(text) < 200:
                continue
            leaf_raw = sec.split(" > ")[-1].strip() if sec else ""
            leaf = NUMBER_PREFIX.sub("", leaf_raw).strip(" .:-")
            words = re.findall(r"[A-Za-z]{2,}", leaf)
            anchor = norm(text)[:80]
            if re.search(r"risk factors", sec, re.I) and len(words) >= 6:
                topic = " ".join(leaf.split()[:14]).rstrip(" .,;:")
                risks.append(dict(topic=topic, section=sec, anchor=anchor, src=src))
            elif good_leaf(leaf, sec):
                secs.append(dict(leaf=leaf, section=sec, anchor=anchor, src=src))
        # per file: kuch hi rakho (yaddasht kam, saal zyada)
        for t in rnd.sample(tabs, min(4, len(tabs))):
            cand["table_number"][co_raw].append(dict(t, year=yr))
        for s in rnd.sample(secs, min(4, len(secs))):
            cand["section_topic"][co_raw].append(dict(s, year=yr))
        for r_ in rnd.sample(risks, min(3, len(risks))):
            cand["risk_factor"][co_raw].append(dict(r_, year=yr))
        for tmpl, rx in NATURAL:
            hits = [s for s in sections_here if re.search(rx, s, re.I)]
            if hits:
                cand["natural_topic"][co_raw].append(dict(tmpl=tmpl, rx=rx, year=yr, example=hits[0]))
        if fi % 50 == 0:
            print(f"  {fi}/{len(files)}")

    quota = dict(table_number=round(a.n * 0.40), section_topic=round(a.n * 0.25), risk_factor=round(a.n * 0.12),
                 natural_topic=round(a.n * 0.15))
    quota["out_of_scope"] = max(20, a.n - sum(quota.values()))

    def round_robin(cat, k):
        pools = {co: rnd.sample(v, len(v)) for co, v in cand[cat].items() if v}
        cos = sorted(pools)
        rnd.shuffle(cos)
        out = []
        while len(out) < k and any(pools.values()):
            for co in cos:
                if pools[co] and len(out) < k:
                    out.append((co, pools[co].pop()))
        return out

    qs = []

    def add(cat, q, co, yr, gold, src=None):
        qs.append(dict(id=f"{cat[:3].upper()}-{len(qs) + 1:04d}", set=SET_NAME, category=cat, question=q,
                       company=co, year=yr, gold=gold, source=src or {}))

    for co, t in round_robin("table_number", quota["table_number"]):
        d = display_name(co)
        lab = t["label"].lower() if t["label"].isupper() else " ".join(
            w if (w.isupper() and len(w) > 1) else w.lower() for w in t["label"].split())   # "Net Income" -> "net income", "SG&A" rehta hai
        q = rnd.choice([f"What was {poss(d)} {lab} in fiscal {t['year']}?", f"{d} {t['year']} {lab}"])
        add("table_number", q, co, t["year"], dict(mode="value", value=t["value"], label=t["label"]), t["src"])
    for co, s in round_robin("section_topic", quota["section_topic"]):
        d = display_name(co)
        q = rnd.choice([f"What does {poss(d)} {s['year']} 10-K say about {s['leaf']}?", f"{d} {s['year']} {s['leaf']}"])
        add("section_topic", q, co, s["year"], dict(mode="section", section=s["section"], anchor=s["anchor"]), s["src"])
    for co, r_ in round_robin("risk_factor", quota["risk_factor"]):
        d = display_name(co)
        q = f"What risk does {d} describe in its {r_['year']} 10-K about: {r_['topic']}?"
        add("risk_factor", q, co, r_["year"], dict(mode="section", section=r_["section"], anchor=r_["anchor"]), r_["src"])
    for co, n_ in round_robin("natural_topic", quota["natural_topic"]):
        d = display_name(co)
        with_year = rnd.random() < 0.6
        q = n_["tmpl"].format(co=d) + (f" according to its {n_['year']} 10-K?" if with_year else "?")
        add("natural_topic", q, co, n_["year"] if with_year else None, dict(mode="section_regex", regex=n_["rx"]),
            dict(example_section=n_["example"]))

    corpus = " ".join(years_by_co).lower()
    oos_cos = [c for c in OOS_COMPANIES if c.lower() not in corpus and c.split()[0].lower() not in corpus
               and c.lower() not in EXTRA_ALIASES]          # Alphabet = Google, Meta = META ... data mein hain
    k_oos = quota["out_of_scope"]
    items = []
    for c in oos_cos:
        items.append(rnd.choice(OOS_TEMPLATES).format(c=c, p=poss(c)))
    some = sorted(years_by_co)
    for _ in range(6):
        co = rnd.choice(some)
        bad = rnd.choice([2031, 2032, 2005, 2008])
        items.append(f"What was {poss(display_name(co))} net income in fiscal {bad}?")
    items += OOS_IRRELEVANT
    rnd.shuffle(items)
    for q in items[:k_oos]:
        add("out_of_scope", q, None, None, dict(mode="none"))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for q in qs:
            fh.write(json.dumps(q, ensure_ascii=False) + "\n")
    cnt = defaultdict(int)
    for q in qs:
        cnt[q["category"]] += 1
    print(f"\n{len(qs)} sawal bane ({SET_NAME}): {dict(cnt)}")
    print(f"Companies: {len({q['company'] for q in qs if q['company']})}")
    print(f"File: {out.resolve()}")
    print("Ab chalao:  python STAGE_1/eval_retriever.py --ablation")


if __name__ == "__main__":
    main()