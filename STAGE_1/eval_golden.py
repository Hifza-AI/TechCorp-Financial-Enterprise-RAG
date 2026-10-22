"""
eval_golden.py  -  Asli users jaise sawalon (golden set) par retriever ka imtihan.

Chalao:   python STAGE_1/eval_golden.py               (retriever.py ke FEATURES jaise hain, waise)
          python STAGE_1/eval_golden.py --ablation    (har naya sudhaar alag naapta hai, ~1-2 min)
          python STAGE_1/eval_golden.py --off fresh,expand   (kuch sudhaar band karke dekho)
Sawal:    STAGE_1/golden_questions.csv   (Excel / Google Sheets mein kholo, review karo, CSV mein save karo)
Nateeja:  terminal mein score + har fail sawal ke top 3 chunks;  STAGE_1/eval/golden_results.csv
          --ablation par:  STAGE_1/eval/golden_ablation.txt

golden_questions.csv ke columns:
  id        G01, G02 ... (Claude ka draft, aap ne review kiya)   H01, H02 ... (aap ke apne haath se likhe)
  question  sawal apne alfaaz mein (jaise analyst / investor poochega). Heading copy mat karo.
  company   kis company ka (Nike, Walmart ...). Do companies: "Costco|Home Depot". Anjaan company: khaali.
  year      wohi fiscal saal jo company statements ke column par likhti hai (2025). Do saal: "2024|2025".
            Jo baat har saal same hai (headquarters ...): khaali chhod do -> koi bhi saal chalega.
  answer    PDF se HU-BA-HU: number jaise chhapa ("51,362", bina $ aur "million" ke),
            ya jawab wale jumle se 3-10 alfaaz ka tukda ("two reportable segments, Square and Cash App").
            Kai jawab "|" se: number/text/latest mein "|" = KOI BHI EK kaafi;  compare/calc mein "|" = SAB chahiye.
            Jis sawal ka jawab data mein hai hi nahi: NOT_FOUND
  type      number / text / compare / calc / latest / out_of_scope
  page      PDF ka page (viewer wala physical page) -- baad mein Gemini ki citation check karne ke kaam aayega
  notes     ye sawal kya parakhta hai (zaroori nahi)

Sahi kab:  top 5 mein koi chunk -> company sahi + saal sahi (agar likha hai) + text mein "answer" mile.
           compare/calc: har hissa (dono saal / dono company ka number) top 5 mein mile.
           NOT_FOUND wale sawal par retriever "nahi mila" kahe to sahi.
"Gol jawab" (alag line mein): number 3,131 (million) ki jagah text mein "$3.1 billion" likha ho to bhi sahi.
           Asal score sakht (exact) wala hi hai; gol wala sirf dikhata hai ke kitne fail asal mein theek the.
Excel "442,111" ko 442111 bana de to bhi chalega -- number commas ke saath ya baghair dono tarah milta hai.
"""
import argparse
import csv
import re
import time
from collections import defaultdict
from pathlib import Path

from retriever import FEATURES, Retriever, ablation_variants, display_name

FIELDS = ["id", "question", "company", "year", "answer", "type", "page", "notes"]
NUMERIC = re.compile(r"^[\s$(]*-?\d[\d,]*(?:\.\d+)?\s*\)?%?$")
ALL_PARTS = {"compare", "calc"}          # in sawalon mein "|" = sab hisse chahiye
TYPES = {"number", "text", "compare", "calc", "latest", "out_of_scope"}
YEAR_OK = re.compile(r"^\s*(?:(?:19|20)\d{2}\s*(?:\|\s*(?:19|20)\d{2}\s*)*)?$")   # "", "2025", "2024|2025"
AUTHORS = {"G": "G = Claude draft + review", "H": "H = haath se likhe"}


def norm(t):
    t = (t or "").replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    t = t.replace("–", "-").replace("—", "-").replace(" ", " ")
    return re.sub(r"\s+", " ", t).strip().lower()


def num_regex(a):
    """'442,111' ya '442111' (Excel) -> dono '442,111' aur '442111' se milta hai, '1442,111' se nahi."""
    s = re.search(r"\d[\d,]*(?:\.\d+)?", a).group(0)
    whole, _, dec = s.partition(".")
    d = whole.replace(",", "")
    groups = []
    while len(d) > 3:
        groups.insert(0, d[-3:])
        d = d[:-3]
    groups.insert(0, d)
    pat = ",?".join(groups) + (r"\." + dec if dec else "")
    return re.compile(r"(?<![\d,.])" + pat + r"(?!\d)(?!,\d)(?!\.\d)")


def rounded_regex(a):
    """3,131 -> '3.1 billion' / '3.13 billion';  442,111 (thousands) -> '442.1 million'.  None agar chhota number."""
    s = re.search(r"\d[\d,]*(?:\.\d+)?", a).group(0)
    n = float(s.replace(",", ""))
    if n < 1000:
        return None
    forms = set()
    for v in (n / 1e3, n / 1e6):
        if v >= 1:
            forms |= {f"{v:.1f}", f"{v:.2f}"}
            if v >= 100:
                forms.add(f"{v:.0f}")
    alts = "|".join(re.escape(f) for f in sorted(forms, key=len, reverse=True))
    return re.compile(r"(?<![\d.,])(?:" + alts + r")\s*(?:billion|million|bn\b|mn\b)", re.I)


def matchers(answer):
    out = []
    for a in [x.strip() for x in (answer or "").split("|") if x.strip()]:
        if NUMERIC.match(a):
            out.append((a, "num", num_regex(a), rounded_regex(a)))
        else:
            out.append((a, "text", norm(a), None))
    return out


def hit(m, text, ntext, lenient=False):
    _, kind, pat, rnd = m
    if kind == "num":
        return bool(pat.search(text or "")) or bool(lenient and rnd is not None and rnd.search(text or ""))
    return pat in ntext


def read_rows(path):
    raw = Path(path).read_bytes()
    for enc in ("utf-8-sig", "cp1252"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    first = text.splitlines()[0] if text else ""
    delim = ";" if first.count(";") > first.count(",") else ","
    rows, problems = [], []
    rd = csv.reader(text.splitlines(), delimiter=delim)
    header = [h.strip().lstrip("﻿").lower() for h in next(rd, [])]
    missing = [f for f in ("id", "question", "answer") if f not in header]
    if missing:
        raise SystemExit(f"CSV ke header mein ye columns nahi mile: {missing}. Header hona chahiye: {','.join(FIELDS)}")
    for ln, cells in enumerate(rd, start=2):
        if not any(c.strip() for c in cells):
            continue
        if len(cells) > len(header) and any(c.strip() for c in cells[len(header):]):
            problems.append(f"line {ln}: {len(cells)} khaane hain, {len(header)} hone chahiye "
                            f"(jawab mein comma ho to Excel mein likho, ya poore khaane ko \"...\" mein band karo)")
            continue
        row = {h: (cells[i].strip() if i < len(cells) else "") for i, h in enumerate(header)}
        if not row.get("question"):
            continue
        typ = (row.get("type") or "").strip().lower()
        if typ and typ not in TYPES:
            problems.append(f"line {ln} ({row.get('id')}): type '{row.get('type')}' samajh nahi aaya -- row ke khaane "
                            f"khisak gaye? (koi khaali khaana ya comma ghayab). Ye sawal score mein NAHI gina.")
            continue
        if not YEAR_OK.match(row.get("year") or ""):
            problems.append(f"line {ln} ({row.get('id')}): year '{row.get('year')}' sahi nahi (2025 ya 2024|2025 ya "
                            f"khaali) -- row ke khaane khisak gaye? Ye sawal score mein NAHI gina.")
            continue
        rows.append(row)
    return rows, problems


def find_rank(resp, ms, comps, years, all_parts, lenient):
    """Sahi jawab kis rank par mila (None = top k mein nahi), aur har hisse ka pehla rank."""
    first = [None] * len(ms)
    for j, res in enumerate(resp["results"], 1):
        if comps and res["company"] not in comps:
            continue
        if years and int(res["year"] or 0) not in years:
            continue
        nt = norm(res["text"])
        for i, m in enumerate(ms):
            if first[i] is None and hit(m, res["text"], nt, lenient):
                first[i] = j
    got = [f for f in first if f]
    if all_parts and len(ms) > 1:
        return (max(first) if len(got) == len(ms) else None), first
    return (min(got) if got else None), first


def evaluate(r, rows, k, warn=None):
    out = []
    for q in rows:
        typ = (q.get("type") or "").strip().lower() or "text"
        ans = q.get("answer", "")
        oos = typ == "out_of_scope" or ans.strip().upper() == "NOT_FOUND"
        comps = set()
        for name in [x.strip() for x in (q.get("company") or "").split("|") if x.strip()]:
            found = r._detect_companies(name) or r._detect_companies(name.title())
            if not found and warn is not None:
                warn.append(f"{q['id']}: company '{name}' index mein pehchani nahi gayi (naam check karo)")
            comps |= found
        years = {int(y) for y in re.findall(r"(?:19|20)\d{2}", q.get("year") or "")}
        resp = r.search(q["question"], top_k=k)
        rank = rank_l = None
        first, ms = [], []
        if oos:
            ok = ok_l = not resp["matched"]
        else:
            ms = matchers(ans)
            if not ms and warn is not None:
                warn.append(f"{q['id']}: 'answer' khaali hai")
            allp = typ in ALL_PARTS
            rank, first = find_rank(resp, ms, comps, years, allp, lenient=False)
            rank_l, _ = find_rank(resp, ms, comps, years, allp, lenient=True)
            ok, ok_l = bool(rank and rank <= 5), bool(rank_l and rank_l <= 5)
        out.append(dict(q=q, typ="out_of_scope" if oos else typ, resp=resp, rank=rank, rank_l=rank_l, ok=ok,
                        ok_l=ok_l, first=first, ms=ms, author=(q["id"][:1].upper() if q.get("id") else "?")))
    return out


def table(title, groups):
    print(f"{title:28s} {'N':>3s} {'Hit@1':>7s} {'Hit@5':>7s} {'Hit@10':>7s} {'MRR':>6s}")
    for name, R in groups:
        if not R:
            continue
        n = len(R)
        print(f"{name:28s} {n:3d} {sum(1 for x in R if x['rank'] == 1) / n:7.1%} "
              f"{sum(1 for x in R if x['rank'] and x['rank'] <= 5) / n:7.1%} {sum(1 for x in R if x['rank']) / n:7.1%} "
              f"{sum(1 / x['rank'] for x in R if x['rank']) / n:6.3f}")


def summary_line(results):
    ins = [x for x in results if x["typ"] != "out_of_scope"]
    O = [x for x in results if x["typ"] == "out_of_scope"]
    n = max(1, len(ins))
    return dict(h5=sum(x["ok"] for x in ins) / n, h5l=sum(x["ok_l"] for x in ins) / n,
                h10=sum(1 for x in ins if x["rank"]) / n, oos=(sum(x["ok"] for x in O), len(O)), n=len(ins))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="STAGE_1/golden_questions.csv")
    ap.add_argument("--store", default="STAGE_1/vector_store")
    ap.add_argument("--out", default="STAGE_1/eval/golden_results.csv")
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--off", default="", help="comma separated sudhaar band karo: " + ",".join(FEATURES))
    ap.add_argument("--ablation", action="store_true", help="har naya sudhaar on/off kar ke alag naapo")
    a = ap.parse_args()

    rows, problems = read_rows(a.questions)
    print(f"{len(rows)} sawal mile ({a.questions}).")
    for p in problems:
        print("  ⚠️ ", p)
    r = Retriever(a.store)
    feats = dict(FEATURES)
    for f in [x.strip() for x in a.off.split(",") if x.strip()]:
        if f not in feats:
            ap.error(f"'{f}' koi sudhaar nahi. In mein se: {', '.join(feats)}")
        feats[f] = False

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if a.ablation:
        L = [f"GOLDEN ABLATION  |  {len(rows)} sawal  |  {a.questions}",
             "(Hit@5 = sakht/exact;  'gol bhi' = '$3.1 billion' jaisa gol jawab bhi sahi;  OOS = 'nahi mila' sahi)", "",
             f"{'Variant':18s} {'Hit@5':>7s} {'gol bhi':>8s} {'Hit@10':>7s} {'dHit@5':>7s} | {'OOS':>6s}"]
        base = None
        for name, fv in ablation_variants():
            r.features = dict(fv)
            s = summary_line(evaluate(r, rows, a.k))
            base = s["h5"] if base is None else base
            L.append(f"{name:18s} {s['h5']:7.1%} {s['h5l']:8.1%} {s['h10']:7.1%} {100 * (s['h5'] - base):+6.1f}  | "
                     f"{s['oos'][0]:>2d}/{s['oos'][1]:<3d}")
            print(f"  {name} ho gaya")
        L += ["", "dHit@5 = pehli line (base) ke muqable kitne points (+ acha, - bura). Jo sudhaar dono test sets mein - de,",
              "use retriever.py ke FEATURES mein False kar dena."]
        text = "\n".join(L)
        print("\n" + text)
        (out.parent / "golden_ablation.txt").write_text(text, encoding="utf-8")

    r.features = feats
    warn = []
    t0 = time.time()
    results = evaluate(r, rows, a.k, warn)
    secs = time.time() - t0

    # ---------------- score ----------------
    off = [f for f, v in feats.items() if not v]
    print(f"\nGOLDEN SET  |  {len(results)} sawal  |  {secs:.0f} sec  |  band sudhaar: {', '.join(off) or 'koi nahi'}\n")
    bytype = defaultdict(list)
    for x in results:
        bytype[x["typ"]].append(x)
    inscope = [x for x in results if x["typ"] != "out_of_scope"]
    table("Qism", [(n, R) for n, R in sorted(bytype.items()) if n != "out_of_scope"] + [("SAB (in-scope)", inscope)])
    s = summary_line(results)
    print(f"\nGol (rounded) jawab bhi sahi maan kar: Hit@5 {s['h5l']:.1%}  (sakht: {s['h5']:.1%})")
    O = bytype.get("out_of_scope", [])
    if O:
        print(f"out_of_scope: {sum(x['ok'] for x in O)}/{len(O)} sahi 'nahi mila' kaha")

    groups = sorted({x["author"] for x in results})
    if len(groups) > 1:
        print()
        table("Kis ne likha (in-scope)", [(AUTHORS.get(g, g), [x for x in inscope if x["author"] == g]) for g in groups])
        for g in groups:
            Og = [x for x in O if x["author"] == g]
            if Og:
                print(f"   {AUTHORS.get(g, g)}: out_of_scope {sum(x['ok'] for x in Og)}/{len(Og)}")
    for w in warn:
        print("  ⚠️ ", w)

    # ---------------- fail sawal ----------------
    fails = [x for x in results if not x["ok"]]
    print(f"\nFAIL ({len(fails)}):" if fails else "\nKoi fail nahi.")
    for x in fails:
        q, resp = x["q"], x["resp"]
        exp = "NOT_FOUND" if x["typ"] == "out_of_scope" else f"{q.get('company', '')} {q.get('year', '')} -> {q.get('answer', '')}"
        if x["ok_l"]:
            where = f"gol jawab #{x['rank_l']} par mila (asal mein theek)"
        elif x["rank"]:
            where = f"#{x['rank']} par mila (top 5 se bahar)"
        elif not resp["matched"]:
            where = "kuch nahi mila: " + (resp["reason"] or "")
        elif x["typ"] == "out_of_scope":
            where = "jawab de diya (asal mein 'nahi mila' kehna tha)"
        elif any(x["first"]):
            where = "adhoora -- " + ", ".join(f"{m[0]} {'#' + str(f) if f else 'NAHI mila'}" for m, f in zip(x["ms"], x["first"]))
        else:
            where = "top 10 mein nahi"
        print(f"\n[{q['id']}] ({x['typ']}) {q['question']}")
        print(f"   sahi jawab: {exp}   |   retriever: {where}")
        for j, res in enumerate(resp["results"][:3], 1):
            pg = res["page_numbers"]
            pg = f"{min(pg)}" if pg else "?"
            print(f"   {j}. {display_name(res['company'])} {res['year']} p{pg} | {(res['section_path'] or '')[-70:]}")
            print(f"      {(res['text'] or '')[:150].replace(chr(10), ' / ')}")

    with open(out, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "type", "question", "company", "year", "answer", "rank", "ok", "rank_rounded", "ok_rounded",
                    "matched", "reason", "top1_company", "top1_year", "top1_page", "top1_section"])
        for x in results:
            q, resp = x["q"], x["resp"]
            t1 = resp["results"][0] if resp["results"] else {}
            w.writerow([q["id"], x["typ"], q["question"], q.get("company", ""), q.get("year", ""), q.get("answer", ""),
                        x["rank"] or "", int(x["ok"]), x["rank_l"] or "", int(x["ok_l"]), int(resp["matched"]),
                        resp["reason"] or "", display_name(t1.get("company", "")) if t1 else "", t1.get("year", ""),
                        t1.get("page_numbers", ""), (t1.get("section_path") or "")[-120:]])
    print(f"\nFile: {out.resolve()}")


if __name__ == "__main__":
    main()