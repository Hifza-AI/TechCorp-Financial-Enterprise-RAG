"""
ask.py  -  retriever se sawal poochho; model sirf EK dafa load hota hai.

Ek ek karke (interactive):   python STAGE_1/ask.py
    Model load hone ke baad "Sawal>" aata hai: sawal likho, Enter dabao, results dekho, agla sawal likho.
    Page bhi check karna ho to:   23 | What does Nike say about its competitors in fiscal 2017?
    (23 = PDF viewer wala page number, jaise "23 / 140" -- page par chhapa hua number nahi)
    Band karne ke liye: exit  (ya khaali Enter)

File se, sab ek saath:      python STAGE_1/ask.py STAGE_1/my_questions.txt
    Har line ek sawal (upar wali shakal mein); "#" wali aur khaali lines chhod di jaati hain.

Dono mein har sawal ke baad nateeja my_questions_results.json mein save hota hai (review page mein khulta hai),
aur aakhir mein marking list + "Page check" ki ginti aati hai. retriever.py ko chhedta nahi.
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retriever import Retriever, _show  # noqa: E402

STOP = {"exit", "quit", "q", "band", "bas"}


def parse_line(line):
    """'23 | sawal' -> (23, 'sawal');  'sawal' -> (None, 'sawal')"""
    m = re.match(r"^(?:p(?:age)?\s*)?(\d{1,4})\s*\|\s*(.+)$", line.strip(), re.I)
    return (int(m.group(1)), m.group(2).strip()) if m else (None, line.strip())


def read_questions(path):
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1252", errors="replace")
    return [parse_line(l) for l in text.splitlines() if l.strip() and not l.strip().startswith("#")]


def ask_one(retriever, page, q, k, log, cat):
    resp = retriever.search(q, top_k=k)
    n = len(log) + 1
    _show(q, resp, label=f"Q{n:02d}")
    entry = {"n": n, "category": cat, "question": q}
    if page:
        ranks = ([i for i, r in enumerate(resp["results"], 1) if page in (r.get("page_numbers") or [])]
                 if resp["matched"] else [])
        entry.update(expected_page=page, page_ranks=ranks)
        print(f"\n    Tumhara page {page}: " + (f"#{ranks[0]} mein mila  ✓" if ranks else f"top {k} mein nahi  ✗"))
    entry.update(resp)
    log.append(entry)


def save(log, out):
    with open(out, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=4, ensure_ascii=False)


def summary(log, k, out):
    if not log:
        return
    print("\n\nMARKING LIST  (company aur saal bhi dekhna;  ✓ jawab mila | ~ adhoora | ✗ nahi mila)")
    for x in log:
        if x["matched"]:
            r = x["results"][0]
            pg = r["page_numbers"]
            got = f"#1: {r['company_display']} {r['year']} p{min(pg) if pg else '?'}"
        else:
            got = "nahi mila"
        q_ = x["question"] if len(x["question"]) <= 60 else x["question"][:57] + "..."
        extra = ""
        if "expected_page" in x:
            extra = (f"  [page {x['expected_page']}: #{x['page_ranks'][0]}]" if x["page_ranks"]
                     else f"  [page {x['expected_page']}: nahi]")
        print(f"[ ] Q{x['n']:02d}  {q_:<60}  -> {got}{extra}")
    withp = [x for x in log if "expected_page" in x]
    if withp:
        top5 = sum(1 for x in withp if x["page_ranks"])
        top1 = sum(1 for x in withp if x["page_ranks"] and x["page_ranks"][0] == 1)
        print(f"\nPage check: {top5}/{len(withp)} sawalon mein tumhara page top {k} mein aaya (#1 par: {top1}).")
        print("Page na mile tab bhi jawab ho sakta hai (agla / pichla page): '>>' wali lines padh kar faisla karo.")
    print(f"\nPoore results: {out}  (review page mein 'JSON file chuno' se kholo)")


def main():
    try:
        sys.stdout.reconfigure(errors="replace")    # output file mein bhejo (> out.txt) to bhi crash na ho
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser(description="Retriever se sawal poochho (model ek dafa load)")
    ap.add_argument("file", nargs="?", help="sawalon ki .txt file; na do to ek ek karke poochho")
    ap.add_argument("-k", type=int, default=5, help="kitne results (default 5)")
    ap.add_argument("--store", default="STAGE_1/vector_store")
    ap.add_argument("--out", default="my_questions_results.json")
    a = ap.parse_args()

    qs = read_questions(a.file) if a.file else None
    if a.file and not qs:
        raise SystemExit(f"{a.file} mein koi sawal nahi mila (har line ek sawal; # wali lines chhod di jaati hain).")
    retriever = Retriever(a.store)
    log = []
    if qs:
        cat = f"Mere sawal ({Path(a.file).name})"
        print(f"{a.file}: {len(qs)} sawal")
        for page, q in qs:
            ask_one(retriever, page, q, a.k, log, cat)
            save(log, a.out)
    else:
        cat = "Mere sawal (ek ek karke)"
        print("\nModel tayyar hai. Sawal likho aur Enter dabao.")
        print("Page bhi check karna ho to:  23 | sawal        Band karne ke liye: exit  (ya khaali Enter)")
        while True:
            try:
                line = input("\nSawal> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line or line.lower() in STOP:
                break
            page, q = parse_line(line)
            ask_one(retriever, page, q, a.k, log, cat)
            save(log, a.out)
    summary(log, a.k, a.out)


if __name__ == "__main__":
    main()