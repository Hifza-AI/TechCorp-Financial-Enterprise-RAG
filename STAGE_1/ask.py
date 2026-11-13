"""
ask.py  -  retriever se sawal poochho; model sirf EK dafa load hota hai.

Ek ek karke (interactive):   python STAGE_1/ask.py
    Model load hone ke baad "Sawal>" aata hai: sawal likho, Enter dabao, results dekho, agla sawal likho.
    Page bhi check karna ho to:   23 | What does Nike say about its competitors in fiscal 2017?
    (23 = PDF viewer wala page number, jaise "23 / 140" -- page par chhapa hua number nahi)
    Poora chunk dekhna ho to:     #2        (pichle sawal ka result [2] poora chhapta hai; #1, #3 ... waise hi)
    Har result hamesha poora dikhana ho to:   python STAGE_1/ask.py --full
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
import time
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


def show_full(resp, i):
    """Pichle sawal ke result [i] ka POORA text (terminal mein sirf 2-3 lines ka preview aata hai)."""
    if not resp or not resp.get("matched"):
        print("    Pichla sawal nahi hai ya us ka koi result nahi tha.")
        return
    rs = resp["results"]
    if not 1 <= i <= len(rs):
        print(f"    #1 se #{len(rs)} tak hi hain.")
        return
    r = rs[i - 1]
    pg = r.get("page_numbers") or []
    pg = f"{min(pg)}-{max(pg)}" if pg and min(pg) != max(pg) else (str(pg[0]) if pg else "?")
    print("\n" + "-" * 70)
    print(f"[{i}] POORA CHUNK  |  {r['company_display']} {r['year']}  |  {r['chunk_type']}  |  PDF page {pg}")
    print("Section:", r["section_path"])
    print("-" * 70)
    print(r["text"])
    print("-" * 70 + f"  ({len(r['text'])} chars)")


def ask_one(retriever, page, q, k, log, cat, full=False):
    resp = retriever.search(q, top_k=k)
    n = len(log) + 1
    _show(q, resp, label=f"Q{n:02d}")
    if full and resp["matched"]:
        for i in range(1, len(resp["results"]) + 1):
            show_full(resp, i)
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
    ap.add_argument("--out", default=None,
                    help="results file (default: my_questions_<tareekh>_<waqt>.json -- har run ki alag file, purani nahi mitti)")
    ap.add_argument("--full", action="store_true", help="har result ka poora chunk chhapo (sirf preview nahi)")
    a = ap.parse_args()
    if not a.out:
        a.out = time.strftime("my_questions_%Y%m%d_%H%M.json")

    qs = read_questions(a.file) if a.file else None
    if a.file and not qs:
        raise SystemExit(f"{a.file} mein koi sawal nahi mila (har line ek sawal; # wali lines chhod di jaati hain).")
    retriever = Retriever(a.store)
    log = []
    if qs:
        cat = f"Mere sawal ({Path(a.file).name})"
        print(f"{a.file}: {len(qs)} sawal")
        for page, q in qs:
            ask_one(retriever, page, q, a.k, log, cat, a.full)
            save(log, a.out)
    else:
        cat = "Mere sawal (ek ek karke)"
        print("\nModel tayyar hai. Sawal likho aur Enter dabao.")
        print("Page bhi check karna ho to:  23 | sawal      Poora chunk:  #2      Band karne ke liye: exit")
        last = None
        while True:
            try:
                line = input("\nSawal> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line or line.lower() in STOP:
                break
            m = re.fullmatch(r"#\s*(\d{1,2})", line)
            if m:
                show_full(last, int(m.group(1)))
                continue
            if line.lower() in ("#", "#all", "full"):
                for i in range(1, len((last or {}).get("results") or []) + 1):
                    show_full(last, i)
                continue
            page, q = parse_line(line)
            ask_one(retriever, page, q, a.k, log, cat, a.full)
            last = log[-1]
            save(log, a.out)
    summary(log, a.k, a.out)


if __name__ == "__main__":
    main()