"""
eval_answers.py  -  RAG ke JAWAB ka imtihan (retrieval ka imtihan retriever.py --test karta hai).

100 fair-test sawal (test_questions.csv, sahi jawab + page ke saath) answer.py se chalata hai, aur har sawal ka
faisla khud karta hai:

  correct         jawab mein sahi number / naam aa gaya
  generation      sahi jawab Gemini ko diye gaye sources mein THA, magar jawab mein nahi aaya (Gemini ki ghalti)
  retrieval       sources (top-k) mein nahi tha, magar top-30 mein tha (ranking ki ghalti)
  not_found       top-30 mein bhi nahi: parsing mein kho gaya, ya retrieval bohat door -> haath se dekhna
  refused         retriever ne "nahi mila" kaha (company / saal pehchana nahi)
  no_model        Gemini ki free limit -- baad mein dobara chalao, ye sawal khud dobara poocha jayega

Free tier ke liye: jo sawal ho chuke wo dobara NAHI poochta (results file se resume), aur --limit se ek run
mein sirf N naye sawal.

    python STAGE_1/eval_answers.py --limit 25        # 25 naye sawal
    python STAGE_1/eval_answers.py --report          # sirf ab tak ka hisaab, koi API call nahi
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import answer as A                                                             # noqa: E402
from retriever import Retriever, load_gold, _gold_pattern, _soft, gold_check  # noqa: E402

DEEP_K = 30
CATS = ["correct", "generation", "retrieval", "not_found", "refused", "no_model"]


def gold_in_answer(row, text):
    soft = _soft(text)
    return any(_gold_pattern(x.strip()).search(soft) for x in (row.get("answer") or "").split("|") if x.strip())


def as_results(sources):
    return [{"company": s["company_key"], "year": s["year"], "text": s["text"]} for s in sources]


def judge(retriever, row, entry):
    if entry.get("error") in ("no_model", "network"):
        return "no_model", None
    if not entry["retrieval_matched"]:
        return "refused", None
    if gold_in_answer(row, entry["answer"]):
        return "correct", None
    src_ranks = gold_check(retriever, row, as_results(entry["sources"]))
    if src_ranks:
        return "generation", src_ranks
    deep = retriever.search(row["question"], top_k=DEEP_K)
    if deep["matched"]:
        deep_ranks = gold_check(retriever, row, deep["results"])
        if deep_ranks:
            return "retrieval", deep_ranks
    return "not_found", None


def report(done, rows):
    n = len(done)
    counts = {c: sum(1 for d in done.values() if d["category"] == c) for c in CATS}
    answered = n - counts["no_model"]
    print("\n" + "=" * 70)
    print(f"JAWAB KA IMTIHAN: {n}/{len(rows)} sawal chal chuke ({answered} ka jawab aaya)")
    print("=" * 70)
    for c in CATS:
        if counts[c]:
            pct = f"{100 * counts[c] / answered:5.1f}%" if answered and c != "no_model" else "      "
            print(f"  {c:<11} {counts[c]:>4}   {pct}")
    if answered:
        print(f"\n  Answer accuracy: {counts['correct']}/{answered} = {100 * counts['correct'] / answered:.1f}%")
    for c in ["generation", "retrieval", "not_found", "refused"]:
        ids = [d["id"] for d in done.values() if d["category"] == c]
        if ids:
            print(f"  {c}: {', '.join(ids)}")
    print()


def main():
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="STAGE_1/test_questions.csv")
    ap.add_argument("--out", default="STAGE_1/eval/answers_t100.json")
    ap.add_argument("--store", default="STAGE_1/vector_store")
    ap.add_argument("--cache", default="STAGE_1/answer_cache.json")
    ap.add_argument("--limit", type=int, default=25, help="is run mein kitne NAYE sawal (default 25)")
    ap.add_argument("--pause", type=float, default=4.0, help="har sawal ke baad kitne sec ruko (minute ki limit)")
    ap.add_argument("--report", action="store_true", help="sirf hisaab dikhao, kuch mat chalao")
    a = ap.parse_args()

    gold = load_gold(a.gold)
    rows = sorted(gold.values(), key=lambda r: int("".join(ch for ch in r.get("id", "0") if ch.isdigit()) or 0))
    if not rows:
        raise SystemExit(f"Test sawal nahi mile: {a.gold}")
    out = Path(a.out)
    done = {d["id"]: d for d in json.loads(out.read_text(encoding="utf-8"))} if out.exists() else {}

    if a.report:
        report(done, rows)
        return

    todo = [r for r in rows if r["id"] not in done or done[r["id"]]["category"] == "no_model"][:a.limit]
    if not todo:
        report(done, rows)
        print("Sab sawal ho chuke.")
        return

    A.load_env()
    client = A.make_client()
    import os
    model = os.environ.get("GEMINI_MODEL") or A.DEFAULT_MODEL
    fb = os.environ.get("GEMINI_FALLBACK_MODELS")
    chain = A.ModelChain([model] + ([x.strip() for x in fb.split(",")] if fb else A.FALLBACK_MODELS))
    retriever = Retriever(a.store)
    cache = A.Cache(a.cache)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{len(todo)} sawal chalenge (pehle se ho chuke: {sum(1 for d in done.values() if d['category'] != 'no_model')})\n")
    for i, row in enumerate(todo, 1):
        entry = A.answer_one(retriever, client, cache, chain, row["question"])
        cat, ranks = judge(retriever, row, entry)
        done[row["id"]] = {
            "id": row["id"], "question": row["question"], "gold": row.get("answer"), "gold_page": row.get("page"),
            "company": row.get("company"), "year": row.get("year"), "type": row.get("type"),
            "category": cat, "gold_ranks": ranks, "answer": entry["answer"], "model": entry.get("model"),
            "cached": entry.get("cached"), "seconds": entry.get("seconds"),
            "sources": [{k: s[k] for k in ("tag", "company", "year", "pages", "section", "type", "text")}
                        for s in entry.get("sources", [])],
        }
        out.write_text(json.dumps(sorted(done.values(), key=lambda d: int(d["id"][1:])), ensure_ascii=False,
                                  indent=1), encoding="utf-8")
        mark = {"correct": "✓", "no_model": "…"}.get(cat, "✗")
        print(f"[{i:>3}/{len(todo)}] {row['id']:<5} {mark} {cat:<11} {row['question'][:60]}")
        if cat == "no_model":
            if entry.get("error") == "network":
                print("\nInternet nahi chal raha. Connection theek karke yahi command dobara chalao -- yahin se shuru hoga.")
            else:
                print("\nGemini ki saari free limits abhi khatam. Baad mein yahi command dobara chalao -- yahin se shuru hoga.")
            break
        if not entry.get("cached"):
            time.sleep(a.pause)

    report(done, rows)


if __name__ == "__main__":
    main()