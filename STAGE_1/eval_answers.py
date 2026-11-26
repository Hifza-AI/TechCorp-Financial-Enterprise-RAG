"""
eval_answers.py  -  RAG ke JAWAB ka imtihan (retrieval ka imtihan retriever.py --test karta hai).

100 fair-test sawal (test_questions.csv, sahi jawab + page ke saath) answer.py se chalata hai, aur har sawal ka
faisla khud karta hai:

  correct         jawab mein sahi number / naam aa gaya
  generation      sahi jawab model ko diye gaye sources mein THA, magar jawab mein nahi aaya (model ki ghalti)
  retrieval       sources (top-k) mein nahi tha, magar top-30 mein tha (ranking ki ghalti)
  not_found       top-30 mein bhi nahi: parsing mein kho gaya, ya retrieval bohat door -> haath se dekhna
  refused         retriever ne "nahi mila" kaha (company / saal pehchana nahi)
  no_model        model ki free limit -- baad mein dobara chalao, ye sawal khud dobara poocha jayega

Free tier ke liye: jo sawal ho chuke wo dobara NAHI poochta (results file se resume), aur --limit se ek run
mein sirf N naye sawal.

    python STAGE_1/eval_answers.py --limit 25        # 25 naye sawal
    python STAGE_1/eval_answers.py --report          # sirf ab tak ka hisaab, koi API call nahi

LLM judge (--judge): exact-match alfaaz dekhta hai, matlab nahi (25 mein se 5 faisle ghalat thay). Judge ko sawal,
sahi jawab, sources aur hamara jawab diya jata hai; wo batata hai: jawab sahi / adhoora / ghalat, aur jawab ki
kaun si baat sources mein NAHI hai (hallucination). Jin jawabon ka judge nahi hua, unka pehle hota hai.
    python STAGE_1/eval_answers.py --judge --limit 25
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import answer as A                                                             # noqa: E402
from retriever import Retriever, load_gold, _gold_pattern, _soft, gold_check  # noqa: E402

DEEP_K = 30
JUDGE_VERSION = 3          # v2: judge ko section / page bhi; v3: naam / abbreviation ke poore form ki khaas jaanch
VERDICTS = ["correct", "partial", "wrong"]

JUDGE_PROMPT = """You are grading an answer produced by a retrieval-augmented system over SEC 10-K filings.

QUESTION: {question}

REFERENCE ANSWER (the key fact; alternatives separated by "|"): {gold}

SOURCES given to the system (the header line of each source, including its section heading, is part of the
source and counts as support):
{sources}

SYSTEM ANSWER:
{answer}

Grade two things:
1. verdict: "correct" if the system answer conveys the reference fact (wording may differ, any language);
   "partial" if it conveys only part of it or mixes it with a wrong statement; "wrong" if it misses it,
   contradicts it, or says the information is not available.
2. unsupported: every statement in the system answer that is NOT supported by the SOURCES. Check carefully:
   - every name of a company, segment, product or person, and every spelled-out form of an abbreviation
     (if the sources only say "CWT", an answer saying "WeatherTech (CWT)" is unsupported);
   - every number and what it measures (a total stated as a change is unsupported);
   - computed values (differences, percentages) are supported only if the arithmetic from cited numbers is right.
   Quote each unsupported piece briefly. Empty list if everything is supported.

Reply with JSON only, no other text:
{{"verdict": "correct|partial|wrong", "unsupported": ["..."], "reason": "one sentence"}}"""
CATS = ["correct", "generation", "retrieval", "not_found", "refused", "no_model"]


def gold_in_answer(row, text):
    soft = _soft(A.clean_answer(text).replace("**", "").replace("__", ""))
    return any(_gold_pattern(x.strip()).search(soft) for x in (row.get("answer") or "").split("|") if x.strip())


_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _nums(text):
    return {n.replace(",", "").rstrip(".") for n in _NUM.findall(text or "")}


def _key_nums(text):
    """Asli numbers: 3+ hindse ya decimal wale; saal (1900-2099) aur tareekh jaise chhote numbers nahi.
    Agar alternative mein sirf saal hi ho (jaise "1925"), to wahi."""
    ns = _nums(text)
    key = {n for n in ns if ("." in n or len(n.replace(".", "")) >= 3) and not re.fullmatch(r"(?:19|20)\d\d", n)}
    return key or {n for n in ns if re.fullmatch(r"(?:19|20)\d\d", n)}


def number_in_answer(row, text):
    """Number wale sawal: gold ke kisi ek alternative ke SAARE numbers jawab mein hain? (jumla nahi, sirf numbers).
    Sawal mein pehle se likhe numbers (saal) gine nahi jaate. None = is sawal par ye check lagu nahi."""
    if (row.get("type") or "").strip().lower() != "number":
        return None
    have, asked = _nums(A.clean_answer(text)), _nums(row.get("question"))
    alts = [x for x in (row.get("answer") or "").split("|") if x.strip()]
    wanted = [n for n in (_key_nums(a) - asked for a in alts) if n]
    return any(n <= have for n in wanted) if wanted else None


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


def make_judge_chain(client):
    """Judge jawab dene wale se ALAG company ka model ho to behtar (model apne jawab ko zyada nambar deta hai).
    JUDGE_MODELS diya ho to wahi; warna Groq key ho to pehle Groq, phir Gemini."""
    if A._env_list("JUDGE_MODELS"):
        return A.build_chain(client, env_name="JUDGE_MODELS")
    chain = A.build_chain(client)
    chain.models = [m for m in chain.models if m.startswith("groq:")] + \
                   [m for m in chain.models if not m.startswith("groq:")]
    return chain


def judge_llm(chain, client, cache, d):
    """Ek jawab par LLM ka faisla -> dict(verdict, unsupported, reason, model) ya None (limit / internet)."""
    srcs = "\n\n".join(
        f"[{s['tag']}] {s['company']} FY{s['year']} | PDF page {A.page_str(s.get('pages'))} | "
        f"section: {s.get('section', '')}\n{s['text']}" for s in d["sources"]) or "(none)"
    prompt = JUDGE_PROMPT.format(question=d["question"], gold=d["gold"], sources=srcs, answer=d["answer"])
    key = A.Cache.key(f"judge-v{JUDGE_VERSION}", prompt)
    hit = cache.get(key)
    if isinstance(hit, dict) and "verdict" in hit:
        return hit
    try:
        model, text = chain.generate(client, prompt, log=lambda m: print(f"    ({m})"), system="")
    except (A.DailyQuota, A.NetworkDown):
        return None
    m = re.search(r"\{.*\}", text, re.S)
    try:
        out = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        out = {}
    verdict = str(out.get("verdict", "")).strip().lower()
    res = {"verdict": verdict if verdict in VERDICTS else "unparsed",
           "unsupported": [str(x) for x in (out.get("unsupported") or []) if str(x).strip()],
           "reason": str(out.get("reason", ""))[:300], "model": model, "version": JUDGE_VERSION}
    if res["verdict"] != "unparsed":
        cache.put(key, res)
    return res


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
        print(f"\n  Exact phrase match: {counts['correct']}/{answered} = {100 * counts['correct'] / answered:.1f}%"
              "   (sakht: gold jumla hoobahoo; bayani jawabon ke liye kam andaza)")
    rowmap = {r["id"]: r for r in rows if isinstance(r, dict)}
    numq = [d for d in done.values() if d["category"] not in ("no_model", "refused")
            and (number_in_answer(rowmap[d["id"]], d["answer"]) if d["id"] in rowmap else d.get("number_match")) is not None]
    if numq:
        ok = [d for d in numq if (number_in_answer(rowmap[d["id"]], d["answer"]) if d["id"] in rowmap
                                  else d.get("number_match"))]
        print(f"  Number sawal (sahi number jawab mein): {len(ok)}/{len(numq)} = {100 * len(ok) / len(numq):.1f}%"
              + (f"   ghalat: {', '.join(sorted({d['id'] for d in numq} - {d['id'] for d in ok}))}" if len(ok) < len(numq) else ""))
    for c in ["generation", "retrieval", "not_found", "refused"]:
        ids = [d["id"] for d in done.values() if d["category"] == c]
        if ids:
            print(f"  {c}: {', '.join(ids)}")
    judged = [d for d in done.values() if (d.get("judge") or {}).get("verdict") in VERDICTS]
    if judged:
        J = len(judged)
        v = {x: [d["id"] for d in judged if d["judge"]["verdict"] == x] for x in VERDICTS}
        hall = [d["id"] for d in judged if d["judge"]["unsupported"]]
        print(f"\n  LLM JUDGE ({J} jawab):")
        print(f"    correct {len(v['correct']):>4}   {100 * len(v['correct']) / J:5.1f}%")
        print(f"    partial {len(v['partial']):>4}   {100 * len(v['partial']) / J:5.1f}%")
        print(f"    wrong   {len(v['wrong']):>4}   {100 * len(v['wrong']) / J:5.1f}%")
        print(f"    Faithfulness (koi unsupported baat nahi): {J - len(hall)}/{J} = {100 * (J - len(hall)) / J:.1f}%")
        for x in ("partial", "wrong"):
            if v[x]:
                print(f"    {x}: {', '.join(v[x])}")
        if hall:
            print(f"    unsupported baatein: {', '.join(hall)}")
        flips = [d["id"] for d in judged if (d["category"] == "correct") != (d["judge"]["verdict"] == "correct")]
        if flips:
            print(f"    exact-match aur judge mein ikhtilaf: {', '.join(flips)}  (inhein haath se dekh lo)")
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
    ap.add_argument("--judge", action="store_true", help="LLM judge bhi (pehle un jawabon ka jin ka judge nahi hua)")
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
    unjudged = [d for d in done.values() if d["category"] not in ("no_model", "refused")
                and (d.get("judge") or {}).get("verdict") not in VERDICTS] if a.judge else []
    if not todo and not unjudged:
        report(done, rows)
        print("Sab sawal ho chuke." + (" Sab ka judge bhi ho chuka." if a.judge else ""))
        return

    A.load_env()
    client = A.make_client()
    chain = A.build_chain(client)
    judge_chain = make_judge_chain(client) if a.judge else None
    retriever = Retriever(a.store)
    cache = A.Cache(a.cache)
    out.parent.mkdir(parents=True, exist_ok=True)

    def save():
        out.write_text(json.dumps(sorted(done.values(), key=lambda d: int(d["id"][1:])), ensure_ascii=False,
                                  indent=1), encoding="utf-8")

    budget = a.limit
    if unjudged:                                         # pehle purane jawabon ka judge
        print(f"\nJudge: {min(len(unjudged), budget)} purane jawab\n")
        for d in sorted(unjudged, key=lambda d: int(d["id"][1:]))[:budget]:
            res = judge_llm(judge_chain, client, cache, d)
            if res is None:
                print("\nModel limit / internet -- judge yahin ruka. Baad mein dobara chalao.")
                save()
                report(done, rows)
                return
            d["judge"] = res
            budget -= 1
            print(f"  {d['id']:<5} judge: {res['verdict']:<8} {'(unsupported: ' + str(len(res['unsupported'])) + ')' if res['unsupported'] else ''}")
            save()
            time.sleep(a.pause)
        todo = todo[:max(budget, 0)]
        if not todo:
            report(done, rows)
            return

    print(f"\n{len(todo)} sawal chalenge (pehle se ho chuke: {sum(1 for d in done.values() if d['category'] != 'no_model')})\n")
    for i, row in enumerate(todo, 1):
        entry = A.answer_one(retriever, client, cache, chain, row["question"])
        cat, ranks = judge(retriever, row, entry)
        done[row["id"]] = {
            "id": row["id"], "question": row["question"], "gold": row.get("answer"), "gold_page": row.get("page"),
            "company": row.get("company"), "year": row.get("year"), "type": row.get("type"),
            "category": cat, "gold_ranks": ranks, "answer": entry["answer"], "model": entry.get("model"),
            "number_match": number_in_answer(row, entry["answer"]) if entry.get("retrieval_matched") else None,
            "cached": entry.get("cached"), "seconds": entry.get("seconds"),
            "structured": entry.get("structured", False),
            "sources": [{k: s[k] for k in ("tag", "company", "year", "pages", "section", "type", "text")}
                        for s in entry.get("sources", [])],
        }
        jtxt = ""
        if a.judge and cat not in ("no_model", "refused"):
            res = judge_llm(judge_chain, client, cache, done[row["id"]])
            if res is not None:
                done[row["id"]]["judge"] = res
                jtxt = f"  judge: {res['verdict']}" + (f" (unsupported: {len(res['unsupported'])})" if res["unsupported"] else "")
        save()
        mark = {"correct": "✓", "no_model": "…"}.get(cat, "✗")
        print(f"[{i:>3}/{len(todo)}] {row['id']:<5} {mark} {cat:<11} {row['question'][:55]}{jtxt}")
        if cat == "no_model":
            if entry.get("error") == "network":
                print("\nInternet nahi chal raha. Connection theek karke yahi command dobara chalao -- yahin se shuru hoga.")
            else:
                print("\nSab models (LLM_MODELS) ki free limits abhi khatam. Baad mein yahi command dobara chalao -- yahin se shuru hoga.")
            break
        if not entry.get("cached"):
            time.sleep(a.pause)

    report(done, rows)


if __name__ == "__main__":
    main()