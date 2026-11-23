"""
answer.py  -  RAG ka "G": retriever ke chunks se Gemini jawab likhta hai, citation ke saath.

Ek ek karke (interactive):   python STAGE_1/answer.py
    Model ek dafa load hota hai, phir "Sawal>" par sawal likho.
    #2   -> pichle jawab ka source [S2] poora dikhao
    exit -> band

Ek sawal:                    python STAGE_1/answer.py "What was Nike's net income in fiscal 2021?"

Har jawab ke saath wo 5 chunks bhi save hote hain jo Gemini ko diye gaye (answers_<tareekh>_<waqt>.json).
Is se ghalat jawab ka faisla ho sakta hai: sahi baat chunks mein thi hi nahi -> retrieval ki ghalti;
chunks mein thi magar jawab ghalat -> Gemini ki ghalti.

API key: repo ke root mein .env file ->  GEMINI_API_KEY=...   (ya environment variable).
Model badalna ho:  --model gemini-2.5-flash   ya .env mein GEMINI_MODEL=...   (--list-models se list)

Cache: wahi sawal + wahi chunks + wahi model -> Gemini ko dobara nahi bhejte (answer_cache.json).
Retriever "nahi mila" kahe (company index mein nahi, jaise Tesla) -> Gemini ko bheja hi nahi jata.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

DEFAULT_MODEL = "gemini-2.5-flash"
# Free tier ki daily limit HAR MODEL ki alag hoti hai. Ek model ki din ki limit khatam -> agla model try.
# .env mein badal sakte ho:  GEMINI_FALLBACK_MODELS=gemini-2.5-flash-lite,gemini-2.0-flash
FALLBACK_MODELS = ["gemini-2.5-flash-lite", "gemini-2.0-flash", "gemini-2.0-flash-lite"]
NOT_STATED = "Not stated in the provided filings."
STOP = {"exit", "quit", "q", "band", "bas"}

SYSTEM_PROMPT = f"""You are a financial analyst assistant answering questions about SEC 10-K annual reports.

Rules:
1. Answer ONLY from the numbered sources provided. Never use outside knowledge, even if you know the answer.
2. Cite every factual sentence with its source tag, e.g. [S1] or [S2][S4].
3. Copy numbers exactly as they appear in the source, with their units (millions, thousands, %, $).
   If a table says "(in millions)" or the figure is clearly in millions, say so.
4. Year labels are FISCAL years as used in the filing. Do not convert them to calendar years.
5. If the sources do not contain the answer, reply exactly: "{NOT_STATED}"
   If they contain only part of it, answer that part and say what is missing.
6. If sources disagree (for example, different fiscal years), say so and cite both.
7. Be concise: 1-4 sentences for a factual question, a short paragraph or bullet list for a "tell me about" question.
"""


# ----------------------------------------------------------------------
# setup
# ----------------------------------------------------------------------
def load_env():
    """Repo root (ya STAGE_1) ki .env file se KEY=VALUE lines -> os.environ (jo pehle se set hain unhein nahi chhedta)."""
    for p in (HERE.parent / ".env", HERE / ".env"):
        if p.exists():
            for line in p.read_text(encoding="utf-8-sig").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def make_client():
    try:
        from google import genai
    except ImportError:
        raise SystemExit("google-genai install nahi hai:  pip install google-genai")
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise SystemExit("API key nahi mili. Repo ke root mein .env file banao:  GEMINI_API_KEY=...  "
                         "(aur .gitignore mein .env likho)")
    return genai.Client(api_key=key)


# ----------------------------------------------------------------------
# prompt
# ----------------------------------------------------------------------
def page_str(pages):
    pages = pages or []
    if not pages:
        return "?"
    return f"{min(pages)}-{max(pages)}" if min(pages) != max(pages) else str(pages[0])


def build_sources(results):
    """Retriever results -> [{tag, company, year, section, pages, type, text}]"""
    out = []
    for i, r in enumerate(results, 1):
        out.append({
            "tag": f"S{i}",
            "company": r["company_display"],
            "year": r["year"],
            "section": r["section_path"],
            "pages": r.get("page_numbers") or [],
            "type": r["chunk_type"],
            "text": r["text"],
        })
    return out


def build_prompt(question, sources):
    blocks = []
    for s in sources:
        blocks.append(f"[{s['tag']}] {s['company']} | fiscal year {s['year']} | {s['type']} | "
                      f"PDF page {page_str(s['pages'])} | section: {s['section']}\n{s['text']}")
    return "SOURCES:\n\n" + "\n\n---\n\n".join(blocks) + f"\n\nQUESTION: {question}\n\nANSWER:"


# ----------------------------------------------------------------------
# cache
# ----------------------------------------------------------------------
class Cache:
    def __init__(self, path):
        self.path = Path(path)
        self.data = {}
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self.data = {}

    @staticmethod
    def key(model, prompt):
        return hashlib.sha256(f"{model}\x00{SYSTEM_PROMPT}\x00{prompt}".encode("utf-8")).hexdigest()

    def get(self, k):
        return self.data.get(k)

    def put(self, k, v):
        self.data[k] = v
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")


# ----------------------------------------------------------------------
# Gemini call (retry on rate limit)
# ----------------------------------------------------------------------
class DailyQuota(Exception):
    """Is model ki aaj ki free limit khatam (dobara koshish ka faida nahi, kal reset hogi)."""


_NO_THINKING = {"ok": True}     # thinking_budget band na ho sake (purana API version) to ek dafa yaad rakho


def call_gemini(client, model, prompt, tries=4):
    from google.genai import types
    # Hidayat (SYSTEM_PROMPT) sawal ke saath hi bheji jaati hai, alag "system_instruction" mein nahi:
    # kuch API versions (v1) mein system_instruction band hai -> 400 INVALID_ARGUMENT.
    prompt = SYSTEM_PROMPT + "\n\n" + prompt

    def cfg():
        # 2.5 models jawab se pehle "sochte" hain aur wo sochna bhi max_output_tokens mein ginta hai -> lamba
        # jawab beech mein kat jata tha. Sources mein se jawab nikaalne ke liye sochna zaroori nahi: band (tez bhi).
        kw = dict(temperature=0.1, max_output_tokens=2048)
        if _NO_THINKING["ok"]:
            kw["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        return types.GenerateContentConfig(**kw)

    wait = 15
    attempt = 0
    while True:
        attempt += 1
        try:
            resp = client.models.generate_content(model=model, contents=prompt, config=cfg())
            text = (resp.text or "").strip()
            try:
                reason = str(resp.candidates[0].finish_reason or "")
            except (AttributeError, IndexError, TypeError):
                reason = ""
            if "MAX_TOKENS" in reason:
                text += "\n[jawab lamba tha, yahan kat gaya]"
            return text
        except Exception as e:
            msg = str(e)
            if _NO_THINKING["ok"] and "thinking" in msg.lower() and ("400" in msg or "INVALID" in msg):
                _NO_THINKING["ok"] = False          # ye version thinking_config nahi leta -> uske baghair dobara
                attempt -= 1
                continue
            is_429 = "429" in msg or "RESOURCE_EXHAUSTED" in msg
            if is_429 and ("PerDay" in msg or "per day" in msg or re.search(r"limit:\s*0\b", msg)):
                raise DailyQuota(model)             # aaj ki limit khatam, ya is model ki free limit hai hi 0
            busy = any(x in msg for x in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "overloaded"))
            if busy and attempt < tries:
                m = re.search(r"retry in (\d+(?:\.\d+)?)s", msg) or re.search(r"retryDelay['\"]?:\s*['\"]?(\d+)", msg)
                sec = int(float(m.group(1))) + 1 if m else wait
                print(f"    (Gemini busy / limit -- {sec} sec ruk kar dobara, koshish {attempt + 1}/{tries})")
                time.sleep(sec)
                wait *= 2
                continue
            if "404" in msg or "NOT_FOUND" in msg:
                raise LookupError(model)
            raise


class ModelChain:
    """Pehla model; us ki din ki limit khatam ya model mojood nahi -> agla. Jo khatam ho chuke unhein yaad rakhta hai."""

    def __init__(self, models):
        self.models = [m for i, m in enumerate(models) if m and m not in models[:i]]
        self.dead = set()

    def current(self):
        for m in self.models:
            if m not in self.dead:
                return m
        return None

    def generate(self, client, prompt):
        while True:
            m = self.current()
            if m is None:
                raise DailyQuota("sab")
            try:
                return m, call_gemini(client, m, prompt)
            except DailyQuota:
                print(f"    ({m}: aaj ki free limit khatam -- agla model try kar raha hoon)")
            except LookupError:
                print(f"    ({m}: is key par ye model nahi -- agla)")
            self.dead.add(m)


# ----------------------------------------------------------------------
# ek sawal
# ----------------------------------------------------------------------
def cited_sources(answer, sources):
    tags = set(re.findall(r"\[(S\d+)\]", answer))
    return [s for s in sources if s["tag"] in tags]


def answer_one(retriever, client, cache, chain, question, k=5):
    t0 = time.time()
    resp = retriever.search(question, top_k=k)
    entry = {"question": question, "model": None, "retrieval_matched": resp["matched"],
             "heading": resp.get("heading")}
    if not resp["matched"]:
        entry.update(answer=f"{NOT_STATED} ({resp['reason']})", not_stated=True, cached=False,
                     sources=[], citations=[], seconds=round(time.time() - t0, 2))
        return entry
    sources = build_sources(resp["results"])
    prompt = build_prompt(question, sources)
    ck = Cache.key("gemini", prompt)          # model ka naam key mein nahi: fallback model ka jawab bhi cache ho
    hit = cache.get(ck)
    if isinstance(hit, dict):
        text, used, cached = hit["text"], hit.get("model"), True
    elif hit is not None:                     # purana cache (sirf text)
        text, used, cached = hit, None, True
    else:
        try:
            used, text = chain.generate(client, prompt)
        except DailyQuota:
            entry.update(answer="[Gemini ki aaj ki free limit sab models par khatam -- kal dobara, ya billing on karo]",
                         not_stated=False, cached=False, sources=sources, citations=[],
                         seconds=round(time.time() - t0, 2), error="daily_quota")
            return entry
        cached = False
        cache.put(ck, {"text": text, "model": used})
    entry["model"] = used
    cites = cited_sources(text, sources)
    entry.update(
        answer=text,
        not_stated=text.strip().startswith(NOT_STATED[:20]),
        cached=cached,
        citations=[{"tag": s["tag"], "company": s["company"], "year": s["year"],
                    "page": page_str(s["pages"]), "section": s["section"]} for s in cites],
        sources=sources,
        seconds=round(time.time() - t0, 2),
    )
    return entry


def show(entry, n):
    print("\n" + "=" * 74)
    print(f"Q{n:02d}: {entry['question']}")
    print("=" * 74)
    if entry.get("heading"):
        print(f"(Section pehchana: {entry['heading']})")
    print("\n" + entry["answer"] + "\n")
    if entry["citations"]:
        print("Sources:")
        for c in entry["citations"]:
            print(f"  [{c['tag']}] {c['company']} FY{c['year']} | PDF page {c['page']} | {c['section'][-80:]}")
    elif entry["sources"] and not entry["not_stated"]:
        print("  (Jawab mein koi [S#] citation nahi -- is jawab par bharosa kam karo)")
    if entry["sources"]:
        others = [s for s in entry["sources"] if s["tag"] not in {c["tag"] for c in entry["citations"]}]
        if others:
            print("Diye gaye magar use nahi hue: " + ", ".join(
                f"[{s['tag']}] {s['company']} FY{s['year']} p{page_str(s['pages'])}" for s in others))
    tail = f"  |  poora source: #1 ... #{len(entry['sources'])}" if entry["sources"] else "  |  Gemini ko bheja nahi gaya"
    if entry.get("model"):
        tail = f"  |  {entry['model']}" + tail
    print(f"\n({entry['seconds']} sec{', cache se' if entry['cached'] else ''}{tail})")


def show_source(entry, i):
    if not entry or not entry.get("sources"):
        print("    Pichle jawab ka koi source nahi.")
        return
    if not 1 <= i <= len(entry["sources"]):
        print(f"    #1 se #{len(entry['sources'])} tak hi hain.")
        return
    s = entry["sources"][i - 1]
    print("\n" + "-" * 74)
    print(f"[{s['tag']}] {s['company']} FY{s['year']} | {s['type']} | PDF page {page_str(s['pages'])}")
    print("Section:", s["section"])
    print("-" * 74)
    print(s["text"])
    print("-" * 74)


# ----------------------------------------------------------------------
def main():
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser(description="RAG: retriever + Gemini jawab, citations ke saath")
    ap.add_argument("question", nargs="?", help="ek sawal; na do to ek ek karke poochho")
    ap.add_argument("-k", type=int, default=5, help="Gemini ko kitne chunks (default 5)")
    ap.add_argument("--model", default=None, help=f"Gemini model (default {DEFAULT_MODEL})")
    ap.add_argument("--store", default="STAGE_1/vector_store")
    ap.add_argument("--cache", default="STAGE_1/answer_cache.json")
    ap.add_argument("--out", default=None, help="default: answers_<tareekh>_<waqt>.json")
    ap.add_argument("--list-models", action="store_true", help="tumhari key par kaun se models hain")
    a = ap.parse_args()

    load_env()
    client = make_client()
    if a.list_models:
        for m in client.models.list():
            acts = getattr(m, "supported_actions", None) or []
            if not acts or "generateContent" in acts:
                print(m.name)
        return
    model = a.model or os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL
    fb = os.environ.get("GEMINI_FALLBACK_MODELS")
    fallbacks = [x.strip() for x in fb.split(",")] if fb else FALLBACK_MODELS
    chain = ModelChain([model] + fallbacks)
    out = a.out or time.strftime("answers_%Y%m%d_%H%M.json")

    from retriever import Retriever
    retriever = Retriever(a.store)
    cache = Cache(a.cache)
    log = []

    def run(q):
        e = answer_one(retriever, client, cache, chain, q, a.k)
        log.append(e)
        show(e, len(log))
        Path(out).write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
        return e

    if a.question:
        run(a.question)
        return

    print(f"\nTayyar. Model: {model} (limit khatam ho to: {', '.join(chain.models[1:])}).  Sawal likho;  #2 = source [S2] poora;  exit = band")
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
            show_source(last, int(m.group(1)))
            continue
        last = run(line)
    if log:
        print(f"\n{len(log)} jawab save: {out}")


if __name__ == "__main__":
    main()