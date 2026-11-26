"""
answer.py  -  RAG ka "G": retriever ke chunks se LLM (Gemini / Groq) jawab likhta hai, citation ke saath.

Ek ek karke (interactive):   python STAGE_1/answer.py
    Model ek dafa load hota hai, phir "Sawal>" par sawal likho.
    #2   -> pichle jawab ka source [S2] poora dikhao
    exit -> band

Ek sawal:                    python STAGE_1/answer.py "What was Nike's net income in fiscal 2021?"

Har jawab ke saath wo 5 chunks bhi save hote hain jo model ko diye gaye (answers_<tareekh>_<waqt>.json).
Is se ghalat jawab ka faisla ho sakta hai: sahi baat chunks mein thi hi nahi -> retrieval ki ghalti;
chunks mein thi magar jawab ghalat -> model ki ghalti.

API keys: repo ke root mein .env file ->  GEMINI_API_KEY=...   aur/ya  GROQ_API_KEY=...
Models: pehle Gemini (GEMINI_MODEL, GEMINI_FALLBACK_MODELS), phir Groq (GROQ_MODELS). Poori tarteeb khud:
LLM_MODELS=groq:openai/gpt-oss-120b,gemini-2.5-flash,...    Pehla model:  --model ...   List:  --list-models

Cache: wahi sawal + wahi chunks + wahi model -> model ko dobara nahi bhejte (answer_cache.json).
Retriever "nahi mila" kahe (company index mein nahi, jaise Tesla) -> model ko bheja hi nahi jata.

Number wale sawal (revenue, net income, EPS ... + company): SQL serving layer (facts.db -> financials, banao:
financials_db.py build) se SEC-verified numbers ek extra source [D1] ban kar model ko jaate hain, % badlao
Python mein hisaab ho kar. facts.db na ho, ya STRUCTURED_DATA=0 -> sirf PDF chunks (pehle jaisa).
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
try:
    from financials_db import FinancialData, detect_metrics   # SQL serving layer (number wale sawal)
except Exception:                               # noqa: BLE001  -- na ho to RAG akela chalta hai
    FinancialData = detect_metrics = None
sys.path.insert(0, str(HERE))

DEFAULT_MODEL = "gemini-2.5-flash"
# Free tier ki daily limit HAR MODEL ki alag hoti hai. Ek model ki din ki limit khatam -> agla model try.
# .env mein badal sakte ho:  GEMINI_FALLBACK_MODELS=gemini-2.5-flash-lite,gemini-2.0-flash
FALLBACK_MODELS = ["gemini-2.5-flash-lite", "gemini-2.0-flash", "gemini-2.0-flash-lite"]
# Groq (dusra provider, free, card nahi; bohat tez). Chain mein "groq:" se pehchane jaate hain.
# .env:  GROQ_API_KEY=...   aur chaho to  GROQ_MODELS=openai/gpt-oss-120b,llama-3.3-70b-versatile
GROQ_URL = "https://api.groq.com/openai/v1"
GROQ_MODELS = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b"]
# Poori tarteeb khud tay karni ho:  LLM_MODELS=groq:openai/gpt-oss-120b,gemini-2.5-flash,...
NOT_STATED = "Not stated in the provided filings."
TRUNCATED = "[jawab lamba tha, yahan kat gaya]"
STOP = {"exit", "quit", "q", "band", "bas"}
REQUEST_TIMEOUT_MS = 45_000   # ek model ko max 45 sec, phir agla
COMPARE_K = 8          # comparison sawal par zyada chunks (dono taraf ke liye jagah)
COMPARE_WORDS = re.compile(
    r"\b(compare[sd]?|comparison|comparing|vs\.?|versus|difference|differ(?:ent|s)?|"
    r"change[sd]? (?:from|between)|grew|growth (?:from|between)|increase[sd]? from|decrease[sd]? from|"
    r"higher|lower|more than|less than|muqabla|farq)\b", re.I)

COMPARE_PROMPT = """This is a COMPARISON question. In addition to the rules:
- State each side separately: company, fiscal year, value or point, with its citation.
- If numbers are compared, add a small markdown table (rows = items, columns = companies or years), then the
  difference: absolute and percent, computed only from the cited numbers. Show the arithmetic in one line,
  e.g. (46,710 - 44,538) / 44,538 = 4.9%. Both numbers must be in the same unit before you subtract.
- If one side is missing from the sources, say exactly which one and do not guess it.
"""
# Jab [D#] database source ho: farq / badlao Python pehle hi hisaab kar chuka -- model khud hisaab na kare.
COMPARE_PROMPT_DATA = """This is a COMPARISON question. In addition to the rules:
- State each side separately: company, fiscal year, value or point, with its citation.
- If numbers are compared, add a small markdown table (rows = items, columns = companies or years).
- For differences and changes, use the values already computed in the [D#] source and cite it. Do not do your
  own arithmetic and do not write formulas.
- If one side is missing from the sources, say exactly which one and do not guess it.
"""

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
8. Use only names, definitions and expansions that appear in the sources. If a source uses an abbreviation
   (for example "CWT") without spelling it out, keep the abbreviation; never guess what it stands for.
9. Keep each number's exact meaning from its source. A total is not a change and a change is not a total:
   "profit was $8,884 million" must never become "profit decreased by $8,884 million".
10. Write in the Latin alphabet only. Answer in English; if the question is in Roman Urdu, you may answer in
    Roman Urdu (Urdu written with English letters). Never use Devanagari, Arabic or any other script.
"""
# v4 (judge calibration ke baad): insaani jaanch mein 8/95 jawabon mein chhoti gharhi baatein mili -- "reputation",
# "supply chain", "partners competitors ban sakte hain" -- aur ek segment ko poori company bana dena (Walmart
# International). PROMPT_VERSION=4 se naya prompt; default abhi 3 (dev par naap kar badlenge).
RULES_V4 = """11. Add nothing the cited source does not say: no extra effects, reasons, examples or conclusions, even
    obvious ones (for example "this could hurt its reputation"). Each tag goes on the source that actually
    contains that sentence. Use quotation marks only for wording copied exactly from a source.
12. Keep each fact's scope as the source states it. If a source describes one segment, region, program or
    year (for example "Walmart International"), say so; do not present it as the whole company. A risk
    factor's "may" or "could" describes a risk, not the company's plan or strategy.
"""
PROMPT_VERSION = os.environ.get("PROMPT_VERSION", "3")   # v4 jaanch ke baad default banega
if PROMPT_VERSION != "3":
    SYSTEM_PROMPT = SYSTEM_PROMPT + RULES_V4


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


class Clients:
    """Har provider ka client / key. Jis ki key na ho wo None (us ke models chain mein nahi aate)."""

    def __init__(self, gemini=None, groq_key=None):
        self.gemini, self.groq_key = gemini, groq_key


def _gemini_client(key):
    try:
        from google import genai
    except ImportError:
        print("    (google-genai install nahi: python -m pip install google-genai -- Gemini models chhor diye)")
        return None
    # Har request ki time-limit, aur library ke apne andar wale retries band: hum khud agla model try karte hain.
    # (Iske baghair ek atki hui request program ko hamesha ke liye rok deti thi.)
    try:
        from google.genai import types
        opts = types.HttpOptions(timeout=REQUEST_TIMEOUT_MS, retry_options=types.HttpRetryOptions(attempts=1))
        return genai.Client(api_key=key, http_options=opts)
    except (AttributeError, TypeError):
        try:
            return genai.Client(api_key=key, http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS))
        except (AttributeError, TypeError):
            return genai.Client(api_key=key)


def make_client():
    gkey = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    qkey = os.environ.get("GROQ_API_KEY")
    if not gkey and not qkey:
        raise SystemExit("Koi API key nahi mili. Repo ke root ki .env file mein  GEMINI_API_KEY=...  aur/ya  "
                         "GROQ_API_KEY=...  likho (aur .gitignore mein .env).")
    return Clients(gemini=_gemini_client(gkey) if gkey else None, groq_key=qkey or None)


def _env_list(name):
    v = os.environ.get(name)
    return [x.strip() for x in v.split(",") if x.strip()] if v else None


def build_chain(clients, first=None, env_name="LLM_MODELS"):
    """Models ki tarteeb: LLM_MODELS (poori list) ya: Gemini (GEMINI_MODEL + GEMINI_FALLBACK_MODELS) phir Groq."""
    models = _env_list(env_name)
    if models is None:
        models = []
        if clients.gemini is not None:
            models += [first or os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL]
            models += _env_list("GEMINI_FALLBACK_MODELS") or FALLBACK_MODELS
        if clients.groq_key:
            models += ["groq:" + m for m in (_env_list("GROQ_MODELS") or GROQ_MODELS)]
    elif first:
        models = [first] + models
    models = [m for m in models if (m.startswith("groq:") and clients.groq_key)
              or (not m.startswith("groq:") and clients.gemini is not None)]
    if not models:
        raise SystemExit("Chain mein koi model nahi (keys check karo).")
    return ModelChain(models)


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
            "company_key": r["company"],
            "year": r["year"],
            "section": r["section_path"],
            "pages": r.get("page_numbers") or [],
            "type": r["chunk_type"],
            "text": r["text"],
        })
    return out


def build_prompt(question, sources, compare=False):
    blocks = []
    for s in sources:
        if s["type"] == "structured":
            blocks.append(f"[{s['tag']}] {s['company']} | {s['section']}\n{s['text']}")
            continue
        blocks.append(f"[{s['tag']}] {s['company']} | fiscal year {s['year']} | {s['type']} | "
                      f"PDF page {page_str(s['pages'])} | section: {s['section']}\n{s['text']}")
    has_data = any(s["type"] == "structured" for s in sources)
    head = (COMPARE_PROMPT_DATA if has_data else COMPARE_PROMPT) + "\n" if compare else ""
    return head + "SOURCES:\n\n" + "\n\n---\n\n".join(blocks) + f"\n\nQUESTION: {question}\n\nANSWER:"


def is_comparison(retriever, question):
    """Do companies, do saal, ya 'compare / vs / difference' jaise alfaaz -> comparison."""
    try:
        comps = retriever._detect_companies(question)
    except Exception:
        comps = set()
    years = {int(y) for y in retriever.YEAR_PATTERN.findall(question)}
    return len(comps) >= 2 or len(years) >= 2 or bool(COMPARE_WORDS.search(question))


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


class NetworkDown(Exception):
    """Internet / DNS nahi -- har model par yahi hoga, is liye agla model try karne ka faida nahi."""


class Busy(Exception):
    """Is model ki is MINUTE ki limit / server busy. args[1] = kitne second baad dobara (andaza)."""


_NO_THINKING = {"ok": True}     # thinking_budget band na ho sake (purana API version) to ek dafa yaad rakho


def _config():
    from google.genai import types
    # 2.5+ models jawab se pehle "sochte" hain aur wo sochna bhi max_output_tokens mein ginta hai -> lamba
    # jawab beech mein kat jata tha. Sources mein se jawab nikaalne ke liye sochna zaroori nahi: band (tez bhi).
    kw = dict(temperature=0.1, max_output_tokens=2048 if _NO_THINKING["ok"] else 8192)
    if _NO_THINKING["ok"]:
        kw["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    return types.GenerateContentConfig(**kw)


def _classify(e, model):
    """Gemini error -> hamari qisam. 'retry' = thinking_config ke baghair dobara; warna exception raise."""
    msg = str(e)
    if any(x in msg for x in ("getaddrinfo", "ConnectError", "Name or service not known", "nodename nor servname",
                              "Network is unreachable", "No route to host")) or type(e).__name__ == "ConnectError":
        raise NetworkDown(msg[:120])
    if _NO_THINKING["ok"] and "thinking" in msg.lower() and ("400" in msg or "INVALID" in msg):
        _NO_THINKING["ok"] = False
        return "retry"
    is_429 = "429" in msg or "RESOURCE_EXHAUSTED" in msg
    if is_429 and ("PerDay" in msg or "per day" in msg or re.search(r"limit:\s*0\b", msg)):
        raise DailyQuota(model)                 # aaj ki limit khatam, ya is model ki free limit hai hi 0
    busy = any(x in msg for x in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "overloaded",
                                  "timed out", "Timeout", "timeout", "504", "DEADLINE"))
    if busy:                                     # minute ki limit / server busy -> wait nahi, agla model
        m = re.search(r"retry in (\d+(?:\.\d+)?)s", msg) or re.search(r"retryDelay['\"]?:\s*['\"]?(\d+)", msg)
        raise Busy(model, int(float(m.group(1))) + 1 if m else 20)
    if "404" in msg or "NOT_FOUND" in msg:
        raise LookupError(model)
    raise e


def _finish_reason(resp):
    try:
        return str(resp.candidates[0].finish_reason or "")
    except (AttributeError, IndexError, TypeError):
        return ""


def call_gemini(client, model, prompt, system=SYSTEM_PROMPT):
    """Poora jawab ek saath (terminal / evaluation ke liye)."""
    # Hidayat (system) sawal ke saath hi bheji jaati hai, alag "system_instruction" mein nahi:
    # kuch API versions (v1) mein system_instruction band hai -> 400 INVALID_ARGUMENT.
    full = (system + "\n\n" + prompt) if system else prompt
    while True:
        try:
            resp = client.models.generate_content(model=model, contents=full, config=_config())
        except Exception as e:
            if _classify(e, model) == "retry":
                continue
        text = (resp.text or "").strip()
        if "MAX_TOKENS" in _finish_reason(resp):
            text += "\n" + TRUNCATED
        return text


def stream_gemini(client, model, prompt, system=SYSTEM_PROMPT):
    """Jawab tukdon mein (app ke liye: lafz lafz screen par). Pehla tukda aane tak errors yahin pakde jaate hain,
    taake ModelChain agla model try kar sake; us ke baad generator baqi tukde deta hai."""
    full = (system + "\n\n" + prompt) if system else prompt
    while True:
        try:
            it = iter(client.models.generate_content_stream(model=model, contents=full, config=_config()))
            first = next(it)
            break
        except StopIteration:
            return iter(())
        except Exception as e:
            if _classify(e, model) == "retry":
                continue

    def gen():
        last = first
        yield first.text or ""
        for part in it:
            last = part
            yield part.text or ""
        if "MAX_TOKENS" in _finish_reason(last):
            yield "\n" + TRUNCATED
    return gen()


# ---------------------------- Groq (OpenAI-jaisa API; sirf Python ki apni urllib, koi naya package nahi) ----
def _groq_request(key, model, prompt, system, stream):
    import urllib.request
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    body = {"model": model, "messages": msgs, "temperature": 0.1, "max_completion_tokens": 2048, "stream": stream}
    if model.startswith("openai/gpt-oss"):
        body["reasoning_effort"] = "low"     # sources se jawab nikaalne ke liye lamba sochna zaroori nahi
    elif "qwen" in model:
        body["reasoning_format"] = "hidden"  # <think> wala sochna jawab mein na aaye
    return urllib.request.Request(
        GROQ_URL + "/chat/completions", data=json.dumps(body).encode("utf-8"), method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                 "User-Agent": "techcorp-rag/1.0"})


def _groq_open(key, model, prompt, system, stream):
    """Request bhejo; error ho to hamari qisam (DailyQuota / Busy / LookupError / NetworkDown) raise."""
    import http.client
    import socket
    import urllib.error
    import urllib.request
    try:
        return urllib.request.urlopen(_groq_request(key, model, prompt, system, stream),
                                      timeout=REQUEST_TIMEOUT_MS / 1000)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        code, low = e.code, body.lower()
        if code == 429:
            if "per day" in low or "(tpd)" in low or "(rpd)" in low:
                raise DailyQuota("groq:" + model)
            ra = e.headers.get("retry-after")
            raise Busy("groq:" + model, int(float(ra)) + 1 if ra and ra.replace(".", "").isdigit() else 10)
        if code == 413 or "request too large" in low:        # prompt is model ki minute-limit se bara
            raise Busy("groq:" + model, 60)
        if code == 404 or "model_not_found" in low or "decommissioned" in low or "does not exist" in low:
            raise LookupError("groq:" + model)
        if code == 401:
            raise SystemExit("GROQ_API_KEY ghalat hai (401). .env check karo.")
        if code >= 500:
            raise Busy("groq:" + model, 10)
        raise RuntimeError(f"Groq {code}: {body[:200]}")
    except (socket.timeout, TimeoutError):
        raise Busy("groq:" + model, 10)
    except urllib.error.URLError as e:
        reason = str(e.reason)
        if "timed out" in reason:
            raise Busy("groq:" + model, 10)
        raise NetworkDown(reason[:120])
    except (http.client.HTTPException, ConnectionError):  # server ne beech mein connection kaat diya: chhoti gharbar
        raise Busy("groq:" + model, 5)


def call_groq(key, model, prompt, system=SYSTEM_PROMPT):
    import http.client
    try:
        with _groq_open(key, model, prompt, system, stream=False) as r:
            data = json.loads(r.read().decode("utf-8"))
    except (http.client.HTTPException, ConnectionError, json.JSONDecodeError):
        raise Busy("groq:" + model, 5)
    ch = (data.get("choices") or [{}])[0]
    text = re.sub(r"<think>.*?</think>", "", (ch.get("message") or {}).get("content") or "", flags=re.S).strip()
    if ch.get("finish_reason") == "length":
        text += "\n" + TRUNCATED
    return text


def stream_groq(key, model, prompt, system=SYSTEM_PROMPT):
    r = _groq_open(key, model, prompt, system, stream=True)

    def gen():
        reason, thinking = None, False
        try:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                ch = (json.loads(payload).get("choices") or [{}])[0]
                reason = ch.get("finish_reason") or reason
                piece = (ch.get("delta") or {}).get("content")
                if piece and "<think>" in piece:
                    thinking, piece = True, piece.split("<think>")[0]
                if thinking and piece and "</think>" in piece:
                    thinking, piece = False, piece.split("</think>", 1)[1]
                elif thinking:
                    piece = ""
                if piece:
                    yield piece
        finally:
            r.close()
        if reason == "length":
            yield "\n" + TRUNCATED
    return gen()


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

    def generate(self, client, prompt, stream=False, rounds=2, log=print, system=SYSTEM_PROMPT):
        """Models ek ek karke. Minute-limit wala (Busy) sirf is sawal ke liye chhodo; din-limit wala hamesha ke liye.
        Sab busy hon to sab se kam wait karke ek dafa aur poori line try.
        Wapas: (model, text)  ya stream=True par  (model, generator)."""
        def fn(m):
            if m.startswith("groq:"):
                return (stream_groq if stream else call_groq)(client.groq_key, m[5:], prompt, system)
            return (stream_gemini if stream else call_gemini)(client.gemini, m, prompt, system)
        for rnd in range(rounds):
            waits = []
            for m in self.models:
                if m in self.dead:
                    continue
                try:
                    return m, fn(m)
                except DailyQuota:
                    log(f"{m}: aaj ki free limit khatam -- agla model")
                    self.dead.add(m)
                except LookupError:
                    log(f"{m}: is key par ye model nahi -- agla")
                    self.dead.add(m)
                except Busy as b:
                    log(f"{m}: abhi busy / minute ki limit -- agla model")
                    waits.append(b.args[1])
            if not waits:
                break
            if rnd + 1 < rounds:
                sec = min(min(waits), 60)
                log(f"sab models busy -- {sec} sec ruk kar dobara")
                time.sleep(sec)
        raise DailyQuota("sab")


# ----------------------------------------------------------------------
# ek sawal:  prepare() -> (LLM) -> finish()      terminal aur app dono yahi use karte hain
# ----------------------------------------------------------------------
NO_MODEL = "[Abhi koi model dastiyab nahi (limits) -- thori der baad dobara poochho]"
NO_NET = "[Internet se rabta nahi (model tak nahi pohanch saka) -- connection check karke dobara poochho]"


_UNI = str.maketrans({"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-",
                      "\u00a0": " ", "\u202f": " ", "\u2009": " ", "\u2007": " ",
                      "\u200b": "", "\u200c": "", "\u200d": "", "\ufeff": ""})


def clean_answer(text):
    """Models ke alag citation / unicode andaaz ek jaise: 【S3】 / 【S1†L1-L9】 -> [S3], [S1, S2] -> [S1][S2], khaas dash/space."""
    text = text.translate(_UNI)
    text = re.sub(r"[【〔［]\s*([SD]\d+)(?:\s*†[^】〕］]*)?\s*[】〕］]", r"[\1]", text)   # 【S1†L1-L9】 bhi
    text = re.sub(r"\(\s*([SD]\d+)\s*\)", r"[\1]", text)                 # (S3) -> [S3]
    text = re.sub(r"\[([SD]\d+(?:\s*[,;]\s*[SD]\d+)+)\]",
                  lambda m: "".join(f"[{t}]" for t in re.findall(r"[SD]\d+", m.group(1))), text)
    return text


def cited_sources(answer, sources):
    tags = set(re.findall(r"\[([SD]\d+)\]", answer))
    return [s for s in sources if s["tag"] in tags]


_FIN = None


def financial_data():
    """Serving layer (facts.db -> financials) ek dafa kholo. STRUCTURED_DATA=0 -> band (ablation / test ke liye)."""
    global _FIN
    if _FIN is None:
        if os.environ.get("STRUCTURED_DATA", "1") == "0" or FinancialData is None:
            _FIN = False
        else:
            fin = FinancialData(os.environ.get("FIN_DB") or HERE / "facts.db")
            _FIN = fin if fin.available else False
    return _FIN or None


def structured_source(retriever, question):
    """Number wale sawal -> [D1] source (SEC se tasdeeq-shuda numbers), warna None."""
    fin = financial_data()
    if not fin:
        return None
    try:
        comps = set(retriever._detect_companies(question))
        if detect_metrics(question):
            # Number wale sawal mein "apple revenue" / "intel vs Amazon" (chhote huroof) bhi company hi hai; koi na
            # mile to spelling ki ghalti ("Microsft"). Ticker (NOW, COST) sirf bare huroof mein -- retriever khud dekhta hai.
            comps |= _loose_companies(retriever, question)
            if not comps:
                comps = retriever._fuzzy_companies(question)[0]
    except Exception:                               # noqa: BLE001
        return None
    years = {int(y) for y in retriever.YEAR_PATTERN.findall(question)}
    return fin.source(question, comps, years)


def _loose_companies(retriever, question):
    q = " " + re.sub(r"[^a-z0-9&]+", " ", question.lower()) + " "
    skip = getattr(retriever, "_upper_only", set())
    found = set()
    for a, ns in retriever.aliases.items():
        if a and a not in skip and re.search(r"\b" + re.escape(a) + r"\b", q):
            found |= ns
    return found


def prepare(retriever, cache, question, k=5, only_model=None):
    """Retrieval + prompt + cache. Wapas: dict(entry, prompt, key, cached_text, cached_model).
    entry["answer"] pehle se bhara ho (retriever ne 'nahi mila' kaha) to model ki zaroorat nahi."""
    t0 = time.time()
    compare = is_comparison(retriever, question)
    resp = retriever.search(question, top_k=max(k, COMPARE_K) if compare else k)
    entry = {"question": question, "model": None, "retrieval_matched": resp["matched"],
             "heading": resp.get("heading"), "comparison": compare, "_t0": t0}
    data = structured_source(retriever, question)
    entry["structured"] = bool(data)
    if not resp["matched"] and not data:
        entry.update(answer=f"{NOT_STATED} ({resp['reason']})", not_stated=True, cached=False,
                     sources=[], citations=[], seconds=round(time.time() - t0, 2))
        entry.pop("_t0")
        return {"entry": entry, "prompt": None, "key": None, "cached_text": None, "cached_model": None}
    sources = build_sources(resp["results"]) if resp["matched"] else []
    if data:
        sources.insert(0, data)            # pehle: model is ko zyada wazan deta hai
    entry["sources"] = sources
    prompt = build_prompt(question, sources, compare)
    # Cache: "kisi bhi model" ka jawab (app / fallback ke liye) aur "is model" ka jawab (model comparison ke liye).
    ck = Cache.key("gemini", prompt)
    if only_model:                            # ek hi model fix -> sirf usi ka cache, warna comparison ghalat
        hit = cache.get(Cache.key("ans|" + only_model, prompt))
    else:
        hit = cache.get(ck)
    text = hit.get("text") if isinstance(hit, dict) else hit
    if text and text.endswith(TRUNCATED):     # purana adhoora jawab -> dobara banao
        text = None
    return {"entry": entry, "prompt": prompt, "key": ck, "prompt_raw": prompt, "cached_text": text,
            "cached_model": hit.get("model") if isinstance(hit, dict) and text else None}


def finish(prep, cache, text, used, cached, error="no_model"):
    """Model ka jawab (ya cache) -> entry mukammal; naya aur poora jawab cache mein."""
    entry = prep["entry"]
    if text is None:
        entry.update(answer=NO_NET if error == "network" else NO_MODEL, not_stated=False, cached=False,
                     citations=[], error=error)
    else:
        text = clean_answer(text).strip()
        if not cached and not text.endswith(TRUNCATED):
            cache.put(prep["key"], {"text": text, "model": used})
            if used:
                cache.put(Cache.key("ans|" + used, prep["prompt_raw"]), {"text": text, "model": used})
        cites = cited_sources(text, entry["sources"])
        entry.update(
            answer=text, model=used, cached=cached,
            not_stated=text.startswith(NOT_STATED[:20]),
            citations=[{"tag": s["tag"], "company": s["company"], "year": s["year"], "type": s["type"],
                        "page": page_str(s["pages"]), "section": s["section"]} for s in cites],
        )
    entry["seconds"] = round(time.time() - entry.pop("_t0"), 2)
    return entry


def answer_one(retriever, client, cache, chain, question, k=5):
    prep = prepare(retriever, cache, question, k, only_model=chain.models[0] if len(chain.models) == 1 else None)
    if prep["prompt"] is None:
        return prep["entry"]
    if prep["cached_text"] is not None:
        return finish(prep, cache, prep["cached_text"], prep["cached_model"], True)
    try:
        used, text = chain.generate(client, prep["prompt"], log=lambda m: print(f"    ({m})"))
    except DailyQuota:
        return finish(prep, cache, None, None, False)
    except NetworkDown:
        return finish(prep, cache, None, None, False, error="network")
    return finish(prep, cache, text, used, False)


def show(entry, n):
    print("\n" + "=" * 74)
    print(f"Q{n:02d}: {entry['question']}")
    print("=" * 74)
    if entry.get("heading"):
        print(f"(Section pehchana: {entry['heading']})")
    if entry.get("comparison"):
        print(f"(Comparison sawal -- {len(entry['sources'])} sources)")
    print("\n" + entry["answer"] + "\n")
    if entry.get("error"):
        return
    if entry["citations"]:
        print("Sources:")
        for c in entry["citations"]:
            if c.get("type") == "structured":
                print(f"  [{c['tag']}] {c['company']} | {c['section']} (database)")
                continue
            print(f"  [{c['tag']}] {c['company']} FY{c['year']} | PDF page {c['page']} | {c['section'][-80:]}")
        for s in entry["sources"]:
            if s["type"] == "structured" and s["tag"] in {c["tag"] for c in entry["citations"]}:
                urls = sorted({r["sec_url"] for r in s.get("rows", []) if r.get("sec_url")})
                if urls:
                    print("  SEC filings: " + "  ".join(urls[:4]))
    elif entry["sources"] and not entry["not_stated"]:
        print("  (Jawab mein koi [S#] citation nahi -- is jawab par bharosa kam karo)")
    if entry["sources"]:
        others = [s for s in entry["sources"] if s["tag"] not in {c["tag"] for c in entry["citations"]}]
        if others:
            print("Diye gaye magar use nahi hue: " + ", ".join(
                f"[{s['tag']}] {s['company']} " + ("(database)" if s["type"] == "structured" else
                                                   f"FY{s['year']} p{page_str(s['pages'])}") for s in others))
    tail = f"  |  poora source: #1 ... #{len(entry['sources'])}" if entry["sources"] else "  |  model ko bheja nahi gaya"
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


def list_models(client):
    if client.gemini is not None:
        print("Model:")
        for m in client.gemini.models.list():
            acts = getattr(m, "supported_actions", None) or []
            if not acts or "generateContent" in acts:
                print("  " + m.name.replace("models/", ""))
    if client.groq_key:
        import urllib.request
        req = urllib.request.Request(GROQ_URL + "/models", headers={
            "Authorization": f"Bearer {client.groq_key}", "User-Agent": "techcorp-rag/1.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.loads(r.read().decode("utf-8"))
        print("Groq (chain mein 'groq:' laga kar):")
        for m in sorted(x["id"] for x in data.get("data", [])):
            print("  groq:" + m)


# ----------------------------------------------------------------------
def main():
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser(description="RAG: retriever + LLM jawab, citations ke saath")
    ap.add_argument("question", nargs="?", help="ek sawal; na do to ek ek karke poochho")
    ap.add_argument("-k", type=int, default=5, help="model ko kitne chunks (default 5)")
    ap.add_argument("--model", default=None, help="pehla model (jaise gemini-2.5-flash ya groq:openai/gpt-oss-120b)")
    ap.add_argument("--store", default="STAGE_1/vector_store")
    ap.add_argument("--cache", default="STAGE_1/answer_cache.json")
    ap.add_argument("--out", default=None, help="default: answers_<tareekh>_<waqt>.json")
    ap.add_argument("--list-models", action="store_true", help="tumhari key par kaun se models hain")
    a = ap.parse_args()

    load_env()
    client = make_client()
    if a.list_models:
        list_models(client)
        return
    chain = build_chain(client, a.model)
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

    print(f"\nTayyar. Model: {chain.models[0]} (limit khatam ho to: {', '.join(chain.models[1:])}).  Sawal likho;  #2 = source [S2] poora;  exit = band")
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