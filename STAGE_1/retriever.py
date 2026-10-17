"""
retriever.py  -  Tumhari wahi Retriever class (wahi search() aur wahi jawab ka format), naye index ke saath.

Chalane ka tareeqa:
    python STAGE_1/retriever.py "Woodward 2025 total assets"
    python STAGE_1/retriever.py "net income" --company Nordson --year 2016
    python STAGE_1/retriever.py "Rockwell Automation 2016 goodwill" --off strip      (ek sudhaar band kar ke dekho)
    python STAGE_1/retriever.py --test          (neeche wale test sawal, retriever_test_results.json banata hai)

Doosre code se (Gemini wale step mein):
    from retriever import Retriever
    r = Retriever()
    resp = r.search("What was Woodward's total assets in 2025?", top_k=5)
    # resp = {"matched": True/False, "reason": ..., "results": [ {score, rrf_score, company, company_display, year,
    #          section_path, page_numbers, chunk_type, text}, ... ]}

v2: eval (441 sawal) ke 66 fail ek ek kar ke dekhe, un se 4 sudhaar. Har sudhaar FEATURES mein on/off hai taake
eval_retriever.py --ablation har ek ka asar alag naap sake:
  guard    Anjaan company ("What risks does Samsung face?") par "nahi mila". Pehle sirf list wale naam pakadta
           tha; ab sawal ka subject (does X face / X's revenue / "X 2024 ...") dekhta hai.
  clutter  Fehrist (table of contents, index, sirf page numbers) aur exhibit list ke bache tukde search mein
           nahi aate. Asli numbers wala (5+ bade numbers) ya signature wala chunk kabhi nahi chhuta.
  strip    Company ka naam (filter ban chuka) search ke alfaaz se nikal do. Warna "Northrop Grumman Corporation
           is a leading..." jaise chunk sirf naam ki wajah se upar aate the. Saal bhi nikalta hai, magar number
           wale sawal mein nahi (wahan saal sahi table / column dhoondne mein madad karta hai).
  stmt     "total assets / revenue / D&A" jaise number wale sawal par asli financial statement (balance sheet,
           income statement, cash flow) ko teesri list (RRF) se vote milta hai. Pehle notes / MD&A jeet jate the.
v2.1 (v2 eval ke baqi 20 fail dekh kar):
  strip    ab "according to its 2024 10-K", "annual report" jaise alfaaz bhi nikalta hai. Har chunk 10-K hi se hai,
           in alfaaz se sirf "This Annual Report on Form 10-K ..." jaise boilerplate chunk upar aate the.
  clutter  XBRL / Exhibit 101 ki fehrist ("(i) Consolidated Statements of ..., (ii) ...", "101.SCH ...") bhi.
  dedup    Bina saal ke sawal par ek hi paragraph 5 saalon se 5 dafa aa jata tha. Ab hu-ba-hu wahi text ek
           dafa aata hai (naye saal wala), baqi jagah doosri cheezein aati hain.
Index dobara banane ki zaroorat NAHI. Pehli baar chalne par chunks ke flags (fehrist? statement?) ek dafa
bante hain aur vector_store/flags.npz mein save ho jate hain (agli baar seedha load).
"""
import argparse
import inspect
import json
import re
import time
import zlib
from pathlib import Path

import faiss
import numpy as np
import scipy.sparse as sp

from build_embeddings import EmbeddingIndexBuilder, tok, section_str, INC_REF, EXHIBIT_REF, FORM_REF

# Har sudhaar on/off. eval_retriever.py --ablation inhein badal badal kar naapta hai.
FEATURES = {"guard": True, "clutter": True, "strip": True, "stmt": True, "dedup": True}


# ---------------------------------------------------------------------------------------------
# Company ke doosre naam / ticker -> index wala naam. Sirf tab lagta hai jab wo company index mein ho.
# (Google = Alphabet, META = Facebook, Block = Square, waghera.)  Naya naam chahiye to yahan likh do.
# ---------------------------------------------------------------------------------------------
EXTRA_ALIASES = {
    "alphabet": "Google", "googl": "Google", "goog": "Google",
    "facebook": "META", "meta platforms": "META",
    "square": "Block",
    "watts": "WATTS WATER TECHNOLOGIES", "watts water": "WATTS WATER TECHNOLOGIES", "wts": "WATTS WATER TECHNOLOGIES",
    "northrop grumman": "NORTHROP", "noc": "NORTHROP",
    "rockwell automation": "Rockwell Auomation", "rok": "Rockwell Auomation",
    "kimberly clark": "KImberly_Clark", "kmb": "KImberly_Clark",
    "colgate palmolive": "Colgate_Palmolive",
    "cmg": "Chipotle Mexican Grill", "panw": "Palo Alto Networks",
    "southwest airlines": "Southwest", "luv": "Southwest", "union pacific": "Union Pacific", "unp": "Union Pacific",
    "home depot": "Home Depot", "the home depot": "Home Depot",
    "aapl": "Apple", "amzn": "Amazon", "msft": "Microsoft", "nvda": "Nvidia", "nflx": "Netflix", "intc": "Intel",
    "adbe": "Adobe", "wday": "Workday", "pypl": "Paypal", "wmt": "Walmart", "sbux": "Starbucks", "nke": "Nike",
    "lmt": "Lockheed Martin", "mmm": "3M", "emr": "Emerson Electric", "rop": "Roper Technologies",
    "ggg": "Graco", "ndsn": "NORDSON", "kmt": "KENNAMETAL", "dci": "Donaldson", "fele": "Franklin Electric",
    "csl": "Carlisle", "xyl": "Xylem Inc", "pnr": "Pentair plc", "wwd": "Woodward Inc", "cvx": "CHEVRON",
    "intu": "Intuit", "salesforce": "SalesForce", "cat": "Caterpillar",
}
# Jawab / citation mein dikhane wala saaf naam (index ka naam waisa hi rehta hai).
DISPLAY_NAMES = {
    "CHEVRON": "Chevron", "NORDSON": "Nordson", "NORTHROP": "Northrop Grumman", "KENNAMETAL": "Kennametal",
    "WATTS WATER TECHNOLOGIES": "Watts Water Technologies", "Colgate_Palmolive": "Colgate-Palmolive",
    "KImberly_Clark": "Kimberly-Clark", "Rockwell Auomation": "Rockwell Automation", "META": "Meta Platforms",
    "SalesForce": "Salesforce", "Ebay": "eBay", "Paypal": "PayPal", "Google": "Alphabet (Google)",
    "Woodward Inc": "Woodward", "Xylem Inc": "Xylem", "Pentair plc": "Pentair",
}
# Ye naam aam English lafz bhi hain -- sirf tab company maano jab sawal mein Capital se likhe hon (Block, Workday...)
AMBIGUOUS = {"block", "workday", "southwest", "apple", "amazon", "meta", "graco", "intel", "nike", "cat"}
# Pehla lafz alias na banao (bohat aam lafz): "union" (labor union), "home", "palo"...
NO_FIRST_WORD = {"union", "home", "general", "united", "american", "first", "national"}


def display_name(company):
    return DISPLAY_NAMES.get(company, company)


# ---------------------------------------------------------------------------------------------
# Chunk flags (flags.npz mein ek dafa bante hain)
# ---------------------------------------------------------------------------------------------
FLAGS_VERSION = 1          # (neeche ke rules badlein to flags.npz khud dobara banti hai -- _rules_sig dekho)

TABLE_TITLE = re.compile(r"^The following table shows (.+?) data for ", re.M)
# Asli financial statement ka unwaan: 1 = balance sheet, 2 = income statement, 3 = cash flow,
# 4 = comprehensive income (is ko vote nahi milta, bas pehchan ke liye)
STMT_TITLE = re.compile(
    r"\b(?:(?P<bs>(?:consolidated\s+|combined\s+)?balance\s+sheets?|(?:consolidated\s+)?statements?\s+of\s+"
    r"(?:consolidated\s+)?financial\s+(?:position|condition)|consolidated\s+financial\s+position)"
    r"|(?P<ci>statements?\s+of\s+(?:consolidated\s+)?comprehensive\s+(?:income|loss|earnings)"
    r"|comprehensive\s+income\s+statements?)"
    r"|(?P<is>statements?\s+of\s+(?:consolidated\s+)?(?:income|operations|earnings)|income\s+statements?"
    r"|consolidated\s+results\s+of\s+operations)"
    r"|(?P<cf>statements?\s+of\s+(?:consolidated\s+)?cash\s+flows?|cash\s+flows?\s+statements?))\b", re.I)
STMT_CODE = {"bs": 1, "is": 2, "cf": 3, "ci": 4}
# Statement ke neeche chhoti sub-heading ("EMERSON ELECTRIC CO. & SUBSIDIARIES", "(In thousands)") -> parent dekho
SUBTITLE = re.compile(r"subsidiar|\(?\s*(?:in |dollars in |amounts in )?(?:millions|thousands)\b|^years? ended|"
                      r"^for the (?:fiscal )?(?:years?|periods?) ended|^as of\b|^(?:at|as at)\s|corporation|\binc\b|"
                      r"\bco\b|&|^\(?unaudited|^(?:january|february|march|april|may|june|july|august|september|"
                      r"october|november|december)\s+\d", re.I)
NOT_STMT = re.compile(r"supplement|condensed|consolidating|guarant|parent|summar|selected|unaudited|pro forma|"
                      r"segment|discussion|quarterly|reconcil|non-gaap|other|additional|details?|classification|"
                      r"included|off-|arrangement|fair value|derivative|\bnote\s*\d|\bin the\b", re.I)
# Statement ke naam ke BAAD ye aaye to wo note / doosri cheez hai ("... Balance Sheet Classification")
BAD_SUFFIX = re.compile(r"^\W*(?:information|info\b|classification|data\b|liabilit|arrangement|risk|the effect|"
                        r"effect|fair value|derivative|as\b|details?|items?\b|captions?|line)", re.I)


def _stmt_code(s):
    s = (s or "").strip()
    m = STMT_TITLE.search(s) if 0 < len(s) <= 100 else None
    if not m or m.start() > 45 or NOT_STMT.search(s[:m.start()]) or BAD_SUFFIX.search(s[m.end():]):
        return 0                                       # "NIKE, Inc. Consolidated Balance Sheets" theek hai
    return STMT_CODE[m.lastgroup]


def statement_type(c):
    """0 = statement nahi. Sirf table chunk jis ka unwaan / heading asli statement ka naam ho."""
    if c.get("chunk_type") != "table":
        return 0
    parts = section_str(c).split(" > ")
    m = TABLE_TITLE.search((c.get("text") or "")[:400])
    cands = ([m.group(1)] if m else []) + [parts[-1]]
    if len(parts) >= 2 and any(SUBTITLE.search(x) for x in cands) and not any(NOT_STMT.search(x) for x in cands):
        cands.append(parts[-2])
    for s in cands:
        code = _stmt_code(s)
        if code:
            return code
    return 0


ITEM_TITLES = re.compile(
    r"\b(business|risk factors|unresolved staff comments|properties|legal proceedings|mine safety disclosures|"
    r"market for (?:the )?registrant|selected financial data|management.s discussion|quantitative and qualitative|"
    r"financial statements and supplementary|changes in and disagreements|controls and procedures|other information|"
    r"directors, executive officers|executive compensation|security ownership|certain relationships|"
    r"principal account\w* fees|form 10-k summary|cybersecurity|exhibits(?:,| and) financial statement)", re.I)
EXH_LEAF = re.compile(r"^(?:\(?[a-z0-9]{1,3}[.)]\s*)*(?:index to |index of |list of )?(?:the )?exhibits?"
                      r"(?: index| list(?:ing)?)?\s*[:*]?\s*$", re.I)
BIG_NUM = re.compile(r"(?<![\d.])\d{1,3}(?:,\d{3})+(?:\.\d+)?|\b\d+\.\d+\b")
PAGE_LINE = re.compile(r"^\s*(?:[^|\d]{0,80}\|\s*)?(?:\$\s*\|\s*)?(\d{1,3})\s*$")
ITEM_TOKEN = re.compile(r"\bItem\s*\d{1,2}[A-C]?\b", re.I)
XBRL_DOC = re.compile(r"\b101\.(?:INS|SCH|CAL|DEF|LAB|PRE)\b|taxonomy extension", re.I)
ROMAN_ITEM = re.compile(r"\((?:i|ii|iii|iv|v|vi|vii|viii|ix|x)\)")
CONS_STMT = re.compile(r"consolidated (?:statements?|balance sheets?)", re.I)


def clutter_reason(c):
    """None = asli content.  'toc' = fehrist (table of contents / index),  'exhibit' = exhibit list ka bacha tukda.
    75 files par parkha: 0.7% chunks pakde, koi asli number wala ya eval ka sahi-jawab wala chunk nahi."""
    t = c.get("text") or ""
    if re.search(r"/s/|attorney-in-fact", t, re.I):          # signatures (naam) -> rakho
        return None
    if len(BIG_NUM.findall(t)) >= 5:                         # asli numbers wala chunk -> rakho
        return None
    sec = section_str(c)
    leaf = sec.split(" > ")[-1].strip()
    pages = [p for p in (c.get("page_numbers") or []) if isinstance(p, int)]
    early = bool(pages) and min(pages) <= 6
    titles = {x.lower() for x in ITEM_TITLES.findall(t[:2000])}
    if early and re.match(r"(?:index|table of contents|contents)\b", leaf, re.I):
        return "toc"
    if c.get("chunk_type") == "table":
        if len(titles) >= 4:                                 # "Business | 4 / Risk Factors | 12 / ..."
            return "toc"
        body = t.split("\n")[1:]
        nums = [int(m.group(1)) for m in (PAGE_LINE.match(x) for x in body) if m]
        if len(nums) >= 5 and len(nums) >= 0.5 * len(body):  # sirf page numbers: 65 / 68 / 69 / 70 ...
            if sum(1 for a, b in zip(nums, nums[1:]) if b >= a) >= 0.8 * (len(nums) - 1):
                return "toc"
    elif early and len(t) <= 400 and len(titles) >= 3 and len(ITEM_TOKEN.findall(sec)) >= 3:
        return "toc"                                         # heading hi "Item 1B. | Item 3. | Item 4." ho
    if EXH_LEAF.match(leaf):
        return "exhibit"
    if XBRL_DOC.search(t):                                   # "101.SCH XBRL Taxonomy Extension Schema ..."
        return "exhibit"
    if (len(ROMAN_ITEM.findall(t)) >= 3 and len(CONS_STMT.findall(t)) >= 2
            and re.search(r"xbrl|notes to (?:the )?consolidated financial statements", t, re.I)):
        return "exhibit"                                     # Exhibit 101: "(i) Consolidated Statements of ..., (ii) ..."
    n_inc, n_ex, n_form = len(INC_REF.findall(t)), len(EXHIBIT_REF.findall(t)), len(FORM_REF.findall(t))
    if re.search(r"\bexhibit", sec, re.I) and ((n_ex >= 1 and (n_inc or n_form)) or n_ex >= 2 or n_inc >= 2):
        return "exhibit"
    return None


# ---------------------------------------------------------------------------------------------
# "stmt": kaun sa sawal kis statement ka hai
# ---------------------------------------------------------------------------------------------
LINE_ITEMS = [
    (1, re.compile(r"\b(?:total (?:current )?assets|total (?:current )?liabilities|current assets|current liabilities|"
                   r"cash and (?:cash )?equivalents|inventor(?:y|ies)|goodwill|long[- ]term debt|total debt|"
                   r"(?:stockholders|shareholders|shareowners)['’]?\s+equity|total equity|accounts receivable|"
                   r"accounts payable|retained earnings|property,? plant,? and equipment)\b", re.I)),
    (2, re.compile(r"\b(?:revenues?|net sales|total sales|cost of (?:sales|revenues?|goods sold|products sold)|"
                   r"gross (?:profit|margin)|operating (?:income|profit|loss|expenses)|selling, general|sg&a|"
                   r"research and development|interest expense|provision for income taxes|"
                   r"income tax (?:expense|provision)|net (?:income|earnings|loss)|earnings per share)\b", re.I)),
    (3, re.compile(r"\b(?:depreciation(?: and amortization)?|capital expenditures?|capex|"
                   r"cash (?:provided by|from|used (?:in|for)) (?:operating|investing|financing)|operating cash flows?|"
                   r"free cash flow|dividends paid)\b", re.I)),
]
QUANT_CUE = re.compile(r"\b(?:what (?:was|were|is|are)|how much|how many|total|amount|value|figure)\b", re.I)
CONCEPT_CUE = re.compile(r"\b(?:how (?:does|do)|why|recogni(?:[sz]\w*|tion)|polic(?:y|ies)|describe|explain|risks?|"
                         r"strateg\w*|factors?|driv\w*|drove)\b", re.I)


def line_item_types(query):
    """Number wala financial sawal hai to statement types {1,2,3}, warna khaali set."""
    types = {t for t, rx in LINE_ITEMS if rx.search(query)}
    if not types or CONCEPT_CUE.search(query):
        return set()
    if QUANT_CUE.search(query) or len(query.split()) <= 8:
        return types
    return set()


# ---------------------------------------------------------------------------------------------
# "guard": sawal ka subject koi anjaan company to nahi?
# ---------------------------------------------------------------------------------------------
# "according to its 2024 10-K", "in the annual report" -- har chunk 10-K hi se hai, ye alfaaz koi maani nahi dete
FILING_WORDS = re.compile(
    r"\b(?:(?:according to|as (?:per|stated in|reported in|disclosed in)|in|from|per)\s+(?:its|their|the|this|his|her)\s+)?"
    r"(?:(?:most recent|latest|recent)\s+)?(?:annual\s+report(?:\s+on\s+form\s*10-?k)?|form\s*10-?k|10-?k)"
    r"(?:\s+(?:filing|report))?(?![\w-])", re.I)

_CAP = r"[A-Z][\w&.\-]*"
SUBJECT_RES = [
    # "What risks does Samsung face?"  "How many employees does Unilever have?"
    re.compile(r"\b(?:[Dd]oes|[Dd]o|[Dd]id|[Ii]s|[Ww]as|[Hh]as|[Hh]ave|[Hh]ad|[Ww]ill|[Cc]an|[Ss]hould|[Ww]ould)\s+"
               r"(?P<e>" + _CAP + r"(?:\s+(?:" + _CAP + r"|&|of|and)){0,3})\s+"
               r"(?:face[sd]?|ha(?:ve|s|d)|pa(?:y|ys|id)|report(?:s|ed)?|owns?|leases?|uses?|compete[sd]?|"
               r"disclose[sd]?|describes?|ma(?:ke|kes|de)|sel(?:l|ls)|sold|employs?|manages?|sa(?:y|ys|id)|"
               r"earn(?:s|ed)?|spen(?:d|ds|t)|generate[sd]?|operates?|expects?|plans?|invest(?:s|ed)?|buy|bought|"
               r"acquire[sd]?|owes?|holds?|record(?:s|ed)?|produces?|manufactures?|depends?|rel(?:y|ies)|"
               r"offers?|provides?|issue[sd]?)\b"),
    # "What was Toyota's revenue?"  "Who audits Siemens' financial statements?"
    re.compile(r"(?P<e>\b" + _CAP + r"(?:\s+(?:" + _CAP + r"|&)){0,3})(?:['’]s\b|(?<=s)['’](?=\s|$))"),
    # "Siemens 2024 revenue"
    re.compile(r"^\s*(?P<e>" + _CAP + r"(?:\s+(?:" + _CAP + r"|&)){0,3})\s+(?:FY\s*)?['’]?(?:19|20)\d{2}\b"),
]
STOP_ENTITY = {
    "the", "a", "an", "what", "which", "who", "whom", "whose", "how", "when", "where", "why", "tell", "show", "give",
    "list", "compare", "describe", "explain", "find", "please", "me", "about", "does", "do", "did", "is", "was",
    "in", "of", "for", "and", "or", "to", "on", "at", "by", "i", "we", "our", "you", "your", "they", "their", "it",
    "its", "this", "that", "these", "those", "there", "he", "she", "his", "her", "my", "company", "companies",
    "corporation", "board", "management", "directors", "group", "segment", "fiscal", "fy", "item", "part", "note",
    "form", "sec", "us", "u.s.", "usa", "uk", "eu", "united states", "america", "europe", "china", "india",
    "gaap", "ifrs", "fasb", "irs", "ceo", "cfo", "coo", "cto", "esg", "ai", "eps", "ebitda", "covid", "covid-19",
    "q1", "q2", "q3", "q4", "january", "february", "march", "april", "may", "june", "july", "august", "september",
    "october", "november", "december", "inc", "inc.", "corp", "corp.", "co", "co.", "ltd", "llc", "plc",
}


class Retriever:

    RECENCY_KEYWORDS = re.compile(
        r"\b(most recent|latest|current|this year|last fiscal year|newest|up[- ]to[- ]date)\b", re.IGNORECASE)
    YEAR_PATTERN = re.compile(r"(?<![\d,.$])(?:19|20)\d{2}(?![\d,])")       # "2024", "FY2024"

    # Dense cosine par check (RRF score par nahi). Model badla hai (bge-small), isliye eval mein dobara
    # parakhna: agar sahi sawal bhi "nahi mila" aaye to ise 0.50 ke qareeb kar dena.
    MIN_CONFIDENCE_SCORE = 0.58
    RRF_K = 60
    POOL = 50
    STMT_POOL = 10

    WELL_KNOWN_COMPANIES = [
        "tesla", "microsoft", "jpmorgan", "citigroup", "pfizer", "coca-cola", "costco", "cvs", "disney",
        "amazon", "google", "alphabet", "nvidia", "walmart", "netflix", "intel", "ibm", "oracle", "exxon",
        "starbucks", "mcdonald", "ford", "general motors", "boeing", "apple", "meta platforms", "facebook",
    ]

    def __init__(self, store_dir="STAGE_1/vector_store"):
        d = Path(store_dir)
        if not (d / "index.faiss").exists():
            raise SystemExit(f"Index nahi mila: {d.resolve()}  (pehle build_embeddings.py chalao)")
        print("Loading FAISS index and metadata...")
        cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
        self.index = faiss.read_index(str(d / "index.faiss"))
        L = np.load(d / "light.npz")
        self.names = cfg["companies"]
        self.type_names = cfg["type_names"]
        self.ccode, self.year, self.tcode = L["company"], L["year"], L["type"]
        self.gid, self.offset = L["gid"], L["offset"]
        self.N = len(self.offset)
        if self.index.ntotal != self.N:
            raise SystemExit("Index aur metadata ka size match nahi karta -- build_embeddings.py dobara chalao.")
        self._mf = open(d / "meta.jsonl", "rb")
        print("Loading keyword (BM25) index...")
        self.X = sp.load_npz(d / "bm25.npz").tocsc()
        self.dl = np.load(d / "doclen.npy")
        self.vocab = json.loads((d / "bm25_vocab.json").read_text(encoding="utf-8"))
        self.avgdl = float(self.dl.mean())
        self.df = np.diff(self.X.indptr)
        self.stmt, self.clutter = self._load_flags(d)

        self.embedder = EmbeddingIndexBuilder(cfg["model"])
        self.features = dict(FEATURES)
        self._qcache = {}

        self.known_companies = sorted(self.names)
        self.latest_year_by_company = {}
        for ci, n in enumerate(self.names):
            ys = self.year[self.ccode == ci]
            ys = ys[ys > 0]
            if ys.size:
                self.latest_year_by_company[n] = int(ys.max())
        self.global_latest_year = max(self.latest_year_by_company.values()) if self.latest_year_by_company else None
        self._years_by_company = {n: sorted({int(y) for y in set(self.year[self.ccode == ci].tolist()) if y > 0})
                                  for ci, n in enumerate(self.names)}
        self._build_aliases()

    def _meta(self, i):
        self._mf.seek(int(self.offset[i]))
        return json.loads(self._mf.readline())

    @staticmethod
    def _rules_sig():
        """Flags ke rules ka fingerprint: rule (regex / function) badla to purani flags.npz khud bekaar."""
        parts = [p.pattern for p in (STMT_TITLE, SUBTITLE, NOT_STMT, BAD_SUFFIX, TABLE_TITLE, ITEM_TITLES, EXH_LEAF,
                                     BIG_NUM, PAGE_LINE, ITEM_TOKEN, XBRL_DOC, ROMAN_ITEM, CONS_STMT,
                                     INC_REF, EXHIBIT_REF, FORM_REF)]
        try:
            parts += [inspect.getsource(f) for f in (statement_type, _stmt_code, clutter_reason)]
        except (OSError, TypeError):
            pass
        return zlib.crc32("\x00".join(parts).encode("utf-8"))

    def _load_flags(self, d):
        """Har chunk: statement type (0-4) aur clutter (True/False). Ek dafa ban kar flags.npz mein."""
        mp, fp = d / "meta.jsonl", d / "flags.npz"
        key = f"{FLAGS_VERSION}|{self._rules_sig()}|{mp.stat().st_size}|{self.N}"
        if fp.exists():
            try:
                with np.load(fp) as z:                # 'with' -> Windows par file band ho jati hai
                    if str(z["key"]) == key:
                        return z["stmt"].copy(), z["clutter"].copy()
            except Exception:
                pass
        print("Pehli baar: chunks ke flags bana raha hoon (fehrist? financial statement?) -- sirf ek dafa...")
        t0 = time.time()
        stmt = np.zeros(self.N, dtype=np.int8)
        clut = np.zeros(self.N, dtype=bool)
        last_g, last = None, (0, False)
        with open(mp, "rb") as fh:
            for i, line in enumerate(fh):
                g = int(self.gid[i])
                if g != last_g:                       # ek chunk ke saare tukde (windows) ek jaise
                    m = json.loads(line)
                    last = (statement_type(m), clutter_reason(m) is not None)
                    last_g = g
                stmt[i], clut[i] = last
        np.savez(fp, key=np.array(key), stmt=stmt, clutter=clut)
        print(f"  ho gaya ({time.time() - t0:.0f} sec): statement tukde {int((stmt > 0).sum())}, "
              f"fehrist/exhibit tukde {int(clut.sum())}  -> {fp.name}")
        return stmt, clut

    # ------------------------------------------------------------------
    # company / saal pehchan
    # ------------------------------------------------------------------
    def _build_aliases(self):
        first = {}
        for n in self.known_companies:
            clean = re.sub(r"\(.*?\)", "", n).strip().lower()
            w = re.findall(r"[a-z0-9&]+", clean)
            first.setdefault(w[0] if w else clean, set()).add(n)
        self.aliases = {}
        for n in self.known_companies:
            clean = re.sub(r"\(.*?\)", "", n).strip().lower()
            clean = re.sub(r"[^a-z0-9& ]+", " ", clean).strip()
            self.aliases[clean] = {n}
            w = re.findall(r"[a-z0-9&]+", clean)
            if w and len(w[0]) >= 3 and len(first[w[0]]) == 1 and w[0] not in NO_FIRST_WORD:
                self.aliases[w[0]] = {n}
            m = re.search(r"\((.*?)\)", n)
            if m and len(m.group(1)) >= 2:
                self.aliases[m.group(1).lower()] = {n}
        for a, target in EXTRA_ALIASES.items():
            if target in self.known_companies:
                self.aliases[a] = self.aliases.get(a, set()) | {target}
        for n, dn in DISPLAY_NAMES.items():              # "Kimberly-Clark", "Watts Water Technologies" ...
            if n in self.known_companies:
                dclean = re.sub(r"[^a-z0-9& ]+", " ", re.sub(r"\(.*?\)", "", dn).lower())
                dclean = re.sub(r"\s+", " ", dclean).strip()
                if dclean:
                    self.aliases[dclean] = self.aliases.get(dclean, set()) | {n}
        self._compact = {re.sub(r"[^a-z0-9]", "", a) for a in self.aliases}

    def _detect_companies(self, query):
        q = " " + re.sub(r"[^a-z0-9&]+", " ", query.lower()) + " "
        found = set()
        for a, ns in self.aliases.items():
            if a and re.search(r"\b" + re.escape(a) + r"\b", q):
                if a in AMBIGUOUS and not re.search(r"\b(?:" + re.escape(a.capitalize()) + "|" + re.escape(a.upper()) + r")\b", query):
                    continue                         # "block of shares" -> Block company nahi
                found |= ns
        return found

    def _detect_company(self, query):          # purane naam ke liye (ek company ya None)
        f = self._detect_companies(query)
        return sorted(f)[0] if len(f) == 1 else None

    def _mentions_unknown_company(self, query):
        q = query.lower()
        for name in self.WELL_KNOWN_COMPANIES:
            if re.search(r"\b" + re.escape(name) + r"\b", q):
                if self._detect_companies(name.title()):      # index mein kisi naam se hai (jaise Alphabet = Google)
                    continue
                return name
        return None

    def _is_known_name(self, e):
        if self._detect_companies(e):
            return True
        ce = re.sub(r"[^a-z0-9]", "", e.lower())          # "Service Now" = servicenow, "Walmart Inc" = walmart
        return any(ce == a or (len(a) >= 4 and ce.startswith(a)) for a in self._compact)

    def _unknown_subject(self, query):
        """Sawal ka subject Capital naam ho jo index mein nahi -> wo naam. (Sirf tab jab koi index company na mili.)"""
        for rx in SUBJECT_RES:
            for m in rx.finditer(query):
                words = re.sub(r"(?:['’]s|['’])$", "", m.group("e")).strip(" .,;:").split()
                while words and words[0].lower().strip(".,") in STOP_ENTITY:      # "What", "The" ...
                    words = words[1:]
                while words and words[-1].lower() in {"of", "and", "&", "the"}:
                    words = words[:-1]
                e = " ".join(words)
                if len(e) < 2 or all(w.lower().strip(".,") in STOP_ENTITY for w in words) or self._is_known_name(e):
                    continue
                return e
        return None

    def _search_text(self, query, comps, years):
        """'strip': filter ban chuke company naam, diye gaye saal aur "10-K / annual report" jaise alfaaz nikal do."""
        s = " " + query + " "
        if comps:
            for a in sorted((a for a, ns in self.aliases.items() if ns & comps), key=len, reverse=True):
                pat = (r"(?<![A-Za-z0-9&])" + r"[\W_]+".join(re.escape(w) for w in a.split()) +
                       r"(?:,?\s+(?:inc|corp|corporation|co|company|plc|ltd)\b\.?)?(?:['’]s\b|['’])?(?![A-Za-z0-9&])")
                s = re.sub(pat, " ", s, flags=re.I)
        for y in sorted(years or []):
            s = re.sub(r"(?:\b(?:in|for|during|of)\s+)?(?:\b(?:the\s+)?(?:fiscal(?:\s+year)?|fy)\s*)?['’]?(?<!\d)" + str(y) +
                       r"(?!\d)", " ", s, flags=re.I)
        s = FILING_WORDS.sub(" ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s if re.search(r"[A-Za-z]{3,}", s) else query          # kuch bacha hi nahi -> asal sawal

    # ------------------------------------------------------------------
    # dense / bm25
    # ------------------------------------------------------------------
    def _embed(self, text):
        v = self._qcache.get(text)
        if v is None:
            v = np.ascontiguousarray(self.embedder.embed_query(text), dtype=np.float32)      # (1, dim)
            if len(self._qcache) > 5000:
                self._qcache.clear()
            self._qcache[text] = v
        return v

    def _dense(self, q, k, ids):
        if ids.size == self.N:
            return self.index.search(q, k)
        ids = np.ascontiguousarray(ids, dtype=np.int64)
        params = faiss.SearchParameters()
        params.sel = faiss.IDSelectorBatch(len(ids), faiss.swig_ptr(ids))
        return self.index.search(q, k, params=params)

    def _bm25(self, query, k1=1.5, b=0.75):
        score = np.zeros(self.N, dtype=np.float32)
        for w in set(tok(query)):
            j = self.vocab.get(w)
            if j is None:
                continue
            s, e = self.X.indptr[j], self.X.indptr[j + 1]
            rows, tf = self.X.indices[s:e], self.X.data[s:e]
            idf = np.log(1 + (self.N - self.df[j] + 0.5) / (self.df[j] + 0.5))
            score[rows] += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * self.dl[rows] / self.avgdl))
        return score

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------
    def search(self, query, top_k=5, company=None, year=None, chunk_type=None):
        F = self.features

        unknown = self._mentions_unknown_company(query)
        if unknown:
            return {"matched": False, "reason": f"No indexed data for '{unknown}'.", "results": []}

        comps = self._detect_companies(query)
        if company:
            comps = {n for n in self.known_companies if company.lower() in n.lower()}
        if F["guard"] and not comps:
            unknown = self._unknown_subject(query)
            if unknown:
                return {"matched": False, "reason": f"No indexed data for '{unknown}'.", "results": []}
        years = {int(y) for y in self.YEAR_PATTERN.findall(query)}
        if year:
            years = {int(year)}
        wants_recent = bool(self.RECENCY_KEYWORDS.search(query)) and not years

        needs_hard_filter = bool(comps or years or wants_recent)
        mask = np.ones(self.N, dtype=bool)

        if comps:
            mask &= np.isin(self.ccode, [self.names.index(c) for c in comps])
            if not mask.any():
                return {"matched": False, "reason": f"No data found for company '{sorted(comps)}'.", "results": []}
        if years:
            m2 = mask & np.isin(self.year, list(years))
            if not m2.any():
                avail = {c: self._years_by_company.get(c, []) for c in (comps or [])}
                extra = f" Available years: {avail}" if avail else ""
                return {"matched": False, "reason": f"No data found for year(s) {sorted(years)}.{extra}", "results": []}
            mask = m2
        elif wants_recent:
            latest = np.array([self.latest_year_by_company.get(n, 0) for n in self.names], dtype=np.int16)[self.ccode]
            mask &= (self.year == latest)
        if chunk_type:
            mask &= (self.tcode == self.type_names.index(chunk_type))
        if F["clutter"]:
            m3 = mask & ~self.clutter
            if m3.any():
                mask = m3

        idx = np.where(mask)[0]
        if idx.size == 0:
            return {"matched": False, "reason": "No sufficiently relevant data found for this query.", "results": []}

        item_types = line_item_types(query)            # number wala sawal? (total assets, revenue, D&A ...)
        sq = self._search_text(query, comps, set() if item_types else years) if F["strip"] else query
        q = self._embed(sq)

        # ---- dense (FAISS) : filter ke andar top POOL ----
        D, I = self._dense(q, int(min(self.POOL, idx.size)), idx)
        dense_hits = [(int(i), float(s)) for i, s in zip(I[0], D[0]) if i >= 0]
        rd = {doc: r for r, (doc, _) in enumerate(dense_hits)}
        dscore = dict(dense_hits)
        best_dense = dense_hits[0][1] if dense_hits else 0.0

        # ---- keyword (BM25) : filter ke andar top POOL ----
        bm = self._bm25(sq)[idx]
        rb = {int(idx[i]): r for r, i in enumerate(np.argsort(-bm)[:self.POOL]) if bm[i] > 0}
        lists = [rd, rb]

        # ---- stmt: number wale sawal par asli statement ki teesri list ----
        types = item_types if F["stmt"] else set()
        if types:
            sel = idx[np.isin(self.stmt[idx], list(types))]
            if sel.size:
                D3, I3 = self._dense(q, int(min(self.STMT_POOL, sel.size)), sel)
                lists.append({int(i): r for r, i in enumerate(I3[0]) if i >= 0})

        rrf = {}
        for L in lists:
            for doc, r in L.items():
                rrf[doc] = rrf.get(doc, 0.0) + 1.0 / (self.RRF_K + r + 1)

        results, seen, same_text = [], set(), {}
        for doc in sorted(rrf, key=lambda x: -rrf[x]):
            g = int(self.gid[doc])
            if g in seen:                    # lambe table ke kai tukde -> ek hi baar
                continue
            seen.add(g)
            m = self._meta(doc)
            item = {
                "score": dscore[doc] if doc in dscore else float(self.index.reconstruct(doc) @ q[0]),
                "rrf_score": rrf[doc],
                "company": m["company"],
                "company_display": display_name(m["company"]),
                "year": m["year"],
                "section_path": m["section_path"],
                "page_numbers": m["page_numbers"],
                "chunk_type": m["chunk_type"],
                "text": m["text"],
            }
            if F["dedup"]:                   # hu-ba-hu wahi paragraph doosre saal ka -> ek hi (naye saal wala)
                sig = re.sub(r"\s+", " ", (m["text"] or "")[:250].lower()).strip()
                j = same_text.get(sig)
                if j is not None:
                    if int(m["year"] or 0) > int(results[j]["year"] or 0):
                        results[j] = item
                    continue
                same_text[sig] = len(results)
            results.append(item)
            if len(results) == top_k:
                break

        if not results:
            return {"matched": False, "reason": "No sufficiently relevant data found for this query.", "results": []}
        if not needs_hard_filter:
            if best_dense < self.MIN_CONFIDENCE_SCORE:
                return {"matched": False, "reason": "No sufficiently relevant data found for this query.", "results": []}
        return {"matched": True, "reason": None, "results": results}


TEST_QUESTIONS = {
    "Numbers (table lookup)": [
        "What was Woodward's total assets in 2025?",
        "What was Nordson's net income in 2016?",
        "What was Caterpillar's total sales and revenues in 2025?",
    ],
    "Conceptual / risk": [
        "What supply chain risks does Caterpillar disclose in 2025?",
        "What competitive conditions does Nordson describe in 2020?",
    ],
    "Business description": [
        "What business segments does Woodward operate?",
    ],
    "Comparison (two years / two companies)": [
        "How did Woodward's net sales change between 2022 and 2023?",
        "Compare Woodward and Nordson total assets in 2024",
    ],
    "Recency": [
        "What is Woodward's most recent total net sales?",
    ],
    "Out-of-scope (should NOT match)": [
        "Who is the CEO of Tesla?",
        "What is the weather in Cupertino?",
        "What was Woodward's revenue in 2031?",
        "What risks does Samsung face?",
    ],
}


def _show(question, response):
    print("\n" + "=" * 70)
    print("Q:", question)
    print("=" * 70)
    if not response["matched"]:
        print("    ⚠️  ", response["reason"])
        return
    for i, r in enumerate(response["results"], 1):
        pg = r["page_numbers"]
        pg = (f"{min(pg)}-{max(pg)}" if pg and min(pg) != max(pg) else (str(pg[0]) if pg else "?"))
        print(f"\n[{i}] dense={r['score']:.3f} rrf={r['rrf_score']:.4f} | {r['company_display']} {r['year']} | "
              f"{r['chunk_type']} | PDF page {pg}")
        print("    Section:", r["section_path"][-110:])
        print("    " + r["text"][:300].replace("\n", " / "))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--company")
    ap.add_argument("--year", type=int)
    ap.add_argument("--type", choices=["text", "table"])
    ap.add_argument("--store", default="STAGE_1/vector_store")
    ap.add_argument("--off", default="", help="comma separated sudhaar band karo: guard,clutter,strip,stmt,dedup")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    retriever = Retriever(a.store)
    for f in [x.strip() for x in a.off.split(",") if x.strip()]:
        if f not in retriever.features:
            ap.error(f"'{f}' koi sudhaar nahi. In mein se: {', '.join(retriever.features)}")
        retriever.features[f] = False
    if a.test:
        log = []
        for cat, qs in TEST_QUESTIONS.items():
            print("\n" + "#" * 70 + f"\n# CATEGORY: {cat}\n" + "#" * 70)
            for qn in qs:
                resp = retriever.search(qn, top_k=3)
                _show(qn, resp)
                log.append({"category": cat, "question": qn, **resp})
        with open("retriever_test_results.json", "w", encoding="utf-8") as f:
            json.dump(log, f, indent=4, ensure_ascii=False)
        print("\n\nFull results also saved to: retriever_test_results.json")
    elif a.query:
        _show(a.query, retriever.search(a.query, top_k=a.k, company=a.company, year=a.year, chunk_type=a.type))
    else:
        ap.error("query do, ya --test lagao")