"""
retriever.py  -  Tumhari wahi Retriever class (wahi search() aur wahi jawab ka format), naye index ke saath.

Chalane ka tareeqa:
    python STAGE_1/retriever.py "Woodward 2025 total assets"
    python STAGE_1/retriever.py "net income" --company Nordson --year 2016
    python STAGE_1/retriever.py "Rockwell Automation 2016 goodwill" --off strip      (ek sudhaar band kar ke dekho)
    python STAGE_1/retriever.py --test          (neeche wale test sawal; VS Code ka ▷ Run bhi yahi chalata hai.
                                                 Sahi jawab STAGE_1/test_questions.csv se; nateeja retriever_test_results.json)

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
v2.2 (golden set ke 56 asli-user jaise sawalon ke 23 fail dekh kar; zor paragraphs par):
  recency  "currently", "now", "today", "nowadays" bhi = sab se naya saal (pehle sirf "current/latest").
  fresh    Sawal mein saal na ho to sab se naye 10-K ke behtareen chunks ko ek extra vote (jaise stmt), kyun ke
           "How many employees does Dell have?" mein user aaj ka number chahta hai. Purane saal bhi aate hain.
  synonyms User ke alfaaz -> PDF ke alfaaz, sirf keyword (BM25) search mein: headcount -> employees,
           bottom line -> net income, audit firm -> auditor / Ernst & Young / Deloitte ...
  stem     Keyword search lafz ke roop bhi pakadta hai: employ = employed = employees = employment.
  names    Company ke naam mein spelling ki ghalti (Wallmart, Micorsoft) bhi pehchanta hai, aur poora
           naam ("Dell Technologies") search ke alfaaz se nikalta hai.
  oos2     Bahar ki mashhoor companies chhote huroof mein bhi pakadta hai (samsung, uber), aur company na
           mile to "kitna milta julta" (confidence) check saal wale sawal par bhi lagta hai (FIFA 2022).
           Ab ye check tab hi lagta hai jab index ki koi company sawal mein na ho ("Apple vs Samsung" theek).
  diverse  Do companies ya do saal ka sawal: dono ke chunks baari baari (pehle sirf ek hi company aati thi).
  expand   Har result ke saath usi section ka pichla aur agla chunk bhi (jawab aksar chunk ki had par toot'ta
           hai, Gemini ko poori baat milni chahiye).
v2.3 (22 Oct 2026) -- 100 sawal wale pehle test (pehla score 80/100) ki ghaltiyan dekh kar:
  intent   "What does X make / sell", "growth plan / strategy", "auditor kaun tha" jaise sawal: Item 1 Business ka
           overview, MD&A ka overview, ya auditor ke signature ("/s/ ... LLP") wala chunk ek extra vote leta hai.
  typo     Sawal ka jo lafz data mein kabhi nahi aata aur kisi aam lafz se bohat milta hai, wo theek: proft -> profit.
  roman    Roman Urdu ke aam alfaaz ka English matlab bhi search mein: banati -> make, bechti -> sell, kab bani -> founded.
  tickers  Chhote / aam-lafz tickers sirf BARE huroof mein (PH, NOW, HD, COST, CRM ...), aur "nike's" / "apple's"
           (possessive) chhote huroof mein bhi company.
  words    Aam user ke kuch aur alfaaz -> 10-K ke alfaaz (sirf keyword search): layoffs -> restructuring workforce
           reduction, profit -> net income, payout -> dividends, climate / carbon -> sustainability ...
Har sudhaar FEATURES mein on/off; eval_retriever.py --ablation aur eval_golden.py --ablation har ek ka asar
alag naapte hain.

Index dobara banane ki zaroorat NAHI. Pehli baar chalne par chunks ke flags (fehrist? statement?) ek dafa
bante hain aur vector_store/flags.npz mein save ho jate hain (agli baar seedha load).
"""
import argparse
import csv
import difflib
import inspect
import json
import re
import sys
import time
import zlib
from pathlib import Path

import faiss
import numpy as np
import scipy.sparse as sp

from build_embeddings import EmbeddingIndexBuilder, tok, section_str, INC_REF, EXHIBIT_REF, FORM_REF

# Har sudhaar on/off. eval_retriever.py --ablation / eval_golden.py --ablation inhein badal badal kar naapte hain.
OLD_FEATURES = ["guard", "clutter", "strip", "stmt", "dedup"]                          # v2 / v2.1
NEW_FEATURES = ["recency", "fresh", "synonyms", "stem", "names", "oos2", "diverse", "expand"]   # v2.2
V23_FEATURES = ["intent", "typo", "roman", "tickers", "words"]                                        # v2.3
FEATURES = {f: True for f in OLD_FEATURES + NEW_FEATURES + V23_FEATURES}


def ablation_variants():
    """v2.2 (v2.3 ke sudhaar band) -> har v2.3 sudhaar akela -> sab on. Dono eval files yahi list use karti hain."""
    base = {f: f not in V23_FEATURES for f in FEATURES}
    return ([("v2.2 (naye off)", base)] + [(f"+ {f}", dict(base, **{f: True})) for f in V23_FEATURES]
            + [("v2.3 (sab on)", {f: True for f in FEATURES})])


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
FLAGS_VERSION = 2          # (neeche ke rules badlein to flags.npz khud dobara banti hai -- _rules_sig dekho)

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


# ---------------------------------------------------------------------------------------------
# v2.2 "oos2": mashhoor companies jo index mein NAHI (chhote huroof mein bhi pakadni hain).
# Sirf wahi naam jo aam English / Roman Urdu lafz nahi ("kia", "shell", "target", "gap", "visa" jaan boojh kar nahi).
# Agar inmein se koi kabhi index mein aa jaye to khud chhoot jata hai (_mentions_unknown_company check karta hai).
# ---------------------------------------------------------------------------------------------
MORE_UNKNOWN_COMPANIES = [
    "samsung", "toyota", "honda", "nissan", "hyundai", "volkswagen", "bmw", "mercedes-benz", "mercedes", "stellantis",
    "ferrari", "porsche", "sony", "panasonic", "nintendo", "xiaomi", "huawei", "lenovo", "alibaba", "tencent", "baidu",
    "bytedance", "tiktok", "nestle", "nestlé", "unilever", "procter & gamble", "procter and gamble", "pepsico", "pepsi",
    "kraft heinz", "mondelez", "novartis", "roche", "astrazeneca", "sanofi", "glaxosmithkline", "bayer", "merck",
    "eli lilly", "abbvie", "amgen", "gilead", "moderna", "biontech", "siemens", "philips", "honeywell", "raytheon",
    "general dynamics", "l3harris", "airbus", "embraer", "bombardier", "lufthansa", "uber", "lyft", "airbnb",
    "doordash", "expedia", "spotify", "shopify", "snapchat", "pinterest", "reddit", "twitter", "mastercard",
    "american express", "goldman sachs", "morgan stanley", "citibank", "bank of america", "wells fargo", "hsbc",
    "barclays", "berkshire hathaway", "blackrock", "exxonmobil", "conocophillips", "aramco", "totalenergies",
    "petrobras", "kroger", "ikea", "adidas", "lululemon", "under armour", "inditex", "lvmh", "gucci", "infosys",
    "wipro", "accenture", "cisco", "qualcomm", "broadcom", "texas instruments", "micron", "asml", "tsmc",
    "hewlett packard", "verizon", "at&t", "t-mobile", "comcast", "warner bros", "fedex", "maersk",
    "john deere", "lowe's", "lowes", "best buy", "walgreens", "unitedhealth", "johnson & johnson", "mcdonald's",
    "burger king", "yum brands", "dominos", "domino's", "zara", "h&m",
]

# v2.2 "synonyms": user ke alfaaz -> 10-K ke alfaaz. Sirf keyword (BM25) query mein jodte hain, dense waisa hi rehta hai.
SYNONYMS = [
    (r"\bhead ?counts?\b|\bworkforce\b|\bstaff(?:ing)?\b|\bpersonnel\b|\bhow many (?:people|workers)\b|"
     r"\bemploy(?:s|ed|ees?)?\b", "employees employed workforce human capital"),
    (r"\bbottom line\b|\bnet profits?\b|\bprofit after tax\b", "net income net earnings"),
    (r"\btop line\b", "revenue net sales"),
    (r"\bcapex\b", "capital expenditures"),
    (r"\br&d\b", "research and development"),
    (r"\bsg&a\b", "selling general administrative"),
    (r"\beps\b", "earnings per share"),
    (r"\bfcf\b", "free cash flow"),
    (r"\bheadquarter(?:s|ed)?\b|\bhq\b|\bhead office\b|\bmain office\b|\bbased in\b|\bwhere is\b.{0,40}\blocated\b",
     "headquarters principal executive offices located"),
    (r"\baudits\b|\bauditors?\b|\baudit firm\b|\baccounting firm\b|\bwho audit",
     "auditor independent registered public accounting firm ernst young deloitte pricewaterhousecoopers kpmg"),
    (r"\border book\b|\bunfilled orders\b|\borders on hand\b", "backlog"),
    (r"\b(?:biggest|largest|major|top|main|key) (?:customers?|clients?|buyers?)\b|\bcustomer concentration\b",
     "largest customer single customer accounted percent net sales"),
    (r"\bbuy ?backs?\b|\brepurchas\w*", "share repurchase program repurchased shares"),
    (r"\blawsuits?\b|\bsued\b|\blitigation\b|\blegal (?:cases?|issues?|matters?)\b", "legal proceedings litigation"),
    (r"\bcyber\w*|\bhack\w*|\bdata breach\w*|\bciso\b|\bsecurity officer\b",
     "cybersecurity chief information security officer"),
    (r"\bcompetitors?\b|\brivals?\b|\bcompet\w*", "competition competitors compete"),
    (r"\bdivisions?\b|\bbusiness (?:units?|lines?)\b|\blines of business\b", "reportable segments"),
    (r"\bfactories\b|\bplants?\b|\bmanufacturing (?:sites?|locations?)\b", "manufacturing facilities plants properties"),
    (r"\bsuppliers?\b|\bsupply chain\b|\braw materials?\b", "suppliers raw materials supply chain sources"),
    (r"\bfounded\b|\bestablished\b|\bhistory\b", "incorporated founded"),
    (r"\bceo\b|\bchief executive\b", "chief executive officer"),
    (r"\bacquisitions?\b|\bacquir\w*|\bbought\b|\btakeover\b", "acquisition acquired business combination"),
    (r"\bstores?\b|\boutlets?\b|\bwarehouses?\b|\bbranches\b", "stores locations operated"),
    (r"\bcountries\b|\bglobal (?:presence|footprint)\b|\binternational presence\b", "countries"),
]
SYNONYMS = [(re.compile(p, re.I), add) for p, add in SYNONYMS]


def synonym_terms(query):
    """Sawal ke alfaaz se milte 10-K ke alfaaz (khaali string agar kuch na mile)."""
    return " ".join(add for rx, add in SYNONYMS if rx.search(query))


# v2.3 "roman": Roman Urdu ke aam alfaaz -> 10-K ke English alfaaz (dense aur keyword dono query mein jurte hain)
ROMAN = [
    (r"\bbana(?:ti|ta|te)\b|\bbnati\b", "make manufacture products"),
    (r"\bbech(?:ti|ta|te|na)\b|\bfarokht\b", "sell sales customers"),
    (r"\bkab (?:bani|bana|qaim|shuru)\b|\bbuniyad\b", "founded incorporated"),
    (r"\bnuqsan\b|\bnuksan\b|\bkhasar[ae]\b", "loss adversely affect harm"),
    (r"\bmunaf[ae]h?\b|\bfaid[ae]\b", "profit net income"),
    (r"\ba+mdani\b|\bkamai\b", "revenue income"),
    (r"\bkharch[ae]?\b", "spending expense"),
    (r"\bmulazi?m\w*|\bmulazmeen\b", "employees"),
    (r"\bdaft[ae]r\b", "office headquarters"),
    (r"\bhiss[ae]y?\b|\bshob[ae]y?\b", "segments"),
    (r"\bmu[qk]abl[ae]\w*", "competition competitors"),
    (r"\bkhatr[ae]\w*", "risk"),
    (r"\bqarz[ae]?\b", "debt borrowings"),
    (r"\bgaa?hak\b|\bkhareedar\w*", "customers"),
]
ROMAN = [(re.compile(p, re.I), add) for p, add in ROMAN]
# "typo" in lafzon ko English ki ghalti samajh kar nahi badalta
ROMAN_WORD = re.compile(r"^(?:kitn[aie]|kahan|kaise|kyun|banati|banata|banate|bechti|bechta|bechte|karti|karta|karte|"
                        r"karein|karen|nuqsan|nuksan|munafa|munafe|aamdani|amdani|kharch\w*|mulazim\w*|mulazmeen|"
                        r"daftar|hisse|shobay|muqabla|mukabla|khatra|khatre|qarza|thein|thiin|wapas|zyada|kitne)$", re.I)


def roman_terms(query):
    return " ".join(add for rx, add in ROMAN if rx.search(query))


# v2.3 "words": synonyms jaisa (sirf keyword query), aam user ke mazeed alfaaz
WORDS = [
    (r"\blay ?offs?\b|\blaid off\b|\bjob cuts?\b|\bcut(?:ting)? jobs\b|\bdownsiz\w*|\bfir(?:e|ed|ing) (?:staff|employees|workers)\b",
     "restructuring reduction workforce severance"),
    (r"(?<!gross )(?<!operating )(?<!segment )\bprofits?\b(?! (?:sharing|margin))|\bearn(?:ed|s)?\b(?! per)",
     "net income earnings"),
    (r"\blos(?:e|t|ing) money\b|\bunprofitable\b", "net loss"),
    (r"\bpay ?outs?\b|\bpaid (?:out )?to (?:shareholders|investors)\b", "dividends paid"),
    (r"\bgreen\b|\bclimate\b|\bcarbon\b|\bnet[- ]zero\b|\beco[- ]friendly\b", "sustainability greenhouse gas emissions climate"),
    (r"\bboss\b|\bwho runs\b|\bin charge of\b", "chief executive officer"),
]
WORDS = [(re.compile(p, re.I), add) for p, add in WORDS]


def words_terms(query):
    return " ".join(add for rx, add in WORDS if rx.search(query))


# v2.3 "intent": sawal kis hisse ka hai -> wahan ke chunks ko ek extra vote (stmt jaisa). Bits: 1 = Item 1 Business
# ka overview / products / strategy, 2 = MD&A ka overview, 4 = auditor ki report ka signature ("/s/ ... LLP").
INTENT_BUSINESS = re.compile(
    r"\bwhat (?:does|do|did)\b[^?]{0,50}\b(?:make|makes|sell|sells|produce|produces|offer|offers|manufactur\w*)\b|"
    r"\bwhat (?:kinds?|types?|sorts?) of (?:products?|business(?:es)?|services?|company)\b|\bwhat (?:products?|services?)\b|"
    r"\bwhat (?:is|are)\b[^?]{0,40}\b(?:main |core )?(?:business|products?|services?)\b|\bmake (?:most of )?(?:its |their )?"
    r"(?:money|revenue)\b|\bsources? of revenue\b|\brevenue (?:sources?|streams?)\b|\bbusiness model\b|"
    r"\bdescribe\b[^?]{0,40}\bbusiness\b|\b(?:company|business) overview\b|\bbana(?:ti|ta|te)\b|\bbech(?:ti|ta|te)\b|"
    r"\bkaam kar(?:ti|ta|te)\b|\bkarobar\b", re.I)
INTENT_STRATEGY = re.compile(
    r"\bgrow(?:th|ing)?\b|\bexpan(?:d|sion)\b|\bstrateg\w*|\b(?:business|growth|expansion) plans?\b|\bplans? to\b|"
    r"\bhow did\b[^?]{0,40}\b(?:do|perform|go)\b|\b(?:overall|financial) performance\b|\bhighlights?\b|\boutlook\b", re.I)
INTENT_AUDITOR = re.compile(r"\baudit(?:or|ors|ed|s)?\b|\baccounting firm\b", re.I)


def intent_bits(query):
    b = 0
    if INTENT_BUSINESS.search(query):
        b |= 1
    if INTENT_STRATEGY.search(query):
        b |= 1 | 2
    if INTENT_AUDITOR.search(query):
        b |= 4
    return b


BIZ_ITEM = re.compile(r"\bitems?\s*1(?![0-9a-z]).{0,25}\bbusiness\b", re.I)
BUSINESS_HEAD = re.compile(r"^\s*(?:business|general|overview|our business|business overview|company overview|the company|"
                           r"our company|description of (?:the )?business|products(?: and services)?|our products|"
                           r"strategy|our strategy|business strategy|who we are)\b", re.I)
MDA = re.compile(r"management['’]?s\s+discussion|results of operations", re.I)
MDA_OVERVIEW = re.compile(r"\b(?:overview|executive (?:summary|overview)|highlights|business outlook|outlook|strategy|"
                          r"our business|summary of results|key (?:developments|events|trends))\b", re.I)
NOT_OVERVIEW_ANY = re.compile(r"risk factors|item\s*1a\b|\bnotes? to (?:the )?(?:consolidated )?financial|accounting polic|"
                              r"report of independent|controls and procedures|\bexhibits?\b|\bsignatures?\b|forward[- ]looking|"
                              r"financial statements and supplementary", re.I)
NOT_OVERVIEW_EL = re.compile(r"^\s*(?:item\s*\w{1,3}\s*[\.:\-–—]?\s*)?(?:risk factors|legal proceedings|properties|"
                             r"(?:information about )?(?:our )?executive officers|available information|competition|"
                             r"human capital|employees|seasonality|backlog|raw materials|intellectual property|patents|"
                             r"trademarks|environmental|government regulation|regulation|cybersecurity|special note|"
                             r"unresolved staff comments|mine safety)", re.I)
AUDIT_SIG = re.compile(r"/s/\s*[A-Z][A-Za-z&.,'’\- ]{1,60}?\bL\.?L\.?P\b")


def section_bits(m):
    """v2.3 "intent" ke liye chunk ki qism (flags.npz mein ek dafa banti hai)."""
    parts = [p.strip() for p in section_str(m).split(" > ") if p.strip()]
    path = " > ".join(parts)
    pages = m.get("page_numbers") or []
    bits = 0
    if not NOT_OVERVIEW_ANY.search(path) and not any(NOT_OVERVIEW_EL.search(p) for p in parts):
        if any(BIZ_ITEM.search(p) for p in parts):
            bits |= 1
        elif pages and min(pages) <= 15 and not MDA.search(path) and any(BUSINESS_HEAD.search(p) for p in parts):
            bits |= 1
        if MDA.search(path) and parts and MDA_OVERVIEW.search(parts[-1]):
            bits |= 2
    if m.get("chunk_type") != "table" and AUDIT_SIG.search(m.get("text") or ""):
        bits |= 4
    return bits


# v2.3 "tickers": ye tickers aam lafz bhi hain -> sirf BARE huroof mein company (PH, NOW, HD ...). ticker -> naam
TICKER_NAMES = {"ph": "parker hannifin", "now": "servicenow", "crm": "salesforce", "hd": "home depot", "cost": "costco",
                "cl": "colgate palmolive", "ba": "boeing", "ame": "ametek", "sq": "block", "xyz": "block"}


# v2.2 "stem": lafz ki jarh (employees -> employ). Sirf keyword search ke liye, query ki taraf; index wahi rehta hai.
_SUFFIXES = ("ations", "ation", "ments", "ment", "ings", "ing", "ers", "ees", "ed", "es", "er", "ee", "s", "ly")


def stem(w):
    """Chhota sa stemmer: employs/employed/employees/employment -> employ, countries -> country, acquired -> acquir."""
    if len(w) < 5 or not w.isalpha():
        return w
    for _ in range(2):
        if len(w) > 5 and w.endswith(("ies", "ied")):
            w = w[:-3] + "y"
            continue
        for s in _SUFFIXES:
            if w.endswith(s) and len(w) - len(s) >= 5:
                if s == "es" and not re.search(r"(?:s|x|z|ch|sh)es$", w):
                    continue                                   # stores -> store (stor nahi), taxes -> tax
                if s == "s" and w.endswith("ss"):
                    continue                                   # business, loss, process -> waise hi
                w = w[:-len(s)]
                break
        else:
            break
    if len(w) > 5 and w.endswith("e"):
        w = w[:-1]                                             # acquire -> acquir (acquired ke barabar)
    return w


# v2.2 "names": "Dell Technologies", "Lockheed Martin Corporation" -> search alfaaz se poora naam niklega
LEGAL_SUFFIX = (r"(?:,?\s+(?:inc|corp|corporation|co|company|plc|ltd|technologies|technology|holdings|group|"
                r"incorporated|international|industries)\b\.?)?")
OLD_LEGAL_SUFFIX = r"(?:,?\s+(?:inc|corp|corporation|co|company|plc|ltd)\b\.?)?"


class Retriever:

    RECENCY_KEYWORDS = re.compile(
        r"\b(most recent|latest|current|this year|last fiscal year|newest|up[- ]to[- ]date)\b", re.IGNORECASE)
    RECENCY_V2 = re.compile(                                                    # v2.2 "recency"
        r"\b(most recent|latest|current(?:ly)?|this year|last fiscal year|newest|up[- ]to[- ]date|now|today|"
        r"nowadays|these days|at present|presently|recent(?:ly)?)\b", re.IGNORECASE)
    YEAR_PATTERN = re.compile(r"(?<![\d,.$])(?:19|20)\d{2}(?![\d,])")       # "2024", "FY2024"
    FRESH_POOL = 3          # "fresh": sab se naye 10-K ke sirf itne behtareen chunks ko extra vote
    EXPAND_MAX = 3200       # "expand": pichla + asal + agla chunk, zyada se zyada itne huroof
    FUZZY_CUTOFF = 0.85     # "names": spelling ki ghalti kitni milti julti ho (1.0 = bilkul wahi)
    FUZZY_MAX_DF = 30       # is se zyada chunks mein aane wala lafz asli English lafz hai, naam ki ghalti nahi

    # Dense cosine par check (RRF score par nahi). Model badla hai (bge-small), isliye eval mein dobara
    # parakhna: agar sahi sawal bhi "nahi mila" aaye to ise 0.50 ke qareeb kar dena.
    MIN_CONFIDENCE_SCORE = 0.58
    RRF_K = 60
    POOL = 50
    STMT_POOL = 10
    INTENT_POOL = 5         # "intent": sahi hisse ke itne behtareen chunks ko extra vote
    TYPO_CUTOFF = 0.85      # "typo": kitna milta julta ho (difflib ratio)
    TYPO_MIN_DF = 20        # "typo": sirf un alfaaz se badlo jo kam az kam itne chunks mein hain

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
        self.stmt, self.clutter, self.sect = self._load_flags(d)

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
        self._latest_arr = np.array([self.latest_year_by_company.get(n, 0) for n in self.names], dtype=np.int16)
        self._stem_ids = None                    # "stem": pehli zaroorat par banta hai (stem -> vocab ids)
        self._typo_idx = None                    # "typo": pehli zaroorat par (pehla harf -> aam alfaaz)
        self._build_aliases()

    def _meta(self, i):
        self._mf.seek(int(self.offset[i]))
        return json.loads(self._mf.readline())

    def _chunk_meta(self, g):
        """Chunk number g (gid) ka meta, ya None."""
        if g < 0:
            return None
        i = int(np.searchsorted(self.gid, g))
        if i >= self.N or int(self.gid[i]) != g:
            return None
        return self._meta(i)

    @staticmethod
    def _join(a, b):
        """Do chunk jodo; agla chunk pichle ke aakhri hisse se shuru hota hai (overlap) to wo dobara nahi."""
        probe = b[:60]
        if len(probe) >= 20:
            p = a.rfind(probe, max(0, len(a) - 1500))
            if p >= 0 and a[p:] == b[:len(a) - p]:
                return a + b[len(a) - p:]
        return a + "\n" + b

    def _expanded_text(self, doc, m, covered=()):
        """"expand": usi file + usi section ka pichla aur agla chunk saath (zyada se zyada EXPAND_MAX huroof).
        Jo padosi chunk pehle hi kisi upar wale result mein aa chuka (covered) wo dobara nahi.
        Wapas: (text, istemal hue chunk numbers, saare page numbers)."""
        g, text = int(self.gid[doc]), m["text"] or ""
        used, pages = {g}, list(m.get("page_numbers") or [])
        prev, nxt = self._chunk_meta(g - 1), self._chunk_meta(g + 1)
        same = lambda x: x and x.get("file") == m.get("file") and x.get("section_path") == m.get("section_path")
        room = max(0, self.EXPAND_MAX - len(text)) // 2
        if room < 200:
            return text, used, pages
        if same(prev) and (g - 1) not in covered:
            joined = self._join(prev["text"] or "", text)
            head = joined[:len(joined) - len(text)]
            text = (("... " + head[-room:]) if len(head) > room else head) + text
            used.add(g - 1)
            pages += prev.get("page_numbers") or []
        if same(nxt) and (g + 1) not in covered:
            joined = self._join(m["text"] or "", nxt["text"] or "")
            tail = joined[len(m["text"] or ""):]
            text = text + ((tail[:room] + " ...") if len(tail) > room else tail)
            used.add(g + 1)
            pages += nxt.get("page_numbers") or []
        return text, used, sorted(set(pages))
        if same(prev):
            joined = self._join(prev["text"] or "", text)
            head = joined[:len(joined) - len(text)]
            text = (("... " + head[-room:]) if len(head) > room else head) + text
        if same(nxt):
            joined = self._join(m["text"] or "", nxt["text"] or "")
            tail = joined[len(m["text"] or ""):]
            text = text + ((tail[:room] + " ...") if len(tail) > room else tail)
        return text

    @staticmethod
    def _rules_sig():
        """Flags ke rules ka fingerprint: rule (regex / function) badla to purani flags.npz khud bekaar."""
        parts = [p.pattern for p in (STMT_TITLE, SUBTITLE, NOT_STMT, BAD_SUFFIX, TABLE_TITLE, ITEM_TITLES, EXH_LEAF,
                                     BIG_NUM, PAGE_LINE, ITEM_TOKEN, XBRL_DOC, ROMAN_ITEM, CONS_STMT,
                                     INC_REF, EXHIBIT_REF, FORM_REF, BIZ_ITEM, BUSINESS_HEAD, MDA, MDA_OVERVIEW,
                                     NOT_OVERVIEW_ANY, NOT_OVERVIEW_EL, AUDIT_SIG)]
        try:
            parts += [inspect.getsource(f) for f in (statement_type, _stmt_code, clutter_reason, section_bits)]
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
                    if str(z["key"]) == key and "sect" in z.files:
                        return z["stmt"].copy(), z["clutter"].copy(), z["sect"].copy()
            except Exception:
                pass
        print("Pehli baar: chunks ke flags bana raha hoon (fehrist? financial statement?) -- sirf ek dafa...")
        t0 = time.time()
        stmt = np.zeros(self.N, dtype=np.int8)
        clut = np.zeros(self.N, dtype=bool)
        sect = np.zeros(self.N, dtype=np.int8)
        last_g, last = None, (0, False, 0)
        with open(mp, "rb") as fh:
            for i, line in enumerate(fh):
                g = int(self.gid[i])
                if g != last_g:                       # ek chunk ke saare tukde (windows) ek jaise
                    m = json.loads(line)
                    last = (statement_type(m), clutter_reason(m) is not None, section_bits(m))
                    last_g = g
                stmt[i], clut[i], sect[i] = last
        np.savez(fp, key=np.array(key), stmt=stmt, clutter=clut, sect=sect)
        print(f"  ho gaya ({time.time() - t0:.0f} sec): statement tukde {int((stmt > 0).sum())}, "
              f"fehrist/exhibit tukde {int(clut.sum())}, overview tukde {int(((sect & 3) > 0).sum())}, "
              f"auditor signature {int(((sect & 4) > 0).sum())}  -> {fp.name}")
        return stmt, clut, sect

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
        # "names": spelling ki ghalti se milane ke liye sirf 5+ huroof wale naam (dell, nike jaise chhote nahi)
        self._fuzzy_keys = sorted(a for a in self.aliases if len(a) >= 5 and not re.search(r"\d", a))
        # v2.3 "tickers": aam-lafz tickers sirf BARE huroof mein (fuzzy / compact list mein nahi)
        self._upper_only = set()
        for t, nm in TICKER_NAMES.items():
            ns = self.aliases.get(nm)
            if ns and t not in self.aliases:
                self.aliases[t] = set(ns)
                self._upper_only.add(t)

    def _detect_companies(self, query):
        q = " " + re.sub(r"[^a-z0-9&]+", " ", query.lower()) + " "
        tick = self.features.get("tickers", True)
        found = set()
        for a, ns in self.aliases.items():
            if a and re.search(r"\b" + re.escape(a) + r"\b", q):
                if a in getattr(self, "_upper_only", ()):           # "PH" haan, "ph" / "Ph" nahi
                    if not tick or not re.search(r"(?<![A-Za-z0-9])" + re.escape(a.upper()) + r"(?![A-Za-z0-9])", query):
                        continue
                elif a in AMBIGUOUS and not re.search(r"\b(?:" + re.escape(a.capitalize()) + "|" + re.escape(a.upper()) + r")\b", query):
                    if not (tick and re.search(r"\b" + re.escape(a) + r"['’]s\b", query, re.I)):
                        continue                     # "block of shares" -> Block company nahi;  "nike's" -> Nike
                found |= ns
        return found

    def _detect_company(self, query):          # purane naam ke liye (ek company ya None)
        f = self._detect_companies(query)
        return sorted(f)[0] if len(f) == 1 else None

    def _mentions_unknown_company(self, query, extended=False):
        q = query.lower()
        if not extended:                                          # v2.1 jaisa
            for name in self.WELL_KNOWN_COMPANIES:
                if re.search(r"\b" + re.escape(name) + r"\b", q):
                    if self._detect_companies(name.title()):      # index mein kisi naam se hai (Alphabet = Google)
                        continue
                    return name
            return None
        for name in self.WELL_KNOWN_COMPANIES + MORE_UNKNOWN_COMPANIES:
            if re.search(r"(?<![\w&])" + re.escape(name) + r"(?![\w&])", q):
                if self._detect_companies(name.title()) or self._detect_companies(name):   # index mein hai
                    continue
                return name
        return None

    def _fuzzy_companies(self, query):
        """"names": spelling ki ghalti wala company naam -> index ki company.  ("Wallmart's" -> Walmart)
        Sirf wo lafz jo data mein (almost) kabhi nahi aate -- asli English lafz kabhi company nahi banta."""
        found, hits = set(), []
        words = re.findall(r"[A-Za-z][A-Za-z&'’.\-]+", query)
        cands = [(w, w) for w in words]
        cands += [(f"{a} {b}", f"{a} {b}") for a, b in zip(words, words[1:])]      # do lafz wale naam
        for raw, w in cands:
            c = re.sub(r"(?:['’]s|['’])$", "", w).lower().strip(".-")
            if len(c) < 5 or c in STOP_ENTITY or c in self.aliases:
                continue
            if " " not in c:
                j = self.vocab.get(c)
                df = int(self.df[j]) if j is not None else 0
                if df > self.FUZZY_MAX_DF:
                    continue
            else:
                df = 0
            best, score = None, 0.0
            for key in self._fuzzy_keys:
                if (" " in key) != (" " in c) or abs(len(key) - len(c)) > 3:
                    continue
                s = difflib.SequenceMatcher(None, c, key).ratio()
                if s > score:
                    best, score = key, s
            if best and (score >= self.FUZZY_CUTOFF or
                         (score >= 0.80 and df == 0 and c[0] == best[0] and abs(len(c) - len(best)) <= 1)):
                found |= self.aliases[best]
                hits.append(re.sub(r"(?:['’]s|['’])$", "", raw))
        return found, hits

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

    def _fix_typos(self, query, skip=()):
        """"typo": 5+ huroof ka jo lafz data mein kabhi nahi aata, aur kisi aam lafz (TYPO_MIN_DF+ chunks) se bohat
        milta hai, us ki jagah wo lafz: proft -> profit, reserch -> research. Company naam, BARE huroof wale lafz
        (tickers) aur Roman Urdu nahi chhedte. Wapas: (naya sawal, {purana: naya})."""
        if self._typo_idx is None:
            ti = {}
            for v, j in self.vocab.items():
                if len(v) >= 4 and v.isalpha() and self.df[j] >= self.TYPO_MIN_DF:
                    ti.setdefault(v[0], []).append(v)
            self._typo_idx = ti
        skip = {w.lower() for s in skip for w in re.findall(r"[A-Za-z]+", s)}
        fixes = {}
        for w in set(re.findall(r"[A-Za-z]{5,}", query)):
            lw = w.lower()
            if w.isupper() or lw in self.vocab or lw in self.aliases or lw in skip or ROMAN_WORD.match(lw):
                continue
            cands = [v for v in self._typo_idx.get(lw[0], ()) if abs(len(v) - len(lw)) <= 2]
            cut = self.TYPO_CUTOFF if len(lw) > 5 else 0.9           # chhota lafz: zyada yaqeen chahiye (thier != tier)
            best = difflib.get_close_matches(lw, cands, n=3, cutoff=cut)
            if best:
                sc = [difflib.SequenceMatcher(None, lw, b).ratio() for b in best]
                keep = [b for b, x in zip(best, sc) if x >= sc[0] - 0.05][:2]   # barabar ke do: distributes distributors
                fixes[w] = " ".join(keep)
        if not fixes:
            return query, {}
        return re.sub(r"[A-Za-z]{5,}", lambda mm: fixes.get(mm.group(0), mm.group(0)), query), fixes

    def _search_text(self, query, comps, years, extra=(), legal=OLD_LEGAL_SUFFIX):
        """'strip': filter ban chuke company naam, diye gaye saal aur "10-K / annual report" jaise alfaaz nikal do.
        extra = spelling ki ghalti wale naam ("Wallmart") jo 'names' ne pehchane -- wo bhi nikalte hain."""
        s = " " + query + " "
        for raw in extra:
            s = re.sub(r"(?<![A-Za-z0-9&])" + re.escape(raw) + r"(?:['’]s\b|['’])?(?![A-Za-z0-9&])", " ", s)
        if comps:
            for a in sorted((a for a, ns in self.aliases.items() if ns & comps), key=len, reverse=True):
                pat = (r"(?<![A-Za-z0-9&])" + r"[\W_]+".join(re.escape(w) for w in a.split()) +
                       legal + r"(?:['’]s\b|['’])?(?![A-Za-z0-9&])")
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

    def _term(self, j, k1, b):
        """Ek vocab lafz ka BM25 hissa: (kin chunks mein, kitna score)."""
        s, e = self.X.indptr[j], self.X.indptr[j + 1]
        rows, tf = self.X.indices[s:e], self.X.data[s:e]
        idf = np.log(1 + (self.N - self.df[j] + 0.5) / (self.df[j] + 0.5))
        return rows, idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * self.dl[rows] / self.avgdl))

    def _variants(self, w):
        """"stem": vocab mein is lafz ke saare roop (employ -> employ, employed, employees, employment ...)."""
        if self._stem_ids is None:
            m = {}
            for v, j in self.vocab.items():
                if len(v) >= 5 and v.isalpha():
                    m.setdefault(stem(v), []).append(j)
            self._stem_ids = m
        ids = list(self._stem_ids.get(stem(w), []))
        j = self.vocab.get(w)
        if j is not None and j not in ids:
            ids.append(j)
        return ids

    def _bm25(self, query, k1=1.5, b=0.75, use_stem=False):
        score = np.zeros(self.N, dtype=np.float32)
        done = set()
        for w in set(tok(query)):
            ids = self._variants(w) if use_stem else ([self.vocab[w]] if w in self.vocab else [])
            key = tuple(sorted(ids))
            if not ids or key in done:                # "employ" aur "employees" dono sawal mein -> ek hi baar
                continue
            done.add(key)
            if len(ids) == 1:
                rows, c = self._term(ids[0], k1, b)
                score[rows] += c
                continue
            g = np.zeros(self.N, dtype=np.float32)    # ek lafz ke kai roop: har chunk mein sab se acha roop
            for j in ids:
                rows, c = self._term(j, k1, b)
                g[rows] = np.maximum(g[rows], c)
            score += g
        return score

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------
    def search(self, query, top_k=5, company=None, year=None, chunk_type=None):
        F = self.features
        NONE = lambda why: {"matched": False, "reason": why, "results": []}

        if not F["oos2"]:                                   # v2.1: anjaan naam ki list sab se pehle
            unknown = self._mentions_unknown_company(query)
            if unknown:
                return NONE(f"No indexed data for '{unknown}'.")

        comps = self._detect_companies(query)
        if company:
            comps = {n for n in self.known_companies if company.lower() in n.lower()}
        typos = []
        if F["oos2"] and not comps:                         # v2.2: list tab hi jab index ki koi company na ho
            unknown = self._mentions_unknown_company(query, extended=True)
            if unknown:
                return NONE(f"No indexed data for '{unknown}'.")
        if F["names"] and not comps:                        # "Wallmart" -> Walmart
            comps, typos = self._fuzzy_companies(query)
        if F["guard"] and not comps:
            unknown = self._unknown_subject(query)
            if unknown:
                return NONE(f"No indexed data for '{unknown}'.")
        years = {int(y) for y in self.YEAR_PATTERN.findall(query)}
        if year:
            years = {int(year)}
        recency_rx = self.RECENCY_V2 if F["recency"] else self.RECENCY_KEYWORDS
        wants_recent = bool(recency_rx.search(query)) and not years

        needs_hard_filter = bool(comps or years or wants_recent)
        mask = np.ones(self.N, dtype=bool)

        if comps:
            mask &= np.isin(self.ccode, [self.names.index(c) for c in comps])
            if not mask.any():
                return NONE(f"No data found for company '{sorted(comps)}'.")
        if years:
            m2 = mask & np.isin(self.year, list(years))
            if not m2.any():
                avail = {c: self._years_by_company.get(c, []) for c in (comps or [])}
                extra = f" Available years: {avail}" if avail else ""
                return NONE(f"No data found for year(s) {sorted(years)}.{extra}")
            mask = m2
        elif wants_recent:
            mask &= (self.year == self._latest_arr[self.ccode])
        if chunk_type:
            mask &= (self.tcode == self.type_names.index(chunk_type))
        if F["clutter"]:
            m3 = mask & ~self.clutter
            if m3.any():
                mask = m3

        idx = np.where(mask)[0]
        if idx.size == 0:
            return NONE("No sufficiently relevant data found for this query.")

        query_t = query
        if F["typo"]:                                  # v2.3: "proft" -> "profit" (company naam nahi chhedte)
            query_t, _ = self._fix_typos(query, skip=typos)
        item_types = line_item_types(query_t)          # number wala sawal? (total assets, revenue, D&A ...)
        if F["strip"]:
            sq = self._search_text(query_t, comps, set() if item_types else years,
                                   extra=typos, legal=LEGAL_SUFFIX if F["names"] else OLD_LEGAL_SUFFIX)
        else:
            sq = query_t
        if F["roman"]:                                 # v2.3: Roman Urdu -> English alfaaz (dense + keyword)
            rt = roman_terms(query_t)
            if rt:
                sq = f"{sq} {rt}"
        q = self._embed(sq)

        # ---- dense (FAISS) : filter ke andar top POOL ----
        D, I = self._dense(q, int(min(self.POOL, idx.size)), idx)
        dense_hits = [(int(i), float(s)) for i, s in zip(I[0], D[0]) if i >= 0]
        rd = {doc: r for r, (doc, _) in enumerate(dense_hits)}
        dscore = dict(dense_hits)
        best_dense = dense_hits[0][1] if dense_hits else 0.0

        # ---- keyword (BM25) : filter ke andar top POOL ----
        bq = sq
        if F["synonyms"]:
            more = synonym_terms(query_t)
            if more:
                bq = f"{sq} {more}"
        if F["words"]:                                 # v2.3: layoffs -> restructuring ... (sirf keyword)
            more2 = words_terms(query_t)
            if more2:
                bq = f"{bq} {more2}"
        bm_all = self._bm25(bq, use_stem=F["stem"])
        bm = bm_all[idx]
        rb = {int(idx[i]): r for r, i in enumerate(np.argsort(-bm)[:self.POOL]) if bm[i] > 0}
        lists = [rd, rb]

        # ---- stmt: number wale sawal par asli statement ki teesri list ----
        types = item_types if F["stmt"] else set()
        if types:
            sel = idx[np.isin(self.stmt[idx], list(types))]
            if sel.size:
                D3, I3 = self._dense(q, int(min(self.STMT_POOL, sel.size)), sel)
                lists.append({int(i): r for r, i in enumerate(I3[0]) if i >= 0})

        # ---- intent (v2.3): overview / strategy / auditor sawal -> us hisse ke behtareen chunks ko ek extra vote ----
        bits = intent_bits(query_t) if (F["intent"] and not item_types) else 0
        if bits:
            sel = idx[(self.sect[idx] & bits) != 0]
            if sel.size:
                Di, Ii = self._dense(q, int(min(self.INTENT_POOL, sel.size)), sel)
                bi = bm_all[sel]
                ri = {}
                for r, i in enumerate(Ii[0]):
                    if i >= 0:
                        ri[int(i)] = ri.get(int(i), 0.0) + 1.0 / (self.RRF_K + r + 1)
                for r, i in enumerate(np.argsort(-bi)[:self.INTENT_POOL]):
                    if bi[i] > 0:
                        d = int(sel[i])
                        ri[d] = ri.get(d, 0.0) + 1.0 / (self.RRF_K + r + 1)
                lists.append({doc: r for r, doc in enumerate(sorted(ri, key=lambda x: -ri[x])[:self.INTENT_POOL])})

        # ---- fresh: saal na diya ho to sab se naye 10-K ke behtareen chunks ko ek extra vote ----
        if F["fresh"] and not years and not wants_recent:
            new_idx = idx[self.year[idx] == self._latest_arr[self.ccode[idx]]]
            if 0 < new_idx.size < idx.size:
                Dn, In = self._dense(q, int(min(self.FRESH_POOL, new_idx.size)), new_idx)
                bn = bm_all[new_idx]
                rn = {}
                for r, i in enumerate(In[0]):
                    if i >= 0:
                        rn[int(i)] = rn.get(int(i), 0.0) + 1.0 / (self.RRF_K + r + 1)
                for r, i in enumerate(np.argsort(-bn)[:self.FRESH_POOL]):
                    if bn[i] > 0:
                        d = int(new_idx[i])
                        rn[d] = rn.get(d, 0.0) + 1.0 / (self.RRF_K + r + 1)
                lists.append({doc: r for r, doc in enumerate(sorted(rn, key=lambda x: -rn[x])[:self.FRESH_POOL])})

        rrf = {}
        for L in lists:
            for doc, r in L.items():
                rrf[doc] = rrf.get(doc, 0.0) + 1.0 / (self.RRF_K + r + 1)

        # ---- diverse: do companies / do saal -> dono ke chunks baari baari ----
        group_by = None
        if F["diverse"]:
            if len(comps) >= 2:
                group_by = "company"
            elif len(years) >= 2:
                group_by = "year"
        want = top_k * 4 if (group_by or F["expand"]) else top_k

        cands, seen, same_text = [], set(), {}
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
                "_doc": doc,
                "_m": m,
            }
            if F["dedup"]:                   # hu-ba-hu wahi paragraph doosre saal ka -> ek hi (naye saal wala)
                sig = re.sub(r"\s+", " ", (m["text"] or "")[:250].lower()).strip()
                j = same_text.get(sig)
                if j is not None:
                    if int(m["year"] or 0) > int(cands[j]["year"] or 0):
                        cands[j] = item
                    continue
                same_text[sig] = len(cands)
            cands.append(item)
            if len(cands) == want:
                break

        if group_by:
            buckets = {}
            for it in cands:                                    # pehle wala group = jis ka sab se acha chunk
                buckets.setdefault(it[group_by], []).append(it)
            order, ordered = list(buckets), []
            while any(buckets[k] for k in order):
                for k in order:
                    if buckets[k]:
                        ordered.append(buckets[k].pop(0))
        else:
            ordered = cands

        results, covered = [], set()
        for it in ordered:
            if len(results) == top_k:
                break
            if F["expand"]:
                g = int(self.gid[it["_doc"]])
                if g in covered:                 # ye chunk upar wale result ke saath pehle hi aa chuka
                    continue
                it["text"], used, it["page_numbers"] = self._expanded_text(it["_doc"], it["_m"], covered)
                covered |= used
            results.append(it)
        for it in cands:
            it.pop("_doc", None)
            it.pop("_m", None)

        if not results:
            return NONE("No sufficiently relevant data found for this query.")
        gate = (not comps) if F["oos2"] else (not needs_hard_filter)
        if gate and best_dense < self.MIN_CONFIDENCE_SCORE:
            return NONE("No sufficiently relevant data found for this query.")
        return {"matched": True, "reason": None, "results": results}


# Test sawal (21 Oct 2026): 50 companies x 2 = 100 sawal, aam user ke andaz mein (casual, Roman Urdu, halki typo,
# ticker, bina saal ...). Files Hifza ne chuni; sawal aur sahi jawab Claude ne PDF padh kar ek pehle se likhe qaide
# ke mutabiq banaye (topic aur andaz PDF padhne se PEHLE tay), aur har jawab PDF mein check kiya. Ye v2.2 ki tuning
# ke BAAD likhe gaye; pehli run se pehle inhein commit (freeze) karna hai, taake result dekh kar sawal na badlein.
# Q01..Q100 = STAGE_1/test_questions.csv ke T01..T100 (wahan sahi jawab + PDF page). Aakhir mein 3 bahar ke sawal.
# "--test" (ya VS Code ka ▷ Run, bina kuch likhe) inhein chalata hai. Sir ya koi bhi apne sawal yahan jod sakta hai
# (category -> list); jis sawal ka jawab CSV mein nahi, us ke results sirf dikhte hain, auto-check nahi hota.
TEST_QUESTIONS = {
    'T01-T20: 3M se Caterpillar tak': [
        'Who does 3M compete with, according to its 2016 annual report?',   # T01
        'how many people work at 3m 2016',   # T02
        'what happens to adobe if its new products dont match what custmers want? 2017',   # T03
        '2017 mein Adobe ka head office kahan tha?',   # T04
        'Does AMZN use temp workers or have unions? (2018)',   # T05
        "Who audited Amazon's financial statements for 2018?",   # T06
        'How was AMD trying to grow its business in 2019?',   # T07
        'Who was the CEO of AMD in 2019?',   # T08
        'what does ametek make 2020',   # T09
        'in how many contries outside the US does ametek manufacture products in 2020?',   # T10
        'Apple apne products kaise bechti hai aur us ke customers kaun log hain? 2021',   # T11
        'AAPL total revenue fiscal 2021',   # T12
        'Does Block make its own Square card readers, or do other companies manufacture them? (2022)',   # T13
        'How much money did Block make or lose in 2022?',   # T14
        "What is the FAA doing about Boeing's quality problems, according to the 2023 10-K?",   # T15
        'boeing backlog 2023',   # T16
        'how many busness segments does carlisle have in 2024 and what are they?',   # T17
        'Carlisle kis state mein incorporated hai? (2024 report)',   # T18
        "How did tariffs hit CAT's profits in 2025?",   # T19
        'when was caterpillar founded',   # T20
    ],
    'T21-T40: Chevron se Google tak': [
        'What cyber risks did Chevron warn about in 2016?',   # T21
        'chevron r&d spending 2016',   # T22
        'What new technology or innovations was Chipotle working on in 2017?',   # T23
        'Kya Chipotle 2017 mein apne shareholders ko dividend deti thi?',   # T24
        'How could climate change hurt Colgate, according to its 2018 report?',   # T25
        'how many employes did colgate have in 2018?',   # T26
        'How does Costco compete, and who are its biggest rivals? (2019)',   # T27
        "Where is Costco's head office, according to its 2019 annual report?",   # T28
        'what does dell worry about with competitors in 2020',   # T29
        "Who was Dell's auditor for fiscal 2020?",   # T30
        'What benefits does Donaldson give its employees? (fiscal 2021)',   # T31
        'Who is the CEO of DCI in 2021?',   # T32
        "whats ebay's growth plan for 2022",   # T33
        'In how many markets is eBay available, per its 2022 report?',   # T34
        'Emerson company kya banati aur bechti hai? 2023',   # T35
        "What were Emerson's total sales in fiscal 2023?",   # T36
        'does franklin electric sell its pumps through distributers or directly? 2024',   # T37
        "What was Franklin Electric's net income in 2024?",   # T38
        'does google make its own phones and servers or use other factories 2025',   # T39
        "how big is google cloud's backlog right now",   # T40
    ],
    'T41-T60: Graco se Netflix tak': [
        'Is Graco dealing with any big lawsuits or legal problems in 2016?',   # T41
        'how many segments does graco have 2016',   # T42
        'How is Home Depot organized into business segments in fiscal 2017?',   # T43
        'Home Depot company kab bani thi? (fiscal 2017 report)',   # T44
        'How did Intel do overall in 2018?',   # T45
        'how much did intel spend on reserch and development in 2018?',   # T46
        'How could hackers or cyberattacks hurt Intuit, according to its fiscal 2019 report?',   # T47
        'How much did Intuit pay in dividends per share in fiscal 2019?',   # T48
        'what does kennametal focus on in r&d 2020',   # T49
        'How many people did Kennametal employ in fiscal 2020?',   # T50
        "What are Kimberly-Clark's environmental goals in 2021?",   # T51
        'Where is KMB headquartered? (2021)',   # T52
        'who does lockheed martin compete against 2022',   # T53
        "Who audited Lockheed Martin's financial statements for 2022?",   # T54
        'Agar log Facebook aur Instagram kam use karein to Meta ko kya nuqsan hoga? 2024',   # T55
        'Who is the CEO of Meta in 2024?',   # T56
        "what is microsoft's company culture like in 2023, acording to its annual report?",   # T57
        "Where are Microsoft's main offices, and how much space does it have there? (fiscal 2023)",   # T58
        "whats netflix's plan to keep growing in 2025",   # T59
        "what is netflix's latest yearly revenue",   # T60
    ],
    'T61-T80: Nike se Roper tak': [
        'What kinds of products does Nike sell in fiscal 2017?',   # T61
        "whats nike's net income for 2017",   # T62
        "Who are Nordson's customers, and how does it sell its products? (2016)",   # T63
        'Nordson ka 2016 mein order backlog kitna tha?',   # T64
        'Does Northrop Grumman rely on outside suppliers and subcontractors for its products? (2018)',   # T65
        'how many segments does northrop grumman have in 2018 and what are thier names?',   # T66
        'Why did shareholders sue Nvidia in fiscal 2019?',   # T67
        'When was Nvidia founded? (fiscal 2019 report)',   # T68
        'is palo alto networks split into different business segments 2020',   # T69
        'How much did Palo Alto Networks spend on research and development in fiscal 2020?',   # T70
        'Did Parker Hannifin make or announce any big acquisitions in fiscal 2021?',   # T71
        'How much did PH pay in dividends in fiscal 2021?',   # T72
        'how does paypal protect itself from hackers 2022',   # T73
        'How many employees did PayPal have at the end of 2022?',   # T74
        'Pentair research and development mein kya karti hai, aur 2023 mein us par kitna kharch kiya?',   # T75
        "Where is Pentair's head office located, according to its 2023 annual report?",   # T76
        "what are rockwell automation's sustainabilty goals in fiscal 2024?",   # T77
        "Who was Rockwell Automation's auditor for fiscal 2024?",   # T78
        'who does roper technologies compete with 2025',   # T79
        'who is the current ceo of roper technologies',   # T80
    ],
    'T81-T100: Salesforce se Xylem tak': [
        'What could happen to Salesforce if hackers break into its systems and steal customer data? (fiscal 2017)',   # T81
        "where are salesforce's offices and data centers 2017",   # T82
        "How big is ServiceNow's workforce, and are its employees in a union? (2016)",   # T83
        'ServiceNow ki 2016 mein total revenue kitni thi?',   # T84
        "What is Southwest Airlines' strategy for staying profitable and growing in 2018?",   # T85
        'how much proft did southwest airlines make in 2018?',   # T86
        'How does Starbucks make its money, and what does it actually sell? (fiscal 2019)',   # T87
        'How many business segments did Starbucks have in fiscal 2019, and what are they?',   # T88
        'what kind of customers does union pacific serve 2020',   # T89
        'How many business segments does Union Pacific report in 2020?',   # T90
        'Where does Walmart get the products it sells, and how do they reach its stores? (fiscal 2021)',   # T91
        'When was WMT founded? (2021 report)',   # T92
        'what regulations do watts water products have to meet 2022',   # T93
        'How much did Watts Water spend on research and development in 2022?',   # T94
        'Woodward ke business segments kaun se hain? (fiscal 2023)',   # T95
        'How much did Woodward spend buying back its own shares in fiscal 2023?',   # T96
        'did workday anounce any layoffs in fiscal 2025?',   # T97
        'How many employees did Workday have at the end of fiscal 2025?',   # T98
        'what is xylem doing about cybersecurity 2025',   # T99
        'where is xylem headquartered',   # T100
    ],
    "Bahar ke sawal (jawab 'nahi mila' hona chahiye)": [
        "What was Tesla's revenue in 2023?",
        "what does spotify say about podcasts",
        "Who won the Cricket World Cup in 2023?",
    ],
}

_PREVIEW_STOP = {
    "what", "which", "who", "whom", "whose", "how", "when", "where", "why", "does", "did", "do", "is", "was", "were",
    "are", "the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "its", "their", "it", "this", "that", "these",
    "those", "about", "say", "says", "said", "tell", "me", "please", "describe", "explain", "much", "many", "has", "have",
    "had", "be", "been", "with", "from", "by", "as", "at", "any", "there", "company", "companies", "fiscal", "year",
    "annual", "report", "main", "mention", "mentions", "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
}


def answer_preview(text, query, n=2, width=320, skip=""):
    """Result ke text mein se wo 1-2 jumle (ya table ki lines) jo sawal se sab se zyada milte hain. Sirf dikhane ke
    liye -- search par koi asar nahi. Sir / Gemini se pehle aankh se check karne mein madad karta hai.
    skip = company ka naam: har jumle mein hota hai, is liye us se jumla nahi chuna jata."""
    skip_set = {stem(w) for w in tok(skip)}
    want = {stem(w) for w in tok(query + " " + synonym_terms(query))
            if w.isalpha() and len(w) > 2 and w not in _PREVIEW_STOP} - skip_set
    if not want:
        return []
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n| / ", text or "") if len(p.strip()) > 25]
    scored = []
    for i, p in enumerate(parts):
        have = {stem(w) for w in tok(p) if w.isalpha()}
        s = len(want & have)
        if s:
            scored.append((s, -i, p))
    best = sorted(scored, reverse=True)[:n]
    best.sort(key=lambda x: -x[1])                      # PDF wali tarteeb mein
    return [p if len(p) <= width else p[:width] + " ..." for _, _, p in best]


# ---- auto-check: test_questions.csv ke sahi jawab se milao (sirf madad; asli faisla aap ka nishan) ----
_GOLD_NUM = re.compile(r"^[\s$(]*-?\d[\d,]*(?:\.\d+)?\s*\)?%?$")
_SOFT = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-",
                       " ": " "})


def _soft(t):
    """eval_golden.norm jaisa (quotes, dash, spaces ek jaise), lekin bare / chhote huroof wahi rehte hain."""
    return re.sub(r"\s+", " ", (t or "").translate(_SOFT)).strip()


def _gold_pattern(alt):
    """'4,240' -> commas ke saath ya baghair ('14,240' nahi);  text -> wahi tukda, bare / chhote huroof ka farq nahi."""
    if _GOLD_NUM.match(alt):
        whole, _, dec = re.search(r"\d[\d,]*(?:\.\d+)?", alt).group(0).partition(".")
        d, groups = whole.replace(",", ""), []
        while len(d) > 3:
            groups.insert(0, d[-3:])
            d = d[:-3]
        groups.insert(0, d)
        return re.compile(r"(?<![\d,.])" + ",?".join(groups) + (r"\." + dec if dec else "") + r"(?!\d)(?!,\d)(?!\.\d)")
    return re.compile(re.escape(_soft(alt)), re.I)


def load_gold(path):
    """test_questions.csv -> {sawal: row}. File na mile to {} (--test phir bhi chalta hai, bas auto-check nahi)."""
    p = Path(path)
    if not p.is_file():
        return {}
    raw = p.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1252", errors="replace")
    lines = text.splitlines()
    delim = ";" if lines and lines[0].count(";") > lines[0].count(",") else ","
    return {(r.get("question") or "").strip(): r for r in csv.DictReader(lines, delimiter=delim)
            if (r.get("question") or "").strip()}


def gold_check(retriever, row, results, width=110):
    """Kin results mein sahi jawab hai -- sahi company + saal ke saath, eval_golden wala hi qaida ("|" = koi bhi ek).
    Aise har result mein 'gold_snippet' jorta hai (jawab ke aas paas ka text, jawab **aise** nishan ke saath).
    Ranks lautata hai: [1, 3] = #1 aur #3 mein jawab hai."""
    comps = set()
    for name in [x.strip() for x in (row.get("company") or "").split("|") if x.strip()]:
        comps |= retriever._detect_companies(name) or retriever._detect_companies(name.title())
    years = {int(y) for y in re.findall(r"(?:19|20)\d{2}", row.get("year") or "")}
    pats = [_gold_pattern(x.strip()) for x in (row.get("answer") or "").split("|") if x.strip()]
    ranks = []
    for j, res in enumerate(results, 1):
        if (comps and res["company"] not in comps) or (years and int(res["year"] or 0) not in years):
            continue
        soft = _soft(res["text"])
        m = next((m for m in (p.search(soft) for p in pats) if m), None)
        if m:
            a, b = max(0, m.start() - width), min(len(soft), m.end() + width)
            res["gold_snippet"] = (("... " if a else "") + soft[a:m.start()] + "**" + m.group(0) + "**"
                                   + soft[m.end():b] + (" ..." if b < len(soft) else ""))
            ranks.append(j)
    return ranks


def _show(question, response, label="Q", gold=None):
    print("\n" + "=" * 70)
    print(f"{label}:", question)
    if gold:
        alts = " ya ".join(f'"{x.strip()}"' for x in (gold.get("answer") or "").split("|") if x.strip())
        print(f"    Jawab hona chahiye ({gold.get('id', '')}): {gold.get('company', '')} {gold.get('year') or 'koi bhi saal'}"
              f" -> {alts}  (PDF page {gold.get('page') or '?'})")
    print("=" * 70)
    if not response["matched"]:
        print("    ⚠️  ", response["reason"])
        return
    for i, r in enumerate(response["results"], 1):
        pg = r["page_numbers"]
        pg = (f"{min(pg)}-{max(pg)}" if pg and min(pg) != max(pg) else (str(pg[0]) if pg else "?"))
        print(f"\n[{i}] {r['company_display']} {r['year']} | {r['chunk_type']} | PDF page {pg} | "
              f"dense={r['score']:.3f} rrf={r['rrf_score']:.4f}")
        print("    Section:", r["section_path"][-110:])
        lines = answer_preview(r["text"], question, skip=f"{r['company']} {r['company_display']}")
        for s in lines:
            print("    >>", s.replace("\n", " / "))
        if not lines:
            print("    " + r["text"][:300].replace("\n", " / "))
        if r.get("gold_snippet"):
            print("    ✓ AUTO: sahi jawab is result mein hai ->", r["gold_snippet"])


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")    # output file mein bhejo (> out.txt) to ✓ / ⚠️ par crash na ho
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?")
    ap.add_argument("-k", type=int, default=None, help="kitne results (default 5)")
    ap.add_argument("--company")
    ap.add_argument("--year", type=int)
    ap.add_argument("--type", choices=["text", "table"])
    ap.add_argument("--store", default="STAGE_1/vector_store")
    ap.add_argument("--off", default="", help="comma separated sudhaar band karo: " + ",".join(FEATURES))
    ap.add_argument("--test", action="store_true", help="neeche wale TEST_QUESTIONS chalao (bina sawal ke bhi yahi)")
    ap.add_argument("--gold", default="STAGE_1/test_questions.csv", help="sahi jawab wali CSV (auto-check ke liye)")
    a = ap.parse_args()
    if not a.query and not a.test:                  # VS Code ka ▷ Run (bina kuch likhe) -> test sawal
        print('Koi sawal nahi diya, is liye --test chal raha hai.  Apna sawal: python STAGE_1/retriever.py "..."')
        a.test = True
    retriever = Retriever(a.store)
    for f in [x.strip() for x in a.off.split(",") if x.strip()]:
        if f not in retriever.features:
            ap.error(f"'{f}' koi sudhaar nahi. In mein se: {', '.join(retriever.features)}")
        retriever.features[f] = False
    if a.test:
        k = a.k or 5
        gold = load_gold(a.gold)
        print(f"Sahi jawab: {a.gold} se {len(gold)} sawal (auto-check)" if gold
              else f"(Sahi jawab wali CSV nahi mili: {a.gold} -- auto-check band, results phir bhi dikhenge)")
        log = []
        for cat, qs in TEST_QUESTIONS.items():
            print("\n" + "#" * 70 + f"\n# CATEGORY: {cat}\n" + "#" * 70)
            for qn in qs:
                resp = retriever.search(qn, top_k=k)
                row = gold.get(qn.strip())
                entry = {"n": len(log) + 1, "category": cat, "question": qn}
                if row:
                    entry["expected"] = {f: (row.get(f) or "").strip() for f in ("id", "company", "year", "answer", "type", "page")}
                    entry["gold_ranks"] = gold_check(retriever, row, resp["results"]) if resp["matched"] else []
                _show(qn, resp, label=f"Q{entry['n']:02d}", gold=row)
                entry.update(resp)
                log.append(entry)
        with open("retriever_test_results.json", "w", encoding="utf-8") as f:
            json.dump(log, f, indent=4, ensure_ascii=False)
        n_ok = sum(1 for x in log if x["matched"])
        print(f"\n\n{len(log)} sawal: {n_ok} par results aaye (top {k}), {len(log) - n_ok} par 'nahi mila'.")
        print("\nMARKING LIST -- upar har sawal ke '>>' padh kar [ ] mein likho:  ✓ jawab mila | ~ adhoora | ✗ nahi mila")
        print("(bahar ke sawal par 'nahi mila' aaye to ✓; sahi company / saal bhi dekhna)")
        for x in log:
            if x["matched"]:
                r = x["results"][0]
                pg = r["page_numbers"]
                got = f"#1: {r['company_display']} {r['year']} p{min(pg) if pg else '?'}"
            else:
                got = "nahi mila"
            q_ = x["question"] if len(x["question"]) <= 62 else x["question"][:59] + "..."
            auto = ""
            if "expected" in x:
                auto = f"  [auto: #{x['gold_ranks'][0]}]" if x["gold_ranks"] else "  [auto: nahi]"
            print(f"[ ] Q{x['n']:02d}  {q_:<62}  -> {got}{auto}")
        if gold:
            g = [x for x in log if "expected" in x]
            print(f"\nAuto-check: {sum(1 for x in g if x['gold_ranks'])}/{len(g)} sawalon ka sahi jawab top {k} mein mila "
                  f"(sahi company + saal ke saath).")
            print("Ye sirf madad hai: jawab doosre alfaaz mein bhi ho sakta hai, is liye har sawal khud dekh kar nishan lagao.")
        print("\nPoore results: retriever_test_results.json")
    else:
        _show(a.query, retriever.search(a.query, top_k=a.k or 5, company=a.company, year=a.year,
                                        chunk_type=a.type))