"""
retriever.py  -  Tumhari wahi Retriever class (wahi search() aur wahi jawab ka format), naye index ke saath.

Chalane ka tareeqa:
    python STAGE_1/retriever.py "Woodward 2025 total assets"
    python STAGE_1/retriever.py "net income" --company Nordson --year 2016
    python STAGE_1/retriever.py --test          (neeche wale test sawal, retriever_test_results.json banata hai)

Doosre code se (Gemini wale step mein):
    from retriever import Retriever
    r = Retriever()
    resp = r.search("What was Woodward's total assets in 2025?", top_k=5)
    # resp = {"matched": True/False, "reason": ..., "results": [ {score, rrf_score, company, year,
    #          section_path, page_numbers, chunk_type, text}, ... ]}

Purani retriever.py se kya badla:
  * FAISS wahi (IndexFlatIP, exact search). Company/saal ka filter FAISS ke andar search se PEHLE lagta hai
    (IDSelector) -> sirf wahi company-saal dhoondi jati hai.
  * BM25 ab sparse matrix hai (rank_bm25 500 files par kai GB RAM leta).
  * Company pehchan: pehla lafz tabhi jab wo sirf ek company ka ho (warna galat company chun leti thi).
    Do companies ka sawal ("Woodward vs Nordson") dono ko filter karta hai.
  * Saal: sawal mein jitne saal likhe hon sab (jaise "2019 and 2020"), sirf pehla nahi.
  * Numbers (1,285) tukde nahi hote, exact number se bhi mil jate hain.
  * Purani cheezein jaisi thin: recency ("latest"), unknown company guard, confidence gate (bina filter wale sawal),
    "No data found for year ..." jawab.
"""
import argparse
import json
import re
from pathlib import Path

import faiss
import numpy as np
import scipy.sparse as sp

from build_embeddings import EmbeddingIndexBuilder, tok


class Retriever:

    RECENCY_KEYWORDS = re.compile(
        r"\b(most recent|latest|current|this year|last fiscal year|newest|up[- ]to[- ]date)\b", re.IGNORECASE)
    YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")

    # Dense cosine par check (RRF score par nahi). Model badla hai (bge-small), isliye eval mein dobara
    # parakhna: agar sahi sawal bhi "nahi mila" aaye to ise 0.50 ke qareeb kar dena.
    MIN_CONFIDENCE_SCORE = 0.58
    RRF_K = 60
    POOL = 50

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

        self.embedder = EmbeddingIndexBuilder(cfg["model"])

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
            if w and len(w[0]) >= 3 and len(first[w[0]]) == 1:
                self.aliases[w[0]] = {n}
            m = re.search(r"\((.*?)\)", n)
            if m and len(m.group(1)) >= 2:
                self.aliases[m.group(1).lower()] = {n}

    def _detect_companies(self, query):
        q = " " + re.sub(r"[^a-z0-9&]+", " ", query.lower()) + " "
        found = set()
        for a, ns in self.aliases.items():
            if a and re.search(r"\b" + re.escape(a) + r"\b", q):
                found |= ns
        return found

    def _detect_company(self, query):          # purane naam ke liye (ek company ya None)
        f = self._detect_companies(query)
        return sorted(f)[0] if len(f) == 1 else None

    def _mentions_unknown_company(self, query):
        q = query.lower()
        known = " ".join(c.lower() for c in self.known_companies)
        for name in self.WELL_KNOWN_COMPANIES:
            if re.search(r"\b" + re.escape(name) + r"\b", q) and name not in known:
                return name
        return None

    # ------------------------------------------------------------------
    # bm25
    # ------------------------------------------------------------------
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

        unknown = self._mentions_unknown_company(query)
        if unknown:
            return {"matched": False, "reason": f"No indexed data for '{unknown}'.", "results": []}

        comps = self._detect_companies(query)
        if company:
            comps = {n for n in self.known_companies if company.lower() in n.lower()}
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

        idx = np.where(mask)[0]
        if idx.size == 0:
            return {"matched": False, "reason": "No sufficiently relevant data found for this query.", "results": []}

        # ---- dense (FAISS) : filter ke andar top POOL ----
        q = np.ascontiguousarray(self.embedder.embed_query(query), dtype=np.float32)      # (1, dim)
        k = int(min(self.POOL, idx.size))
        if idx.size == self.N:
            D, I = self.index.search(q, k)
        else:
            ids = np.ascontiguousarray(idx, dtype=np.int64)
            params = faiss.SearchParameters()
            params.sel = faiss.IDSelectorBatch(len(ids), faiss.swig_ptr(ids))
            D, I = self.index.search(q, k, params=params)
        dense_hits = [(int(i), float(s)) for i, s in zip(I[0], D[0]) if i >= 0]
        rd = {doc: r for r, (doc, _) in enumerate(dense_hits)}
        dscore = dict(dense_hits)
        best_dense = dense_hits[0][1] if dense_hits else 0.0

        # ---- keyword (BM25) : filter ke andar top POOL ----
        bm = self._bm25(query)[idx]
        rb = {int(idx[i]): r for r, i in enumerate(np.argsort(-bm)[:self.POOL]) if bm[i] > 0}

        rrf = {}
        for doc in set(rd) | set(rb):
            rrf[doc] = (1.0 / (self.RRF_K + rd[doc] + 1) if doc in rd else 0.0) + \
                       (1.0 / (self.RRF_K + rb[doc] + 1) if doc in rb else 0.0)

        results, seen = [], set()
        for doc in sorted(rrf, key=lambda x: -rrf[x]):
            g = int(self.gid[doc])
            if g in seen:                    # lambe table ke kai tukde -> ek hi baar
                continue
            seen.add(g)
            m = self._meta(doc)
            results.append({
                "score": dscore[doc] if doc in dscore else float(self.index.reconstruct(doc) @ q[0]),
                "rrf_score": rrf[doc],
                "company": m["company"],
                "year": m["year"],
                "section_path": m["section_path"],
                "page_numbers": m["page_numbers"],
                "chunk_type": m["chunk_type"],
                "text": m["text"],
            })
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
        print(f"\n[{i}] dense={r['score']:.3f} rrf={r['rrf_score']:.4f} | {r['company']} {r['year']} | "
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
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    retriever = Retriever(a.store)
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