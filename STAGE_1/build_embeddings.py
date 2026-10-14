"""
build_embeddings.py  -  Tumhari wahi file, naye (500-file ke liye) tareeqe se.

Chalane ka tareeqa (project root se):
  1) Chhota test:   python STAGE_1/build_embeddings.py --only Nordson
  2) Poora:         python STAGE_1/build_embeddings.py
     (beech mein ruk jaye to wahi command dobara chalao -- jo files ho chuki hain wo dobara nahi banti)

Kya badla (purani file se):
  * Model: bge-small (bge-base se ~3 guna tez; 500 files par base ~1 din leta, small ~9-10 ghante).
    Chaho to  --model BAAI/bge-base-en-v1.5  de sakti ho.
  * Har file ki embeddings alag save hoti hain (STAGE_1/vector_store/cache) -> crash par kuch zaya nahi.
  * Vector index: FAISS (IndexFlatIP = exact cosine search, FYP report wala). File-by-file add hota hai, RAM kam.
  * Keyword search (BM25) bhi yahin banta hai -- rank_bm25 ki jagah sparse matrix (rank_bm25 3 lakh chunks par
    kai GB RAM leta). Numbers (jaise 1,285) poore rehte hain.
  * Bekaar chunks index mein nahi jaate: sirf EXHIBIT LISTS (doosre documents ki fehrist) aur
    "See accompanying notes." jaisi khaali lines. Signatures, power of attorney aur chhote chunks RAKHE jaate hain
    (un mein CEO/CFO/directors ke naam aur kuch asli numbers hote hain). Sab wapas daalne ke liye --keep-junk.
  * Embedding text = "Company | Saal 10-K | Section" + chunk text (section retrieval mein madad karta hai).
    Headings baad mein theek hon to sirf wahi files dobara embed hongi jin ke labels badle (cache khud pehchanta hai).
    Section ke baghair chahiye to --no-section.
  * Lambe chunks (zyada-tar bade tables) tukdon (windows) mein embed hote hain, dum gayab nahi hoti.

Install (ek baar):  python -m pip install sentence-transformers faiss-cpu numpy scipy
"""
import argparse
import json
import re
import time
import zlib
from array import array
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sp

TOKEN_RE = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?|[a-z]+")


def tok(text):
    return TOKEN_RE.findall(text.lower())


def section_str(c):
    s = c.get("section_path", "")
    parts = s if isinstance(s, list) else str(s or "").split(" > ")
    parts = [str(x).strip() for x in parts if x is not None and str(x).strip() not in ("", "None")]
    return " > ".join(parts)


# ---------------- bekaar (junk) chunks ----------------
# Sirf EXHIBIT LISTS (doosre documents ki fehrist: "3.1 Restated Certificate ... incorporated by reference to
# Exhibit 3.1 of Form 10-Q ...") aur "See accompanying notes." jaisi khaali lines nikalti hain.
# SIGNATURES aur POWER OF ATTORNEY nahi nikalte: in mein CEO / CFO / directors ke naam aur ohde hote hain.
# Chhote chunks bhi nahi nikalte: kuch mein asli data hai ("Total long-term debt $ 15,696").
EXHIBIT_LEAF = re.compile(r"^(index to )?exhibits?( index| list)?\.?$", re.I)
EXHIBIT_TABLE = re.compile(r"Description of Document|Incorporated (?:by|herein by) Reference|Exhibit Number", re.I)
INC_REF = re.compile(r"incorporated\s+(?:herein\s+)?by\s+reference", re.I)
EXHIBIT_REF = re.compile(r"\bExhibit\s+\d{1,3}(?:\.\d{1,3})?", re.I)
FORM_REF = re.compile(r"\bForm\s+(?:10-?K|10-?Q|8-?K|S-\d)", re.I)
SIGNATURE_LIKE = re.compile(r"/s/|attorney-in-fact|Pursuant to the requirements of Section 13", re.I)
BOILERPLATE = re.compile(
    r"^(?:(?:see|refer to) (?:the )?accompanying notes(?: to (?:the )?(?:consolidated )?financial statements)?"
    r"|the accompanying notes are an integral part of (?:these|the) (?:consolidated )?financial statements"
    r"|\(?(?:in|dollars in|\$ in) (?:millions|thousands)[^)]{0,40}\)?)\.?$", re.I)


def junk_reason(c):
    """None = rakho. Warna wajah (string)."""
    t = (c.get("text") or "").strip()
    if not t:
        return "empty"
    if len(t) < 120 and BOILERPLATE.match(t):
        return "boilerplate"
    sec = section_str(c)
    if EXHIBIT_LEAF.match(sec.split(">")[-1].strip()):
        return "exhibit list"
    if c.get("chunk_type") == "table" and EXHIBIT_TABLE.search(t[:600]):
        return "exhibit list"
    if SIGNATURE_LIKE.search(t):          # signatures / power of attorney: naam aur ohde -> rakho
        return None
    n_inc, n_ex, n_form = len(INC_REF.findall(t)), len(EXHIBIT_REF.findall(t)), len(FORM_REF.findall(t))
    if n_ex >= 3 or (n_ex >= 2 and n_inc >= 2):                       # "filed as Exhibit 10.4 to Form 8-K ..."
        return "exhibit list"
    if re.search(r"exhibit", sec, re.I) and (n_inc >= 2 or n_ex + n_form >= 4):
        return "exhibit list"
    return None


def is_junk(c):
    return junk_reason(c) is not None


class _Stub:
    """Sirf code test karne ke liye (asli search ke liye nahi)."""
    dim = 384

    def encode(self, texts, batch_size=32):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in tok(t):
                h = zlib.crc32(w.encode())
                out[i, h % self.dim] += 1.0 if (h >> 9) & 1 else -1.0
        n = np.linalg.norm(out, axis=1, keepdims=True)
        n[n == 0] = 1
        return out / n


class EmbeddingIndexBuilder:
    MODEL_NAME = "BAAI/bge-small-en-v1.5"

    # BGE ka asymmetric rule: passages jaise hain, QUERY ke aage ye instruction lagti hai.
    QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

    # Model 512 token se zyada chupke se kaat deta hai, isliye lambe chunks tukdon mein.
    # Tables mein numbers bohat tokens lete hain, isliye table ke tukde chhote (150 words).
    WINDOW_WORDS_TEXT = 300
    WINDOW_WORDS_TABLE = 150
    OVERLAP_WORDS = 25

    def __init__(self, model_name=None, batch_size=32, max_len=512, include_section=True):
        self.model_name = model_name or self.MODEL_NAME
        self.batch_size = batch_size
        self.include_section = include_section
        print(f"Loading embedding model: {self.model_name} ...")
        self._enc = self._load(self.model_name, max_len)
        self.dimension = int(self._enc(["test"]).shape[1])
        print(f"Model loaded. Embedding dimension: {self.dimension}")

    # ---------------- model ----------------
    def _load(self, name, max_len):
        if name == "stub":
            m = _Stub()
            return lambda texts: m.encode(texts)
        try:
            from sentence_transformers import SentenceTransformer
            try:
                m = SentenceTransformer(name, device="cpu", local_files_only=True)   # net ki zaroorat nahi
            except Exception:
                m = SentenceTransformer(name, device="cpu")
            m.max_seq_length = max_len

            def enc(texts):
                return np.asarray(m.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                           show_progress_bar=False), dtype=np.float32)
            return enc
        except ImportError:
            pass
        try:
            from fastembed import TextEmbedding
            try:
                m = TextEmbedding(model_name=name, local_files_only=True)
            except Exception:
                m = TextEmbedding(model_name=name)

            def enc(texts):
                v = np.array(list(m.embed(texts, batch_size=self.batch_size)), dtype=np.float32)
                n = np.linalg.norm(v, axis=1, keepdims=True)
                n[n == 0] = 1
                return v / n
            return enc
        except ImportError:
            raise SystemExit("Koi embedding library nahi mili. Chalao:  python -m pip install sentence-transformers")

    def embed_passages(self, texts):
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.vstack([self._enc(texts[j:j + 256]) for j in range(0, len(texts), 256)]).astype(np.float32)

    def embed_query(self, query_text):
        prefix = self.QUERY_INSTRUCTION if "bge" in self.model_name.lower() else ""
        return self._enc([prefix + query_text]).astype(np.float32)      # shape (1, dim)

    # ---------------- chunk -> embeddable units ----------------
    def header(self, c):
        """Embedding ke text ke upar: company | saal | section (heading). Section retrieval mein madad karta hai."""
        base = f"{c.get('company', '')} | {c.get('year', '')} 10-K"
        sec = section_str(c)
        return f"{base} | {sec}\n" if (self.include_section and sec) else f"{base}\n"

    def windows(self, c):
        words = c["text"].split()
        size = self.WINDOW_WORDS_TABLE if c.get("chunk_type") == "table" else self.WINDOW_WORDS_TEXT
        head = self.header(c)
        if len(words) <= size:
            return [head + c["text"]]
        out, st = [], 0
        while st < len(words):
            out.append(head + " ".join(words[st:st + size]))
            if st + size >= len(words):
                break
            st += size - self.OVERLAP_WORDS
        return out

    @staticmethod
    def pages_str(c):
        p = c.get("page_numbers") or []
        if not p:
            return ""
        return str(min(p)) if min(p) == max(p) else f"{min(p)}-{max(p)}"

    # ---------------- poora build ----------------
    def build(self, chunks_dir="STAGE_1/chunks", out_dir="STAGE_1/vector_store", only="", keep_junk=False):
        try:
            import faiss                      # shuru mein hi check (9 ghante baad error na aaye)
        except ImportError:
            raise SystemExit("FAISS nahi mila. Chalao:  python -m pip install faiss-cpu")
        out = Path(out_dir)
        cache = out / "cache"
        cache.mkdir(parents=True, exist_ok=True)
        files = sorted(Path(chunks_dir).rglob("*_chunks.json"))
        if only:
            keys = [k.strip().lower() for k in only.split(",") if k.strip()]
            files = [f for f in files if any(k in f.name.lower() for k in keys)]
        if not files:
            raise SystemExit(f"Koi *_chunks.json nahi mili: {Path(chunks_dir).resolve()}")
        print(f"{len(files)} files mileen.")

        t0 = time.time()
        TYPE_NAMES = ["text", "table", "other"]
        comp_names, comp_code = [], {}
        a_comp, a_year, a_type, a_gid, a_off = array("h"), array("h"), array("b"), array("i"), array("q")
        a_dl = array("f")
        vocab = {}
        idx_parts, val_parts, len_parts, emb_files = [], [], [], []
        skipped, chunk_counter, pos = {}, 0, 0
        meta_fh = open(out / "meta.jsonl", "wb")

        for i, f in enumerate(files, start=1):
            safe = re.sub(r"[^A-Za-z0-9_.-]", "_", f.stem)
            cache_npy, cache_sig = cache / (safe + ".npy"), cache / (safe + ".sig")
            try:
                chunks = json.load(open(f, encoding="utf-8"))
            except Exception as e:
                print("  Parh nahi saka:", f.name, e)
                continue
            if keep_junk:
                keep = chunks
            else:
                keep = []
                for c in chunks:
                    r = junk_reason(c)
                    if r:
                        skipped[r] = skipped.get(r, 0) + 1
                    else:
                        keep.append(c)
            meta, texts = [], []
            for c in keep:
                for wi, wt in enumerate(self.windows(c)):
                    meta.append(dict(company=str(c.get("company", "")),
                                     year=int(c["year"]) if str(c.get("year", "")).isdigit() else 0,
                                     chunk_type=c.get("chunk_type", ""), page=self.pages_str(c),
                                     page_numbers=c.get("page_numbers") or [], section_path=section_str(c),
                                     file=f.name, win=wi, text=c["text"], _gid=chunk_counter))
                    texts.append(wt)
                chunk_counter += 1
            sig = str(zlib.crc32("\x00".join(texts).encode("utf-8")))   # text badla to cache purana maana jayega
            ok = False
            if cache_npy.exists() and cache_sig.exists() and cache_sig.read_text(encoding="utf-8") == sig:
                ok = np.load(cache_npy, mmap_mode="r").shape[0] == len(meta)
            if ok:
                how = "cache se"
            else:
                np.save(cache_npy, self.embed_passages(texts))
                cache_sig.write_text(sig, encoding="utf-8")
                how = f"{(time.time() - t0) / 60:.1f} min guzre"
            emb_files.append((cache_npy, len(meta)))

            cols, vals, lens = [], [], []
            for m in meta:
                gid = m.pop("_gid")
                line = (json.dumps(m, ensure_ascii=False) + "\n").encode("utf-8")
                meta_fh.write(line)
                a_off.append(pos)
                pos += len(line)
                if m["company"] not in comp_code:
                    comp_code[m["company"]] = len(comp_names)
                    comp_names.append(m["company"])
                a_comp.append(comp_code[m["company"]])
                a_year.append(m["year"])
                a_type.append(TYPE_NAMES.index(m["chunk_type"]) if m["chunk_type"] in TYPE_NAMES else 2)
                a_gid.append(gid)
                toks = tok(f"{m['company']} {m['year']} {m['section_path']} {m['text']}")
                a_dl.append(len(toks))
                cnt = Counter(toks)
                lens.append(len(cnt))
                for w, n in cnt.items():
                    cols.append(vocab.setdefault(w, len(vocab)))
                    vals.append(n)
            idx_parts.append(np.array(cols, dtype=np.int32))
            val_parts.append(np.array(vals, dtype=np.float32))
            len_parts.append(np.array(lens, dtype=np.int64))
            del cols, vals, lens, meta, texts, chunks, keep
            print(f"[{i}/{len(files)}] {f.name}: {how}")

        meta_fh.close()
        N = len(a_off)
        if N == 0:
            raise SystemExit("Koi chunk nahi bacha.")

        dim = next(np.load(p, mmap_mode="r").shape[1] for p, n in emb_files if n > 0)
        print("FAISS index bana raha hoon...")
        index = faiss.IndexFlatIP(dim)        # normalized vectors -> inner product = cosine (exact search)
        for p, n in emb_files:
            if n:
                index.add(np.ascontiguousarray(np.load(p), dtype=np.float32))
        if index.ntotal != N:
            raise SystemExit(f"Gadbad: FAISS mein {index.ntotal} vectors, meta mein {N}")
        faiss.write_index(index, str(out / "index.faiss"))
        del index

        print("BM25 bana raha hoon...")
        indices, data = np.concatenate(idx_parts), np.concatenate(val_parts)
        lens_all = np.concatenate(len_parts)
        del idx_parts, val_parts, len_parts
        indptr = np.zeros(N + 1, dtype=np.int64)
        np.cumsum(lens_all, out=indptr[1:])
        X = sp.csr_matrix((data, indices, indptr), shape=(N, len(vocab))).tocsc()
        del data, indices
        sp.save_npz(out / "bm25.npz", X, compressed=False)
        np.save(out / "doclen.npy", np.frombuffer(a_dl, dtype=np.float32))
        (out / "bm25_vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
        np.savez(out / "light.npz", company=np.frombuffer(a_comp, dtype=np.int16), year=np.frombuffer(a_year, dtype=np.int16),
                 type=np.frombuffer(a_type, dtype=np.int8), gid=np.frombuffer(a_gid, dtype=np.int32),
                 offset=np.frombuffer(a_off, dtype=np.int64))
        (out / "config.json").write_text(json.dumps(dict(model=self.model_name, n_chunks=N, dim=int(dim),
                                                         companies=comp_names, type_names=TYPE_NAMES,
                                                         section_in_embedding=self.include_section,
                                                         skipped=skipped)), encoding="utf-8")
        print(f"\nHo gaya. {N} vectors index mein.")
        print(f"Nahi daale (bekaar): {sum(skipped.values())}  {skipped}")
        print(f"Folder: {out.resolve()}")
        print('Ab test:  python STAGE_1/retriever.py "Woodward 2025 total assets"')


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="STAGE_1/chunks")
    ap.add_argument("--out", default="STAGE_1/vector_store")
    ap.add_argument("--model", default=EmbeddingIndexBuilder.MODEL_NAME, help="'stub' = sirf test ke liye")
    ap.add_argument("--only", default="", help="comma separated: sirf in companies ki files, jaise Woodward,Nordson")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--keep-junk", action="store_true", help="exhibit lists waghera bhi daalo")
    ap.add_argument("--no-section", action="store_true", help="section ka naam embedding mein mat daalo")
    a = ap.parse_args()
    print("\n====================================")
    print(" Embedding Index Builder Started")
    print("====================================\n")
    EmbeddingIndexBuilder(a.model, a.batch, include_section=not a.no_section).build(a.chunks, a.out, a.only, a.keep_junk)