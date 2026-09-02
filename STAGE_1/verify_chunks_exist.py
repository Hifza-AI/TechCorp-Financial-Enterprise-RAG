"""
verify_chunks_exist.py

DIAGNOSTIC (not a retriever test) -- for each known-failed query from
the last retriever run, this checks the RAW metadata.pkl directly
(bypassing dense/BM25/RRF ranking entirely) to answer ONE question:

    Does a chunk containing the correct answer actually EXIST
    in the corpus for this company/year, regardless of whether
    the retriever surfaced it?

This separates two very different problems that look identical from
the retriever's output alone:
  (a) RETRIEVER BUG -- the right chunk exists, but ranking/fusion
      failed to surface it in top-3.
  (b) CONTENT GAP -- the right chunk was never built at all (e.g.
      fragmented into unusable pieces during chunk_builder).

For each check, prints:
  - Whether a matching candidate chunk was found at all
  - Its FULL text (not truncated) so you can see if the actual
    answer is really in there
  - Its exact index position, so a follow-up check can see what
    rank the retriever's dense/BM25/RRF scoring gave it

Saves ALL output to verify_chunks_results.txt (not just terminal --
terminal truncates on long runs; the file will not).

USAGE:
    python verify_chunks_exist.py
"""

import pickle
import re
from pathlib import Path


STORE_DIR = Path("STAGE_1/vector_store")
OUTPUT_FILE = "verify_chunks_results.txt"


# Each entry: (label, company, year, chunk_type filter or None,
#              list of keyword-patterns that MUST all appear in the
#              text for it to count as "the right chunk")
CHECKS = [
    (
        "Intel net income 2024",
        "Intel", 2024, "table",
        [r"\bNet income\b", r"2024"],
    ),
    (
        "Intel total assets (balance sheet)",
        "Intel", None, "table",
        [r"\bTotal assets\b"],
    ),
    (
        "Meta Family of Apps revenue",
        "META", None, "table",
        [r"Family of Apps", r"\bRevenue\b"],
    ),
    (
        "Meta lease liability maturity schedule",
        "META", None, None,
        [r"[Ll]ease", r"[Mm]aturit"],
    ),
    (
        "Nvidia inventory valuation policy",
        "Nvidia", None, "text",
        [r"[Ii]nventor", r"(valuation|first-in|FIFO|net realizable)"],
    ),
    (
        "Intel manufacturing risk factors",
        "Intel", None, "text",
        [r"[Mm]anufactur", r"[Rr]isk"],
    ),
    (
        "Meta competitive risks",
        "META", None, "text",
        [r"[Cc]ompetit", r"[Rr]isk"],
    ),
    (
        "Google latest revenue (2025)",
        "Google", 2025, "table",
        [r"\bRevenue", r"\d{2,3},\d{3}"],  # a real dollar figure
    ),
    (
        "Nvidia most recent quarterly revenue",
        "Nvidia", None, "table",
        [r"[Qq]uarter", r"[Rr]evenue"],
    ),
]


def main():

    with open(STORE_DIR / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    lines_out = []

    def log(msg=""):
        print(msg)
        lines_out.append(msg)

    log(f"Total chunks in corpus: {len(metadata)}")
    log("=" * 80)

    for label, company, year, chunk_type, patterns in CHECKS:

        log(f"\n{'=' * 80}")
        log(f"CHECK: {label}")
        log(f"  company={company!r}  year={year!r}  chunk_type={chunk_type!r}")
        log(f"  required patterns: {patterns}")
        log("-" * 80)

        matches = []

        for idx, chunk in enumerate(metadata):

            if company and chunk.get("company") != company:
                continue

            if year and chunk.get("year") != year:
                continue

            if chunk_type and chunk.get("chunk_type") != chunk_type:
                continue

            text = chunk.get("text", "")

            if all(re.search(p, text) for p in patterns):
                matches.append((idx, chunk))

        if not matches:
            log("  RESULT: NOT FOUND -- no chunk matches these patterns at all.")
            log("  ==> Likely a CONTENT GAP, not a retriever bug.")
        else:
            log(f"  RESULT: FOUND {len(matches)} matching chunk(s).")
            log("  ==> Content exists. If retriever didn't surface it, "
                 "that's a RETRIEVER/RANKING bug.")
            for idx, chunk in matches[:3]:
                log(f"\n  --- chunk index {idx} ---")
                log(f"  company={chunk.get('company')} year={chunk.get('year')} "
                    f"chunk_type={chunk.get('chunk_type')}")
                log(f"  section_path={chunk.get('section_path')}")
                log(f"  FULL TEXT:")
                log(f"  {chunk.get('text','')}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_out))

    print(f"\n\nFull results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()