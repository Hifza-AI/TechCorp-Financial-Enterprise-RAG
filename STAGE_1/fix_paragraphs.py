import json
import re
from pathlib import Path


def _is_page_progress_indicator(text):
    return bool(re.fullmatch(r"\d{1,4}\s*/\s*\d{1,4}", text.strip()))


def _is_standalone_page_footer_number(text):
    """
    Some companies (e.g. Alphabet/Google) print their OWN internal
    page number at the bottom of every page, separate from the PDF's
    overall page count -- e.g. a lone "53." or "88." sitting by
    itself with nothing else on the line. This is never real 10-K
    content, just a footer artifact -- but unlike the repeated-
    footer-text case (which _normalize_for_repeat_check already
    catches by stripping the trailing number and comparing), THIS
    footer is the number and NOTHING else, so there's no surrounding
    text left to match against -- every single occurrence is a
    unique, one-off string ("53." only ever appears once, "54." only
    once, etc.), so the frequency-based boilerplate check can't catch
    it either. Confirmed on Google/Alphabet 2025: 79 separate
    standalone "N." paragraphs (numbers 1 through 98) scattered
    throughout the Notes to Financial Statements section.
    A real paragraph is never JUST a bare number with a period and
    nothing else, so this is safe to drop unconditionally.
    """
    return bool(re.fullmatch(r"\d{1,4}\.", text.strip()))


def _normalize_for_repeat_check(text):
    return re.sub(r"\d+\s*$", "#", text.strip())


def fix_paragraph_file(data):

    total_pages = len(data["pages"])

    text_counts = {}

    for page in data["pages"]:
        seen = set()
        for block in page["blocks"]:
            if block["block_type"] != "paragraph":
                continue
            text = block["text"].strip()
            if not text:
                continue
            normalized = _normalize_for_repeat_check(text)
            if normalized not in seen:
                seen.add(normalized)
                text_counts[normalized] = text_counts.get(normalized, 0) + 1

    threshold = max(5, total_pages * 0.3)
    boilerplate = {t for t, c in text_counts.items() if c >= threshold}

    numbering_pattern = re.compile(
        r"^(Item\s+\d+[A-Za-z]?\.?|PART\s+[IVXLCDM]+\.?|[IVXLCDM]+\.|[A-Za-z]?\d+\.)$",
        re.IGNORECASE,
    )

    fixed_pages = []

    for page in data["pages"]:

        new_blocks = []

        for block in page["blocks"]:

            text = block["text"].strip()

            if block["block_type"] == "paragraph":

                if _is_page_progress_indicator(text):
                    continue

                if _is_standalone_page_footer_number(text):
                    continue

                normalized = _normalize_for_repeat_check(text)
                if normalized in boilerplate:
                    continue

            if block["block_type"] == "heading":

                words = text.split()

                looks_like_fragment = (
                    len(words) <= 4
                    and text.endswith(".")
                    and not numbering_pattern.match(text)
                )

                if looks_like_fragment:
                    block = dict(block)
                    block["block_type"] = "paragraph"
                    block.pop("level", None)

            new_blocks.append(block)

        fixed_pages.append({
            "page_number": page["page_number"],
            "blocks": new_blocks,
        })

    data["pages"] = fixed_pages

    return data


def process_folder(input_dir="STAGE_1/paragraphs", output_dir="STAGE_1/paragraphs_fixed"):

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    total_footer_removed = 0
    total_fragments_fixed = 0

    for company_dir in sorted(input_dir.iterdir()):

        if not company_dir.is_dir():
            continue

        out_company_dir = output_dir / company_dir.name
        out_company_dir.mkdir(parents=True, exist_ok=True)

        for json_file in sorted(company_dir.glob("*_paragraphs.json")):

            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            before_p = sum(
                1 for p in data["pages"] for b in p["blocks"]
                if b["block_type"] == "paragraph"
            )
            before_h = sum(
                1 for p in data["pages"] for b in p["blocks"]
                if b["block_type"] == "heading"
            )

            fixed = fix_paragraph_file(data)

            after_p = sum(
                1 for p in fixed["pages"] for b in p["blocks"]
                if b["block_type"] == "paragraph"
            )
            after_h = sum(
                1 for p in fixed["pages"] for b in p["blocks"]
                if b["block_type"] == "heading"
            )

            out_file = out_company_dir / json_file.name

            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(fixed, f, indent=4, ensure_ascii=False)

            print(
                f"{json_file.name}: "
                f"headings {before_h}->{after_h}, "
                f"paragraphs {before_p}->{after_p}"
            )

    print("\nDone. Fixed files saved to:", output_dir)


if __name__ == "__main__":
    process_folder()