import re


class CleaningRules:

    # =====================================================
    # 1. URLs
    # =====================================================

    @staticmethod
    def remove_urls(text):
        return re.sub(
            r"https?://\S+",
            "",
            text,
            flags=re.IGNORECASE,
        )

    # =====================================================
    # 2. Browser timestamps
    # =====================================================

    @staticmethod
    def remove_timestamps(text):
        return re.sub(
            r"\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}(?::\d{2})?\s?(?:AM|PM)?\b",
            "",
            text,
            flags=re.IGNORECASE,
        )

    # =====================================================
    # 3. HTML files
    # =====================================================

    @staticmethod
    def remove_html_files(text):
        return re.sub(
            r"[A-Za-z0-9_-]+\.html?",
            "",
            text,
            flags=re.IGNORECASE,
        )

    # =====================================================
    # 4. SEC IDs
    # =====================================================

    @staticmethod
    def remove_sec_ids(text):
        return re.sub(
            r"\b[a-z]{2,10}-\d{8}\b",
            "",
            text,
            flags=re.IGNORECASE,
        )

    # =====================================================
    # 5. Checkboxes
    # =====================================================

    @staticmethod
    def remove_checkboxes(text):
        return re.sub(
            r"[☐☑☒□■✓✔]",
            "",
            text,
        )

    # =====================================================
    # 6. Isolated Table of Contents Navigation Link
    # =====================================================

    @staticmethod
    def remove_toc_link(text):
        return re.sub(
            r"^\s*Table\s+of\s+Contents\s*$",
            "",
            text,
            flags=re.IGNORECASE,
        )

    # =====================================================
    # 7. Unicode Normalize
    # =====================================================

    @staticmethod
    def normalize_unicode(text):

        replacements = {
            "•": "-",
            "–": "-",
            "—": "-",
            "…": "...",
            "\u00a0": " ",
            "\u201c": '"',
            "\u201d": '"',
            "\u2018": "'",
            "\u2019": "'",
            "\t": " ",
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        return text

    # =====================================================
    # 8. Normalize Spaces
    # =====================================================

    @staticmethod
    def normalize_spaces(text):

        text = re.sub(r"[ ]{2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text

    # =====================================================
    # MASTER CLEANER (per-string, position-independent stuff)
    #
    # NOTE: page-fraction removal ("6/65") aur footer-byline
    # ("Company | Year Form 10-K | N") REMOVED from here --
    # ye ab sirf _remove_browser_export_block() mein, 5-line
    # GROUP ke context mein hi handle hote hain. Standalone
    # global regex se exhibit dates ("6/6/14") corrupt ho rahi
    # thi -- isliye ye responsibility yahan se hata di gayi hai.
    # =====================================================

    @staticmethod
    def clean_text(text):

        if not isinstance(text, str):
            return text

        text = CleaningRules.remove_urls(text)
        text = CleaningRules.remove_timestamps(text)
        text = CleaningRules.remove_html_files(text)
        text = CleaningRules.remove_sec_ids(text)
        text = CleaningRules.remove_checkboxes(text)
        text = CleaningRules.remove_toc_link(text)
        text = CleaningRules.normalize_unicode(text)
        text = CleaningRules.normalize_spaces(text)

        return text.strip()

    # =====================================================
    # PAGE-LEVEL: browser-export 5-line footer block
    # =====================================================

    @staticmethod
    def _is_footer_byline(text):
        return bool(re.fullmatch(
            r".{1,80}\|\s*\d{4}\s*Form\s*10-K\s*\|\s*\d{1,4}",
            text.strip(),
            re.IGNORECASE,
        ))

    @staticmethod
    def _is_browser_timestamp(text):
        return bool(re.fullmatch(
            r"\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}(?::\d{2})?\s?(?:AM|PM)?",
            text.strip(),
            re.IGNORECASE,
        ))

    @staticmethod
    def _is_document_word(text):
        return text.strip().lower() == "document"

    @staticmethod
    def _is_url_line(text):
        return bool(re.search(r"https?://", text.strip(), re.IGNORECASE))

    @staticmethod
    def _is_page_fraction(text):
        return bool(re.fullmatch(r"\d{1,4}\s*/\s*\d{1,4}", text.strip()))

    @staticmethod
    def _remove_browser_export_block(lines):
        """
        Jab PDF browser se export/print hoti hai, ye 5 lines ka
        FIXED block page ke end mein aata hai:
          1) Company footer byline ("Apple Inc. | 2016 Form 10-K | 22")
          2) Browser timestamp ("5/16/26, 9:56 AM")
          3) "Document"
          4) SEC URL
          5) Page fraction ("25/84")

        SIRF tab poora block remove karte hain jab paanchon lines
        EK SAATH, isi order mein milti hain -- taake:
        - Exhibit dates ("6/6/14") jo akeli fraction-jaisi dikhti
          hain, kabhi na katein
        - Exhibit-index fractions ("6/65") bhi safe rahein jab tak
          unke saath baaki 4 lines na hon
        """

        result = []
        i = 0
        n = len(lines)

        while i < n:

            if i + 4 < n:

                texts = [
                    (lines[i + k].get("text") or "").strip()
                    for k in range(5)
                ]

                if (
                    CleaningRules._is_footer_byline(texts[0])
                    and CleaningRules._is_browser_timestamp(texts[1])
                    and CleaningRules._is_document_word(texts[2])
                    and CleaningRules._is_url_line(texts[3])
                    and CleaningRules._is_page_fraction(texts[4])
                ):
                    i += 5
                    continue

            result.append(lines[i])
            i += 1

        return result

    # =====================================================
    # PAGE-LEVEL: standalone digit / "Page N" footer
    # (last-line-only, position-independent)
    # =====================================================

    @staticmethod
    def _is_page_dict(data):
        return (
            isinstance(data, dict)
            and "lines" in data
            and "height" in data
        )

    @staticmethod
    def _clean_page_dict(page):

        lines = list(page.get("lines", []))

        # Step 1: 5-line browser-export block hatao (agar hai)
        lines = CleaningRules._remove_browser_export_block(lines)

        # Step 2: aakhri line agar standalone digit/"Page N" hai to hatao
        if lines:

            last_line = lines[-1]
            text = (last_line.get("text") or "").strip()

            is_footer_pattern = bool(
                re.fullmatch(r"(?:page\s*)?\d{1,4}", text, re.IGNORECASE)
            )

            if is_footer_pattern:
                lines = lines[:-1]

        cleaned_page = {}

        for key, value in page.items():
            if key == "lines":
                cleaned_page[key] = [
                    CleaningRules.clean_dict_structure(line) for line in lines
                ]
            elif key == "text":
                cleaned_page[key] = CleaningRules.clean_text(value)
            else:
                cleaned_page[key] = CleaningRules.clean_dict_structure(value)

        return cleaned_page

    # =====================================================
    # Recursive Cleaner
    # =====================================================

    @staticmethod
    def clean_dict_structure(data):

        if CleaningRules._is_page_dict(data):
            return CleaningRules._clean_page_dict(data)

        if isinstance(data, dict):

            cleaned = {}

            for key, value in data.items():
                if key == "text":
                    cleaned[key] = CleaningRules.clean_text(value)
                else:
                    cleaned[key] = CleaningRules.clean_dict_structure(value)

            return cleaned

        elif isinstance(data, list):
            return [CleaningRules.clean_dict_structure(item) for item in data]

        else:
            return data