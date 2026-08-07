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
            r"\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}\s?(AM|PM)",
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
    # 5. Page Numbers
    # =====================================================

    @staticmethod
    def remove_page_numbers(text):

        text = re.sub(
            r"\b\d+\s*/\s*\d+\b",
            "",
            text,
        )

        return text

    # =====================================================
    # 6. Checkboxes
    # =====================================================

    @staticmethod
    def remove_checkboxes(text):
        return re.sub(
            r"[☐☑☒□■✓✔]",
            "",
            text,
        )

    # =====================================================
    # 7. Headers / Footers
    # =====================================================

    @staticmethod
    def remove_headers(text):

        patterns = [
            r"Table\s+of\s+Contents",
            r"Document",
        ]

        for p in patterns:

            text = re.sub(
                p,
                "",
                text,
                flags=re.IGNORECASE,
            )

        return text

    # =====================================================
    # 8. Unicode Normalize
    # =====================================================

    @staticmethod
    def normalize_unicode(text):

        replacements = {
            "•": "-",
            "–": "-",
            "—": "-",
            "…": "...",
            "\u00a0": " ",
            "“": '"',
            "”": '"',
            "‘": "'",
            "’": "'",
            "\t": " ",
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        return text

    # =====================================================
    # 9. Normalize Spaces
    # =====================================================

    @staticmethod
    def normalize_spaces(text):

        text = re.sub(
            r"[ ]{2,}",
            " ",
            text,
        )

        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text,
        )

        return text

    # =====================================================
    # MASTER CLEANER
    # =====================================================

    @staticmethod
    def clean_text(text):

        if not isinstance(text, str):
            return text

        text = CleaningRules.remove_urls(text)

        text = CleaningRules.remove_timestamps(text)

        text = CleaningRules.remove_html_files(text)

        text = CleaningRules.remove_sec_ids(text)

        text = CleaningRules.remove_page_numbers(text)

        text = CleaningRules.remove_checkboxes(text)

        text = CleaningRules.remove_headers(text)

        text = CleaningRules.normalize_unicode(text)

        text = CleaningRules.normalize_spaces(text)

        return text.strip()

    # =====================================================
    # Recursive Cleaner
    # =====================================================

    @staticmethod
    def clean_dict_structure(data):

        if isinstance(data, dict):

            cleaned = {}

            for key, value in data.items():

                if key == "text":

                    cleaned[key] = CleaningRules.clean_text(value)

                else:

                    cleaned[key] = CleaningRules.clean_dict_structure(value)

            return cleaned

        elif isinstance(data, list):

            return [
                CleaningRules.clean_dict_structure(item) for item in data
            ]

        else:

            return data