import re

class FinancialTableAnalyzer:

    def __init__(self, section_name, text):
        self.section_name = section_name
        self.text = text

    def detect_table(self):
        """
        Enterprise table detector
        """

        score = 0

        lines = [
            l.strip()
            for l in self.text.split("\n")
            if l.strip()
        ]

        # ---------- Financial numbers ----------
        number_lines = 0

        for line in lines:

            if len(re.findall(r"\d", line)) >= 3:
                number_lines += 1

        if number_lines >= 3:
            score += 3

        # ---------- Currency ----------
        if "$" in self.text:
            score += 2

        # ---------- Multiple years ----------
        years = re.findall(r"\b20\d{2}\b", self.text)

        if len(set(years)) >= 2:
            score += 2

        # ---------- Accounting formatting ----------
        accounting = re.findall(r"\(?[\d,]+\)?", self.text)

        if len(accounting) >= 5:
            score += 2

        # ---------- Statement headings ----------
        heading = self.section_name.upper()

        statement_words = [
            "BALANCE SHEET",
            "STATEMENTS",
            "STATEMENT",
            "CASH FLOWS",
            "OPERATIONS",
            "EQUITY",
            "REVENUE",
            "DEBT",
            "SEGMENT",
            "INVENTORY",
            "LEASE",
            "FAIR VALUE",
            "TAX",
            "SHAREHOLDERS",
        ]

        if any(word in heading for word in statement_words):
            score += 2

        return score >= 5