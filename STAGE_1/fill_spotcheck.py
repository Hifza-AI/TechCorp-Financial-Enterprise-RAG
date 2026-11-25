"""
fill_spotcheck.py  -  insaani spot-check ke faisle (PDF dekh kar, 25 Nov 2026) STAGE_1/eval/spotcheck.csv mein bharo.
Har row Hifza ne PDF khol kar dekhi; yahan sirf wahi likha hai jo screenshot / PDF mein mila.

    python STAGE_1/fill_spotcheck.py
    python STAGE_1/spotcheck.py report
"""
import csv
import sys
from pathlib import Path

PATH = Path(sys.argv[1] if len(sys.argv) > 1 else "STAGE_1/eval/spotcheck.csv")

# id: (pdf_value, verdict, comment)
DONE = {
    "Q01": ("123,249", "Y", "p67"),
    "Q02": ("52,868", "Y", "p96, thousands"),
    "Q03": ("(1.61)", "Y", "p59, net loss year"),
    "Q04": ("(364)", "Y", "p45 total capex 364 = 360 + 4 real estate; cash flow line alone 360"),
    "Q05": ("111,820", "N", "ours 118,861 = FY2020 value (next year's filing); bug found and fixed"),
    "Q06": ("472.8", "Y", "p46"),
    "Q07": ("619,003", "Y", "p53"),
    "Q08": ("5,933", "Y", "p62"),
    "Q09": ("47,344", "Y", "p97"),
    "Q10": ("(3,226)", "Y", "p58; HD calls it Fiscal 2023 (ends Jan 2024), we label FY2024"),
    "Q11": ("141,925", "Y", "p55; no total liabilities line, = total liab & equity 133,625 - equity (8,300)"),
    "Q12": ("347.4", "Y", "p60"),
    "Q13": ("10,278", "Y", "p67; reported 2017 (restated under ASC 606 = 10,344)"),
    "Q14": ("8,885", "Y", "p28 MD&A results of operations"),
    "Q15": ("47,087", "Y", "p130, thousands"),
    "Q16": ("9,478.0", "Y", "p85; = 10,241.6 - temporary equity 129.1 - equity 634.5"),
    "Q17": ("290,164", "Y", "p34, thousands"),
    "Q18": ("61,491", "Y", "p74, consolidated 2016 column"),
    "Q19": ("13.64", "Y", "p51"),
    "Q20": ("82.1", "Y", "p58"),
    "Q21": ("5,007", "Y", "attributable to Costco (incl. NCI = 5,079)"),
    "Q22": ("613.2", "Y", "p57; total (continuing only = 613.6)"),
    "Q23": ("(1,574)", "Y", "p43; Dell Technologies stockholders' deficit (excl. NCI)"),
    "Q24": ("1.14", "Y", "p38; after 50-for-1 split in 2024"),
    "Q25": ("1,635", "Y", "p33; common stockholders (incl. NCI = 1,664)"),
    "Q26": ("78,745", "N", "p32 thousands; ours = 2015 value 72,945 (52/53-week year); bug found and fixed"),
    "Q27": ("(5,543)", "Y", "p66; attributable to Chevron (incl. NCI = 5,561)"),
    "Q28": ("20,156,447", "Y", "p45, thousands"),
    "Q29": ("743", "Y", "p137 historical summary; Colgate shareholders' equity (excl. NCI)"),
    "Q30": ("1,929.8", "N", "p91; no GAAP operating income line; SEC tag = total segment operating earnings"),
    "Q31": ("1,039,905", "N", "p34 thousands; ours = 2015 value 996,111; same bug as Q26, fixed"),
    "Q32": ("0.40", "Y", "p76"),
    "Q33": ("2,278", "Y", "p28; attributable to K-C (incl. NCI = 2,319)"),
    "Q34": ("4,634", "Y", "p97"),
    "Q35": ("105,590", "Y", "p78; Intel stockholders' equity (incl. NCI = 109,965)"),
    "Q36": ("14,476", "Y", "p74"),
    "Q37": ("9,752", "Y", "p82; FY ends Jan 30, 2022"),
    "Q38": ("(102)", "Y", "Colgate_2020 p137 summary table"),
    "Q39": ("5,865", "Y", "p78"),
    "Q40": ("5,275", "Y", "p75; FY ends Jan 29, 2021"),
    "Q41": ("2,581.8", "Y", "p40; FY ends July 31"),
    "Q42": ("13.77", "Y", "p108"),
    "Q43": ("(3,333)", "Y", "p81; 'Operating loss', FY ends Feb 2, 2018"),
    "Q44": ("47,806", "Y", "p64"),
    "Q45": ("144,623", "Y", "p42 thousands; attributable to Kennametal (incl. NCI = 150,201)"),
    "Q46": ("83,253", "Y", "Walmart shareholders' equity (incl. NCI = 91,891); FY ends Jan 31, 2022"),
    "Q47": ("7,637", "Y", "Item 6 selected financial data, 2018"),
    "Q48": ("22.1", "Y", "Item 1 business, R&D table"),
    "Q49": ("1,252,577", "Y", "p43 thousands; Kennametal shareholders' equity (incl. NCI = 1,291,247)"),
    "Q50": ("120,067", "Y", "p53; FY ends Jan 25, 2026"),
    "Q51": ("3,543.0", "Y", "p52"),
    "Q52": ("17,222,879", "Y", "p94 thousands; attributable to common (incl. NCI = 17,251,355)"),
    "Q53": ("180,838", "Y", "p67 EPS note, thousands"),
    "Q54": ("1,912", "Y", "p37; total (continuing only = 2,690)"),
    "Q55": ("3,219.8", "Y", "p53"),
    "Q56": ("3,103", "Y", "p83; labelled 'Technology and development'"),
    "Q57": ("24,156.4", "Y", "p55"),
    "Q58": ("275,235", "Y", "p38; total revenue (net sales = 269,912)"),
    "Q59": ("4,604", "N", "p84; ours 4,992 = FY2020 value; same bug as Q05, fixed"),
    "Q60": ("12,299", "Y", "p66; attributable to Chevron (incl. NCI = 12,485)"),
}


def main():
    if not PATH.exists():
        raise SystemExit(f"{PATH} nahi mila. Pehle: python STAGE_1/spotcheck.py make")
    with PATH.open(encoding="utf-8-sig") as f:
        rd = csv.DictReader(f)
        fields, rows = rd.fieldnames, list(rd)
    filled, missing = 0, []
    for r in rows:
        v = DONE.get(r["id"])
        if not v:
            missing.append(r["id"])
            continue
        r["pdf_value"], r["verdict"], r["comment"] = v
        filled += 1
    with PATH.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"{PATH}: {filled} rows bhari" + (f", in ka faisla nahi mila: {missing}" if missing else ""))


if __name__ == "__main__":
    main()