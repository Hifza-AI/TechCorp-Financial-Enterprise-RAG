"""
fix_years.py  -  Jin files ka 'year' label andar ke data se match nahi karta, unka year theek karta hai.

Pehle sirf dekho (kuch nahi badalta):   python STAGE_1/fix_years.py
Phir sach mein badlo:                   python STAGE_1/fix_years.py --apply

Kya karta hai (--apply par):
  * Har badalne wali file ki asli copy pehle STAGE_1/chunks_backup_yearfix/ mein rakhta hai (wapas ja sakti ho).
  * Har chunk ka "year" badalta hai, aur table chunk ki pehli line ("... data for Home Depot 2018:") ka saal bhi.
  * File ka NAAM nahi badalta (Home Depot_2018_10K_chunks.json ka year 2017 ho sakta hai) -- sab kuch
    STAGE_1/eval/year_fixes.json mein likha jata hai.
  * Uske baad:  python STAGE_1/build_embeddings.py   -> sirf yehi files dobara embed hongi.

Faisla kaise:
  * Agar kisi company ki 70%+ files ek hi tarah khiski hon (jaise Home Depot: sab 1 saal aage), to company ki
    SAARI files par wahi farq lagta hai (company ka apna naam-rakhne ka tareeqa alag hai).
  * Warna sirf wahi file theek hoti hai jo alag hai (jaise Southwest 2026 -> 2025).
  * Agar theek karne se ek company mein ek saal do baar ban jaye, to kuch nahi badalta aur report karta hai.
"""
import argparse
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from check_data import data_year


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="STAGE_1/chunks")
    ap.add_argument("--apply", action="store_true", help="sach mein badlo (warna sirf dikhata hai)")
    ap.add_argument("--backup", default="STAGE_1/chunks_backup_yearfix")
    ap.add_argument("--log", default="STAGE_1/eval/year_fixes.json")
    a = ap.parse_args()

    root = Path(a.chunks)
    files = sorted(root.rglob("*_chunks.json"))
    if not files:
        raise SystemExit(f"Koi *_chunks.json nahi mili: {root.resolve()}")

    info = defaultdict(list)            # company -> [dict]
    for f in files:
        ch = json.load(open(f, encoding="utf-8"))
        if not ch:
            continue
        comp = str(ch[0].get("company", "")).strip()
        label = int(str(ch[0].get("year", "0")).strip() or 0)
        dy, why = data_year(ch)
        info[comp].append(dict(file=f, label=label, data=dy, why=why,
                               off=(dy - label) if dy is not None else None))

    plan, problems = [], []
    for comp, L in sorted(info.items()):
        offs = Counter(x["off"] for x in L if x["off"] not in (0, None))
        if not offs:
            continue
        common, k = offs.most_common(1)[0]
        if k >= 0.7 * len(L):
            new = {x["file"]: x["label"] + common for x in L}
            rule = f"company rule: sab files {common:+d}"
        else:
            new = {x["file"]: (x["label"] + x["off"] if x["off"] not in (0, None) else x["label"]) for x in L}
            rule = "sirf alag wali file"
        years = list(new.values())
        if len(set(years)) != len(years):
            dup = sorted(y for y, n in Counter(years).items() if n > 1)
            detail = "\n".join(f"        {Path(x['file']).name}: label {x['label']} -> data {x['data']}  [{x['why']}]"
                               for x in sorted(L, key=lambda z: z["label"]))
            problems.append(f"{comp}: theek karne se saal do baar banta ({dup}) -- kuch nahi badla. Files:\n{detail}")
            continue
        for x in L:
            if new[x["file"]] != x["label"]:
                plan.append(dict(company=comp, file=str(x["file"]), old=x["label"], new=new[x["file"]],
                                 rule=rule, evidence=x["why"]))

    if not plan and not problems:
        print("Sab theek hai -- koi year badalna nahi.")
        return
    print(f"{len(plan)} files ka year badlega:\n")
    for p in plan:
        print(f"  {Path(p['file']).name:42s} {p['old']} -> {p['new']}   [{p['rule']}]")
        print(f"      saboot: {p['evidence']}")
    for pr in problems:
        print("  ! " + pr)

    if not a.apply:
        print("\nAbhi kuch nahi badla. Theek lage to chalao:  python STAGE_1/fix_years.py --apply")
        return

    bk = Path(a.backup)
    for p in plan:
        f = Path(p["file"])
        dst = bk / f.relative_to(root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(f, dst)
        ch = json.load(open(f, encoding="utf-8"))
        old, new = p["old"], p["new"]
        for c in ch:
            c["year"] = str(new) if isinstance(c.get("year"), str) else new
            t = c.get("text") or ""
            if t.startswith("The following table shows"):
                first, _, rest = t.partition("\n")
                first = re.sub(rf"^(The following table shows .* data for .*?)\b{old}\b(\s*:)", rf"\g<1>{new}\2", first, count=1)
                c["text"] = first + ("\n" + rest if rest else "")
        tmp = f.with_suffix(".tmp")
        json.dump(ch, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        tmp.replace(f)
    log = Path(a.log)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nHo gaya. Backup: {bk.resolve()}\nLog: {log.resolve()}")
    print("Ab chalao:  python STAGE_1/build_embeddings.py   (sirf ye files dobara banengi)")


if __name__ == "__main__":
    main()