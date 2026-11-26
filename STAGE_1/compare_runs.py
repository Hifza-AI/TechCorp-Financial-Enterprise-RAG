"""
compare_runs.py  -  do eval runs ka muqabla, sawal-ba-sawal (ablation: SQL [D1] ke baghair vs saath).

    python STAGE_1/compare_runs.py STAGE_1/eval/cmp_gptoss120b_v3.json STAGE_1/eval/cmp_gptoss120b_sql.json

Dikhata hai: dono mein judge kitna sahi, number sawal kitne sahi, faithfulness -- aur sirf woh sawal jin ka
judge badla (behtar / bura). Sawal dono runs mein mile tabhi gine jaate hain (barabar muqabla).
"""
import json
import math
import sys
from pathlib import Path

RANK = {"correct": 2, "partial": 1, "wrong": 0}


def load(p):
    return {d["id"]: d for d in json.loads(Path(p).read_text(encoding="utf-8"))}


def wilson(k, n, z=1.96):
    if not n:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def verdict(d):
    return (d.get("judge") or {}).get("verdict")


def stats(rows):
    j = [d for d in rows if verdict(d) in RANK]
    ok = sum(verdict(d) == "correct" for d in j)
    faith = sum(not (d.get("judge") or {}).get("unsupported") for d in j)
    num = [d for d in rows if d.get("number_match") is not None]
    nok = sum(bool(d["number_match"]) for d in num)
    return len(j), ok, faith, len(num), nok


def line(label, k, n):
    lo, hi = wilson(k, n)
    return f"  {label:<34}{k:>4}/{n:<4}{100 * k / n if n else 0:6.1f}%   (95% CI {100 * lo:.1f}-{100 * hi:.1f})"


def main():
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    a, b = load(sys.argv[1]), load(sys.argv[2])
    ids = sorted(set(a) & set(b), key=lambda i: int(i[1:]))
    ids = [i for i in ids if verdict(a[i]) in RANK and verdict(b[i]) in RANK]
    if not ids:
        raise SystemExit("Dono runs mein judge kiye hue mushtarka sawal nahi mile.")
    A, B = [a[i] for i in ids], [b[i] for i in ids]
    print(f"\nMUQABLA ({len(ids)} mushtarka sawal, dono ka judge hua)\n  A = {sys.argv[1]}\n  B = {sys.argv[2]}\n")
    for name, rows in (("A", A), ("B", B)):
        n, ok, faith, nn, nok = stats(rows)
        print(f"{name}:")
        print(line("judge: correct", ok, n))
        print(line("faithfulness (koi unsupported nahi)", faith, n))
        if nn:
            print(line("number sawal: sahi number jawab mein", nok, nn))
    sd = [i for i in ids if b[i].get("structured")]
    print(f"\nB mein [D1] (SQL) wale sawal: {len(sd)}" + (f"  ({', '.join(sd)})" if sd else ""))
    if sd:
        sa = sum(verdict(a[i]) == "correct" for i in sd)
        sb = sum(verdict(b[i]) == "correct" for i in sd)
        print(f"  in sawalon par judge correct:  A {sa}/{len(sd)}  ->  B {sb}/{len(sd)}")
    better = [i for i in ids if RANK[verdict(b[i])] > RANK[verdict(a[i])]]
    worse = [i for i in ids if RANK[verdict(b[i])] < RANK[verdict(a[i])]]
    print(f"\nBehtar hue ({len(better)}):")
    for i in better:
        print(f"  {i:<5} {verdict(a[i])} -> {verdict(b[i])}  {'[D1] ' if b[i].get('structured') else ''}{a[i]['question'][:70]}")
    print(f"Bure hue ({len(worse)}):")
    for i in worse:
        print(f"  {i:<5} {verdict(a[i])} -> {verdict(b[i])}  {'[D1] ' if b[i].get('structured') else ''}{a[i]['question'][:70]}")
    print()


if __name__ == "__main__":
    main()