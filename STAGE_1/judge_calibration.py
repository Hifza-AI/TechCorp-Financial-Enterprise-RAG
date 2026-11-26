"""
judge_calibration.py  -  LLM judge ki jaanch: insaan (Hifza) ke faisle vs Qwen judge ke faisle.

    python STAGE_1/judge_calibration.py make     -> STAGE_1/eval/judge_review.html  (browser mein kholo)
    python STAGE_1/judge_calibration.py report   -> STAGE_1/eval/judge_labels.csv se agreement + kappa

Kyun: hamara 88% "LLM judge" ka number hai. Judge khud kitna sahi hai, ye insaan ke faislon se naapte hain.
Report mein likhte hain: "LLM judge agreed with human labels on X% (Cohen's kappa = Y)".

BLIND: review page par judge ka faisla aur exact-match NAHI dikhaya jata -- warna insaan us ki taraf jhuk jata hai.
Page par har sawal: sawal, sahi jawab (gold, PDF page), model ka jawab, aur jo sources model ko diye gaye.
Tum har jawab par: correct / partial / wrong, aur "gharhi hui baat" (sources mein nahi) ho to us ka nishan.
Progress browser mein khud save hoti hai; aakhir mein "Download labels CSV" -> STAGE_1/eval/judge_labels.csv
"""
import argparse
import csv
import html
import json
import math
from collections import Counter
from pathlib import Path

VERDICTS = ("correct", "partial", "wrong")


def load_run(path):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return [d for d in rows if (d.get("judge") or {}).get("verdict") in VERDICTS]


def make(run, out):
    rows = load_run(run)
    cards = []
    for d in rows:
        cited = []
        for s in d.get("sources", []):
            pages = "-".join(str(p) for p in (s.get("pages") or [])[:2]) or "database"
            head = f"[{s['tag']}] {s.get('company', '')} FY{s.get('year', '')} | p{pages} | {s.get('section', '')[-90:]}"
            body = html.escape(s.get("text", ""))
            used = f"[{s['tag']}]" in (d.get("answer") or "")
            cited.append(f"<details{' open' if used else ''}><summary>{html.escape(head)}"
                         f"{' <b>(jawab mein cite hua)</b>' if used else ''}</summary><pre>{body}</pre></details>")
        gold = html.escape(str(d.get("gold") or "-"))
        cards.append(f"""
<section class="card" id="{d['id']}" data-id="{d['id']}">
  <h2>{d['id']} <span class="q">{html.escape(d['question'])}</span></h2>
  <div class="gold"><b>Sahi jawab (gold):</b> {gold} <span class="pg">(PDF page {html.escape(str(d.get('gold_page') or '?'))},
     {html.escape(str(d.get('company') or ''))} {html.escape(str(d.get('year') or ''))})</span></div>
  <div class="ans"><b>Model ka jawab:</b><pre>{html.escape(d.get('answer') or '')}</pre></div>
  <div class="src"><b>Model ko diye gaye sources</b> (cite hue khule hain):{''.join(cited)}</div>
  <div class="lab">
    <label><input type="radio" name="v_{d['id']}" value="correct"> correct</label>
    <label><input type="radio" name="v_{d['id']}" value="partial"> partial</label>
    <label><input type="radio" name="v_{d['id']}" value="wrong"> wrong</label>
    <label class="uns"><input type="checkbox" name="u_{d['id']}"> gharhi hui baat (sources mein nahi)</label>
    <input class="note" name="n_{d['id']}" placeholder="note (zaroori nahi)">
  </div>
</section>""")
    page = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>Judge calibration -- {len(rows)} jawab</title>
<style>
body{{font-family:system-ui,Segoe UI,sans-serif;max-width:980px;margin:0 auto;padding:16px 16px 90px;background:#fafafa;color:#222}}
.card{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:14px 18px;margin:16px 0}}
.card.done{{border-left:6px solid #2e7d32}}
h2{{font-size:16px;margin:0 0 8px}} .q{{font-weight:400}}
pre{{white-space:pre-wrap;font-family:inherit;margin:6px 0;line-height:1.45}}
.gold{{background:#eef6ee;padding:8px;border-radius:6px}} .pg{{color:#666;font-size:13px}}
.ans{{margin:10px 0}} .ans pre{{background:#f4f6fb;padding:8px;border-radius:6px}}
details{{margin:4px 0;font-size:13px}} details pre{{background:#fbfbf6;padding:6px;border-radius:4px}}
.lab{{margin-top:10px;display:flex;gap:16px;align-items:center;flex-wrap:wrap}}
.lab label{{cursor:pointer}} .uns{{color:#a33}} .note{{flex:1;min-width:200px;padding:4px}}
.bar{{position:fixed;bottom:0;left:0;right:0;background:#222;color:#fff;padding:10px;text-align:center}}
.bar button{{margin-left:12px;padding:6px 14px;font-size:14px}}
.help{{background:#fff8e1;border:1px solid #f0d58a;padding:10px 14px;border-radius:8px;font-size:14px}}
</style></head><body>
<h1>Judge calibration ({len(rows)} jawab)</h1>
<div class="help"><b>Kaise:</b> har jawab ko sahi jawab (gold) aur sources se milao.
<b>correct</b> = sawal ka jawab sahi aur poora. <b>partial</b> = sahi magar adhoora, ya ek chhoti ghalti.
<b>wrong</b> = ghalat ya "nahi mila" jab ke jawab sources mein tha.
<b>gharhi hui baat</b> = jawab mein koi aisi baat jo diye gaye sources mein nahi (chahe sahi lage).
Gold sirf ek misaal hai -- doosre alfaaz mein sahi jawab bhi <b>correct</b> hai. Progress khud save hoti hai.</div>
{''.join(cards)}
<div class="bar"><span id="prog"></span><button onclick="dl()">Download labels CSV</button></div>
<script>
const KEY="judge_calib_v1";
let st={{}};try{{st=JSON.parse(localStorage.getItem(KEY)||"{{}}")}}catch(e){{st={{}}}}
function save(){{try{{localStorage.setItem(KEY,JSON.stringify(st))}}catch(e){{}}prog()}}
function prog(){{const cs=document.querySelectorAll(".card");let n=0;cs.forEach(c=>{{const d=st[c.dataset.id];
 const ok=d&&d.v;c.classList.toggle("done",!!ok);if(ok)n++}});document.getElementById("prog").textContent=n+" / "+cs.length+" ho gaye"}}
document.querySelectorAll(".card").forEach(c=>{{const id=c.dataset.id,d=st[id]||{{}};
 c.querySelectorAll("input").forEach(i=>{{
  if(i.type==="radio"&&d.v===i.value)i.checked=true;
  if(i.type==="checkbox"&&d.u)i.checked=true;
  if(i.classList.contains("note")&&d.n)i.value=d.n;
  i.addEventListener("change",()=>{{const x=st[id]||{{}};
   if(i.type==="radio")x.v=i.value; if(i.type==="checkbox")x.u=i.checked; if(i.classList.contains("note"))x.n=i.value;
   st[id]=x;save()}});
 }})}});
prog();
function dl(){{let rows=[["id","human_verdict","human_unsupported","note"]];
 document.querySelectorAll(".card").forEach(c=>{{const d=st[c.dataset.id]||{{}};
  rows.push([c.dataset.id,d.v||"",d.u?1:0,(d.n||"").replace(/"/g,"'")])}});
 const csv=rows.map(r=>r.map(x=>'"'+x+'"').join(",")).join("\\n");
 const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([csv],{{type:"text/csv"}}));
 a.download="judge_labels.csv";a.click()}}
</script></body></html>"""
    out.write_text(page, encoding="utf-8")
    print(f"{out}: {len(rows)} jawab. Browser mein kholo (double-click), faisle lagao, aakhir mein "
          f"'Download labels CSV' -> STAGE_1/eval/judge_labels.csv")


def kappa(pairs, labels):
    n = len(pairs)
    if not n:
        return float("nan")
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[l] * cb[l] for l in labels) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def wilson(k, n, z=1.96):
    if not n:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def report(run, labels_path):
    if not labels_path.exists():
        raise SystemExit(f"{labels_path} nahi mila. Review page se 'Download labels CSV' karke yahan rakho.")
    runs = {d["id"]: d for d in load_run(run)}
    with labels_path.open(encoding="utf-8-sig") as f:
        lab = {r["id"]: r for r in csv.DictReader(f) if r.get("human_verdict") in VERDICTS}
    ids = [i for i in sorted(lab, key=lambda x: int(x[1:])) if i in runs]
    if not ids:
        raise SystemExit("Koi labelled jawab nahi mila.")
    pairs = [(lab[i]["human_verdict"], runs[i]["judge"]["verdict"]) for i in ids]
    n = len(pairs)
    agree = sum(a == b for a, b in pairs)
    bin_pairs = [(a == "correct", b == "correct") for a, b in pairs]
    bagree = sum(a == b for a, b in bin_pairs)
    lo, hi = wilson(agree, n)
    blo, bhi = wilson(bagree, n)
    print(f"\nJUDGE CALIBRATION -- {n} jawab (insaan vs Qwen judge)\n")
    print(f"  3 qism (correct/partial/wrong) mutabiqat: {agree}/{n} = {100 * agree / n:.1f}%  (95% CI {100*lo:.1f}-{100*hi:.1f})")
    print(f"  Cohen's kappa (3 qism):                    {kappa(pairs, VERDICTS):.2f}")
    print(f"  Sirf correct vs baqi mutabiqat:           {bagree}/{n} = {100 * bagree / n:.1f}%  (95% CI {100*blo:.1f}-{100*bhi:.1f})")
    print(f"  Cohen's kappa (correct vs baqi):           {kappa(bin_pairs, (True, False)):.2f}")
    hc = sum(a == "correct" for a, _ in pairs)
    jc = sum(b == "correct" for _, b in pairs)
    print(f"\n  Insaan ke mutabiq correct: {hc}/{n} = {100 * hc / n:.1f}%     Judge ke mutabiq: {jc}/{n} = {100 * jc / n:.1f}%")
    print("\n  Confusion (rows = insaan, cols = judge):")
    print("              " + "".join(f"{v:>10}" for v in VERDICTS))
    for h in VERDICTS:
        print(f"    {h:<10}" + "".join(f"{sum(1 for a, b in pairs if a == h and b == j):>10}" for j in VERDICTS))
    hu = {i for i in ids if lab[i].get("human_unsupported") in ("1", "true", "True")}
    ju = {i for i in ids if runs[i]["judge"].get("unsupported")}
    print(f"\n  Faithfulness (koi gharhi baat nahi): insaan {n - len(hu)}/{n}, judge {n - len(ju)}/{n};"
          f" dono ne pakra {len(hu & ju)}, sirf insaan {sorted(hu - ju)}, sirf judge {sorted(ju - hu)}")
    dis = [i for i, (a, b) in zip(ids, pairs) if a != b]
    if dis:
        print(f"\n  Ikhtilaf ({len(dis)}):")
        for i in dis:
            note = lab[i].get("note") or ""
            print(f"    {i:<5} insaan {lab[i]['human_verdict']:<8} judge {runs[i]['judge']['verdict']:<8} "
                  f"{runs[i]['question'][:55]}" + (f"  | {note[:60]}" if note else ""))
    print()


def main():
    ap = argparse.ArgumentParser(description="LLM judge vs insaani faisle")
    ap.add_argument("--run", default="STAGE_1/eval/cmp_gptoss120b_v3.json")
    ap.add_argument("--dir", default="STAGE_1/eval")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("make")
    sub.add_parser("report")
    a = ap.parse_args()
    d = Path(a.dir)
    if a.cmd == "make":
        make(a.run, d / "judge_review.html")
    else:
        report(a.run, d / "judge_labels.csv")


if __name__ == "__main__":
    main()