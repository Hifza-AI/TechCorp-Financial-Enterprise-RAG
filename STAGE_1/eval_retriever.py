"""
eval_retriever.py  -  Saare sawal retriever par chalata hai aur khud check karta hai ke sahi jawab aaya ya nahi.

Chalao:   python STAGE_1/eval_retriever.py              (retriever.py ke FEATURES jaise hain, waise)
          python STAGE_1/eval_retriever.py --ablation   (har sudhaar ka asar alag naapta hai: 6 dafa, ~2-3 min)
Pehle:    python STAGE_1/make_eval_set.py   (sawal banane ke liye)

Banata (STAGE_1/eval/ mein):
  summary.txt     -> score card (Hit@1, Hit@5, MRR har qism ke liye + kamzor companies). Ye mujhe bhejna.
  ablation.txt    -> (sirf --ablation par) har sudhaar on/off karke score ki table. Ye bhi bhejna.
  results.csv     -> har sawal ki ek line (Excel mein khulti hai)
  failures.html   -> sirf FAIL hue sawal, cards ki shakal mein. Browser mein kholo, har card par
                     "asal wajah" chuno (retriever / chunk / heading / naam-saal / sawal galat), phir CSV download.

Score ka matlab:
  Hit@1  = sahi chunk pehle number par aaya       Hit@5 = sahi chunk top 5 mein aaya (Gemini ko 5 milenge)
  MRR    = sahi chunk kitna upar tha (1 = hamesha pehla, 0.5 = aksar doosra)
  out_of_scope ke liye "sahi" = retriever ne "nahi mila" kaha.

v2 checker (pehle wala kuch sahi jawabon ko bhi FAIL keh deta tha -- 66 fail haath se dekh kar pakda):
  section sawal  -> wahi section, YA us ke andar ki sub-heading (child), YA wahi heading isi filing mein kahin
                    aur (jaise "Productivity and Business Processes" Item 1 aur Item 7 dono mein) = sahi.
  natural sawal  -> section ka naam match kare, YA heading text ke shuru mein chipki ho (Capital alfaaz
                    se shuru: Chevron "Operating Segments and Geographic Data Although each ...") = sahi.
  table sawal    -> wahi number us company ke chunk mein (pehle jaisa).
"""
import argparse
import csv
import html
import json
import re
import time
from collections import defaultdict
from pathlib import Path

from retriever import FEATURES, Retriever

CATS = ["table_number", "section_topic", "risk_factor", "natural_topic"]
NUM_PREFIX = re.compile(r"^(?:(?:item|note)\s*\d+[a-z]?\s*[.:-]?\s*|\(?[a-z0-9]{1,3}\)\s*|\d+[.)-]\s*)", re.I)
# Text ke shuru mein Capital alfaaz ka tukda = chipki hui heading ("Operating Segments and Geographic Data ...")
_W = r"(?:[A-Z][\w,&'’\-]*|and|of|the|for|to|in|on|or|&)"
TEXT_HEADING = re.compile(r"[A-Z][\w,&'’\-]*(?:\s+" + _W + r"){1,9}")


def norm(t):
    return re.sub(r"\s+", " ", (t or "").replace("’", "'")).strip().lower()


def leaf_norm(s):
    return norm(NUM_PREFIX.sub("", (s or "").strip()).strip(" .:-"))


def value_re(v):
    return re.compile(r"(?<![\d,.])" + re.escape(v) + r"(?!\d)(?!,\d)(?!\.\d)")


def is_hit(q, r):
    g = q["gold"]
    if r["company"] != q["company"]:
        return False
    if g["mode"] == "value":                      # pichle/agle saal ki 10-K mein bhi wahi number ho to bhi sahi
        return bool(value_re(g["value"]).search(r["text"] or ""))
    if q.get("year") and int(r["year"]) != int(q["year"]):
        return False
    sec = r["section_path"] or ""
    if g["mode"] == "section":
        gs = g["section"]
        if sec == gs or sec.startswith(gs + " > "):                      # wahi section, ya us ke andar
            return True
        gl = leaf_norm(gs.split(" > ")[-1])
        if len(gl) >= 8 and any(gl in leaf_norm(p) for p in sec.split(" > ")):   # wahi heading isi filing mein
            return True
        return bool(g.get("anchor") and g["anchor"] in norm(r["text"]))
    if g["mode"] == "section_regex":
        if re.search(g["regex"], sec, re.I):
            return True
        head = TEXT_HEADING.match(r["text"] or "")             # heading text ke shuru mein chipki ho
        return bool(head and re.search(g["regex"], head.group(0), re.I))
    return False


HINT_LABEL = {
    "oos_answered": "Anjaan/bekaar sawal par bhi jawab de diya (hona chahiye tha 'nahi mila')",
    "nothing_found": "Kuch nahi mila (naam/saal ka label ya filter)",
    "wrong_company": "Galat company aayi (company pehchan)",
    "wrong_year": "Galat saal aaya (saal pehchan ya label)",
    "near_miss": "Qareeb tha: sahi chunk #6-10 par (ranking)",
    "not_in_top10": "Sahi company/saal, lekin sahi chunk top 10 mein nahi (ranking ya chunk/table)",
}


def diagnose(q, resp, rank):
    """(type, text) -- type khaali = koi masla nahi."""
    res = resp["results"]
    if q["category"] == "out_of_scope":
        if not resp["matched"]:
            return "", ""
        return "oos_answered", f"{HINT_LABEL['oos_answered']}: {res[0]['company']} {res[0]['year']}"
    if not resp["matched"]:
        return "nothing_found", f"{HINT_LABEL['nothing_found']}: {resp['reason']}"
    if rank and rank <= 5:
        return "", ""
    top = res[0]
    if top["company"] != q["company"]:
        return "wrong_company", f"{HINT_LABEL['wrong_company']}: {top['company']}"
    if q.get("year") and int(top["year"]) != int(q["year"]):
        return "wrong_year", f"{HINT_LABEL['wrong_year']}: {top['year']}"
    if rank:
        return "near_miss", f"{HINT_LABEL['near_miss']} -- yahan #{rank}"
    return "not_in_top10", HINT_LABEL["not_in_top10"]


def run(r, qs, k, label=""):
    t0 = time.time()
    rows = []
    for i, q in enumerate(qs, 1):
        resp = r.search(q["question"], top_k=k)
        rank = None
        if q["category"] != "out_of_scope":
            for j, res in enumerate(resp["results"], 1):
                if is_hit(q, res):
                    rank = j
                    break
        ok_oos = q["category"] == "out_of_scope" and not resp["matched"]
        htype, hint = diagnose(q, resp, rank)
        rows.append(dict(q=q, resp=resp, rank=rank, ok_oos=ok_oos, htype=htype, hint=hint))
        if i % 100 == 0 or i == len(qs):
            print(f"  {label}{i}/{len(qs)}  ({time.time() - t0:.0f} sec)")
    return rows, time.time() - t0


def score(R):
    n = len(R)
    if not n:
        return None
    return dict(n=n, h1=sum(1 for x in R if x["rank"] == 1) / n,
                h5=sum(1 for x in R if x["rank"] and x["rank"] <= 5) / n,
                h10=sum(1 for x in R if x["rank"]) / n,
                mrr=sum(1 / x["rank"] for x in R if x["rank"]) / n)


def scores(rows):
    s = {c: score([x for x in rows if x["q"]["category"] == c]) for c in CATS}
    s["in"] = score([x for x in rows if x["q"]["category"] in CATS])
    s["in_norisk"] = score([x for x in rows if x["q"]["category"] in CATS and x["q"]["category"] != "risk_factor"])
    O = [x for x in rows if x["q"]["category"] == "out_of_scope"]
    s["oos"] = (sum(x["ok_oos"] for x in O), len(O))
    return s


VARIANTS = [
    ("pehle jaisa (sab off)", dict(guard=False, clutter=False, strip=False, stmt=False)),
    ("+ guard", dict(guard=True, clutter=False, strip=False, stmt=False)),
    ("+ clutter", dict(guard=False, clutter=True, strip=False, stmt=False)),
    ("+ strip", dict(guard=False, clutter=False, strip=True, stmt=False)),
    ("+ stmt", dict(guard=False, clutter=False, strip=False, stmt=True)),
    ("sab on", dict(guard=True, clutter=True, strip=True, stmt=True)),
]


def ablation_table(results, nq, qset):
    L = [f"ABLATION: har sudhaar ka asar alag  |  {nq} sawal  |  sawal: {qset}",
         "(Hit@1/Hit@5/Hit@10/MRR = saare in-scope sawal;  table..natural = us qism ka Hit@5;  OOS = 'nahi mila' sahi)", "",
         f"{'Variant':22s} {'Hit@1':>6s} {'Hit@5':>6s} {'Hit@10':>6s} {'MRR':>6s} {'dHit@5':>7s} | "
         f"{'table':>6s} {'section':>7s} {'risk':>6s} {'natural':>7s} | {'OOS':>6s}"]
    base = results[0][1]["in"]["h5"]
    for name, s in results:
        a = s["in"]
        cat = [f"{s[c]['h5']:6.1%}" if s[c] else f"{'-':>6s}" for c in CATS]
        L.append(f"{name:22s} {a['h1']:6.1%} {a['h5']:6.1%} {a['h10']:6.1%} {a['mrr']:6.3f} {100 * (a['h5'] - base):+6.1f}  | "
                 f"{cat[0]} {cat[1]:>7s} {cat[2]} {cat[3]:>7s} | {s['oos'][0]:>2d}/{s['oos'][1]:<3d}")
    L += ["", "dHit@5 = 'pehle jaisa' ke muqable kitne points (+ acha, - bura). Jo sudhaar - de, use retriever.py ke",
          "FEATURES mein False kar dena."]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="STAGE_1/eval/questions.jsonl")
    ap.add_argument("--store", default="STAGE_1/vector_store")
    ap.add_argument("--out", default="STAGE_1/eval")
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--ablation", action="store_true", help="har sudhaar on/off kar ke alag naapo")
    a = ap.parse_args()

    qs = [json.loads(l) for l in open(a.questions, encoding="utf-8") if l.strip()]
    qset = qs[0].get("set", "v1") if qs else "?"
    print(f"{len(qs)} sawal mile (sawal: {qset}).")
    r = Retriever(a.store)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rows, secs = None, 0.0
    if a.ablation:
        results = []
        for name, feats in VARIANTS:
            r.features = dict(feats)
            print(f"\n== {name} ==")
            rws, sc = run(r, qs, a.k, label=f"[{name}] ")
            results.append((name, scores(rws)))
            if feats == FEATURES:
                rows, secs = rws, sc
        table = ablation_table(results, len(qs), qset)
        print("\n" + table)
        (out / "ablation.txt").write_text(table, encoding="utf-8")
    if rows is None:
        r.features = dict(FEATURES)
        rows, secs = run(r, qs, a.k)

    # ---------------- score card ----------------
    S = scores(rows)
    on = [f for f, v in FEATURES.items() if v]
    lines = []
    P = lines.append
    P(f"RETRIEVER EVAL  |  {len(qs)} sawal (sawal: {qset})  |  index: {Path(a.store).resolve()}")
    P(f"Sudhaar on: {', '.join(on) if on else 'koi nahi'}")
    P("")
    P(f"{'Qism':16s} {'N':>4s} {'Hit@1':>7s} {'Hit@5':>7s} {'Hit@10':>7s} {'MRR':>6s}")
    for c in CATS:
        s = S[c]
        if s:
            P(f"{c:16s} {s['n']:4d} {s['h1']:7.1%} {s['h5']:7.1%} {s['h10']:7.1%} {s['mrr']:6.3f}")
    for key, name in (("in", "SAB (in-scope)"), ("in_norisk", "SAB (risk bina)")):
        s = S[key]
        if s:
            P(f"{name:16s} {s['n']:4d} {s['h1']:7.1%} {s['h5']:7.1%} {s['h10']:7.1%} {s['mrr']:6.3f}")
    P("  (risk_factor aasaan hai: sawal risk heading se copy hota hai, isliye 'risk bina' wala zyada asli hai)")
    if S["oos"][1]:
        P(f"\nout_of_scope: {S['oos'][0]}/{S['oos'][1]} sahi 'nahi mila' kaha")

    P("\nFAIL KI WAJAH (khud-ba-khud andaza; asal wajah failures.html mein tum khud chunogi):")
    why = defaultdict(int)
    for x in rows:
        if x["htype"]:
            why[x["htype"]] += 1
    for k_, v in sorted(why.items(), key=lambda kv: -kv[1]):
        P(f"  {v:4d}  {HINT_LABEL[k_]}")

    P("\nSABSE KAMZOR COMPANIES (Hit@5, kam se kam 4 sawal):")
    byco = defaultdict(list)
    for x in rows:
        if x["q"]["category"] in CATS:
            byco[x["q"]["company"]].append(x)
    worst = sorted(((sum(1 for x in v if x["rank"] and x["rank"] <= 5) / len(v), co, len(v))
                    for co, v in byco.items() if len(v) >= 4))[:10]
    for h5, co, n in worst:
        P(f"  {co:30s} {h5:6.1%}  ({n} sawal)")
    P(f"\nWaqt: {secs:.0f} sec ({secs / max(1, len(qs)):.2f} sec/sawal)")

    text = "\n".join(lines)
    print("\n" + text)
    (out / "summary.txt").write_text(text, encoding="utf-8")

    # ---------------- results.csv ----------------
    with open(out / "results.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "category", "question", "company", "year", "hit_rank", "hit@1", "hit@5", "matched",
                    "reason", "auto_hint", "top1_company", "top1_year", "top1_page", "top1_section"])
        for x in rows:
            q, resp, rank = x["q"], x["resp"], x["rank"]
            t1 = resp["results"][0] if resp["results"] else {}
            ok5 = (rank is not None and rank <= 5) if q["category"] != "out_of_scope" else x["ok_oos"]
            w.writerow([q["id"], q["category"], q["question"], q["company"], q["year"], rank or "",
                        int(rank == 1), int(ok5), int(resp["matched"]), resp["reason"] or "", x["hint"],
                        t1.get("company", ""), t1.get("year", ""), t1.get("page_numbers", ""),
                        (t1.get("section_path") or "")[-120:]])

    # ---------------- failures.html ----------------
    fails = []
    for x in rows:
        q, resp, rank = x["q"], x["resp"], x["rank"]
        bad = (not x["ok_oos"]) if q["category"] == "out_of_scope" else not (rank and rank <= 5)
        if not bad:
            continue
        g = q["gold"]
        exp = {"value": f"number {g.get('value')} ({g.get('label', '')})", "section": f"section: {g.get('section', '')}",
               "section_regex": f"section mein ye ho: /{g.get('regex', '')}/", "none": "'nahi mila' hona chahiye tha"}[g["mode"]]
        mark = g.get("value") or ""
        top = [dict(rank=j, company=res["company"], year=res["year"], type=res["chunk_type"],
                    page=res["page_numbers"], section=(res["section_path"] or "")[-140:],
                    text=(res["text"] or "")[:420], hit=bool(q["category"] != "out_of_scope" and is_hit(q, res)))
               for j, res in enumerate(resp["results"][:5], 1)]
        fails.append(dict(id=q["id"], category=q["category"], question=q["question"],
                          company=q["company"] or "", year=q["year"] or "", expected=exp, mark=mark,
                          source=q.get("source", {}), hint=x["hint"], rank=rank or "", reason=resp["reason"] or "",
                          top=top))
    data = json.dumps(fails, ensure_ascii=False).replace("</", "<\\/")
    page = HTML.replace("__DATA__", data).replace("__SUMMARY__", html.escape(text))
    (out / "failures.html").write_text(page, encoding="utf-8")
    print(f"\nFiles: {(out / 'summary.txt').resolve()}\n       {(out / 'results.csv').resolve()}\n"
          f"       {(out / 'failures.html').resolve()}   ({len(fails)} fail cards)")
    if a.ablation:
        print(f"       {(out / 'ablation.txt').resolve()}")


HTML = r"""<!doctype html>
<html lang="ur"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Retriever Failures</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#1c2430;--mute:#5b6675;--line:#dfe3ea;--acc:#2f6fdf;--ok:#1a8f4c;--bad:#c2410c;--hl:#fff3b0}
@media (prefers-color-scheme:dark){:root{--bg:#14171c;--card:#1d222a;--ink:#e8ecf2;--mute:#9aa6b6;--line:#2c333d;--acc:#7aa7ff;--ok:#4cc38a;--bad:#ff8a5c;--hl:#5a4a00}}
*{box-sizing:border-box}body{overflow-wrap:anywhere;margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,Segoe UI,Arial,sans-serif}
.wrap{max-width:980px;margin:0 auto;padding:16px}
h1{font-size:21px;margin:6px 0 10px}
details.sum{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:10px 14px;margin-bottom:12px}
pre{white-space:pre-wrap;margin:8px 0 0;font:12.5px/1.45 ui-monospace,Consolas,monospace}
.bar{position:sticky;top:0;z-index:5;background:var(--bg);padding:8px 0;display:flex;gap:8px;flex-wrap:wrap;align-items:center;border-bottom:1px solid var(--line)}
select,button,textarea{font:inherit;color:inherit;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:6px 10px}
button{cursor:pointer}button.pri{background:var(--acc);border-color:var(--acc);color:#fff}
.card{background:var(--card);border:1px solid var(--line);border-left:5px solid var(--bad);border-radius:12px;padding:14px;margin:14px 0}
.card.done{border-left-color:var(--ok)}
.head{display:flex;gap:10px;flex-wrap:wrap;align-items:baseline}
.id{font-weight:700}.pill{font-size:12px;padding:2px 8px;border-radius:99px;border:1px solid var(--line);color:var(--mute)}
.q{font-size:16.5px;font-weight:600;margin:6px 0}
.lbl{font-size:12px;color:var(--mute);text-transform:uppercase;letter-spacing:.04em;margin-top:8px}
.hint{color:var(--bad);font-weight:600}
.src{font-size:13px;color:var(--mute)}
ol{padding-left:22px;margin:6px 0}li{margin:6px 0}
.res{font-size:13.5px}.res .meta{font-weight:600}.res.hit .meta{color:var(--ok)}
.txt{color:var(--mute);word-break:break-word}
mark{background:var(--hl);color:inherit;padding:0 2px;border-radius:3px}
.ctl{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}.ctl select,.bar select{max-width:100%}.ctl textarea{flex:1;min-width:220px;min-height:38px}
</style></head><body><div class="wrap">
<h1>Retriever ke FAIL sawal — asal wajah dhoondo</h1>
<details class="sum"><summary>Score card (summary.txt)</summary><pre>__SUMMARY__</pre></details>
<div class="bar">
 <select id="fcat"><option value="">Saari qismein</option></select>
 <select id="fdone"><option value="">Sab cards</option><option value="todo">Sirf jo baqi hain</option><option value="done">Sirf jo ho gaye</option></select>
 <span id="prog" class="pill"></span>
 <button class="pri" id="dl">CSV download</button>
</div>
<div id="list"></div>
</div>
<script type="application/json" id="data">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById('data').textContent);
const CAUSES = ["", "Retriever ranking (sahi chunk tha, neeche reh gaya)", "Chunk ya table ka masla (text/label/number gadbad)",
  "Heading / section galat", "Company ya saal ka naam/label galat", "Sawal hi galat ya ajeeb", "Asal mein sahi tha (checker ki galti)", "Doosri wajah"];
const KEY = 'retriever_failures_v2';
let saved = {};
try { saved = JSON.parse(localStorage.getItem(KEY) || '{}'); } catch (e) { saved = {}; }
function save(){ try { localStorage.setItem(KEY, JSON.stringify(saved)); } catch (e) {} prog(); }
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function hl(t, m){ const e = esc(t); if(!m) return e; return e.split(esc(m)).join('<mark>'+esc(m)+'</mark>'); }
const cats = [...new Set(DATA.map(d => d.category))];
const fcat = document.getElementById('fcat');
cats.forEach(c => { const o = document.createElement('option'); o.value = c; o.textContent = c + ' (' + DATA.filter(d=>d.category===c).length + ')'; fcat.appendChild(o); });
function prog(){ const n = DATA.filter(d => (saved[d.id]||{}).cause).length; document.getElementById('prog').textContent = n + ' / ' + DATA.length + ' cards ki wajah likhi'; }
function render(){
  const fc = fcat.value, fd = document.getElementById('fdone').value, L = document.getElementById('list');
  L.innerHTML = '';
  DATA.forEach(d => {
    const s = saved[d.id] || {}, done = !!s.cause;
    if (fc && d.category !== fc) return;
    if (fd === 'todo' && done) return; if (fd === 'done' && !done) return;
    const el = document.createElement('div'); el.className = 'card' + (done ? ' done' : '');
    const src = d.source || {};
    el.innerHTML = `<div class="head"><span class="id">${esc(d.id)}</span><span class="pill">${esc(d.category)}</span>
      <span class="pill">${esc(d.company)} ${esc(d.year)}</span>${d.rank ? `<span class="pill">sahi chunk #${esc(d.rank)}</span>` : ''}</div>
      <div class="q">${esc(d.question)}</div>
      <div class="hint">${esc(d.hint)}</div>
      <div class="lbl">Sahi jawab kya tha</div><div>${esc(d.expected)}</div>
      ${src.file ? `<div class="src">Source: ${esc(src.file)} · page ${esc((src.page||[]).join(', '))} · ${esc(src.section)}</div>
      <div class="src txt">${hl(src.preview, d.mark)}</div>` : ''}
      <div class="lbl">Retriever ne kya diya (top 5)</div>
      ${d.top.length ? '<ol>' + d.top.map(r => `<li class="res ${r.hit ? 'hit' : ''}"><div class="meta">${r.hit ? '✓ ' : ''}${esc(r.company)} ${esc(r.year)} · ${esc(r.type)} · page ${esc((r.page||[]).join(', '))}</div>
        <div class="src">${esc(r.section)}</div><div class="txt">${hl(r.text, d.mark)}</div></li>`).join('') + '</ol>'
        : `<div class="txt">Kuch nahi diya. ${esc(d.reason)}</div>`}
      <div class="ctl"><select data-id="${esc(d.id)}">${CAUSES.map(c => `<option ${s.cause===c?'selected':''}>${esc(c)}</option>`).join('')}</select>
      <textarea data-id="${esc(d.id)}" placeholder="Notes (kya dekha)">${esc(s.notes||'')}</textarea></div>`;
    L.appendChild(el);
  });
  L.querySelectorAll('select[data-id]').forEach(x => x.onchange = e => { const id = e.target.dataset.id; saved[id] = Object.assign(saved[id]||{}, {cause: e.target.value}); save(); e.target.closest('.card').classList.toggle('done', !!e.target.value); });
  L.querySelectorAll('textarea[data-id]').forEach(x => x.oninput = e => { const id = e.target.dataset.id; saved[id] = Object.assign(saved[id]||{}, {notes: e.target.value}); save(); });
  prog();
}
fcat.onchange = render; document.getElementById('fdone').onchange = render;
document.getElementById('dl').onclick = () => {
  const q = v => '"' + String(v ?? '').replace(/"/g, '""') + '"';
  const rows = [['id','category','question','company','year','auto_hint','cause','notes'].join(',')];
  DATA.forEach(d => { const s = saved[d.id] || {}; rows.push([d.id,d.category,d.question,d.company,d.year,d.hint,s.cause||'',s.notes||''].map(q).join(',')); });
  const blob = new Blob(['﻿' + rows.join('\r\n')], {type: 'text/csv'});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'failure_review.csv'; a.click();
};
render();
</script></body></html>"""

if __name__ == "__main__":
    main()