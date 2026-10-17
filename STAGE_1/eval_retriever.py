"""
eval_retriever.py  -  Saare sawal retriever par chalata hai aur khud check karta hai ke sahi jawab aaya ya nahi.

Chalao:   python STAGE_1/eval_retriever.py
Pehle:    python STAGE_1/make_eval_set.py   (sawal banane ke liye)

Banata (STAGE_1/eval/ mein):
  summary.txt     -> score card (Hit@1, Hit@5, MRR har qism ke liye + kamzor companies). Ye mujhe bhejna.
  results.csv     -> har sawal ki ek line (Excel mein khulti hai)
  failures.html   -> sirf FAIL hue sawal, cards ki shakal mein. Browser mein kholo, har card par
                     "asal wajah" chuno (retriever / chunk / heading / naam-saal / sawal galat), phir CSV download.

Score ka matlab:
  Hit@1  = sahi chunk pehle number par aaya       Hit@5 = sahi chunk top 5 mein aaya (Gemini ko 5 milenge)
  MRR    = sahi chunk kitna upar tha (1 = hamesha pehla, 0.5 = aksar doosra)
  out_of_scope ke liye "sahi" = retriever ne "nahi mila" kaha.
"""
import argparse
import csv
import html
import json
import re
import time
from collections import defaultdict
from pathlib import Path

from retriever import Retriever


def norm(t):
    return re.sub(r"\s+", " ", (t or "").replace("’", "'")).strip().lower()


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
    if g["mode"] == "section":
        return r["section_path"] == g["section"] or (g.get("anchor") and g["anchor"] in norm(r["text"]))
    if g["mode"] == "section_regex":
        return bool(re.search(g["regex"], r["section_path"] or "", re.I))
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="STAGE_1/eval/questions.jsonl")
    ap.add_argument("--store", default="STAGE_1/vector_store")
    ap.add_argument("--out", default="STAGE_1/eval")
    ap.add_argument("-k", type=int, default=10)
    a = ap.parse_args()

    qs = [json.loads(l) for l in open(a.questions, encoding="utf-8") if l.strip()]
    print(f"{len(qs)} sawal mile.")
    r = Retriever(a.store)
    t0 = time.time()
    rows = []
    for i, q in enumerate(qs, 1):
        resp = r.search(q["question"], top_k=a.k)
        rank = None
        if q["category"] != "out_of_scope":
            for j, res in enumerate(resp["results"], 1):
                if is_hit(q, res):
                    rank = j
                    break
        ok_oos = q["category"] == "out_of_scope" and not resp["matched"]
        htype, hint = diagnose(q, resp, rank)
        rows.append(dict(q=q, resp=resp, rank=rank, ok_oos=ok_oos, htype=htype, hint=hint))
        if i % 25 == 0 or i == len(qs):
            print(f"  {i}/{len(qs)}  ({time.time() - t0:.0f} sec)")

    # ---------------- score card ----------------
    cats = ["table_number", "section_topic", "risk_factor", "natural_topic"]
    lines = []
    P = lines.append
    P(f"RETRIEVER EVAL  |  {len(qs)} sawal  |  index: {Path(a.store).resolve()}")
    P("")
    P(f"{'Qism':16s} {'N':>4s} {'Hit@1':>7s} {'Hit@5':>7s} {'Hit@10':>7s} {'MRR':>6s}")
    allin = []
    for c in cats:
        R = [x for x in rows if x["q"]["category"] == c]
        allin += R
        if not R:
            continue
        n = len(R)
        h1 = sum(1 for x in R if x["rank"] == 1) / n
        h5 = sum(1 for x in R if x["rank"] and x["rank"] <= 5) / n
        h10 = sum(1 for x in R if x["rank"]) / n
        mrr = sum(1 / x["rank"] for x in R if x["rank"]) / n
        P(f"{c:16s} {n:4d} {h1:7.1%} {h5:7.1%} {h10:7.1%} {mrr:6.3f}")
    if allin:
        n = len(allin)
        P(f"{'SAB (in-scope)':16s} {n:4d} {sum(1 for x in allin if x['rank'] == 1) / n:7.1%} "
          f"{sum(1 for x in allin if x['rank'] and x['rank'] <= 5) / n:7.1%} "
          f"{sum(1 for x in allin if x['rank']) / n:7.1%} {sum(1 / x['rank'] for x in allin if x['rank']) / n:6.3f}")
    O = [x for x in rows if x["q"]["category"] == "out_of_scope"]
    if O:
        P(f"\nout_of_scope: {sum(x['ok_oos'] for x in O)}/{len(O)} sahi 'nahi mila' kaha")

    P("\nFAIL KI WAJAH (khud-ba-khud andaza; asal wajah failures.html mein tum khud chunogi):")
    why = defaultdict(int)
    for x in rows:
        if x["htype"]:
            why[x["htype"]] += 1
    for k, v in sorted(why.items(), key=lambda kv: -kv[1]):
        P(f"  {v:4d}  {HINT_LABEL[k]}")

    P("\nSABSE KAMZOR COMPANIES (Hit@5, kam se kam 4 sawal):")
    byco = defaultdict(list)
    for x in allin:
        byco[x["q"]["company"]].append(x)
    worst = sorted(((sum(1 for x in v if x["rank"] and x["rank"] <= 5) / len(v), co, len(v))
                    for co, v in byco.items() if len(v) >= 4))[:10]
    for h5, co, n in worst:
        P(f"  {co:30s} {h5:6.1%}  ({n} sawal)")
    P(f"\nWaqt: {time.time() - t0:.0f} sec ({(time.time() - t0) / max(1, len(qs)):.2f} sec/sawal)")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
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
const KEY = 'retriever_failures_v1';
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