"""
make_audit_sheet.py  -  Blind random audit (tumhari apni aankhon ke liye)

Maqsad: 500 files mein se RANDOM 30 jagahen chunni jo tum khud PDF se milao.
Koi cheez haath se nahi chuni jaati (na tum, na Claude), isliye natija imaandar hota hai.

Chalao (project root se):
    python STAGE_1/make_audit_sheet.py

Ye DO files banata hai:
    audit_sheet.html   <-- ISE CHROME MEIN KHOLO (aasan, buttons wala, score khud dikhata hai)
    audit_sheet.csv    <-- Excel wala purana tareeqa (zaroorat nahi, backup)

HTML mein har item ke liye PDF ka naam aur page milta hai. PDF mein wo page kholo, tukda milao,
phir S / T / G dabao:
    S = sahi    T = thora ghalat (number sahi, label thora off)    G = ghalat (number ghalat/gayab, text kata)

Score HTML ke upar khud dikhta hai. (CSV wale tareeqe ke liye:  python STAGE_1/make_audit_sheet.py --score)

Dobara naya random sample:  python STAGE_1/make_audit_sheet.py --seed 7 --n 30
(Same seed = same 30 jagahen, isliye pehle wali sheet dobara wahi banegi.)
"""
import argparse
import csv
import json
import math
import random
import re
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--chunks", default="STAGE_1/chunks")
ap.add_argument("--pdfs", default="STAGE_1/data")
ap.add_argument("--n", type=int, default=30)
ap.add_argument("--seed", type=int, default=2026)
ap.add_argument("--out", default="audit_sheet.csv")
ap.add_argument("--html", default="audit_sheet.html")
ap.add_argument("--score", action="store_true")
args = ap.parse_args()


def wilson_upper(g, n, z=1.96):
    p = g / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre + margin) / denom


def score(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    done = [r for r in rows if r.get("verdict", "").strip().upper() in ("S", "T", "G")]
    if not done:
        print("Abhi koi verdict nahi likha (S / T / G).")
        return
    n = len(done)
    s = sum(1 for r in done if r["verdict"].strip().upper() == "S")
    t = sum(1 for r in done if r["verdict"].strip().upper() == "T")
    g = sum(1 for r in done if r["verdict"].strip().upper() == "G")
    print(f"Check ho chuke: {n} / {len(rows)}")
    print(f"  S (sahi)          : {s}  ({100*s/n:.0f}%)")
    print(f"  T (thora ghalat)  : {t}  ({100*t/n:.0f}%)")
    print(f"  G (ghalat)        : {g}  ({100*g/n:.0f}%)")
    print(f"\nHonest andaza: {n} random checks mein {g} serious ghalti.")
    print(f"95% bharose ke saath poore data mein serious ghalti {100*wilson_upper(g, n):.0f}% se zyada nahi hogi.")
    print("\nGhalat/thora ghalat wali rows:")
    for r in done:
        if r["verdict"].strip().upper() in ("T", "G"):
            print(f"  {r['id']}  {r['company']} {r['year']}  p{r['page']}  [{r['verdict'].upper()}]  {r.get('note','')}")


if args.score:
    score(args.out)
    raise SystemExit

root = Path(args.chunks)
files = sorted(root.rglob("*_chunks.json"))
if not files:
    raise SystemExit(f"Koi *_chunks.json nahi mili: {root.resolve()}")

pdf_index = {}
for p in Path(args.pdfs).rglob("*.pdf"):
    pdf_index[re.sub(r"[^a-z0-9]", "", p.stem.lower())] = p.name


def find_pdf(file_name):
    k = re.sub(r"[^a-z0-9]", "", file_name.lower())
    if k in pdf_index:
        return pdf_index[k]
    for kk, v in pdf_index.items():
        if kk.startswith(k[:18]) and k[-8:] == kk[-8:]:
            return v
    return "(PDF khud dhoondo: %s)" % file_name


rnd = random.Random(args.seed)
tables, texts = [], []
for f in files:
    try:
        ch = json.load(open(f, encoding="utf-8"))
    except Exception:
        continue
    for i, c in enumerate(ch):
        t = c.get("text", "")
        if c.get("chunk_type") == "table" and len(re.findall(r"\d", t)) >= 20:
            tables.append((f, i, c))
        elif c.get("chunk_type") == "text" and len(t) > 400:
            texts.append((f, i, c))

n_t = args.n // 2
picked = rnd.sample(tables, min(n_t, len(tables))) + rnd.sample(texts, min(args.n - n_t, len(texts)))
rnd.shuffle(picked)

items = []
for k, (f, i, c) in enumerate(picked, start=1):
    pages = c.get("page_numbers") or []
    pg = f"{min(pages)}-{max(pages)}" if len(pages) > 1 else (pages[0] if pages else "?")
    t = c["text"]
    if c["chunk_type"] == "table":
        lines = t.splitlines()
        snippet = "\n".join(lines[:8])
        start, end = "\n".join(lines[:8]), ""
        todo = "Pehli 4 rows ke numbers PDF page par milao (columns ke naam bhi)"
    else:
        snippet = t[:350] + " ... " + t[-150:]
        start, end = t[:350], t[-150:]
        todo = "Pehla aur aakhri jumla PDF par milao; text kata/garbled to nahi"
    items.append(dict(
        id=f"A{k:02d}", company=str(c.get("company", "")), year=str(c.get("year", "")),
        type=c["chunk_type"], page=str(pg),
        pdf_file=find_pdf(c.get("file_name", f.stem.replace("_chunks", ""))),
        section=c.get("section_path", "")[-90:], snippet=snippet, start=start, end=end,
        todo=todo))

with open(args.out, "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.writer(fh)
    w.writerow(["id", "company", "year", "type", "page", "pdf_file", "section", "snippet", "kya_milana_hai", "verdict", "note"])
    for it in items:
        w.writerow([it["id"], it["company"], it["year"], it["type"], it["page"], it["pdf_file"],
                    it["section"], it["snippet"], it["todo"], "", ""])

HTML = r"""<!doctype html>
<html lang="ur"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Random Audit</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#1c2430;--mute:#5b6675;--line:#dfe3ea;--ok:#1d7a46;--mid:#a86a00;--bad:#b3261e;--acc:#2456d6}
@media (prefers-color-scheme:dark){:root{--bg:#14171c;--card:#1d222a;--ink:#e8ecf2;--mute:#9aa6b5;--line:#2e3540;--ok:#4cc28a;--mid:#e0a43a;--bad:#ff7b72;--acc:#7da2ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,Segoe UI,Roboto,sans-serif}
.wrap{max-width:900px;margin:0 auto;padding:16px}
h1{font-size:22px;margin:8px 0}
.how{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin-bottom:14px}
.how ol{margin:6px 0 0 18px;padding:0}.how li{margin:4px 0}
.bar{position:sticky;top:0;z-index:5;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:10px 14px;margin-bottom:14px;display:flex;flex-wrap:wrap;gap:8px 18px;align-items:center}
.bar b{font-size:18px}.pill{padding:2px 10px;border-radius:99px;border:1px solid var(--line);font-size:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin-bottom:14px}
.card.done{border-left:5px solid var(--ok)}.card.T.done{border-left-color:var(--mid)}.card.G.done{border-left-color:var(--bad)}
.head{display:flex;flex-wrap:wrap;gap:6px 14px;align-items:baseline}
.id{font-weight:700;font-size:18px}.meta{color:var(--mute)}
.pdf{font-family:ui-monospace,Consolas,monospace;background:var(--bg);padding:2px 8px;border-radius:6px;border:1px solid var(--line);word-break:break-all}
.todo{margin:8px 0;color:var(--acc);font-weight:600}
.lbl{font-size:13px;color:var(--mute);margin:8px 0 2px;text-transform:uppercase;letter-spacing:.04em}
pre{white-space:pre-wrap;word-break:break-word;margin:0;background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:10px 12px;font:14px/1.45 ui-monospace,Consolas,monospace;max-height:320px;overflow:auto}
.btns{display:flex;gap:8px;margin-top:10px;flex-wrap:wrap}
.btns button{font:inherit;padding:8px 18px;border-radius:8px;border:2px solid var(--line);background:var(--card);color:var(--ink);cursor:pointer}
.btns button.sel.S{background:var(--ok);border-color:var(--ok);color:#fff}.btns button.sel.T{background:var(--mid);border-color:var(--mid);color:#fff}.btns button.sel.G{background:var(--bad);border-color:var(--bad);color:#fff}
input.note{flex:1;min-width:200px;font:inherit;padding:8px 10px;border-radius:8px;border:1px solid var(--line);background:var(--bg);color:var(--ink)}
.dl{font:inherit;padding:8px 14px;border-radius:8px;border:0;background:var(--acc);color:#fff;cursor:pointer}
</style></head><body><div class="wrap">
<h1>Random Audit: apni aankhon se tasalli</h1>
<div class="how"><b>Kaise karna hai (har item ke liye):</b>
<ol>
<li>Item mein likha <b>PDF ka naam</b> dhoondo (<code>STAGE_1/data</code> folder) aur Chrome mein kholo.</li>
<li><b>Page number</b> par jao. Na mile to 1-2 page <b>pehle</b> dekho (chunk ka page kabhi 1-2 aage hota hai).</li>
<li>Neeche wala tukda PDF mein dhoondo (Ctrl+F mein kuch alfaaz likh kar bhi mil jata hai).</li>
<li><b>Table</b> ho: pehli 4 rows ke numbers PDF se milao. <b>Text</b> ho: shuruaat aur aakhir ke jumle milao, aur dekho text kata ya garbled to nahi.</li>
<li>Button dabao: <b>S</b> = sahi, <b>T</b> = thora ghalat (number sahi, label/naam thora off), <b>G</b> = ghalat (number ghalat ya gayab, ya text kata hua). Jo S nahi, uski ek line note mein likho.</li>
</ol>
<div style="margin-top:6px;color:var(--mute)">Tumhara kiya hua is page mein khud save hota rehta hai (isi computer ke Chrome mein). Band karke baad mein dobara kholo to wahin se jaari.</div></div>
<div class="bar"><b id="prog">0 / 0</b><span class="pill" id="cS">S: 0</span><span class="pill" id="cT">T: 0</span><span class="pill" id="cG">G: 0</span>
<span id="bound" style="color:var(--mute)"></span><button class="dl" id="dl">Result CSV download</button></div>
<div id="list"></div>
<script>
const ITEMS=__ITEMS__;
let h=5381;ITEMS.map(x=>x.id+x.company+x.year+x.page+x.pdf_file).join("|").split("").forEach(ch=>{h=((h<<5)+h+ch.charCodeAt(0))|0});const KEY="audit_"+h;
let st={};try{st=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
function save(){try{localStorage.setItem(KEY,JSON.stringify(st))}catch(e){}}
function wilson(g,n){if(!n)return 0;const z=1.96,p=g/n,d=1+z*z/n,c=p+z*z/(2*n),m=z*Math.sqrt(p*(1-p)/n+z*z/(4*n*n));return (c+m)/d}
function upd(){let s=0,t=0,g=0;ITEMS.forEach(i=>{const v=(st[i.id]||{}).v;if(v=="S")s++;if(v=="T")t++;if(v=="G")g++});
const n=s+t+g;document.getElementById("prog").textContent=n+" / "+ITEMS.length+" check";
document.getElementById("cS").textContent="S: "+s;document.getElementById("cT").textContent="T: "+t;document.getElementById("cG").textContent="G: "+g;
document.getElementById("bound").textContent=n?("95% bharose se serious ghalti "+Math.round(wilson(g,n)*100)+"% se zyada nahi ("+n+" checks, "+g+" G)"):"";
ITEMS.forEach(i=>{const el=document.getElementById("c_"+i.id);const v=(st[i.id]||{}).v;el.className="card"+(v?" done "+v:"");
el.querySelectorAll(".btns button").forEach(b=>{b.classList.toggle("sel",b.dataset.v==v)})})}
const list=document.getElementById("list");
ITEMS.forEach(i=>{const d=document.createElement("div");d.id="c_"+i.id;d.className="card";
d.innerHTML='<div class="head"><span class="id"></span><span class="meta"></span></div><div style="margin:6px 0">PDF: <span class="pdf"></span> &nbsp; Page: <b class="pg"></b></div><div class="meta sec"></div><div class="todo"></div><div class="lbl">Chunk ka text (jo humare data mein hai)</div><pre class="s1"></pre><div class="lbl endl">Aakhir ka hissa</div><pre class="s2"></pre><div class="btns"><button data-v="S" class="S">S sahi</button><button data-v="T" class="T">T thora ghalat</button><button data-v="G" class="G">G ghalat</button><input class="note" placeholder="Note (jo S nahi uski ek line)"></div>';
d.querySelector(".id").textContent=i.id;d.querySelector(".meta").textContent=i.company+" "+i.year+" | "+(i.type=="table"?"TABLE":"TEXT");
d.querySelector(".pdf").textContent=i.pdf_file;d.querySelector(".pg").textContent=i.page;d.querySelector(".sec").textContent="Section: "+i.section;
d.querySelector(".todo").textContent=i.todo;d.querySelector(".s1").textContent=i.start;
if(i.end){d.querySelector(".s2").textContent=i.end}else{d.querySelector(".s2").style.display="none";d.querySelector(".endl").style.display="none"}
const note=d.querySelector(".note");note.value=(st[i.id]||{}).n||"";
note.addEventListener("input",()=>{st[i.id]=Object.assign(st[i.id]||{},{n:note.value});save()});
d.querySelectorAll(".btns button").forEach(b=>b.addEventListener("click",()=>{const cur=(st[i.id]||{}).v;st[i.id]=Object.assign(st[i.id]||{},{v:cur==b.dataset.v?"":b.dataset.v});save();upd()}));
list.appendChild(d)});
document.getElementById("dl").addEventListener("click",()=>{const q=s=>'"'+String(s==null?"":s).replace(/"/g,'""')+'"';
const rows=[["id","company","year","type","page","pdf_file","section","snippet","kya_milana_hai","verdict","note"].join(",")];
ITEMS.forEach(i=>{const o=st[i.id]||{};rows.push([i.id,i.company,i.year,i.type,i.page,i.pdf_file,i.section,i.snippet,i.todo,o.v||"",o.n||""].map(q).join(","))});
const b=new Blob(["﻿"+rows.join("\n")],{type:"text/csv"});const a=document.createElement("a");a.href=URL.createObjectURL(b);a.download="audit_sheet.csv";a.click()});
upd();
</script></div></body></html>"""

Path(args.html).write_text(HTML.replace("__ITEMS__", json.dumps(items, ensure_ascii=False).replace("</", "<\\/")), encoding="utf-8")

print(f"{len(items)} random jagahen ban gayin:")
print(f"  {args.html}  <-- ise CHROME mein kholo (double-click)")
print(f"  {args.out}   (Excel wala backup)")