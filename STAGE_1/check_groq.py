"""Groq ki limit check karo: ek chhota sa request bhej kar Groq ka apna jawab dikhata hai.

Chalao:  python STAGE_1/check_groq.py                 (koi bhi jagah hai?)
         python STAGE_1/check_groq.py --need 5000     (ek poore jawab ki jagah hai? nahi to kab hogi)
         python STAGE_1/check_groq.py qwen/qwen3.8-27b --need 5000   (judge model)
Key repo-root .env se GROQ_API_KEY uthata hai (chat mein key kabhi mat dalo).
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ap = argparse.ArgumentParser(description="Groq ki limit check")
ap.add_argument("model", nargs="?", default="openai/gpt-oss-120b")
ap.add_argument("--need", type=int, default=1,
                help="itne tokens ki jagah maango (jaise 5000 = ek jawab). Model sirf 'OK' likhta hai, is liye "
                     "asal kharcha bohat kam; jagah na ho to Groq exact waqt batata hai")
ARGS = ap.parse_args()
MODEL = ARGS.model


def load_key():
    key = os.environ.get("GROQ_API_KEY")
    if key:
        return key
    env = Path(__file__).resolve().parent.parent / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("GROQ_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("GROQ_API_KEY .env mein nahi mili")


req = urllib.request.Request(
    "https://api.groq.com/openai/v1/chat/completions",
    data=json.dumps({"model": MODEL, "max_tokens": ARGS.need,
                     "messages": [{"role": "user", "content": "Reply with the single word OK."}]}).encode(),
    headers={"Authorization": f"Bearer {load_key()}",
             "Content-Type": "application/json",
             # Cloudflare default "Python-urllib" agent ko 403/1010 deta hai
             "User-Agent": "techcorp-rag-check/1.0"},
)
try:
    with urllib.request.urlopen(req, timeout=30) as r:
        print(f"{MODEL}: OK -- {ARGS.need} tokens ki jagah hai, eval chala sakti ho")
        for h in ("x-ratelimit-remaining-tokens", "x-ratelimit-remaining-requests"):
            print(f"  {h}: {r.headers.get(h)}")
except urllib.error.HTTPError as e:
    body = e.read().decode(errors="replace")
    try:
        msg = json.loads(body)["error"]["message"]
    except Exception:
        msg = body
    print(f"{MODEL}: HTTP {e.code}")
    print(f"  retry-after: {e.headers.get('retry-after')} sec")
    print(f"  Groq ka message: {msg}")