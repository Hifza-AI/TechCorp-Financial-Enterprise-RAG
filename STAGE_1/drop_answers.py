"""Results file se kuch sawal nikaalo taake eval_answers.py unhein dobara chalaye (pehle .bak copy banti hai).

Chalao:  python STAGE_1/drop_answers.py STAGE_1/eval/cmp_gptoss120b_v3.json T96 T97 T98 T99 T100
"""
import json
import shutil
import sys
from pathlib import Path

path, ids = Path(sys.argv[1]), set(sys.argv[2:])
rows = json.loads(path.read_text(encoding="utf-8"))
shutil.copy(path, path.with_suffix(path.suffix + ".bak"))
keep = [d for d in rows if d["id"] not in ids]
path.write_text(json.dumps(keep, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"{len(rows) - len(keep)} sawal nikaale ({', '.join(sorted(ids))}); baqi {len(keep)}. Backup: {path.name}.bak")