"""Merge a task aggregated separately (aggregate.py --tasks ... --out SRC) into an existing summary directory:
copies summary_<task>.{json,csv} and adds the task to summary_all.json.
Usage: merge_summary.py SRC DST [task ...]"""
import json
import shutil
import sys
from pathlib import Path

src, dst, tasks = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
allf = dst / "summary_all.json"
summ = json.loads(allf.read_text())
for task in tasks or [p.stem[len("summary_"):] for p in src.glob("summary_*.json") if p.name != "summary_all.json"]:
    for ext in ("json", "csv"):
        shutil.copy(src / f"summary_{task}.{ext}", dst / f"summary_{task}.{ext}")
    summ[task] = json.loads((src / f"summary_{task}.json").read_text())
    print(f"{task}: {len(summ[task])} rows -> {dst}")
allf.write_text(json.dumps(summ, indent=1, default=float))
