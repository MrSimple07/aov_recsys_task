"""Run  python3 check_install.py  if the app or train.py crashes in a strange way.

It checks that every project file is the current version (mixed old/new files cause errors
like "'Bundle' object has no attribute 'seq'") and that no leftover files will pollute training.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MUST_CONTAIN = {                       # file -> text that only the current version has
    "src/artifacts.py": "SCHEMA = 2",
    "src/training.py": "SequentialModel",
    "src/models.py": "class LinUCBPolicy",
    "src/simulator.py": "warm_start",
    "src/experiment.py": "REFERENCE",
    "src/data.py": "_read_excel_cached",
    "app.py": 'st.form("settings")',
    "train.py": "--no-cancel-filter",
    "run_experiment.py": "--price-limit",
    "tests/test_pipeline.py": "test_all_models_respect_the_rules",
}
MUST_NOT_EXIST = ["src/models", "src/features.py", "src/synthetic.py"]

bad = []
for f, text in MUST_CONTAIN.items():
    p = ROOT / f
    if not p.exists():
        bad.append(f"MISSING   {f}")
    elif text not in p.read_text(encoding="utf-8", errors="ignore"):
        bad.append(f"OLD       {f}  (replace it with the file from the latest aov-recsys.zip)")
for f in MUST_NOT_EXIST:
    if (ROOT / f).exists():
        bad.append(f"LEFTOVER  {f}  (delete it)")
raw = ROOT / "data" / "raw"
if raw.exists():
    for p in raw.iterdir():
        if "synthetic" in p.name.lower():
            bad.append(f"POLLUTES  data/raw/{p.name}  (fake orders would end up in your test set - delete it)")
if (ROOT / "artifacts" / "meta.json").exists():
    import json
    schema = json.loads((ROOT / "artifacts" / "meta.json").read_text()).get("schema")
    if schema != 2:
        bad.append(f"STALE     artifacts/ were trained by an older version (schema {schema}): "
                   f"delete the folder contents and run  python3 train.py")

if bad:
    print("PROBLEMS FOUND:\n  " + "\n  ".join(bad))
    sys.exit(1)
print("OK - all files are the current version.")
