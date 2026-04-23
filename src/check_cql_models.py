"""
CQL Model Diagnostic
=====================
Run this FIRST to see exactly what files d3rlpy saved.
It will tell you the correct loading method automatically.

Run:
  python check_cql_models.py
"""

import pickle
from pathlib import Path

MODEL_DIR = Path("models")

print("=" * 55)
print("CQL Model Diagnostic")
print("=" * 55)

print(f"\nAll files in {MODEL_DIR}/:")
print("-" * 40)
for f in sorted(MODEL_DIR.iterdir()):
    size = f.stat().st_size / 1024
    kind = "DIR" if f.is_dir() else "FILE"
    print(f"  [{kind}]  {f.name:<35} {size:>8.1f} KB")

# Check each CQL path
for tag in ["cql_D_policy", "cql_Dp_policy"]:
    path = MODEL_DIR / tag
    print(f"\nChecking: {path}")

    if path.is_dir():
        print(f"  Type   : FOLDER")
        for child in sorted(path.iterdir()):
            print(f"    {child.name}")

    elif path.is_file():
        print(f"  Type   : SINGLE FILE  ({path.stat().st_size/1024:.1f} KB)")

    else:
        # Search for similar filenames
        matches = list(MODEL_DIR.glob(f"{tag}*"))
        if matches:
            print(f"  Not found exactly, but similar files:")
            for m in matches:
                print(f"    {m.name}  ({m.stat().st_size/1024:.1f} KB)")
        else:
            print(f"  NOT FOUND — CQL was not saved or failed during training.")

# Try loading d3rlpy
print("\n" + "=" * 55)
print("d3rlpy version check")
print("=" * 55)
try:
    import d3rlpy
    print(f"  d3rlpy version: {d3rlpy.__version__}")
    print(f"  Available load methods:")
    cql_cls = d3rlpy.algos.DiscreteCQL
    for method in ["from_json", "from_file", "load"]:
        has = hasattr(cql_cls, method)
        print(f"    DiscreteCQL.{method}  →  {'YES' if has else 'NO'}")
except ImportError:
    print("  d3rlpy NOT installed.")
    print("  Install: pip install d3rlpy torch")

print("\nShare the output above — it will tell us exactly how to fix the loader.")