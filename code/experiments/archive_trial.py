"""Archive one trial's final artifacts under versions/<trial>/ with a sha256 manifest and metrics.json.

usage:
  python archive_trial.py <trial> --metrics '{"val_macro_f05": 0.97}' [--move a.tsv ...] [--copy b.ipynb ...]

Large files (tsv/npz/npy) are git-ignored but their sha256 goes into MANIFEST.md, which is tracked, so every
submission file can be matched to the commit and tag that produced it.
"""
import argparse
import datetime as dt
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(r"D:\AmazonMLChallenge2026")


def sha256(path, bufsize=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trial")
    ap.add_argument("--metrics", default="{}")
    ap.add_argument("--move", nargs="*", default=[])
    ap.add_argument("--copy", nargs="*", default=[])
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    dest = ROOT / "versions" / a.trial
    dest.mkdir(parents=True, exist_ok=True)
    for src in a.move:
        shutil.move(str(ROOT / src), dest / Path(src).name)
    for src in a.copy:
        shutil.copy2(ROOT / src, dest / Path(src).name)
    metrics = json.loads(a.metrics)
    metrics["archived_at"] = dt.datetime.now().isoformat(timespec="seconds")
    if a.note:
        metrics["note"] = a.note
    (dest / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    rows = ["| file | bytes | sha256 |", "|---|---|---|"]
    for f in sorted(dest.iterdir()):
        if f.name in ("MANIFEST.md",):
            continue
        rows.append(f"| {f.name} | {f.stat().st_size:,} | `{sha256(f)}` |")
    (dest / "MANIFEST.md").write_text(f"# {a.trial}\n\n" + "\n".join(rows) + "\n", encoding="utf-8")
    print(f"archived {a.trial} -> {dest}")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
