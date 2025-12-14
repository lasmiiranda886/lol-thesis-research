#!/usr/bin/env python3
"""
Inventory & schema discovery for the LoL data repo.

Outputs (under ./inventory/):
- inventory_report.md      : Markdown summary (tree, counts, sizes)
- file_index.csv           : All files with size, ext, mtime
- schema_samples.json      : Column/schema samples for JSON/CSV/Parquet
- summary.json             : Machine-readable summary

Dependencies:
- Standard library only. (Optional) pandas/pyarrow if installed for richer CSV/Parquet introspection.
Graceful fallback if not installed.

Usage:
  python3 inventory.py --root . --json-sample 300 --csv-sample-rows 200 --parquet-sample-rows 200
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
import math
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

# Optional deps
try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None

try:
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
except Exception:
    pa = None
    pq = None


def human_bytes(n: int) -> str:
    if n is None:
        return "0B"
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    f = float(n)
    while f >= 1024 and i < len(units) - 1:
        f /= 1024.0
        i += 1
    return f"{f:.2f} {units[i]}"


def walk_files(root: Path) -> list[dict]:
    rows = []
    for p in root.rglob("*"):
        if p.is_file():
            try:
                stat = p.stat()
            except Exception:
                continue
            rows.append(
                {
                    "path": str(p.relative_to(root)),
                    "size": stat.st_size,
                    "ext": p.suffix.lower(),
                    "mtime": stat.st_mtime,
                }
            )
    return rows


def sample_json_schema(paths: list[Path], max_files: int = 200) -> dict:
    """
    For Riot match JSONs, we expect top-level keys like 'metadata' and 'info'.
    We show a union of keys and a shallow sample of participant keys if found.
    """
    import random
    schemas = {"top_level_keys": Counter(), "info_keys": Counter(), "participant_keys": Counter(), "examples": []}
    for fp in random.sample(paths, min(max_files, len(paths))):
        try:
            with fp.open("r") as f:
                obj = json.load(f)
        except Exception:
            continue
        if isinstance(obj, dict):
            schemas["top_level_keys"].update(obj.keys())
            info = obj.get("info", {})
            if isinstance(info, dict):
                schemas["info_keys"].update(info.keys())
                parts = info.get("participants", [])
                if isinstance(parts, list) and parts:
                    if isinstance(parts[0], dict):
                        schemas["participant_keys"].update(parts[0].keys())
        # keep 3 tiny examples (filenames only)
        if len(schemas["examples"]) < 3:
            schemas["examples"].append(fp.name)
    # convert counters to dicts
    for k in ["top_level_keys", "info_keys", "participant_keys"]:
        schemas[k] = dict(schemas[k])
    return schemas


def sample_csv_schema(path: Path, nrows: int = 200) -> dict:
    out = {"path": str(path), "columns": None, "dtypes": None, "sample_head": None, "note": ""}
    if pd is None:
        out["note"] = "pandas not installed; only filename captured."
        return out
    try:
        df = pd.read_csv(path, nrows=nrows)
        out["columns"] = df.columns.tolist()
        out["dtypes"] = {c: str(t) for c, t in df.dtypes.items()}
        out["sample_head"] = df.head(min(5, len(df))).to_dict(orient="records")
    except Exception as e:
        out["note"] = f"error reading CSV: {e}"
    return out


def sample_parquet_schema(path: Path, nrows: int = 200) -> dict:
    out = {"path": str(path), "schema": None, "columns": None, "sample_head": None, "note": ""}
    # Prefer pyarrow (fast), fallback to pandas if available.
    if pq is not None:
        try:
            pqf = pq.ParquetFile(path)
            out["schema"] = pqf.schema_arrow.to_string()
            # sample first row group into pandas if possible
            if pd is not None:
                table = pqf.read_row_group(0) if pqf.num_row_groups > 0 else pqf.read()
                df = table.to_pandas()
                if len(df) > nrows:
                    df = df.sample(nrows, random_state=0)
                out["columns"] = df.columns.tolist()
                out["sample_head"] = df.head(min(5, len(df))).to_dict(orient="records")
            return out
        except Exception as e:
            out["note"] = f"pyarrow read failed: {e}"
    if pd is not None:
        try:
            df = pd.read_parquet(path)
            if len(df) > nrows:
                df = df.sample(nrows, random_state=0)
            out["columns"] = df.columns.tolist()
            out["sample_head"] = df.head(min(5, len(df))).to_dict(orient="records")
            return out
        except Exception as e:
            out["note"] = f"pandas read_parquet failed: {e}"
    if out["note"] == "":
        out["note"] = "no reader available (install pyarrow and/or pandas)."
    return out


def summarize_tree(file_rows: list[dict], root: Path, max_lines: int = 300) -> str:
    """
    Simple tree view limited to max_lines to keep report small.
    """
    # group by top-level dir
    by_dir = defaultdict(list)
    for r in file_rows:
        parts = r["path"].split(os.sep)
        top = parts[0] if parts else "."
        by_dir[top].append(r)

    lines = []
    total = len(file_rows)
    total_size = sum(r["size"] for r in file_rows)
    lines.append(f"Root: {root.resolve()}")
    lines.append(f"Total files: {total:,} | Total size: {human_bytes(total_size)}")
    lines.append("")

    for top in sorted(by_dir.keys()):
        group = by_dir[top]
        size = human_bytes(sum(g["size"] for g in group))
        lines.append(f"{top}/  ({len(group):,} files, {size})")
        # show few example files per group
        examples = sorted(group, key=lambda r: r["path"])[:10]
        for ex in examples:
            lines.append(f"  └─ {ex['path']}  ({human_bytes(ex['size'])})")
        lines.append("")
        if len(lines) >= max_lines:
            lines.append("... (truncated)")
            break
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".", help="root folder to inventory")
    ap.add_argument("--json-sample", type=int, default=300, help="max JSON files to sample for schema")
    ap.add_argument("--csv-sample-rows", type=int, default=200, help="rows to sample from CSVs")
    ap.add_argument("--parquet-sample-rows", type=int, default=200, help="rows to sample from Parquet")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    outdir = root / "inventory"
    outdir.mkdir(exist_ok=True)

    t0 = time.time()
    files = walk_files(root)
    scan_secs = time.time() - t0

    # Basic aggregates
    by_ext = Counter([r["ext"] for r in files])
    total_size = sum(r["size"] for r in files)
    largest = sorted(files, key=lambda r: r["size"], reverse=True)[:15]

    # Detect likely Riot JSONs
    json_paths = [root / r["path"] for r in files if r["ext"] == ".json"]
    riot_json_guess = [p for p in json_paths if "matches" in str(p).lower() or "riot" in str(p).lower() or "data_collection" in str(p).lower()]

    # Schema sampling
    json_schema = sample_json_schema(riot_json_guess, max_files=args.json_sample) if riot_json_guess else {}

    # CSV/Parquet sampling (limit number of files to keep report small)
    csv_paths = [root / r["path"] for r in files if r["ext"] == ".csv"][:10]
    parquet_paths = [root / r["path"] for r in files if r["ext"] in (".parquet", ".pq")]  # limit later

    csv_schemas = [sample_csv_schema(p, args.csv_sample_rows) for p in csv_paths]

    parquet_schemas = []
    for p in parquet_paths[:10]:
        parquet_schemas.append(sample_parquet_schema(p, args.parquet_sample_rows))

    # Write file index CSV (no pandas needed)
    idx_csv = outdir / "file_index.csv"
    with idx_csv.open("w", encoding="utf-8") as f:
        f.write("path,size_bytes,ext,modified_iso\n")
        for r in files:
            f.write(f"{r['path']},{r['size']},{r['ext']},{datetime.utcfromtimestamp(r['mtime']).isoformat()}Z\n")

    # Write schema samples
    schema_json = {
        "json_schema_union": json_schema,
        "csv_samples": csv_schemas,
        "parquet_samples": parquet_schemas,
        "notes": {
            "pandas_available": bool(pd is not None),
            "pyarrow_available": bool(pq is not None),
        },
    }
    with (outdir / "schema_samples.json").open("w", encoding="utf-8") as f:
        json.dump(schema_json, f, indent=2)

    # Text tree
    tree_txt = summarize_tree(files, root)
    # Markdown report
    report_md = outdir / "inventory_report.md"
    with report_md.open("w", encoding="utf-8") as f:
        f.write(f"# Repository Inventory Report\n\n")
        f.write(f"- Root: `{root}`\n")
        f.write(f"- Scanned in: {scan_secs:.2f}s\n")
        f.write(f"- Total files: **{len(files):,}**\n")
        f.write(f"- Total size: **{human_bytes(total_size)}**\n")
        f.write(f"- pandas available: **{bool(pd is not None)}** | pyarrow available: **{bool(pq is not None)}**\n")
        f.write("\n## File type counts\n\n")
        for ext, cnt in sorted(by_ext.items(), key=lambda x: (-x[1], x[0])):
            f.write(f"- `{ext or '(no ext)'}`: {cnt:,}\n")
        f.write("\n## Largest files\n\n")
        for r in largest:
            f.write(f"- {r['path']}  ({human_bytes(r['size'])})\n")
        f.write("\n## Likely Riot match JSON sampling\n\n")
        f.write("Top-level key union / info keys / participant keys were sampled from candidate JSONs.\n")
        f.write("See `inventory/schema_samples.json` for details.\n")
        f.write("\n## Directory overview (truncated)\n\n")
        f.write("```\n")
        f.write(tree_txt)
        f.write("\n```\n")

    # Machine-readable summary
    summary = {
        "root": str(root),
        "total_files": len(files),
        "total_size_bytes": total_size,
        "by_ext": dict(by_ext),
        "largest_files": largest,
        "json_schema_union_present": bool(json_schema),
        "outputs": {
            "report_md": str(report_md),
            "file_index_csv": str(idx_csv),
            "schema_samples_json": str(outdir / "schema_samples.json"),
            "summary_json": str(outdir / "summary.json"),
        },
    }
    with (outdir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Done. See:\n - {report_md}\n - {idx_csv}\n - {outdir/'schema_samples.json'}\n - {outdir/'summary.json'}")


if __name__ == "__main__":
    sys.exit(main())
