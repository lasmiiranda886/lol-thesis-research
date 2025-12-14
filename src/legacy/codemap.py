#!/usr/bin/env python3
"""
Code & Data Mapper (stdlib only)

Outputs (under ./inventory/):
- codemap_report.md      : Markdown summary of Python files and data folders
- codemap_index.csv      : Table of Python files with imports, functions, classes, CLI
- codemap_imports.mmd    : Mermaid graph (module import edges)

Usage:
  python3 codemap.py --roots src . --data-dirs data data_collection match_details
"""
from __future__ import annotations
import argparse, ast, os, sys, time
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

def human_bytes(n:int)->str:
    units=["B","KB","MB","GB","TB"]; i=0; f=float(n)
    while f>=1024 and i<len(units)-1: f/=1024; i+=1
    return f"{f:.2f} {units[i]}"

def safe_read_text(path:Path)->str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""

def py_info(path:Path)->dict:
    """Parse a .py file: imports, functions, classes, has_cli."""
    src = safe_read_text(path)
    info = {
        "path": str(path),
        "module": str(path.with_suffix("")).replace(os.sep, "."),
        "size": path.stat().st_size if path.exists() else 0,
        "mtime": path.stat().st_mtime if path.exists() else 0.0,
        "n_lines": src.count("\n")+1,
        "imports": [],
        "from_imports": [],
        "functions": [],
        "classes": [],
        "has_main": False,
        "uses_argparse": "argparse" in src,
        "top_doc": "",
    }
    try:
        tree = ast.parse(src or "", filename=str(path))
    except SyntaxError:
        return info

    # module docstring
    info["top_doc"] = (ast.get_docstring(tree) or "").strip()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                info["imports"].append(n.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            info["from_imports"].append(mod)
        elif isinstance(node, ast.FunctionDef):
            info["functions"].append(node.name)
        elif isinstance(node, ast.ClassDef):
            info["classes"].append(node.name)
        elif isinstance(node, ast.If):
            # detect: if __name__ == "__main__":
            try:
                test = ast.get_source_segment(src, node.test) or ""
            except Exception:
                test = ""
            if "__name__" in test and "__main__" in test:
                info["has_main"] = True
    return info

def scan_py_files(roots:list[Path])->list[dict]:
    rows=[]
    for root in roots:
        for p in root.rglob("*.py"):
            # skip venvs and site-packages
            if "venv" in p.parts or "site-packages" in p.parts:
                continue
            rows.append(py_info(p))
    return rows

def data_overview(dirs:list[Path])->dict:
    out={}
    for d in dirs:
        if not d.exists(): 
            continue
        total=0; size=0
        by_ext=Counter()
        examples=[]
        for p in d.rglob("*"):
            if p.is_file():
                total+=1; size+=p.stat().st_size
                by_ext[p.suffix.lower()]+=1
                if len(examples)<10:
                    examples.append(str(p.relative_to(d)))
        out[str(d)] = {
            "total_files": total,
            "total_size_bytes": size,
            "total_size_human": human_bytes(size),
            "by_ext": dict(by_ext.most_common(20)),
            "examples": examples,
        }
    return out

def write_csv(rows:list[dict], path:Path)->None:
    cols = ["path","module","size","n_lines","modified_iso","has_main","uses_argparse","n_imports","n_from_imports","n_functions","n_classes","imports_sample","from_imports_sample","functions_sample","classes_sample"]
    path.parent.mkdir(exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(cols)+"\n")
        for r in rows:
            modified_iso = datetime.utcfromtimestamp(r["mtime"]).isoformat()+"Z" if r["mtime"] else ""
            row = {
                "path": r["path"],
                "module": r["module"],
                "size": str(r["size"]),
                "n_lines": str(r["n_lines"]),
                "modified_iso": modified_iso,
                "has_main": str(r["has_main"]),
                "uses_argparse": str(r["uses_argparse"]),
                "n_imports": str(len(set(r["imports"]))),
                "n_from_imports": str(len(set(r["from_imports"]))),
                "n_functions": str(len(r["functions"])),
                "n_classes": str(len(r["classes"])),
                "imports_sample": ";".join(sorted(set(r["imports"]))[:10]),
                "from_imports_sample": ";".join(sorted(set(r["from_imports"]))[:10]),
                "functions_sample": ";".join(sorted(r["functions"])[:10]),
                "classes_sample": ";".join(sorted(r["classes"])[:10]),
            }
            f.write(",".join(row[c] for c in cols)+"\n")

def write_mermaid_import_graph(rows:list[dict], path:Path)->None:
    """
    Very rough import graph (module → imported module). Best-effort since Python imports can be dynamic.
    """
    edges=set()
    for r in rows:
        src = r["module"]
        for t in set(r["imports"]+r["from_imports"]):
            if not t: 
                continue
            # skip stdlib-ish short names to reduce noise
            if t.split(".")[0] in {"os","sys","re","json","time","typing","pathlib","argparse","subprocess","logging","random","math","itertools","collections","datetime"}:
                continue
            edges.add((src, t))
    path.parent.mkdir(exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("graph LR\n")
        for a,b in sorted(edges):
            f.write(f"  {a.replace('.','_')} --> {b.replace('.','_')}\n")

def write_report(rows:list[dict], data_info:dict, path:Path)->None:
    path.parent.mkdir(exist_ok=True)
    rows_sorted = sorted(rows, key=lambda r: (-r["has_main"], -r["uses_argparse"], -r["size"]))
    # quick buckets
    with_main=[r for r in rows_sorted if r["has_main"]]
    with_cli=[r for r in rows_sorted if r["uses_argparse"]]
    top_imports=Counter()
    for r in rows:
        top_imports.update(set([i.split(".")[0] for i in r["imports"]+r["from_imports"] if i]))
    top_imports = Counter({k:v for k,v in top_imports.items() if k not in {"os","sys","re","json","time","typing","pathlib","argparse","subprocess","logging","random","math","itertools","collections","datetime"}})

    with path.open("w", encoding="utf-8") as f:
        f.write("# Code & Data Map\n\n")
        f.write(f"Scanned at: {datetime.utcnow().isoformat()}Z\n\n")
        f.write("## Python files summary\n")
        f.write(f"- Total .py files scanned: **{len(rows)}**\n")
        f.write(f"- With `if __name__ == \"__main__\"`: **{len(with_main)}**\n")
        f.write(f"- Using `argparse`: **{len(with_cli)}**\n")
        f.write("\n### Likely entrypoints (sorted by size)\n")
        for r in with_main[:20]:
            f.write(f"- `{r['path']}`  (lines: {r['n_lines']}, size: {human_bytes(r['size'])})\n")
        f.write("\n### Heavyweight modules (top 20 by size)\n")
        for r in sorted(rows, key=lambda r: r["size"], reverse=True)[:20]:
            f.write(f"- `{r['path']}`  ({human_bytes(r['size'])})\n")
        f.write("\n### Most common non-stdlib imports\n")
        for name,count in top_imports.most_common(30):
            f.write(f"- {name}: {count}\n")

        f.write("\n## Data folders\n")
        for d,info in data_info.items():
            f.write(f"\n### {d}\n")
            f.write(f"- Files: {info['total_files']:,}\n")
            f.write(f"- Size: {info['total_size_human']}\n")
            f.write(f"- Top extensions: {info['by_ext']}\n")
            f.write("- Examples:\n")
            for ex in info["examples"]:
                f.write(f"  - {ex}\n")

        f.write("\n---\n_This report was generated by codemap.py_\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=["src"], help="folders to scan for .py")
    ap.add_argument("--data-dirs", nargs="*", default=["data"], help="folders to summarize for data")
    args = ap.parse_args()

    roots=[Path(r).resolve() for r in args.roots]
    data_dirs=[Path(d).resolve() for d in args.data_dirs]

    t0=time.time()
    rows = scan_py_files(roots)
    data_info = data_overview(data_dirs)
    dt=time.time()-t0

    inv = Path("inventory"); inv.mkdir(exist_ok=True)
    write_csv(rows, inv/"codemap_index.csv")
    write_mermaid_import_graph(rows, inv/"codemap_imports.mmd")
    write_report(rows, data_info, inv/"codemap_report.md")

    print(f"Done in {dt:.2f}s\n- {inv/'codemap_report.md'}\n- {inv/'codemap_index.csv'}\n- {inv/'codemap_imports.mmd'}")

if __name__ == "__main__":
    sys.exit(main())
