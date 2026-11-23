#!/usr/bin/env python3
"""Compare every Python file of this folder with the original (frozen) copy of the same relative path.

For each .py file present in both trees it reports one of:
  identical      the bytes are the same;
  ast-identical  the files differ only in comments, docstrings or layout (same syntax tree once docstrings
                 are removed; line numbers are not compared);
  strings        the syntax tree is the same except for some string constants, which are listed;
  DIFFERENT      the code differs in anything else (exit status 1).

Usage: python3 scripts/release/check_ast.py ORIGINAL_ROOT [--root .] [--json report.json]
"""
import argparse, ast, json, os, sys


def strip_docstrings(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            b = node.body[0]
            if isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant) and isinstance(b.value.value, str):
                node.body = node.body[1:] or [ast.Pass()]
    return tree


def strings_blanked(tree):
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            found.append(node.value)
            node.value = ""
    return found


def compare(orig_path, new_path):
    a, b = open(orig_path, "rb").read(), open(new_path, "rb").read()
    if a == b:
        return "identical", []
    ta, tb = strip_docstrings(ast.parse(a)), strip_docstrings(ast.parse(b))
    if ast.dump(ta) == ast.dump(tb):
        return "ast-identical", []
    sa, sb = strings_blanked(ta), strings_blanked(tb)
    if ast.dump(ta) == ast.dump(tb) and len(sa) == len(sb):
        return "strings", [(x, y) for x, y in zip(sa, sb) if x != y]
    return "DIFFERENT", []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("original_root")
    ap.add_argument("--root", default=".")
    ap.add_argument("--json")
    a = ap.parse_args()
    counts, report, missing = {}, [], []
    for dp, dn, fn in os.walk(a.root):
        dn[:] = sorted(d for d in dn if d not in ("__pycache__", ".git"))
        for f in sorted(fn):
            if not f.endswith(".py"):
                continue
            rel = os.path.relpath(os.path.join(dp, f), a.root)
            orig = os.path.join(a.original_root, rel)
            if not os.path.exists(orig):
                missing.append(rel)
                continue
            status, changed = compare(orig, os.path.join(a.root, rel))
            counts[status] = counts.get(status, 0) + 1
            if status != "identical":
                report.append({"file": rel, "status": status, "changed_strings": changed})
    for r in report:
        print("%-14s %s" % (r["status"], r["file"]))
        for x, y in r["changed_strings"]:
            print("    - %r\n    + %r" % (x, y))
    print("summary:", json.dumps(counts, sort_keys=True), "| not in the original tree:", len(missing))
    for m in missing:
        print("  (no original) %s" % m)
    if a.json:
        json.dump({"counts": counts, "files": report, "no_original": missing}, open(a.json, "w"), indent=1,
                  ensure_ascii=False)
    sys.exit(1 if counts.get("DIFFERENT") else 0)


if __name__ == "__main__":
    main()
