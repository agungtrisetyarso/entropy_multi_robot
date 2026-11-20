"""Fills the @@placeholders@@ of ../paper/main_src.tex from
../results/numbers.json and writes ../paper/main.tex.
usage: python3 build.py [--draft]   (--draft tolerates missing numbers)"""
import json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "..", "paper", "main_src.tex")).read()
N = json.load(open(os.path.join(HERE, "..", "results", "numbers.json")))
missing = []
def sub(m):
    k = m.group(1)
    if k in N:
        return N[k]
    missing.append(k)
    return "" if k.endswith("_ROWS") else r"\textbf{??}"
out = re.sub(r"@@([A-Za-z0-9_]+)@@", sub, src)
if missing and "--draft" not in sys.argv:
    sys.exit("missing numbers: " + ", ".join(sorted(set(missing))))
open(os.path.join(HERE, "..", "paper", "main.tex"), "w").write(out)
print("written main.tex; missing:", sorted(set(missing)))
