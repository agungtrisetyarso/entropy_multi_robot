"""Wall-clock time of policy synthesis (run on an otherwise idle machine).
usage: python3 timing.py   -> ../results/T.json"""
import sys, os, json, time, platform
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from fg import *

out = {}
m, lam = 4, 1.0
for n in [32, 128]:
    rng = np.random.default_rng(5000 + n)
    ts = []
    for t in range(3):
        inst = random_instance(n, m, rng, graph="ladder", coupling="general", gamma=0.5)
        order = kruskal_order(n, inst.weights())
        t0 = time.perf_counter()
        P, _ = forest_gibbs(inst, order, lam)
        P.fcd(max_sweeps=400, tol=1e-10)
        ts.append(time.perf_counter() - t0)
    out[str(n)] = float(np.median(ts))
    print(n, ts)
out["machine"] = platform.processor() or platform.machine()
out["python"] = platform.python_version()
out["numpy"] = np.__version__
json.dump(out, open(os.path.join(HERE, "..", "results", "T.json"), "w"), indent=1)
print(out)
