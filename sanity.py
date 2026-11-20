import sys, time
import numpy as np
import os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fg import *
rng = np.random.default_rng(7)
# variance law quick look
for coupling in ["congestion", "general"]:
    inst = random_instance(5, 3, rng, graph="random", coupling=coupling, p=0.4)
    en = Enum(inst); w = inst.weights(); V = sum(w.values()); order = kruskal_order(5, w)
    print(coupling, "edges", len(inst.edges), "V_int", round(V, 4), "forest edges", len(order))
    for lam in [0.5, 1, 4, 16, 64, 256]:
        _, Js = en.gibbs(lam)
        row = []
        for k in [0, 2, len(order)]:
            P = solve_forest(inst, order[:k], lam, tol=1e-14)
            D = P.objective()[0] - Js
            Vres = V - sum(w[e] for e in order[:k])
            row.append((D, D * 2 * lam / Vres if Vres > 1e-12 else float('nan')))
        print("  lam %6.1f " % lam + "  ".join("k: D=%.3e ratio=%.4f" % r for r in row))
# closed form for congestion weights
inst = random_instance(4, 5, rng, graph="complete", coupling="congestion")
for e in inst.edges[:2]:
    wv = inst.W[e][0, 0]
    print("v_e", inst.v(e), "closed form", wv ** 2 * (5 - 1) / 25)
# timing at scale
inst = random_instance(128, 4, rng, graph="ladder", coupling="general")
w = inst.weights(); order = kruskal_order(128, w)
t = time.time(); P = ForestPolicy(inst, order, 1.0); print("build", time.time() - t, "depth", P.rounds())
t = time.time(); h = P.fcd(record=True, max_sweeps=3, tol=0); print("3 sweeps", time.time() - t, h)
t = time.time(); lz = exact_logZ(inst, 1.0); print("logZ", time.time() - t, -lz, "gap", h[-1] + lz)
