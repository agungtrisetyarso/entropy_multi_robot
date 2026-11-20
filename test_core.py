"""Unit checks of the core library against brute-force enumeration."""
import sys
import numpy as np
import os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fg import *

rng = np.random.default_rng(1)
worst = dict(obj=0, phi=0, mono=0, tree_exact=0, logZ=0, samp=0, fcd_vs_lbfgs=[])
for trial in range(40):
    n = int(rng.integers(3, 7)); m = int(rng.integers(2, 4))
    inst = random_instance(n, m, rng, graph=["random", "cycle", "complete"][trial % 3],
                           coupling=["general", "congestion"][trial % 2],
                           hetero_prior=(trial % 4 == 0))
    lam = float(rng.choice([0.3, 1.0, 3.0]))
    en = Enum(inst)
    wts = inst.weights()
    order = kruskal_order(n, wts)
    k = int(rng.integers(0, len(order) + 1))
    F = order[:k]
    P = ForestPolicy(inst, F, lam)
    # random parameters
    P.T = rng.dirichlet(np.ones(m), size=(n, m)); P.q0 = rng.dirichlet(np.ones(m), size=n); P.refresh()
    q = P.joint(en.X)
    assert abs(q.sum() - 1) < 1e-12
    Jt, ect, klt = P.objective(); Je, ece, kle = en.J(q, lam)
    worst["obj"] = max(worst["obj"], abs(Jt - Je), abs(ect - ece), abs(klt - kle))
    # brute-force Phi for each block: E_q[C + lam*sum_{k!=j} l_k | x_pa=a, x_j=b]
    f = en.C + lam * (np.log(q) - en.logmu)
    for j in range(n):
        p = P.pa[j]
        if p >= 0:
            own = np.log(P.T[j][en.X[:, p], en.X[:, j]]) - np.log(inst.mu[j][en.X[:, j]])
            key = en.X[:, p] * m + en.X[:, j]
            num = np.bincount(key, weights=q * (f - lam * own), minlength=m * m).reshape(m, m)
            den = np.bincount(key, weights=q, minlength=m * m).reshape(m, m)
        else:
            own = np.log(P.q0[j][en.X[:, j]]) - np.log(inst.mu[j][en.X[:, j]])
            num = np.bincount(en.X[:, j], weights=q * (f - lam * own), minlength=m)
            den = np.bincount(en.X[:, j], weights=q, minlength=m)
        brute = num / den
        ph = P.phi(j)
        diff = brute - ph
        diff = diff - diff.mean(-1, keepdims=True)   # constants in b are irrelevant
        worst["phi"] = max(worst["phi"], np.abs(diff).max())
    # monotone descent
    hist = P.fcd(record=True, max_sweeps=300)
    worst["mono"] = max(worst["mono"], max(np.diff(hist).max(), 0))
    # compare with L-BFGS global-ish optimum
    Pb = solve_forest(inst, F, lam, restarts=4, seed=trial)
    _, Jl = en.opt_forest(F, lam, restarts=6, seed=trial)
    worst["fcd_vs_lbfgs"].append(Pb.objective()[0] - Jl)
    # sampling law
    Xs = Pb.sample(200000, rng)
    worst["samp"] = max(worst["samp"], abs(inst.cost(Xs).mean() - Pb.objective()[1]) /
                        (inst.cost(Xs).std() / np.sqrt(len(Xs)) + 1e-12))
    # exact logZ by elimination
    _, Jstar = en.gibbs(lam)
    worst["logZ"] = max(worst["logZ"], abs(-lam * exact_logZ(inst, lam) - Jstar))
    assert Pb.objective()[0] >= Jstar - 1e-9

# exactness on forests after ONE backward sweep
for trial in range(20):
    n = int(rng.integers(3, 8)); m = int(rng.integers(2, 4))
    inst0 = random_instance(n, m, rng, graph="random", p=0.0, coupling="general", hetero_prior=True)
    lam = float(rng.choice([0.2, 1.0, 5.0]))
    en = Enum(inst0); _, Jstar = en.gibbs(lam)
    P = ForestPolicy(inst0, inst0.edges, lam)
    P.T = rng.dirichlet(np.ones(m), size=(n, m)); P.q0 = rng.dirichlet(np.ones(m), size=n); P.refresh()
    for j in reversed(P.order):
        P.block_update(j)
    worst["tree_exact"] = max(worst["tree_exact"], abs(P.objective()[0] - Jstar))

d = np.array(worst.pop("fcd_vs_lbfgs"))
print({k: float(v) for k, v in worst.items()})
print("FCD - LBFGS: min %.2e  median %.2e  max %.2e ; frac |diff|<1e-7: %.2f ; frac FCD worse by >1e-6: %.2f"
      % (d.min(), np.median(d), d.max(), np.mean(np.abs(d) < 1e-7), np.mean(d > 1e-6)))
