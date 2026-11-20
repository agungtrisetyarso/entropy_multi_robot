"""Numerical checks of statements that the experiments do not exercise directly:
Corollary 1 (congestion closed form, non-uniform prior), identity (13) for the
forest-restricted Gibbs policy, and the exact best-of-S formula.
usage: python3 verify_claims.py"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from scipy.special import logsumexp
from fg import *

rng = np.random.default_rng(11)
worst = dict(cor1=0.0, mgf=0.0, bestS_z=0.0, resampling=0.0)
for trial in range(30):
    n = int(rng.integers(3, 6)); m = int(rng.integers(2, 5))
    inst = random_instance(n, m, rng, graph="random", coupling=["congestion", "general"][trial % 2],
                           hetero_prior=True)
    lam = float(rng.choice([0.3, 1.0, 4.0]))
    en = Enum(inst)
    # Corollary 1
    if trial % 2 == 0:
        for (i, j) in inst.edges:
            w = inst.W[(i, j)][0, 0]
            mi, mj = inst.mu[i], inst.mu[j]
            s = float(mi @ mj)
            closed = w ** 2 * (s + s ** 2 - float(np.sum(mi * mj * (mi + mj))))
            worst["cor1"] = max(worst["cor1"], abs(closed - inst.v((i, j))))
    # identity (13): J(q~_F) - J* = lam log E_{q~}[exp(-(R - E R)/lam)]
    order = kruskal_order(n, inst.weights())
    F = order[: int(rng.integers(0, len(order) + 1))]
    P, _ = forest_gibbs(inst, F, lam)
    q = P.joint(en.X)
    inst2, const = inst.residual_instance(F)
    R = en.C - (inst2.cost(en.X) + const)
    _, Js = en.gibbs(lam)
    lhs = P.objective()[0] - Js
    ER = float(q @ R)
    rhs = lam * float(logsumexp(np.log(q) - (R - ER) / lam))
    worst["mgf"] = max(worst["mgf"], abs(lhs - rhs))
    # best-of-S: exact formula against Monte Carlo, and the two-sided resampling law
    qJ, _ = en.gibbs(lam)
    S_list = [1, 3, 8]
    exact = en.best_of_S(qJ, S_list)
    idx = rng.choice(en.d, size=(40000, max(S_list)), p=qJ)
    cs = en.C[idx]
    for S, ex in zip(S_list, exact):
        mins = cs[:, :S].min(1)
        se = mins.std() / np.sqrt(len(mins))
        if se > 1e-9:
            worst["bestS_z"] = max(worst["bestS_z"], abs(mins.mean() - ex) / se)
        else:                                 # degenerate sample: compare directly
            assert abs(mins.mean() - ex) < 1e-4, (mins.mean(), ex)   # rare events unseen by MC
        Cmin = en.C.min()
        pmin = qJ[en.C <= Cmin + 1e-12].sum()
        vals = np.unique(np.round(en.C, 12))
        delta = vals[1] - vals[0]
        lo, hi = delta * (1 - pmin) ** S, (en.C.max() - Cmin) * (1 - pmin) ** S
        worst["resampling"] = max(worst["resampling"], lo - (ex - Cmin), (ex - Cmin) - hi)
print({k: float(v) for k, v in worst.items()})
assert worst["cor1"] < 1e-12 and worst["mgf"] < 1e-10 and worst["bestS_z"] < 4.5 and worst["resampling"] < 1e-10
print("all checks passed")

# ---- Proposition 3 (re-centring at a blockwise factorized optimum):
#      J(q) - J* = lam * log E_q[ exp(-R^q / lam) ],  R^q = sum_e W~_e centred under q
worst_rc = 0.0
for trial in range(30):
    n = int(rng.integers(3, 7)); m = int(rng.integers(2, 4))
    inst = random_instance(n, m, rng, graph="random", coupling=["congestion", "general"][trial % 2],
                           hetero_prior=(trial % 3 == 0))
    lam = float(rng.choice([0.5, 1.0, 4.0]))
    en = Enum(inst)
    P = solve_forest(inst, [], lam, restarts=2, seed=trial, tol=1e-15, max_sweeps=5000)
    for _ in range(200):                       # polish to a blockwise optimum
        for j in range(n):
            P.block_update(j)
    q = P.q
    instq = Instance(inst.c, inst.W, q)
    R = np.zeros(en.d)
    for (i, j) in inst.edges:
        R += instq.centred((i, j))[en.X[:, i], en.X[:, j]]
    qx = P.joint(en.X)
    _, Js = en.gibbs(lam)
    lhs = P.objective()[0] - Js
    rhs = lam * float(logsumexp(np.log(qx) - R / lam))
    worst_rc = max(worst_rc, abs(lhs - rhs) / max(1.0, abs(lhs)))
    assert lhs <= instq.bd_sq_sum() / (8 * lam) + 1e-9          # a posteriori bound (17)
    assert lhs <= (R.max() - R.min()) + 1e-9
print("re-centring identity, worst relative error:", worst_rc)
assert worst_rc < 1e-8
