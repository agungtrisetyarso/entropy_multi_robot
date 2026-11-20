"""Experiment F: warehouse AMR dispatch with geometry-derived costs.
Pairwise costs come from shared cells of obstacle-aware shortest paths
(aisle interference) plus queueing at a common station, so the coupling is
NOT a pure duplicate-task penalty.   usage: python3 warehouse.py small|large|example"""
import sys, json, time
from collections import deque
import numpy as np
import os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fg import *

OUT = os.path.join(HERE, "..", "results") + os.sep
os.makedirs(OUT, exist_ok=True)
H, WD = 14, 20
TAU, GAMMA, WQ = 0.1, 0.15, 1.0     # cost per cell, per shared cell, per queued pair


def layout():
    free = np.ones((H, WD), bool)
    for c in range(WD):
        if c % 3 == 1:
            free[2:6, c] = False
            free[8:12, c] = False
    return free


def bfs(free, src):
    dist = -np.ones(free.shape, int)
    prev = {}
    dist[src] = 0
    dq = deque([src])
    while dq:
        u = dq.popleft()
        for dr, dc in ((1, 0), (0, 1), (-1, 0), (0, -1)):      # fixed tie-breaking
            v = (u[0] + dr, u[1] + dc)
            if 0 <= v[0] < H and 0 <= v[1] < WD and free[v] and dist[v] < 0:
                dist[v] = dist[u] + 1
                prev[v] = u
                dq.append(v)
    return dist, prev


def make_instance(n, m, rng):
    free = layout()
    stations = [(H - 1, int(round((k + 0.5) * WD / m - 0.5))) for k in range(m)]
    cells = [(r, c) for r in range(0, H - 2) for c in range(WD) if free[r, c]]
    idx = rng.choice(len(cells), size=n, replace=False)
    robots = [cells[k] for k in idx]
    c = np.zeros((n, m))
    paths = [[None] * m for _ in range(n)]
    for a, s in enumerate(stations):
        dist, prev = bfs(free, s)
        for i, r in enumerate(robots):
            c[i, a] = TAU * dist[r]
            p = set()
            u = r
            while u != s:
                u = prev[u]            # walk towards the station
                p.add(u)
            paths[i][a] = p            # cells entered on the way, station included
    W = {}
    for i in range(n):
        for j in range(i + 1, n):
            w = np.zeros((m, m))
            for a in range(m):
                for b in range(m):
                    w[a, b] = GAMMA * len(paths[i][a] & paths[j][b])
            w += WQ * np.eye(m)
            W[(i, j)] = w
    return Instance(c, W)


def blind_softmax(inst, lam):
    """interaction-blind independent softmax: q_i ~ mu_i exp(-c_i/lam)"""
    P = ForestPolicy(inst, [], lam)
    a = np.log(inst.mu) - inst.c / lam
    P.q0 = np.exp(a - logsumexp(a, axis=1, keepdims=True))
    P.refresh()
    return P


def match_feasible(inst, F, h, lo=0.02, hi=200.0, iters=60):
    """Bisection on lambda for a class-F policy whose randomisation is at
    least the required one: returns the policy with the largest KL(q||mu)
    that does not exceed h.  Its expected cost is an upper bound on the best
    cost of the class at level h."""
    warm = None
    best = None
    for _ in range(iters):
        mid = np.sqrt(lo * hi)
        P = solve_forest(inst, F, mid, restarts=1, seed=1, warm=warm, tol=1e-13)
        kl = P.objective()[2]
        if kl <= h and (best is None or kl > best[1]):
            best = (P, kl, mid)
        if kl > h:
            lo = mid
        else:
            hi = mid
            warm = (P.q0, P.T)
        if 0 <= h - kl < 1e-7 * max(1.0, h):
            break
    return best


def expected_queue_pairs(P):
    tot = 0.0
    for (i, j) in P.inst.edges:
        tot += float(np.einsum("c,cc->", P.q[i], P.A[i, j]))
    return tot


def icm(inst, x, sweeps=50):
    """iterated conditional modes (deterministic local search)"""
    x = x.copy()
    for _ in range(sweeps):
        changed = False
        for i in range(inst.n):
            cost = inst.c[i].copy()
            for j in range(inst.n):
                if j != i:
                    cost += inst.pair(i, j)[:, x[j]]
            b = int(np.argmin(cost))
            if b != x[i]:
                x[i] = b
                changed = True
        if not changed:
            break
    return x


def describe(P, Js=None):
    J, ec, kl = P.objective()
    d = dict(J=J, ec=ec, H=P.entropy(), queue=expected_queue_pairs(P), msgs=P.messages(),
             rounds=P.rounds())
    if Js is not None:
        d["D"] = J - Js
    return d


def small():
    n, m, NI = 6, 4, 30
    lams = [0.25, 0.5, 1.0]
    res = dict(n=n, m=m, NI=NI, lams=lams, rows=[], params=dict(TAU=TAU, GAMMA=GAMMA, WQ=WQ, H=H, W=WD))
    rng = np.random.default_rng(7000)
    S_list = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    res["S_list"] = S_list
    from exp import product_starts
    for t in range(NI):
        inst = make_instance(n, m, rng)
        en = Enum(inst)
        w = inst.weights()
        order = kruskal_order(n, w)
        V = sum(w.values())
        Cmin = float(en.C.min())
        co = sum((en.X[:, i] == en.X[:, j]).astype(float) for (i, j) in inst.edges)
        row = dict(Cmin=Cmin, V=V, vshare_tree=sum(w[e] for e in order) / V, lam={})
        starts = product_starts(inst, en, np.geomspace(0.1, 20, 12), seed=t)
        for lam in lams:
            qJ, Js = en.gibbs(lam)
            _, ecJ, klJ = en.J(qJ, lam)
            r = dict(joint=dict(J=Js, ec=ecJ, H=en.entropy(qJ), queue=float(qJ @ co), D=0.0))
            r["blind"] = describe(blind_softmax(inst, lam), Js)
            pols = {}
            for k in range(0, n):
                if k == 0:
                    pols[k] = best_product(inst, en, lam, seed=t)
                else:
                    # nested forests: warm start from the previous optimum, so
                    # that the reported gaps are monotone in k
                    pols[k] = solve_forest(inst, order[:k], lam, restarts=3, seed=t,
                                           init="both", warm_policies=[pols[k - 1]])
                r["k%d" % k] = describe(pols[k], Js)
            # matched-randomisation comparison (same KL as the joint optimum)
            # best factorized policy with KL <= KL of the joint optimum (SLSQP
            # on the enumerated family), and a feasible spanning-forest policy
            qMp, _, klp, _ = en.constrained([], klJ, starts)
            Mt, klt, _ = match_feasible(inst, order, klJ)
            qMt = Mt.joint(en.X)
            r["matched"] = dict(prod_ec=float(qMp @ en.C), tree_ec=float(qMt @ en.C), joint_ec=ecJ,
                                prod_queue=float(qMp @ co), tree_queue=float(qMt @ co),
                                joint_queue=float(qJ @ co), klp_err=float(klp - klJ),
                                klt_err=float(klt - klJ))
            r["BS_joint"] = (en.best_of_S(qJ, S_list) - Cmin).tolist()
            r["BS_prod"] = (en.best_of_S(qMp, S_list) - Cmin).tolist()
            r["BS_tree"] = (en.best_of_S(qMt, S_list) - Cmin).tolist()
            row["lam"][str(lam)] = r
        res["rows"].append(row)
        print("small", t, flush=True)
    json.dump(res, open(OUT + "F_small.json", "w"), indent=1)


def large():
    n, m, NI, lam = 30, 6, 10, 0.5
    ks = [0, 5, 10, 15, 20, 25, 29]
    res = dict(n=n, m=m, NI=NI, lam=lam, ks=ks, rows=[])
    rng = np.random.default_rng(7100)
    for t in range(NI):
        inst = make_instance(n, m, rng)
        w = inst.weights()
        order = kruskal_order(n, w)
        V = sum(w.values())
        row = dict(V=V, E=len(inst.edges))
        row["blind"] = describe(blind_softmax(inst, lam))
        # best deterministic assignment found by ICM from several starts (heuristic)
        r2 = np.random.default_rng(t)
        best = np.inf
        starts = [np.argmin(inst.c, axis=1)] + [r2.integers(0, m, n) for _ in range(200)]
        for x0 in starts:
            x = icm(inst, np.array(x0))
            best = min(best, float(inst.cost(x[None, :])[0]))
        row["det_icm"] = best
        prev = None
        for k in ks:
            t0 = time.time()
            P, _ = forest_gibbs(inst, order[:k], lam)
            hist = P.fcd(record=True, max_sweeps=400, tol=1e-9)
            if prev is not None:
                # the forests are nested, so the previous policy is feasible:
                # continue FCD from it and keep the better of the two results
                P2 = ForestPolicy(inst, order[:k], lam).embed(prev)
                hist2 = P2.fcd(record=True, max_sweeps=400, tol=1e-9)
                if P2.objective()[0] < P.objective()[0]:
                    P, hist = P2, hist2
            prev = P
            el = time.time() - t0
            d = describe(P)
            X = P.sample(20000, np.random.default_rng(100 + t))
            cs = inst.cost(X)
            d.update(time=el, sweeps=P.sweeps, mc_mean=float(cs.mean()),
                     mc_se=float(cs.std() / np.sqrt(len(cs))),
                     mono=float(max(np.diff(hist).max(), 0.0)),
                     pred_gain=float(sum(w[e] for e in order[:k]) / (2 * lam)))
            row["k%d" % k] = d
            print("large", t, k, round(d["J"], 3), round(d["ec"], 3), P.sweeps, round(el, 1), flush=True)
        res["rows"].append(row)
        json.dump(res, open(OUT + "F_large.json", "w"), indent=1)


def example():
    """Two-robot, two-dock running example (all numbers quoted in the paper)."""
    c = np.array([[0.2, 1.0], [0.8, 0.4]])
    W = {(0, 1): 2.0 * np.eye(2)}
    inst = Instance(c, W)
    en = Enum(inst)
    out = dict(C=en.C.tolist(), X=en.X.tolist(), v=inst.v((0, 1)), lam={})
    for lam in [0.5, 1.0, 2.0, 4.0]:
        qJ, Js = en.gibbs(lam)
        P = solve_forest(inst, [], lam, restarts=5)
        J, ec, kl = P.objective()
        _, ecJ, klJ = en.J(qJ, lam)
        M, klm, lamm = match_feasible(inst, [], klJ)
        qM = M.joint(en.X)
        out["lam"][str(lam)] = dict(
            joint=dict(pi=qJ.tolist(), J=Js, ec=ecJ, kl=klJ, H=en.entropy(qJ),
                       collide=float(qJ[0] + qJ[3])),
            prod=dict(q1=P.q[0].tolist(), q2=P.q[1].tolist(), J=J, ec=ec, kl=kl, H=P.entropy(),
                      collide=float(P.q[0] @ P.q[1])),
            gap=J - Js, var_law=inst.v((0, 1)) / (2 * lam),
            matched=dict(q1=M.q[0].tolist(), q2=M.q[1].tolist(), ec=float(qM @ en.C), kl=klm,
                         H=M.entropy(), collide=float(M.q[0] @ M.q[1]), lam=lamm))
    json.dump(out, open(OUT + "G_example.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    t0 = time.time()
    {"small": small, "large": large, "example": example}[sys.argv[1]]()
    print("elapsed", time.time() - t0)
