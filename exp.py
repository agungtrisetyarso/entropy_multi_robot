"""Experiments A-E.  usage: python3 exp.py A|B|C|E
All numbers are exact (enumeration / variable elimination) unless a Monte
Carlo estimate is explicitly named.  Results are written to ../results/*.json"""
import sys, json, time, itertools
import numpy as np
import os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fg import *

OUT = os.path.join(HERE, "..", "results") + os.sep
os.makedirs(OUT, exist_ok=True)


def q3(a):
    a = np.asarray(a, float)
    return [float(np.nanpercentile(a, 25)), float(np.nanmedian(a)), float(np.nanpercentile(a, 75))]


# ---------------------------------------------------------------- A
def exp_A():
    """Variance law and uniform bounds."""
    configs = [
        dict(name="congestion, random graph, (n,m)=(5,3)", n=5, m=3, graph="random", coupling="congestion", p=0.4),
        dict(name="general, random graph, (n,m)=(5,3)", n=5, m=3, graph="random", coupling="general", p=0.4),
        dict(name="general, cycle, (n,m)=(6,3)", n=6, m=3, graph="cycle", coupling="general", p=0),
        dict(name="congestion, complete, (n,m)=(6,4)", n=6, m=4, graph="complete", coupling="congestion", p=0),
        dict(name="general, random graph, non-uniform prior, (n,m)=(5,4)", n=5, m=4, graph="random",
             coupling="general", p=0.4, hetero_prior=True),
    ]
    lams = [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256]
    NI = 30
    res = dict(lams=lams, configs=[])
    max_viol = 0.0
    for ci, cf in enumerate(configs):
        rng = np.random.default_rng(1000 + ci)
        rat = {key: np.full((NI, len(lams)), np.nan) for key in ["prod", "half", "span"]}
        rem = {key: np.full((NI, len(lams)), np.nan) for key in ["prod", "half", "span"]}
        gap = {key: np.full((NI, len(lams)), np.nan) for key in ["prod", "half", "span"]}
        tight = np.full((NI, len(lams)), np.nan)
        for t in range(NI):
            inst = random_instance(cf["n"], cf["m"], rng, graph=cf["graph"], coupling=cf["coupling"],
                                   p=cf["p"], hetero_prior=cf.get("hetero_prior", False))
            en = Enum(inst)
            w = inst.weights()
            V = sum(w.values())
            order = kruskal_order(inst.n, w)
            ks = dict(prod=0, half=len(order) // 2, span=len(order))
            for li, lam in enumerate(lams):
                _, Js = en.gibbs(lam)
                prev = []
                for key, k in ks.items():
                    F = order[:k]
                    if k == 0:
                        P = best_product(inst, en, lam, seed=t, restarts=(6 if lam < 2 else 1))
                        P.fcd(tol=1e-15)
                    else:
                        P = solve_forest(inst, F, lam, restarts=(3 if lam < 2 else 0), seed=t,
                                         tol=1e-15, init="both", warm_policies=prev)
                    prev = prev + [P]
                    D = P.objective()[0] - Js
                    Vres = V - sum(w[e] for e in F)
                    gap[key][t, li] = D
                    if Vres > 1e-9:
                        rat[key][t, li] = D * 2 * lam / Vres
                        rem[key][t, li] = (D - Vres / (2 * lam)) * lam ** 2
                    # uniform bounds with exact osc(R_F)
                    inst2, const = inst.residual_instance(F)
                    R = en.C - (inst2.cost(en.X) + const)
                    osc = R.max() - R.min()
                    b = min(osc, osc ** 2 / (8 * lam))
                    if k == 0:
                        b = min(b, inst.bd_sq_sum() / (8 * lam))
                        tight[t, li] = D / b
                    max_viol = max(max_viol, D - b)
        res["configs"].append(dict(
            name=cf["name"],
            ratio={k: [q3(rat[k][:, li]) for li in range(len(lams))] for k in rat},
            rem_l2={k: [q3(rem[k][:, li]) for li in range(len(lams))] for k in rem},
            gap={k: [q3(gap[k][:, li]) for li in range(len(lams))] for k in gap},
            tight_prod=[q3(tight[:, li]) for li in range(len(lams))],
            ratio_min={k: [float(np.nanmin(rat[k][:, li])) for li in range(len(lams))] for k in rat},
            ratio_max={k: [float(np.nanmax(rat[k][:, li])) for li in range(len(lams))] for k in rat},
        ))
        print(cf["name"], "done", flush=True)
    res["max_bound_violation"] = float(max_viol)
    res["n_instances"] = NI
    json.dump(res, open(OUT + "A.json", "w"), indent=1)


# ---------------------------------------------------------------- B
def exp_B():
    """Greedy max-interaction-variance forest versus exhaustive search."""
    lams = [0.5, 1.0, 4.0]
    NI = 12
    n, m = 6, 3
    res = dict(lams=lams, n=n, m=m, NI=NI, classes={})
    # optional third argument selects one family (the two families use
    # independent random streams and can be run in parallel)
    families = [sys.argv[2]] if len(sys.argv) > 2 else ["congestion", "general"]
    for coupling in families:
        rng = np.random.default_rng(2000 + (coupling == "general"))
        rows = []
        for t in range(NI):
            inst = random_instance(n, m, rng, graph="random", coupling=coupling, p=0.25,
                                   w_range=(0.3, 2.5))
            en = Enum(inst)
            w = inst.weights()
            V = sum(w.values())
            order = kruskal_order(n, w)
            E = inst.edges
            for lam in lams:
                _, Js = en.gibbs(lam)
                best = {frozenset(): best_product(inst, en, lam, seed=t)}
                D0 = best[frozenset()].objective()[0] - Js
                # weights re-centred at the factorized optimum (Corollary 2)
                qp = np.maximum(best[frozenset()].q, 1e-300)
                qp = qp / qp.sum(1, keepdims=True)
                order_ad = kruskal_order(n, Instance(inst.c, inst.W, qp).weights())
                F_fs = ()
                for k in range(1, len(order) + 1):
                    vals = {}
                    for F in itertools.combinations(E, k):
                        if not is_forest(n, F):
                            continue
                        # warm start from the best sub-forest optimum: the value of
                        # F is then at most the value of every forest it contains
                        subs = [best[frozenset(F) - {e}] for e in F]
                        sub = min(subs, key=lambda P_: P_.objective()[0])
                        P = solve_forest(inst, list(F), lam, restarts=1, seed=t, init="gibbs",
                                         warm_policies=[sub])
                        best[frozenset(F)] = P
                        vals[F] = P.objective()[0] - Js
                    Fg = tuple(sorted(order[:k]))
                    Dg = vals[Fg]
                    Db = min(vals.values())
                    Dad = vals[tuple(sorted(order_ad[:k]))]
                    # forward selection: add the single link with the best exact value
                    F_fs = min((F for F in vals if set(F_fs) <= set(F)), key=lambda F: vals[F])
                    rows.append(dict(inst=t, lam=lam, k=k, nforests=len(vals), D0=D0, Dg=Dg, Db=Db,
                                     Dad=Dad, Dfs=vals[F_fs],
                                     Dmean=float(np.mean(list(vals.values()))),
                                     Dworst=float(max(vals.values())),
                                     rank=int(sorted(vals.values()).index(Dg)) + 1,
                                     pred=float((V - sum(w[e] for e in order[:k])) / V),
                                     predbest=float(min((V - sum(w[e] for e in F)) / V for F in vals)),
                                     E=len(E)))
            print(coupling, t, flush=True)
            res["classes"][coupling] = rows
            json.dump(res, open(OUT + "B_%s.json" % "_".join(families), "w"), indent=1)


# ---------------------------------------------------------------- C (+D)
def lower_envelope(kl, ec):
    """Lower convex envelope of points (kl, ec) seen as cost-vs-KL-budget;
    returns a function h -> min cost with KL <= h (convexified)."""
    pts = sorted(zip(kl, ec))
    # nonincreasing running minimum (a larger KL budget can never hurt)
    xs, ys = [], []
    best = np.inf
    for x, y in pts:
        if y < best - 1e-15:
            best = y
            xs.append(x)
            ys.append(y)
    # lower convex hull
    hull = []
    for p in zip(xs, ys):
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) <= 0:
                hull.pop()
            else:
                break
        hull.append(p)
    hx = np.array([p[0] for p in hull])
    hy = np.array([p[1] for p in hull])
    return lambda h: float(np.interp(h, hx, hy))


def path_solve(inst, F, lams):
    """continuation over a descending lambda grid; returns list of policies"""
    out = []
    warm = None
    for lam in lams:
        P = solve_forest(inst, F, lam, restarts=(2 if lam < 1 else 0), seed=0, warm=warm, tol=1e-13)
        warm = (P.q0, P.T)
        out.append(P)
    return out


def match_kl(inst, F, h, lo=0.02, hi=200.0, iters=50):
    """bisection on lambda so that the class-F optimum has KL(q||mu) = h"""
    warm = None
    best = None
    for _ in range(iters):
        mid = np.sqrt(lo * hi)
        P = solve_forest(inst, F, mid, restarts=1, seed=1, warm=warm, tol=1e-13)
        kl = P.objective()[2]
        if best is None or abs(kl - h) < abs(best[1] - h):
            best = (P, kl, mid)
        if kl > h:
            lo = mid
        else:
            hi = mid
        if abs(kl - h) < 1e-6 * max(1.0, h):
            break
    return best


def best_scalar(inst, en, F, lam, seed=0, Pprod=None):
    """best-found optimum of J_lam over the forest class (see fg.best_product);
    forests are warm-started from the product optimum, so that the forest
    value never exceeds the product value."""
    if len(F) == 0:
        P = best_product(inst, en, lam, seed=seed)
        return P.joint(en.X), P.objective()[0], P
    P = solve_forest(inst, F, lam, restarts=4, seed=seed, init="both",
                     warm_policies=([Pprod] if Pprod is not None else []))
    qL, JL = en.opt_forest(F, lam, restarts=4, seed=seed)
    if JL < P.objective()[0] - 1e-10:
        return qL, JL, P
    return P.joint(en.X), P.objective()[0], P


def product_starts(inst, en, lam_grid, seed=0):
    """exponential-family parameters of product optima over a lambda grid"""
    ths = [np.zeros(inst.n * inst.m)]
    for lam in lam_grid:
        P = solve_forest(inst, [], lam, restarts=1, seed=seed)
        ths.append((np.log(P.q0) - np.log(inst.mu)).ravel())
        _, _, th = en.opt_forest([], lam, restarts=1, seed=seed, return_theta=True)
        ths.append(th)
    return ths


def exp_C():
    """Matched-randomisation premium in one-shot dispatch, and resampling law."""
    n, m = 6, 3
    NI = 20
    lam0s = [0.5, 1.0, 2.0]
    S_list = [2 ** k for k in range(0, 11)]
    res = dict(n=n, m=m, NI=NI, lam0s=lam0s, S_list=S_list, classes={})
    # resume support: instances are regenerated from the same random stream,
    # and instances whose rows are already on disk are not recomputed
    done = {}
    if os.path.exists(OUT + "C.json"):
        old = json.load(open(OUT + "C.json"))
        if (old.get("n"), old.get("m")) == (n, m):
            done = {k: [r for r in v["rows"] if r["inst"] < NI] for k, v in old["classes"].items()}
    for coupling in ["congestion", "general"]:
        rng = np.random.default_rng(3000 + (coupling == "general"))
        rows = list(done.get(coupling, []))
        have = set(r["inst"] for r in rows)
        for t in range(NI):
            inst = random_instance(n, m, rng, graph="random", coupling=coupling, p=0.4)
            if t in have:
                continue
            en = Enum(inst)
            w = inst.weights()
            order = kruskal_order(n, w)
            Cmin, Cmax = float(en.C.min()), float(en.C.max())
            co = sum((en.X[:, i] == en.X[:, j]).astype(float) for (i, j) in inst.edges)
            starts = product_starts(inst, en, np.geomspace(0.1, 20, 12), seed=t)
            for lam0 in lam0s:
                qJ, Js = en.gibbs(lam0)
                _, ecJ, hJ = en.J(qJ, lam0)
                qP, JP, Pp = best_scalar(inst, en, [], lam0, seed=t)
                qT, JT, _ = best_scalar(inst, en, order, lam0, seed=t, Pprod=Pp)
                Dp, Dt = JP - Js, JT - Js
                # best factorized policy at the randomisation level of the joint optimum
                qMp, ecMp, klMp, _ = en.constrained([], hJ, starts)
                # forest policy at the same level (bisection on lambda; an upper
                # bound on the best forest cost at that level)
                Mt, klt, lamt = match_kl(inst, order, hJ)
                qMt = Mt.joint(en.X)
                row = dict(inst=t, lam0=lam0, hJ=hJ, ecJ=ecJ, Cmin=Cmin, Cmax=Cmax, Dp=Dp, Dt=Dt,
                           prem_p=float(ecMp - ecJ), prem_t=float(qMt @ en.C - ecJ),
                           klp=float(klMp), klt_err=float(klt - hJ),
                           co_joint=float(qJ @ co), co_prod=float(qMp @ co), co_tree=float(qMt @ co),
                           ec_prod_same=float(qP @ en.C), kl_prod_same=float(en.J(qP, lam0)[2]),
                           ec_tree_same=float(qT @ en.C), kl_tree_same=float(en.J(qT, lam0)[2]),
                           Vint=float(sum(w.values())),
                           Vtree=float(sum(w[e] for e in order)),
                           pstar_joint=float(qJ[en.C <= Cmin + 1e-12].sum()),
                           pstar_prod=float(qMp[en.C <= Cmin + 1e-12].sum()),
                           delta2=float(np.sort(np.unique(np.round(en.C, 12)))[1] - Cmin))
                row["BS_joint"] = (en.best_of_S(qJ, S_list) - Cmin).tolist()
                row["BS_prod_matched"] = (en.best_of_S(qMp, S_list) - Cmin).tolist()
                row["BS_tree_matched"] = (en.best_of_S(qMt, S_list) - Cmin).tolist()
                row["BS_prod_same"] = (en.best_of_S(qP, S_list) - Cmin).tolist()
                row["BS_tree_same"] = (en.best_of_S(qT, S_list) - Cmin).tolist()
                rows.append(row)
            print(coupling, t, flush=True)
            res["classes"][coupling] = dict(rows=rows)
            json.dump(res, open(OUT + "C.json", "w"), indent=1)
    json.dump(res, open(OUT + "C.json", "w"), indent=1)


def exp_D():
    """Frontier curves (expected one-shot cost versus randomisation) for one
    representative instance of each coupling class; used for the figure."""
    n, m = 6, 3
    out = {}
    for coupling in ["congestion", "general"]:
        rng = np.random.default_rng(3000 + (coupling == "general"))
        inst = random_instance(n, m, rng, graph="random", coupling=coupling, p=0.4)
        en = Enum(inst)
        w = inst.weights()
        order = kruskal_order(n, w)
        hmax = float(n * np.log(m))
        lams = np.geomspace(60, 0.04, 60)
        joint = []
        for l in lams:
            q, _ = en.gibbs(l)
            _, ec, kl = en.J(q, l)
            joint.append([kl, ec])
        tree = [[P.objective()[2], P.objective()[1]] for P in path_solve(inst, order, lams)]
        starts = product_starts(inst, en, np.geomspace(0.1, 20, 12))
        hs = np.linspace(0.02, 0.98, 33) * hmax
        prod = []
        for h in hs:
            b = en.constrained([], h, starts)
            prod.append([b[2], b[1]])
            starts = starts + [b[3]]
        out[coupling] = dict(hmax=hmax, Cmin=float(en.C.min()), joint=joint, tree=tree, prod=prod,
                             Emu=float(np.exp(en.logmu) @ en.C))
        print(coupling, "done", flush=True)
    json.dump(out, open(OUT + "D.json", "w"), indent=1)


# ---------------------------------------------------------------- E
def exp_E():
    """Scale: ladder interaction graphs with exact reference by variable
    elimination; budget curve at n=64; Monte Carlo check of one-shot dispatch."""
    m, lam = 4, 1.0
    NI = 10
    res = dict(m=m, lam=lam, NI=NI, sizes=[], budget=None)
    for n in [8, 16, 32, 64, 128]:
        rng = np.random.default_rng(5000 + n)
        rows = []
        for t in range(NI):
            inst = random_instance(n, m, rng, graph="ladder", coupling="general", gamma=0.5)
            Js = -lam * exact_logZ(inst, lam)
            w = inst.weights()
            V = sum(w.values())
            order = kruskal_order(n, w)
            out = dict(V=V, Js=Js, E=len(inst.edges))
            for name, F in [("prod", []), ("tree", order)]:
                t0 = time.time()
                P, _ = forest_gibbs(inst, F, lam)
                J_init = P.objective()[0]
                hist = P.fcd(record=True, max_sweeps=400, tol=1e-10)
                el = time.time() - t0
                J, ec, kl = P.objective()
                X = P.sample(20000, np.random.default_rng(t))
                cs = inst.cost(X)
                out[name] = dict(D=J - Js, D_init=J_init - Js, sweeps=P.sweeps, time=el,
                                 time_per_sweep=el / max(P.sweeps, 1), ec=ec, kl=kl,
                                 mc_mean=float(cs.mean()), mc_se=float(cs.std() / np.sqrt(len(cs))),
                                 msgs=P.messages(), rounds=P.rounds(),
                                 mono=float(max(np.diff(hist).max(), 0.0)),
                                 pred=float((V - sum(w[e] for e in F)) / (2 * lam)),
                                 bound=float(min(inst.osc_sum(F), inst.osc_sum(F) ** 2 / (8 * lam))))
            out["prod"]["bound_bd"] = float(inst.bd_sq_sum() / (8 * lam))
            rows.append(out)
            print(n, t, round(out["prod"]["D"], 3), round(out["tree"]["D"], 3), out["tree"]["sweeps"],
                  round(out["tree"]["time"], 2), flush=True)
        res["sizes"].append(dict(n=n, rows=rows))
        json.dump(res, open(OUT + "E.json", "w"), indent=1)
    # budget curve at n = 64
    n = 64
    rng = np.random.default_rng(5000 + n)
    ks = [0, 4, 8, 16, 24, 32, 40, 48, 56, 63]
    curve = []
    for t in range(NI):
        inst = random_instance(n, m, rng, graph="ladder", coupling="general", gamma=0.5)
        Js = -lam * exact_logZ(inst, lam)
        w = inst.weights()
        V = sum(w.values())
        order = kruskal_order(n, w)
        rnd = np.random.default_rng(99 + t)
        row = dict(greedy=[], pred=[], random=[])
        for k in ks:
            P = solve_forest(inst, order[:k], lam, tol=1e-10)
            row["greedy"].append(P.objective()[0] - Js)
            row["pred"].append((V - sum(w[e] for e in order[:k])) / (2 * lam))
            # random forest with k links: Kruskal on shuffled weights
            wr = {e: rnd.random() for e in inst.edges}
            Fr = kruskal_order(n, wr)[:k]
            row["random"].append(solve_forest(inst, Fr, lam, tol=1e-10).objective()[0] - Js)
        curve.append(row)
        print("budget", t, flush=True)
    res["budget"] = dict(n=n, ks=ks, rows=curve)
    json.dump(res, open(OUT + "E.json", "w"), indent=1)


# ---------------------------------------------------------------- H
def exp_H():
    """A posteriori certificate of Proposition 2 at the factorized policy that
    FCD returns, on the instances of experiment E (exact reference available)."""
    m, lam = 4, 1.0
    NI = 10
    res = dict(m=m, lam=lam, NI=NI, sizes=[])
    for n in [8, 16, 32, 64, 128]:
        rng = np.random.default_rng(5000 + n)
        rows = []
        for t in range(NI):
            inst = random_instance(n, m, rng, graph="ladder", coupling="general", gamma=0.5)
            Js = -lam * exact_logZ(inst, lam)
            P, _ = forest_gibbs(inst, [], lam)
            P.fcd(max_sweeps=2000, tol=1e-13)
            q = np.maximum(P.q, 1e-300)
            q = q / q.sum(1, keepdims=True)
            instq = Instance(inst.c, inst.W, q)
            D = P.objective()[0] - Js
            # Monte Carlo evaluation of the exact identity lam*log E_q[exp(-R^q/lam)]
            X = P.sample(200000, np.random.default_rng(t))
            R = np.zeros(X.shape[0])
            for (i, j) in inst.edges:
                R += instq.centred((i, j))[X[:, i], X[:, j]]
            mc = lam * float(logsumexp(-R / lam) - np.log(len(R)))
            rows.append(dict(D=D, cert=instq.bd_sq_sum() / (8 * lam), prior_bound=inst.bd_sq_sum() / (8 * lam),
                             second=instq.V_int() / (2 * lam), mc=mc))
            print(n, t, round(D, 3), round(rows[-1]["cert"], 3), round(rows[-1]["second"], 3), round(mc, 3), flush=True)
        res["sizes"].append(dict(n=n, rows=rows))
    json.dump(res, open(OUT + "H.json", "w"), indent=1)


if __name__ == "__main__":
    t0 = time.time()
    {"A": exp_A, "B": exp_B, "C": exp_C, "D": exp_D, "E": exp_E, "H": exp_H}[sys.argv[1]]()
    print("elapsed", time.time() - t0)
