"""Collects every number quoted in the manuscript from ../results/*.json and
writes ../results/numbers.json (placeholder -> string).  build.py substitutes
them into the LaTeX source, so the text cannot drift from the results.
usage: python3 make_numbers.py"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results") + os.sep
N = {}


def load(name):
    return json.load(open(RES + name + ".json"))


def load_B():
    """experiment B is run per family (exp.py B congestion / exp.py B general)"""
    B = load("B_congestion")
    B["classes"].update(load("B_general")["classes"])
    return B


def f(x, d=2):
    return "%.*f" % (d, x)


def sci(x):
    e = int(np.floor(np.log10(abs(x))))
    return r"%.1f\times10^{%d}" % (x / 10 ** e, e)


def med(a):
    return float(np.median(a))


def iqr(a):
    return float(np.percentile(a, 75) - np.percentile(a, 25))


def pct(x, d=0):
    return "%.*f" % (d, 100 * x)


# ------------------------------------------------------------------ example
def example():
    G = load("G_example")
    N["EX_v"] = f(G["v"], 2)
    rows = []
    for lam in ["0.5", "1.0", "2.0", "4.0"]:
        g = G["lam"][lam]
        prem = g["matched"]["ec"] - g["joint"]["ec"]
        assert prem >= g["gap"] - 1e-6
        assert g["matched"]["kl"] <= g["joint"]["kl"] + 1e-9
        rows.append("$%s$ & $%s$ & $%s$ & $%s$ & $%s$ & $%s$ & $%s$ & $%s$\\\\" % (
            lam.rstrip("0").rstrip(".") if lam != "0.5" else "0.5",
            f(g["gap"], 3), f(g["var_law"], 3), f(g["joint"]["ec"], 3), f(g["joint"]["collide"], 3),
            f(g["matched"]["ec"], 3), f(g["matched"]["collide"], 3), f(prem, 3)))
    N["TAB_EXAMPLE_ROWS"] = "\n".join(rows)
    g = G["lam"]["1.0"]
    pi = g["joint"]["pi"]
    N["EX_pi"] = "(%s,\\,%s,\\,%s,\\,%s)" % tuple(f(p, 3) for p in pi)
    N["EX_piprod_a"] = f(pi[0] * pi[3], 4)
    N["EX_piprod_b"] = f(pi[1] * pi[2], 4)
    N["EX_J_ec"] = f(g["joint"]["ec"], 3)
    N["EX_J_co"] = f(g["joint"]["collide"], 3)
    N["EX_J_kl"] = f(g["joint"]["kl"], 3)
    N["EX_P_q1"] = "(%s,\\,%s)" % tuple(f(p, 3) for p in g["prod"]["q1"])
    N["EX_P_q2"] = "(%s,\\,%s)" % tuple(f(p, 3) for p in g["prod"]["q2"])
    N["EX_P_ec"] = f(g["prod"]["ec"], 3)
    N["EX_P_kl"] = f(g["prod"]["kl"], 3)
    N["EX_P_co"] = f(g["prod"]["collide"], 3)
    N["EX_gap"] = f(g["gap"], 3)
    N["EX_M_q1"] = "(%s,\\,%s)" % tuple(f(p, 3) for p in g["matched"]["q1"])
    N["EX_M_q2"] = "(%s,\\,%s)" % tuple(f(p, 3) for p in g["matched"]["q2"])
    N["EX_M_ec"] = f(g["matched"]["ec"], 3)
    N["EX_M_co"] = f(g["matched"]["collide"], 3)
    N["EX_prem"] = f(g["matched"]["ec"] - g["joint"]["ec"], 3)


# ------------------------------------------------------------------ A
def exp_A():
    A = load("A")
    lams = A["lams"]
    li = {l: i for i, l in enumerate(lams)}

    def rng(key, lam):
        v = [c["ratio"][key][li[lam]][1] for c in A["configs"]]
        return min(v), max(v)
    lo, hi = rng("prod", 16); N["A_r16_lo"], N["A_r16_hi"] = f(lo, 2), f(hi, 2)
    lo, hi = rng("prod", 256); N["A_r256_lo"] = f(np.floor(lo * 1000) / 1000, 3)
    lo, hi = rng("span", 16); N["A_s16_lo"], N["A_s16_hi"] = f(lo, 2), f(hi, 2)
    lo, hi = rng("span", 256); N["A_s256_lo"] = f(np.floor(lo * 1000) / 1000, 3)
    lo, hi = rng("prod", 1); N["A_r1_lo"], N["A_r1_hi"] = f(lo, 2), f(hi, 2)
    N["A_ncases"] = str(len(A["configs"]) * A["n_instances"] * len(lams) * 3)
    v = A["max_bound_violation"]
    N["A_viol"] = "$%s$" % sci(v) if v > 0 else "$0$"
    assert v < 1e-9
    t = [c["tight_prod"][li[l]][1] for c in A["configs"] for l in lams if l >= 16]
    N["A_tight_lo"], N["A_tight_hi"] = f(min(t), 2), f(max(t), 2)
    # every ratio must tend to 1
    for c in A["configs"]:
        for key in ("prod", "half", "span"):
            assert abs(c["ratio"][key][-1][1] - 1) < 0.01, (c["name"], key)


# ------------------------------------------------------------------ B
def exp_B():
    B = load_B()
    for cl in ("congestion", "general"):
        assert len(set(r["inst"] for r in B["classes"][cl])) == B["NI"], cl
    N["B_NI"] = str(B["NI"])
    N["B_nprob"] = str(B["NI"] * 5)
    allrows = [r for rows in B["classes"].values() for r in rows]
    N["B_Emin"] = str(min(r["E"] for r in allrows))
    N["B_Emax"] = str(max(r["E"] for r in allrows))
    N["B_nforests"] = "%d" % np.median([r["nforests"] for r in allrows])
    N["B_nforests_max"] = "%d" % max(r["nforests"] for r in allrows)
    lines = []
    st = {}
    rules = [("Dg", "g"), ("Dad", "a"), ("Dfs", "f")]
    for cl in ["congestion", "general"]:
        for lam in B["lams"]:
            rr = [r for r in B["classes"][cl] if r["lam"] == lam]
            assert len(rr) == B["NI"] * 5
            assert all(r["Db"] <= min(r["Dg"], r["Dad"], r["Dfs"]) + 1e-9 for r in rr)
            assert all(r["Dworst"] <= r["D0"] + 1e-7 for r in rr), "nesting violated"
            cells = []
            for key, tag in rules:
                opt = np.mean([r[key] <= r["Db"] + 1e-7 * max(1, r["D0"]) for r in rr])
                exc = np.array([(r[key] - r["Db"]) / r["D0"] for r in rr])
                st[(cl, lam, tag)] = dict(opt=opt, med=med(exc), p90=float(np.percentile(exc, 90)), mx=float(exc.max()))
                cells.append("$%s$ / $%s$" % (pct(opt), pct(np.percentile(exc, 90), 1)))
            excm = np.array([(r["Dmean"] - r["Db"]) / r["D0"] for r in rr])
            st[(cl, lam, "m")] = dict(med=med(excm))
            full = [r for r in rr if r["k"] == 5]
            cb = med([1 - r["Db"] / r["D0"] for r in full])
            lines.append("%s & $%g$ & %s & $%s$ & $%s$\\\\" % (
                cl if lam == B["lams"][0] else "", lam, " & ".join(cells), pct(med(excm), 1), pct(cb)))
        if cl == "congestion":
            lines.append("\\addlinespace")
    N["TAB_LINKS_ROWS"] = "\n".join(lines)
    g = lambda cl, lam, tag, k: st[(cl, lam, tag)][k]
    N["B_opt4_c"] = pct(g("congestion", 4.0, "g", "opt")); N["B_opt4_g"] = pct(g("general", 4.0, "g", "opt"))
    N["B_exc4_max"] = pct(max(g("congestion", 4.0, "g", "mx"), g("general", 4.0, "g", "mx")), 1)
    N["B_opt05_c"] = pct(g("congestion", 0.5, "g", "opt")); N["B_opt05_g"] = pct(g("general", 0.5, "g", "opt"))
    N["B_p90_05_c"] = pct(g("congestion", 0.5, "g", "p90")); N["B_p90_05_g"] = pct(g("general", 0.5, "g", "p90"))
    N["B_ad_p90_05_c"] = pct(g("congestion", 0.5, "a", "p90")); N["B_ad_p90_05_g"] = pct(g("general", 0.5, "a", "p90"))
    fs_opt = [st[k]["opt"] for k in st if k[2] == "f"]
    fs_p90 = [st[k]["p90"] for k in st if k[2] == "f"]
    N["B_fs_opt_lo"], N["B_fs_opt_hi"] = pct(min(fs_opt)), pct(max(fs_opt))
    N["B_fs_p90_max"] = pct(max(fs_p90), 1)
    N["B_fs_mx"] = pct(max(st[k]["mx"] for k in st if k[2] == "f"), 1)
    mm = [st[k]["med"] for k in st if k[2] == "m"]
    N["B_mean_lo"], N["B_mean_hi"] = pct(min(mm)), pct(max(mm))
    return st


# ------------------------------------------------------------------ C
def exp_C():
    C = load("C")
    N["C_NI"] = str(C["NI"])
    N["C_nstarts"] = "25"
    S = C["S_list"]
    lines = []
    rec = []
    ncase = 0
    minslack = np.inf
    for cl in ["congestion", "general"]:
        rows = C["classes"][cl]["rows"]
        assert len(rows) == 3 * C["NI"], (cl, len(rows))
        for lam in C["lam0s"]:
            rr = [r for r in rows if r["lam0"] == lam]
            ncase += len(rr)
            minslack = min(minslack, min(r["prem_p"] - r["Dp"] for r in rr))
            assert all(r["prem_p"] >= r["Dp"] - 1e-6 for r in rr)
            assert all(r["prem_t"] >= r["Dt"] - 1e-4 for r in rr)
            assert all(r["klp"] <= r["hJ"] + 1e-6 for r in rr)
            assert all(abs(r["klt_err"]) < 1e-4 for r in rr)
            exJ = [r["ecJ"] - r["Cmin"] for r in rr]
            rec.append(med([1 - r["prem_t"] / r["prem_p"] for r in rr]))
            lines.append("%s & $%g$ & $%s$ & $%s$ & $%s$ & $%s$ & $%s$ & $%s$ & $%s$ / $%s$ / $%s$\\\\" % (
                cl if lam == C["lam0s"][0] else "", lam, f(med(exJ)), f(med([r["Dp"] for r in rr])),
                f(med([r["prem_p"] for r in rr])),
                pct(med([r["prem_p"] / (r["ecJ"] - r["Cmin"]) for r in rr])),
                f(med([r["Dt"] for r in rr])), f(med([r["prem_t"] for r in rr])),
                f(med([r["co_joint"] for r in rr])), f(med([r["co_tree"] for r in rr])),
                f(med([r["co_prod"] for r in rr]))))
            if lam == 1.0:
                tag = "c" if cl == "congestion" else "g"
                N["C_rel1_" + tag] = pct(med([r["prem_p"] / (r["ecJ"] - r["Cmin"]) for r in rr]))
                N["C_co1_%s_p" % tag] = f(med([r["co_prod"] for r in rr]))
                N["C_co1_%s_j" % tag] = f(med([r["co_joint"] for r in rr]))
                if cl == "congestion":
                    d = np.array([np.array(r["BS_prod_matched"]) - np.array(r["BS_joint"]) for r in rr])
                    md = np.median(d, 0)
                    N["C_bs1"] = f(md[S.index(1)])
                    N["C_bs16"] = f(md[S.index(16)], 2)
                    N["C_bs64"] = f(md[S.index(64)], 3)
                    worst = max(np.median(np.array([r[k] for r in rr]), 0)[S.index(1024)]
                                for k in ("BS_joint", "BS_prod_matched", "BS_tree_matched"))
                    N["C_bs1024"] = "$%s$" % sci(worst) if worst > 1e-6 else "$10^{-6}$"
                    N["C_bs1024_raw"] = repr(worst)
        if cl == "congestion":
            lines.append("\\addlinespace")
    N["TAB_ONESHOT_ROWS"] = "\n".join(lines)
    N["C_ncases"] = str(ncase)
    N["C_minslack"] = "$%s$" % sci(minslack) if minslack > 0 else "$0$"
    N["C_rec_lo"], N["C_rec_hi"] = pct(min(rec)), pct(max(rec))
    kle = max(abs(r["klt_err"]) for cl in C["classes"].values() for r in cl["rows"])
    N["C_klerr"] = "$%s$" % sci(kle)


# ------------------------------------------------------------------ E
def exp_E():
    E = load("E")
    N["E_NI"] = str(E["NI"])
    lines = []
    dpn, closed, sw, z = [], [], [], []
    sec, pri, cert, mcr, swp = [], [], [], [], []
    tt = {}
    Hs = {h["n"]: h["rows"] for h in load("H")["sizes"]}
    for s in E["sizes"]:
        n, rows = s["n"], s["rows"]
        g = lambda sub, k: med([r[sub][k] for r in rows])
        cl = med([1 - r["tree"]["D"] / r["prod"]["D"] for r in rows])
        dpn.append(g("prod", "D") / n); closed.append(cl); sw.append(g("tree", "sweeps"))
        z += [abs(r[sub]["mc_mean"] - r[sub]["ec"]) / r[sub]["mc_se"] for r in rows for sub in ("prod", "tree")]
        assert max(max(r["prod"]["mono"], r["tree"]["mono"]) for r in rows) <= 1e-9
        assert all(r["prod"]["D"] <= r["prod"]["bound_bd"] for r in rows)
        assert all(r["tree"]["D"] <= r["tree"]["bound"] + 1e-9 for r in rows)
        tt[n] = g("tree", "time")
        hrows = Hs[n]
        # experiment H re-solves the factorized class on the same instances
        assert all(abs(h["D"] - r["prod"]["D"]) < 1e-6 * max(1, r["prod"]["D"]) for h, r in zip(hrows, rows))
        assert all(h["D"] <= h["cert"] for h in hrows)
        sec.append(med([h["second"] / h["D"] for h in hrows]))
        pri.append(med([r["prod"]["pred"] / r["prod"]["D"] for r in rows]))
        cert.append(med([h["cert"] / h["D"] for h in hrows]))
        mcr.append(med([h["mc"] / h["D"] for h in hrows]))
        lines.append("$%d$ & $%s$ & $%s$ & $%s$ & $%s$ & $%s$ & $%d$ & $%d$\\\\" % (
            n, f(g("prod", "D")), f(g("prod", "pred")), f(med([h["second"] for h in hrows])),
            f(g("tree", "D")), pct(cl), g("tree", "msgs"), g("tree", "rounds")))
        swp.append(g("prod", "sweeps"))
    N["TAB_SCALE_ROWS"] = "\n".join(lines)
    N["E_dpn_lo"], N["E_dpn_hi"] = f(min(dpn)), f(max(dpn))
    N["E_closed_lo"], N["E_closed_hi"] = pct(min(closed)), pct(max(closed))
    N["E_sw_lo"], N["E_sw_hi"] = "%d" % min(sw), "%d" % max(sw)
    N["E_swp_lo"], N["E_swp_hi"] = "%d" % min(swp), "%d" % max(swp)
    N["H_sec_lo"], N["H_sec_hi"] = pct(min(sec)), pct(max(sec))
    N["H_pri_lo"], N["H_pri_hi"] = f(min(pri), 1), f(max(pri), 1)
    N["H_cert_lo"], N["H_cert_hi"] = "%d" % min(cert), "%d" % max(cert)
    N["H_mc_lo"], N["H_mc_hi"] = pct(min(mcr)), pct(max(mcr))
    N["E_zmax"] = f(max(z), 1)
    N["E_nmc"] = str(len(z))
    b = E["budget"]
    g = np.array([r["greedy"] for r in b["rows"]]); rn = np.array([r["random"] for r in b["rows"]])
    inner = slice(1, len(b["ks"]) - 1)
    N["E_gfrac"] = pct(np.mean(g[:, inner] <= rn[:, inner] + 1e-9))
    gm, rm = np.median(g, 0), np.median(rn, 0)
    kh = [k for k, v in zip(b["ks"], gm) if v <= rm[-1]][0]
    N["E_khalf"] = str(kh)
    N["E_closed_all"] = "%s--%s\\%%" % (pct(min(closed)), pct(max(closed)))
    T = load("T") if os.path.exists(RES + "T.json") else None
    if T:
        N["E_t32"], N["E_t128"] = f(T["32"], 1), f(T["128"], 1)
    else:
        N["E_t32"], N["E_t128"] = f(tt[32], 1), f(tt[128], 1)


# ------------------------------------------------------------------ F
def exp_F():
    F = load("F_small")
    rows = F["rows"]
    N["F_NI"] = str(F["NI"])
    N["F_vshare"] = pct(med([r["vshare_tree"] for r in rows]))
    lines, closed, blind = [], [], []
    for lam in F["lams"]:
        L = [r["lam"][str(lam)] for r in rows]
        m_ = lambda fn: med([fn(l) for l in L])
        for l in L:
            ds = [l["k%d" % k]["D"] for k in range(6)]
            assert all(ds[i + 1] <= ds[i] + 1e-7 for i in range(5)), ds
            assert l["matched"]["klp_err"] <= 1e-6 and l["matched"]["klt_err"] <= 1e-9
            assert l["matched"]["prod_ec"] - l["matched"]["joint_ec"] >= l["k0"]["D"] - 1e-5
            assert l["matched"]["tree_ec"] - l["matched"]["joint_ec"] >= l["k5"]["D"] - 1e-5
        cl = m_(lambda l: 1 - l["k5"]["D"] / l["k0"]["D"]); closed.append(cl)
        blind.append(m_(lambda l: l["blind"]["D"] - l["k0"]["D"]))
        pp = m_(lambda l: l["matched"]["prod_ec"] - l["matched"]["joint_ec"])
        pt = m_(lambda l: l["matched"]["tree_ec"] - l["matched"]["joint_ec"])
        lines.append("$%g$ & $%s$ & $%s$ & $%s$ & $%s$ & $%s$ & $%s$ & $%s$ & $%s$ & $%s$ / $%s$ / $%s$\\\\" % (
            lam, f(m_(lambda l: l["blind"]["D"])), f(m_(lambda l: l["k0"]["D"])),
            f(m_(lambda l: l["k1"]["D"])), f(m_(lambda l: l["k3"]["D"])), f(m_(lambda l: l["k5"]["D"])),
            pct(cl), f(pp), f(pt),
            f(m_(lambda l: l["matched"]["joint_queue"])), f(m_(lambda l: l["matched"]["tree_queue"])),
            f(m_(lambda l: l["matched"]["prod_queue"]))))
        if lam == 1.0:
            N["F_premp_1"], N["F_premt_1"] = f(pp), f(pt)
            N["F_qp_1"] = f(m_(lambda l: l["matched"]["prod_queue"]))
            N["F_qt_1"] = f(m_(lambda l: l["matched"]["tree_queue"]))
            N["F_qj_1"] = f(m_(lambda l: l["matched"]["joint_queue"]))
    N["TAB_WAREHOUSE_ROWS"] = "\n".join(lines)
    N["F_closed_lo"], N["F_closed_hi"] = pct(min(closed)), pct(max(closed))
    N["F_blind_lo"], N["F_blind_hi"] = f(min(blind)), f(max(blind))
    N["F_klslack"] = "$%s$" % sci(max(max(-l["matched"]["klt_err"], 1e-12)
                                       for r in rows for l in r["lam"].values()))



def exp_FL():
    FL = load("F_large")
    rows = FL["rows"]
    N["FL_NI"] = str(len(rows))
    ks = FL["ks"]
    lines = []
    m_ = lambda key, sub: med([r[key][sub] for r in rows])
    lines.append("interaction-blind & $0$ & $%s$ & $%s$ & $%s$ & $%s$\\\\" % (
        f(m_("blind", "J")), f(m_("blind", "ec")), f(m_("blind", "H")), f(m_("blind", "queue"))))
    for k in ks:
        key = "k%d" % k
        assert max(r[key]["mono"] for r in rows) <= 1e-9
        lines.append("%s & $%d$ & $%s$ & $%s$ & $%s$ & $%s$\\\\" % (
            "factorized (FCD)" if k == 0 else ("forest $F_k$" if k == ks[1] else ""), k,
            f(m_(key, "J")), f(m_(key, "ec")), f(m_(key, "H")), f(m_(key, "queue"))))
    N["TAB_WAREHOUSE_LARGE_ROWS"] = "\n".join(lines)
    kmax = "k%d" % ks[-1]
    dJ = [r["k0"]["J"] - r[kmax]["J"] for r in rows]
    dH = [r[kmax]["H"] - r["k0"]["H"] for r in rows]
    dE = [r[kmax]["ec"] - r["k0"]["ec"] for r in rows]
    dQ = [r["k0"]["queue"] - r[kmax]["queue"] for r in rows]
    N["FL_dJ"] = f(med(dJ)); N["FL_dJ_min"] = f(min(dJ)); N["FL_dJ_max"] = f(max(dJ))
    N["FL_dH"] = f(med(dH), 1); N["FL_dQ"] = f(med(dQ))
    N["FL_dE"] = "$%+.2f$ in the median (between $%+.2f$ and $%+.2f$)" % (med(dE), min(dE), max(dE))
    assert min(dJ) > 0
    N["FL_blind_dJ"] = f(med([r["blind"]["J"] - r["k0"]["J"] for r in rows]))
    N["FL_det"] = f(med([r["det_icm"] for r in rows]))
    N["FL_vshare"] = pct(med([r[kmax]["pred_gain"] * 2 * FL["lam"] / r["V"] for r in rows]))
    N["FL_time"] = f(med([r[kmax]["time"] for r in rows]), 1)
    N["FL_sweeps"] = "%d" % med([r[kmax]["sweeps"] for r in rows])
    z = [abs(r["k%d" % k]["mc_mean"] - r["k%d" % k]["ec"]) / r["k%d" % k]["mc_se"] for r in rows for k in ks]
    N["FL_zmax"] = f(max(z), 1)
    N["FL_mono_frac"] = pct(np.mean([all(r["k%d" % ks[i + 1]]["J"] <= r["k%d" % ks[i]]["J"] + 1e-9
                                         for i in range(len(ks) - 1)) for r in rows]))


if __name__ == "__main__":
    import sys
    draft = "--draft" in sys.argv
    for fn in (example, exp_A, exp_B, exp_C, exp_E, exp_F, exp_FL):
        if not draft:
            fn()                      # final build: every check must pass
            print("ok", fn.__name__)
            continue
        try:
            fn()
            print("ok", fn.__name__)
        except (FileNotFoundError, AssertionError, KeyError) as e:
            print("INCOMPLETE", fn.__name__, repr(e)[:200])
    json.dump(N, open(RES + "numbers.json", "w"), indent=1)
    for k in sorted(N):
        if not k.startswith("TAB_"):
            print("%-16s %s" % (k, N[k]))
