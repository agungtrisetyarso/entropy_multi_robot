"""Figures of the paper (vector PDF).  usage: python3 make_figs.py
Reads ../results/{A,B,C,D,E}.json and writes ../paper/fig_*.pdf"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results") + os.sep
OUT = os.path.join(HERE, "..", "paper") + os.sep

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.6, "lines.linewidth": 1.1, "lines.markersize": 3.2,
    "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "cm",
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.linewidth": 0.4, "grid.color": "#d9d9d9",
})
# colour-blind-safe palette (Okabe-Ito); every series also has its own marker / dash
COL = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#555555"]
MK = ["o", "s", "^", "D", "v"]
LS = ["-", "--", "-.", ":", (0, (5, 1, 1, 1, 1, 1))]


def load(name):
    return json.load(open(RES + name + ".json"))


def load_B():
    """experiment B is run per family (exp.py B congestion / exp.py B general)"""
    B = load("B_congestion")
    B["classes"].update(load("B_general")["classes"])
    return B


def fig_variance():
    A = load("A")
    lams = np.array(A["lams"], float)
    labels = ["congestion, random graph, $(n,m)=(5,3)$", "general, random graph, $(5,3)$",
              "general, cycle, $(6,3)$", "congestion, complete graph, $(6,4)$",
              "general, random graph, non-uniform prior, $(5,4)$"]
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 2.55), sharey=True)
    for ax, key, title in zip(axs, ["prod", "span"], ["(a) product class", "(b) spanning forest"]):
        for ci, cf in enumerate(A["configs"]):
            q = np.array(cf["ratio"][key])
            ax.fill_between(lams, q[:, 0], q[:, 2], color=COL[ci], alpha=0.12, lw=0)
            ax.plot(lams, q[:, 1], color=COL[ci], marker=MK[ci], ls=LS[ci], label=labels[ci])
        ax.axhline(1.0, color="k", lw=0.5)
        ax.set_xscale("log", base=2)
        ax.set_xticks([0.25, 1, 4, 16, 64, 256])
        ax.set_xticklabels(["1/4", "1", "4", "16", "64", "256"])
        ax.set_xlabel(r"regularization weight $\lambda$")
        ax.set_title(title)
        ax.grid(True, axis="y")
        ax.set_ylim(0, 1.08)
    axs[0].set_ylabel("exact gap / prediction")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=1, frameon=False, bbox_to_anchor=(0.5, -0.02),
               handlelength=3.2, labelspacing=0.25)
    fig.subplots_adjust(left=0.15, right=0.99, top=0.92, bottom=0.47, wspace=0.08)
    fig.savefig(OUT + "fig_variance.pdf")
    plt.close(fig)


def fig_links():
    B = load_B()
    rows = B["classes"]["general"]
    lams = [1.0, 4.0]
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 2.15), sharey=True)
    for ax, lam, tag in zip(axs, lams, ["(a)", "(b)"]):
        ks = sorted(set(r["k"] for r in rows))
        series = dict(greedy=[], best=[], mean=[], pred=[], fs=[])
        for k in ks:
            rk = [r for r in rows if r["lam"] == lam and r["k"] == k]
            series["greedy"].append(np.median([1 - r["Dg"] / r["D0"] for r in rk]))
            series["best"].append(np.median([1 - r["Db"] / r["D0"] for r in rk]))
            series["mean"].append(np.median([1 - r["Dmean"] / r["D0"] for r in rk]))
            series["pred"].append(np.median([1 - r["pred"] for r in rk]))
            series["fs"].append(np.median([1 - r["Dfs"] / r["D0"] for r in rk]))
        ax.plot(ks, series["best"], color=COL[1], marker="s", ls="--", label="best forest (exhaustive)")
        ax.plot(ks, series["fs"], color=COL[3], marker="D", ls=(0, (1, 1)), label="forward selection")
        ax.plot(ks, series["greedy"], color=COL[0], marker="o", ls="-", label="variance forest $F_k$")
        ax.plot(ks, series["mean"], color=COL[2], marker="^", ls=":", label="average forest")
        ax.plot(ks, series["pred"], color=COL[4], marker="", ls="-.", label="leading-order prediction")
        ax.set_xlabel("number of links $k$")
        ax.set_title(r"%s $\lambda=%g$" % (tag, lam))
        ax.set_xticks(ks)
        ax.set_ylim(0, 1)
        ax.grid(True, axis="y")
    axs[0].set_ylabel("fraction of gap closed")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02),
               handlelength=2.8, columnspacing=1.0, labelspacing=0.25)
    fig.subplots_adjust(left=0.13, right=0.99, top=0.91, bottom=0.42, wspace=0.08)
    fig.savefig(OUT + "fig_links.pdf")
    plt.close(fig)


def fig_frontier():
    D = load("D")["congestion"]
    C = load("C")
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 2.05))
    ax = axs[0]
    hmax = D["hmax"]
    for key, lab, c, ls in [("prod", "factorized", COL[1], ":"), ("tree", "forest", COL[2], "--"),
                            ("joint", "joint", COL[0], "-")]:
        a = np.array(sorted(D[key]))
        ax.plot(hmax - a[:, 0], a[:, 1], color=c, ls=ls, label=lab)
    ax.axhline(D["Cmin"], color="k", lw=0.4)
    ax.text(hmax * 0.98, D["Cmin"], r"$C_{\min}$", ha="right", va="bottom", fontsize=7)
    ax.set_xlabel("randomization $H$ (nats)")
    ax.set_ylabel(r"expected one-shot cost")
    ax.set_title("(a)")
    ax.grid(True, axis="y")
    ax.legend(frameon=False, loc="upper left", handlelength=2.2, borderaxespad=0.1)
    ax = axs[1]
    S = np.array(C["S_list"])
    rows = [r for r in C["classes"]["congestion"]["rows"] if r["lam0"] == 1.0]
    for key, lab, c, ls, mk in [("BS_prod_matched", "factorized", COL[1], ":", "s"),
                                ("BS_tree_matched", "forest", COL[2], "--", "^"),
                                ("BS_joint", "joint", COL[0], "-", "o")]:
        med = np.median(np.array([r[key] for r in rows]), 0)
        ok = (med > 1e-4) & (S <= 256)
        ax.plot(S[ok], med[ok], color=c, ls=ls, marker=mk, label=lab)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([1, 4, 16, 64, 256])
    ax.set_xticklabels(["1", "4", "16", "64", "256"])
    ax.set_xlabel("samples $S$")
    ax.set_ylabel(r"$B_S-C_{\min}$")
    ax.set_title("(b)")
    ax.grid(True, axis="y")
    fig.subplots_adjust(left=0.13, right=0.99, top=0.90, bottom=0.19, wspace=0.42)
    fig.savefig(OUT + "fig_frontier.pdf")
    plt.close(fig)


def fig_budget():
    E = load("E")["budget"]
    ks = np.array(E["ks"])
    g = np.array([r["greedy"] for r in E["rows"]])
    rn = np.array([r["random"] for r in E["rows"]])
    pr = np.array([r["pred"] for r in E["rows"]])
    g0 = g[:, :1]
    fig, ax = plt.subplots(1, 1, figsize=(3.2, 1.9))
    for arr, lab, c, ls, mk in [(rn / g0, "random forest", COL[1], "--", "s"),
                                (g / g0, "greedy forest $F_k$", COL[0], "-", "o")]:
        q = np.percentile(arr, [25, 50, 75], axis=0)
        ax.fill_between(ks, q[0], q[2], color=c, alpha=0.15, lw=0)
        ax.plot(ks, q[1], color=c, ls=ls, marker=mk, label=lab)
    ax.plot(ks, np.median(pr / pr[:, :1], 0), color=COL[4], ls="-.", label="leading-order prediction")
    ax.set_xlabel("dispatch messages (links) $k$")
    ax.set_ylabel(r"residual gap $\Delta_{F}/\Delta_\otimes$")
    ax.set_ylim(0, 1.02)
    ax.set_xticks([0, 16, 32, 48, 63])
    ax.grid(True, axis="y")
    ax.legend(frameon=False, loc="lower left", handlelength=2.6)
    fig.subplots_adjust(left=0.16, right=0.98, top=0.97, bottom=0.21)
    fig.savefig(OUT + "fig_budget.pdf")
    plt.close(fig)


if __name__ == "__main__":
    for f in (fig_variance, fig_links, fig_frontier, fig_budget):
        try:
            f()
            print("ok", f.__name__)
        except FileNotFoundError as e:
            print("skipped", f.__name__, e)
