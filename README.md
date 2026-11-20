# Code for "The Factorization Gap in Entropy-Regularized Multi-Robot Task Allocation"

Python 3 with NumPy, SciPy and Matplotlib. Every number, table and figure of the
evaluation section is produced by these scripts; all random seeds are fixed.

| File | Purpose |
|---|---|
| `fg.py` | Core library: instances, exact enumeration, interaction variances, greedy forest (Kruskal), forest policies, forest coordinate descent (FCD), exact log-partition by variable elimination |
| `test_core.py` | Unit tests of the forest inference, the block update, monotone descent and exactness on forests against brute-force enumeration |
| `verify_claims.py` | Numerical checks of Corollary 1, identity (14), Proposition 2 and the best-of-S formula |
| `exp.py` | Experiments `A` (variance law, bounds), `B <family>` (link selection vs. exhaustive search; families `congestion`, `general`), `C` (matched-randomization dispatch, best-of-S), `D` (frontier curves), `E` (scaling on ladders), `H` (re-centred estimate and a posteriori bound on the instances of `E`) |
| `warehouse.py` | `example` (two-robot running example), `small` (6 robots, 4 stations, exact), `large` (30 robots, 6 stations) |
| `timing.py` | Wall-clock time of synthesis; run it on an idle machine |
| `make_numbers.py` | Collects every number quoted in the paper into `results/numbers.json` and checks the inequalities the paper states (it fails if one is violated) |
| `make_figs.py` | Figures (vector PDF) |
| `check_previous_formulation.py` | Reviewer 2's two checks on the two-robot example of the previous (operator) formulation |
| `independent_check.py` | Re-derivation of Table II and of the variance law that does not use `fg.py` |
| `build.py` | Fills the `@@placeholders@@` of `paper/main_src.tex` and writes `paper/main.tex` |

Run order:

    python3 test_core.py
    python3 verify_claims.py
    python3 warehouse.py example
    python3 exp.py A
    python3 exp.py B congestion
    python3 exp.py B general
    python3 exp.py C
    python3 exp.py D
    python3 exp.py E
    python3 exp.py H
    python3 warehouse.py small
    python3 warehouse.py large
    python3 timing.py
    python3 make_numbers.py
    python3 make_figs.py
    python3 build.py

Results are written to `../results/`. Experiment `C` resumes from an existing
`C.json`; delete the file to recompute from scratch.

Layout: `code/` (this folder), `results/` (outputs of the scripts), `paper_source/main_src.tex` (manuscript with `@@placeholders@@`).
`build.py` reads `../paper/main_src.tex`; copy `paper_source/` to `paper/` (with the four figure files) before running it.
The response letter cross-references the manuscript through `main.aux` (package `xr`).
