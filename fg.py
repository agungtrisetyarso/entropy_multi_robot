"""
fg.py -- core library for
"The Factorization Gap in Entropy-Regularized Multi-Robot Task Allocation"

Model
-----
n robots, m tasks, joint assignment x in [m]^n.
Team cost      C(x) = sum_i c_i(x_i) + sum_{(i,j) in E} W_ij(x_i, x_j)
Product prior  mu(x) = prod_i mu_i(x_i)   (full support)
Objective      J(pi) = E_pi[C] + lam * KL(pi || mu)

Policy classes
--------------
joint          all distributions on [m]^n            (Gibbs optimum, exact)
forest F       distributions Markov on a forest F    (F = {} is the product class)

Everything in this file is exact (no sampling) except ForestPolicy.sample.
"""
import itertools
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import minimize


# --------------------------------------------------------------------------
# Instance
# --------------------------------------------------------------------------
class Instance:
    def __init__(self, c, W, mu=None):
        """c: (n,m) array.  W: dict {(i,j): (m,m) array}, i<j, W[(i,j)][a,b]
        is the pairwise cost when x_i=a and x_j=b.  mu: (n,m) prior."""
        self.c = np.asarray(c, float)
        self.n, self.m = self.c.shape
        self.W = {}
        for (i, j), w in W.items():
            w = np.asarray(w, float)
            if i > j:
                i, j, w = j, i, w.T
            assert i != j and (i, j) not in self.W
            self.W[(i, j)] = w
        self.edges = sorted(self.W.keys())
        self.mu = (np.full((self.n, self.m), 1.0 / self.m) if mu is None
                   else np.asarray(mu, float))
        assert np.all(self.mu > 0) and np.allclose(self.mu.sum(1), 1)

    def pair(self, i, j):
        """cost table oriented as [x_i, x_j]"""
        return self.W[(i, j)] if i < j else self.W[(j, i)].T

    # ---- interaction variance (second-order weights) ----
    def centred(self, e):
        i, j = e
        w = self.W[e]
        r = w @ self.mu[j]                 # E[W | x_i=a]
        col = self.mu[i] @ w               # E[W | x_j=b]
        g = self.mu[i] @ w @ self.mu[j]
        return w - r[:, None] - col[None, :] + g

    def v(self, e):
        i, j = e
        wt = self.centred(e)
        return float(self.mu[i] @ (wt ** 2) @ self.mu[j])

    def weights(self):
        return {e: self.v(e) for e in self.edges}

    def V_int(self):
        return sum(self.weights().values())

    # ---- quantities entering the uniform (all-lambda) bounds ----
    def osc_sum(self, F=()):
        """sum over off-forest edges of osc(W~_e) >= osc(R_F)"""
        Fs = set(tuple(sorted(e)) for e in F)
        tot = 0.0
        for e in self.edges:
            if e not in Fs:
                wt = self.centred(e)
                tot += wt.max() - wt.min()
        return tot

    def bd_sq_sum(self):
        """sum_i c_i^2 with c_i >= max change of R_0 = sum_e W~_e when x_i changes"""
        ci = np.zeros(self.n)
        for (i, j) in self.edges:
            wt = self.centred((i, j))
            ci[i] += (wt.max(0) - wt.min(0)).max()
            ci[j] += (wt.max(1) - wt.min(1)).max()
        return float(np.sum(ci ** 2))

    def residual_instance(self, F):
        """Instance with every off-forest table replaced by its additive part:
        C~ = C - R_F.  Returns (instance, constant) with C~ = cost_inst + const."""
        Fs = set(tuple(sorted(e)) for e in F)
        c2 = self.c.copy()
        const = 0.0
        W2 = {}
        for e in self.edges:
            if e in Fs:
                W2[e] = self.W[e]
            else:
                i, j = e
                w = self.W[e]
                c2[i] += w @ self.mu[j]
                c2[j] += self.mu[i] @ w
                const -= float(self.mu[i] @ w @ self.mu[j])
        return Instance(c2, W2, self.mu), const

    # ---- deterministic cost of assignments ----
    def cost(self, X):
        """X: (..., n) int array -> cost array"""
        X = np.asarray(X)
        out = np.zeros(X.shape[:-1])
        for i in range(self.n):
            out += self.c[i][X[..., i]]
        for (i, j), w in self.W.items():
            out += w[X[..., i], X[..., j]]
        return out


# --------------------------------------------------------------------------
# Exact enumeration engine (small instances)
# --------------------------------------------------------------------------
class Enum:
    def __init__(self, inst):
        self.inst = inst
        n, m = inst.n, inst.m
        self.X = np.array(list(itertools.product(range(m), repeat=n)), dtype=np.int64)
        self.d = self.X.shape[0]
        self.C = inst.cost(self.X)
        self.logmu = np.zeros(self.d)
        for i in range(n):
            self.logmu += np.log(inst.mu[i][self.X[:, i]])

    # joint (Gibbs) optimum
    def gibbs(self, lam):
        a = self.logmu - self.C / lam
        lz = logsumexp(a)
        return np.exp(a - lz), -lam * lz

    def J(self, q, lam):
        q = np.asarray(q)
        nz = q > 0
        kl = float(np.sum(q[nz] * (np.log(q[nz]) - self.logmu[nz])))
        ec = float(q @ self.C)
        return ec + lam * kl, ec, kl

    def entropy(self, q):
        nz = q > 0
        return float(-np.sum(q[nz] * np.log(q[nz])))

    # ---- global-ish optimum over forest class by L-BFGS on the exp. family
    def _features(self, F):
        n, m = self.inst.n, self.inst.m
        idx = [self.X[:, i] + i * m for i in range(n)]
        off = n * m
        for (i, j) in F:
            idx.append(off + self.X[:, i] * m + self.X[:, j])
            off += m * m
        return np.stack(idx, 1), off

    def opt_forest(self, F, lam, restarts=6, seed=0, extra_inits=(), return_theta=False):
        idx, P = self._features(F)
        inst = self.inst
        n, m = inst.n, inst.m

        def unpack(th):
            return self.logmu + th[idx].sum(1)

        def fun(th):
            a = unpack(th)
            lz = logsumexp(a)
            lq = a - lz
            q = np.exp(lq)
            f = self.C + lam * (lq - self.logmu)
            Jv = float(q @ f)
            gx = q * (f - Jv)
            g = np.zeros(P)
            for col in range(idx.shape[1]):
                g += np.bincount(idx[:, col], weights=gx, minlength=P)
            return Jv, g

        inits = []
        th0 = np.zeros(P)
        th0[: n * m] = (-inst.c / lam).ravel()
        off = n * m
        for (i, j) in F:
            th0[off: off + m * m] = (-inst.pair(i, j) / lam).ravel()
            off += m * m
        inits.append(th0)
        inits.append(np.zeros(P))
        rng = np.random.default_rng(seed)
        for _ in range(restarts):
            inits.append(th0 + rng.normal(0, 1.0, P) * max(1.0, 1.0 / lam))
        inits.extend(extra_inits)
        best = None
        for th in inits:
            res = minimize(fun, th, jac=True, method="L-BFGS-B",
                           options=dict(maxiter=5000, ftol=1e-15, gtol=1e-10))
            if best is None or res.fun < best[0]:
                best = (res.fun, res.x)
        a = unpack(best[1])
        q = np.exp(a - logsumexp(a))
        if return_theta:
            return q, best[0], best[1]
        return q, best[0]

    def constrained(self, F, h, starts, tol=1e-10):
        """min E_q[C] over the forest class s.t. KL(q||mu) <= h (SLSQP on the
        exponential-family parameters, several starts); returns (q, EC, KL)."""
        idx, Pn = self._features(F)

        def dist(th):
            a = self.logmu + th[idx].sum(1)
            lq = a - logsumexp(a)
            return np.exp(lq), lq

        def back(gx):
            g = np.zeros(Pn)
            for col in range(idx.shape[1]):
                g += np.bincount(idx[:, col], weights=gx, minlength=Pn)
            return g

        def f(th):
            q, _ = dist(th)
            ec = float(q @ self.C)
            return ec, back(q * (self.C - ec))

        def con(th):
            q, lq = dist(th)
            return h - float(q @ (lq - self.logmu))

        def conj(th):
            q, lq = dist(th)
            r = lq - self.logmu
            return -back(q * (r - float(q @ r)))

        best = None
        for th0 in starts:
            res = minimize(f, th0, jac=True, method="SLSQP",
                           constraints=[dict(type="ineq", fun=con, jac=conj)],
                           options=dict(maxiter=2000, ftol=tol))
            q, lq = dist(res.x)
            kl = float(q @ (lq - self.logmu))
            if kl <= h + 1e-7 and (best is None or res.fun < best[1]):
                best = (q, float(res.fun), kl, res.x)
        return best

    # best-of-S expected cost, exact
    def best_of_S(self, q, S_list):
        order = np.argsort(self.C, kind="stable")
        cs = self.C[order]
        qs = q[order]
        # survival: G_k = P(C > c_k) using distinct sorted values
        tail = 1.0 - np.cumsum(qs)            # P(index > k)
        tail = np.clip(tail, 0.0, 1.0)
        prev = np.concatenate([[1.0], tail[:-1]])  # P(index >= k)
        out = []
        for S in S_list:
            pk = prev ** S - tail ** S           # P(min index == k)
            out.append(float(pk @ cs))
        return np.array(out)


# --------------------------------------------------------------------------
# Greedy maximum-interaction-variance forest (Kruskal)
# --------------------------------------------------------------------------
def kruskal_order(n, weights, allowed=None):
    """Return forest edges in greedy order (descending weight).  The first k
    entries are a maximum-weight forest with at most k edges."""
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    order = []
    for e in sorted(weights, key=lambda e: (-weights[e], e)):
        if allowed is not None and e not in allowed:
            continue
        ra, rb = find(e[0]), find(e[1])
        if ra != rb:
            parent[ra] = rb
            order.append(e)
    return order


def is_forest(n, F):
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for (i, j) in F:
        ra, rb = find(i), find(j)
        if ra == rb:
            return False
        parent[ra] = rb
    return True


# --------------------------------------------------------------------------
# Forest-structured policy and forest coordinate descent (FCD)
# --------------------------------------------------------------------------
class ForestPolicy:
    """q(x) = prod_roots q_r(x_r) * prod_{j non-root} T_j(x_j | x_pa(j))."""

    def __init__(self, inst, F, lam, roots=None):
        self.inst, self.lam = inst, lam
        n, m = inst.n, inst.m
        self.n, self.m = n, m
        self.F = [tuple(sorted(e)) for e in F]
        assert is_forest(n, self.F)
        assert all(e in inst.W for e in self.F), "forest links must be interaction edges"
        adj = [[] for _ in range(n)]
        for (i, j) in self.F:
            adj[i].append(j)
            adj[j].append(i)
        self.pa = -np.ones(n, dtype=int)
        self.comp = -np.ones(n, dtype=int)
        self.depth = np.zeros(n, dtype=int)
        self.order = []                       # BFS order (roots first)
        self.children = [[] for _ in range(n)]
        seen = np.zeros(n, bool)
        root_pref = list(roots) if roots is not None else []
        cid = 0
        for s in root_pref + list(range(n)):
            if seen[s]:
                continue
            # choose a centre of the component as root unless given
            if s not in root_pref:
                s = self._centre(s, adj)
            seen[s] = True
            self.comp[s] = cid
            queue = [s]
            while queue:
                u = queue.pop(0)
                self.order.append(u)
                for w in adj[u]:
                    if not seen[w]:
                        seen[w] = True
                        self.pa[w] = u
                        self.comp[w] = cid
                        self.depth[w] = self.depth[u] + 1
                        self.children[u].append(w)
                        queue.append(w)
            cid += 1
        self.roots = [u for u in range(n) if self.pa[u] < 0]
        # Euler intervals for subtree tests
        self.tin = np.zeros(n, int)
        self.tout = np.zeros(n, int)
        t = 0
        for r in self.roots:
            stack = [(r, 0)]
            while stack:
                u, st = stack.pop()
                if st == 0:
                    self.tin[u] = t
                    t += 1
                    stack.append((u, 1))
                    for w in self.children[u]:
                        stack.append((w, 0))
                else:
                    self.tout[u] = t
        self.sub = [np.array([k for k in range(n) if self.in_sub(k, j)]) for j in range(n)]
        self.desc = [np.array([k for k in self.sub[j] if k != j], dtype=int) for j in range(n)]
        self.notsub = [np.array([k for k in range(n) if not self.in_sub(k, j)], dtype=int)
                       for j in range(n)]
        self.off = [e for e in inst.edges if e not in set(self.F)]
        # tree-edge cost oriented [parent, child]
        self.Wt = np.zeros((n, m, m))
        for j in range(n):
            if self.pa[j] >= 0:
                self.Wt[j] = inst.pair(self.pa[j], j)
        self.Woff = np.array([inst.W[e] for e in self.off]) if self.off else np.zeros((0, m, m))
        self.offk = np.array([e[0] for e in self.off], dtype=int)
        self.offl = np.array([e[1] for e in self.off], dtype=int)
        self.lca = np.array([self._lca(k, l) for (k, l) in self.off], dtype=int)
        # per-block classification of off-tree edges
        self.cls = []
        for j in range(n):
            ink = np.array([self.in_sub(k, j) for k in self.offk], dtype=bool)
            inl = np.array([self.in_sub(l, j) for l in self.offl], dtype=bool)
            both = np.where(ink & inl)[0]
            k_in = np.where(ink & ~inl)[0]
            l_in = np.where(~ink & inl)[0]
            self.cls.append((both, k_in, l_in))
        # parameters: start at the prior
        self.q0 = inst.mu.copy()                       # used for roots only
        self.T = np.repeat(inst.mu[:, None, :], m, axis=1)  # T[j][a,b]
        self.logmu = np.log(inst.mu)
        self.refresh()

    # ---- structure helpers ----
    @staticmethod
    def _centre(s, adj):
        def far(a):
            dist = {a: 0}
            par = {a: None}
            queue = [a]
            last = a
            while queue:
                u = queue.pop(0)
                last = u
                for w in adj[u]:
                    if w not in dist:
                        dist[w] = dist[u] + 1
                        par[w] = u
                        queue.append(w)
            return last, dist, par
        a, _, _ = far(s)
        b, dist, par = far(a)
        path = [b]
        while par[path[-1]] is not None:
            path.append(par[path[-1]])
        return path[len(path) // 2]

    def in_sub(self, k, j):
        return self.tin[j] <= self.tin[k] < self.tout[j]

    def _lca(self, k, l):
        if self.comp[k] != self.comp[l]:
            return -1
        while not self.in_sub(l, k):
            k = self.pa[k]
        return k

    # ---- inference on the forest ----
    def refresh(self):
        n, m = self.n, self.m
        q = np.zeros((n, m))
        for u in self.order:
            q[u] = self.q0[u] if self.pa[u] < 0 else q[self.pa[u]] @ self.T[u]
        self.q = q
        # reverse conditionals B[c][x_c, x_pa]
        B = np.zeros((n, m, m))
        for c in range(n):
            p = self.pa[c]
            if p >= 0:
                B[c] = (q[p][:, None] * self.T[c]).T / q[c][:, None]
        # all-pairs conditionals A[u,l][x_u, x_l] = P(x_l | x_u)
        A = np.empty((n, n, m, m))
        A[:] = q[None, :, None, :]                 # different components: marginal
        I = np.eye(m)
        for u in range(n):
            A[u, u] = I
        for z in reversed(self.order):             # upward
            for c in self.children[z]:
                us = self.sub[c]
                A[us, z] = A[us, c] @ B[c]
        for z in self.order:                       # downward
            p = self.pa[z]
            if p < 0:
                continue
            us = self.notsub[z]
            us = us[self.comp[us] == self.comp[z]]
            A[us, z] = A[us, p] @ self.T[z]
        self.A = A

    def embed(self, other):
        """Load a policy `other` that is Markov on a sub-forest of self.F
        (exact re-parametrisation; the distribution is unchanged)."""
        assert set(other.F) <= set(self.F)
        for j in range(self.n):
            p = self.pa[j]
            if p < 0:
                self.q0[j] = other.q[j]
            else:
                self.T[j] = other.A[p, j]
        self.refresh()
        return self

    def _t(self):
        """t_k(e) = sum_c T_k(c|e) [c_k(c) + Wt_k(e,c) + lam log(T_k(c|e)/mu_k(c))]"""
        with np.errstate(divide="ignore", invalid="ignore"):
            lr = np.where(self.T > 0, np.log(self.T) - self.logmu[:, None, :], 0.0)
        return np.einsum("kec,kec->ke", self.T,
                         self.inst.c[:, None, :] + self.Wt + self.lam * lr)

    def phi(self, j):
        """Phi_j(a,b) (or Phi_j(b) for a root), up to terms constant in b."""
        A, lam = self.A, self.lam
        p = self.pa[j]
        t = self._t()
        d = self.desc[j]
        U = np.einsum("kbe,ke->b", A[j, self.pa[d]], t[d]) if len(d) else np.zeros(self.m)
        both, k_in, l_in = self.cls[j]
        if len(both):
            u = self.lca[both]
            g = np.einsum("eac,ecd,ead->ea", A[u, self.offk[both]], self.Woff[both],
                          A[u, self.offl[both]])
            U = U + np.einsum("eba,ea->b", A[j, u], g)
        phi = self.inst.c[j] + U
        if p >= 0:
            M = self.Wt[j].copy()
            if len(k_in):
                M += np.einsum("ead,ecd,ebc->ab", A[p, self.offl[k_in]], self.Woff[k_in],
                               A[j, self.offk[k_in]])
            if len(l_in):
                M += np.einsum("eac,ecd,ebd->ab", A[p, self.offk[l_in]], self.Woff[l_in],
                               A[j, self.offl[l_in]])
            return phi[None, :] + M
        if len(k_in):
            phi = phi + np.einsum("ebc,ecd,ed->b", A[j, self.offk[k_in]], self.Woff[k_in],
                                  self.q[self.offl[k_in]])
        if len(l_in):
            phi = phi + np.einsum("ec,ecd,ebd->b", self.q[self.offk[l_in]], self.Woff[l_in],
                                  A[j, self.offl[l_in]])
        return phi

    def block_update(self, j):
        ph = self.phi(j)
        a = self.logmu[j] - ph / self.lam
        a = a - logsumexp(a, axis=-1, keepdims=True)
        if self.pa[j] >= 0:
            self.T[j] = np.exp(a)
        else:
            self.q0[j] = np.exp(a)
        self.refresh()

    def objective(self):
        inst, lam, q, A = self.inst, self.lam, self.q, self.A
        ec = float(np.sum(q * inst.c))
        kl = 0.0
        for j in range(self.n):
            p = self.pa[j]
            if p >= 0:
                ec += float(q[p] @ (self.T[j] * self.Wt[j]).sum(1))
                with np.errstate(divide="ignore", invalid="ignore"):
                    lr = np.where(self.T[j] > 0, np.log(self.T[j]) - self.logmu[j][None, :], 0.0)
                kl += float(q[p] @ (self.T[j] * lr).sum(1))
            else:
                nz = self.q0[j] > 0
                kl += float(np.sum(self.q0[j][nz] * (np.log(self.q0[j][nz]) - self.logmu[j][nz])))
        if len(self.off):
            ec += float(np.einsum("ec,ecd,ecd->", q[self.offk], A[self.offk, self.offl], self.Woff))
        return ec + lam * kl, ec, kl

    def fcd(self, max_sweeps=200, tol=1e-12, record=False, perturb=None, rng=None):
        """Forest coordinate descent: alternate leaves->root and root->leaves
        sweeps of exact block minimisations."""
        if perturb is not None:
            rng = rng or np.random.default_rng(0)
            g = rng.gumbel(size=self.T.shape) * perturb
            T = np.exp(np.log(self.T) + g)
            self.T = T / T.sum(-1, keepdims=True)
            g0 = rng.gumbel(size=self.q0.shape) * perturb
            q0 = np.exp(np.log(self.q0) + g0)
            self.q0 = q0 / q0.sum(-1, keepdims=True)
            self.refresh()
        hist = [self.objective()[0]]
        back = list(reversed(self.order))
        sweeps = 0
        for s in range(max_sweeps):
            for j in (back if s % 2 == 0 else self.order):
                self.block_update(j)
            sweeps += 1
            hist.append(self.objective()[0])
            if hist[-2] - hist[-1] < tol * max(1.0, abs(hist[-1])):
                break
        self.sweeps = sweeps
        return (hist if record else hist[-1])

    # ---- exact joint vector on an enumeration (testing only) ----
    def joint(self, X):
        lq = np.zeros(X.shape[0])
        for j in range(self.n):
            p = self.pa[j]
            if p >= 0:
                lq += np.log(self.T[j][X[:, p], X[:, j]])
            else:
                lq += np.log(self.q0[j][X[:, j]])
        return np.exp(lq)

    # ---- one-shot dispatch (ancestral sampling) ----
    def sample(self, S, rng):
        X = np.zeros((S, self.n), dtype=np.int64)
        for u in self.order:
            p = self.pa[u]
            P = (np.repeat(self.q0[u][None, :], S, 0) if p < 0 else self.T[u][X[:, p]])
            cum = np.cumsum(P, axis=1)
            r = rng.random(S)[:, None]
            X[:, u] = np.minimum((r > cum).sum(1), self.m - 1)
        return X

    def messages(self):
        return len(self.F)

    def rounds(self):
        return int(self.depth.max())

    def entropy(self):
        """H(q) = -sum q log q, exact on the forest."""
        H = 0.0
        for j in range(self.n):
            p = self.pa[j]
            if p >= 0:
                with np.errstate(divide="ignore", invalid="ignore"):
                    lt = np.where(self.T[j] > 0, np.log(self.T[j]), 0.0)
                H -= float(self.q[p] @ (self.T[j] * lt).sum(1))
            else:
                nz = self.q0[j] > 0
                H -= float(np.sum(self.q0[j][nz] * np.log(self.q0[j][nz])))
        return H


def forest_gibbs(inst, F, lam):
    """q~_F : exact Gibbs policy of the forest-restricted cost C - R_F, obtained
    by one leaves-to-root sweep (soft dynamic programming).  Returned as a
    ForestPolicy on the FULL instance, together with J~* = min of the restricted
    problem (including its additive constant)."""
    inst2, const = inst.residual_instance(F)
    P2 = ForestPolicy(inst2, F, lam)
    for j in reversed(P2.order):
        P2.block_update(j)
    Jt = P2.objective()[0] + const
    P = ForestPolicy(inst, F, lam)
    P.q0, P.T = P2.q0.copy(), P2.T.copy()
    P.refresh()
    return P, Jt


def solve_forest(inst, F, lam, restarts=0, seed=0, max_sweeps=400, tol=1e-12,
                 warm=None, init="gibbs", warm_policies=()):
    """FCD from q~_F (default) and/or the prior, optional warm starts (raw
    parameters `warm`, or policies on sub-forests `warm_policies`) and
    perturbed restarts; returns the best policy found."""
    rng = np.random.default_rng(seed)
    cands = []
    if warm is not None:
        P = ForestPolicy(inst, F, lam)
        P.q0, P.T = warm[0].copy(), warm[1].copy()
        P.refresh()
        P.fcd(max_sweeps=max_sweeps, tol=tol)
        cands.append(P)
    for Pw in warm_policies:
        P = ForestPolicy(inst, F, lam).embed(Pw)
        P.fcd(max_sweeps=max_sweeps, tol=tol)
        cands.append(P)
    if init in ("gibbs", "both"):
        P, _ = forest_gibbs(inst, F, lam)
        P.fcd(max_sweeps=max_sweeps, tol=tol)
        cands.append(P)
    if init in ("prior", "both"):
        P = ForestPolicy(inst, F, lam)
        P.fcd(max_sweeps=max_sweeps, tol=tol)
        cands.append(P)
    for r in range(restarts):
        P = ForestPolicy(inst, F, lam)
        P.fcd(max_sweeps=max_sweeps, tol=tol, perturb=2.0, rng=rng)
        cands.append(P)
    vals = [P.objective()[0] for P in cands]
    return cands[int(np.argmin(vals))]


def best_product(inst, en, lam, seed=0, restarts=6):
    """best-found product optimum: FCD (two initialisations and perturbed
    restarts) and L-BFGS on the enumerated exponential family."""
    P = solve_forest(inst, [], lam, restarts=restarts, seed=seed, init="both")
    qL, JL = en.opt_forest([], lam, restarts=restarts, seed=seed)
    if JL < P.objective()[0] - 1e-10:
        P2 = ForestPolicy(inst, [], lam)
        for i in range(inst.n):
            P2.q0[i] = np.bincount(en.X[:, i], weights=qL, minlength=inst.m)
        P2.refresh()
        P2.fcd(tol=1e-13)
        if P2.objective()[0] < P.objective()[0]:
            P = P2
    return P


# --------------------------------------------------------------------------
# Exact log-partition function by variable elimination (low tree-width graphs)
# --------------------------------------------------------------------------
def exact_logZ(inst, lam, order=None):
    """log sum_x mu(x) exp(-C(x)/lam) by bucket elimination in the given order.
    Exponential in the induced width of the order; exact."""
    n, m = inst.n, inst.m
    factors = []
    for i in range(n):
        factors.append(((i,), np.log(inst.mu[i]) - inst.c[i] / lam))
    for (i, j), w in inst.W.items():
        factors.append(((i, j), -w / lam))
    order = list(range(n)) if order is None else list(order)
    for v in order:
        bucket = [f for f in factors if v in f[0]]
        factors = [f for f in factors if v not in f[0]]
        scope = sorted(set().union(*[set(f[0]) for f in bucket]))
        tot = np.zeros([m] * len(scope))
        for sc, tab in bucket:
            # tab axes follow sc; move to scope order with broadcasting
            perm = np.argsort([scope.index(s) for s in sc])
            tab2 = np.transpose(tab, perm)
            shape = [m if s in sc else 1 for s in scope]
            tot = tot + tab2.reshape(shape)
        ax = scope.index(v)
        red = logsumexp(tot, axis=ax)
        new_scope = tuple(s for s in scope if s != v)
        factors.append((new_scope, red))
    return float(sum(np.asarray(f[1]).sum() for f in factors))


# --------------------------------------------------------------------------
# Instance generators
# --------------------------------------------------------------------------
def graph_edges(kind, n, rng, p=0.5, deg=3):
    if kind == "cycle":
        return [(i, (i + 1) % n) if i < (i + 1) % n else ((i + 1) % n, i) for i in range(n)]
    if kind == "complete":
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    if kind == "path":
        return [(i, i + 1) for i in range(n - 1)]
    if kind == "random":
        # connected random graph: random spanning tree plus extra edges w.p. p
        perm = rng.permutation(n)
        E = set()
        for k in range(1, n):
            a, b = perm[k], perm[rng.integers(0, k)]
            E.add((min(a, b), max(a, b)))
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < p:
                    E.add((i, j))
        return sorted((int(a), int(b)) for a, b in E)
    if kind == "ladder":
        L = n // 2
        E = [(2 * t, 2 * t + 1) for t in range(L)]
        E += [(2 * t, 2 * t + 2) for t in range(L - 1)]
        E += [(2 * t + 1, 2 * t + 3) for t in range(L - 1)]
        return sorted(E)
    raise ValueError(kind)


def random_instance(n, m, rng, graph="random", coupling="general", p=0.5,
                    w_range=(0.5, 2.0), gamma=1.0, hetero_prior=False):
    """coupling = 'congestion': W = w 1[a=b]
       coupling = 'general'   : W = w 1[a=b] + gamma * G,  G_ab ~ N(0,1) per pair
                                (heterogeneous interference / synergy tables)"""
    c = rng.uniform(0.0, 1.0, (n, m))
    E = graph_edges(graph, n, rng, p=p)
    W = {}
    for e in E:
        w = rng.uniform(*w_range) * np.eye(m)
        if coupling == "general":
            w = w + gamma * rng.normal(0, 1.0, (m, m))
        W[e] = w
    mu = None
    if hetero_prior:
        mu = rng.dirichlet(np.full(m, 4.0), size=n)
    return Instance(c, W, mu)
