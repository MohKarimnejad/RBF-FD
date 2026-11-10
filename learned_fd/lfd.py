"""
lfd -- learned meshless finite differences with free, polynomially exact weights.

Setting (same discretization as PHS_fd_TP1.m / PHS_fd_TP2.m, solver = 'FD'):
  elliptic interface problem  -div(alpha grad u) = f  in Omega+ and Omega-,
  u = g on dOmega,  [u] = phi,  [alpha du/dn] = psi  on Gamma.
  Unknowns: u+ at X+ = [Xint+; Xbd+; XG], u- at X- = [Xint-; Xbd-; XG].
  Rows: Laplacian rows (Xint+, Xint-), Dirichlet rows, jump rows, flux rows.
  Every derivative row uses one stencil = the n nearest nodes of the same side.

Weights of one stencil (row point x, scaled coordinates y_hat = (y - x)/r, r = stencil radius):
    w_hat = w_base + s * (I - Pi) v
  w_base : RBF-FD weights (PHS r^beta + polynomials of degree deg), the baseline;
  Pi     : orthogonal projector onto range(P), P = polynomials at the stencil nodes,
           so (I - Pi) v lies in the null space of P' and w_hat is exact on the
           polynomials of degree <= deg for EVERY v (no kernel is reproduced);
  s      : rms of w_base (the correction is relative to the size of the baseline);
  v      : free correction, one number per stencil node, given either by
           - FreeCorrection : one trainable number per (row, node) of ONE node set
             (no network; the best this loss can reach on that node set), or
           - NetCorrection  : a permutation-equivariant network of the stencil geometry.
  v = 0 gives the baseline.

Loss (manufactured solutions on the training node sets; index 0 = baseline):
    L = ||E||^2/||E_0||^2 + lam_tau * mean_g ||tau_g||^2/||tau_0,g||^2 + lam_l1 * rho/rho_0
  E     = -A^{-1} tau : error of the discrete solution (the PDE is solved inside the loss),
  tau   = A U* - b    : truncation error, per row group g (Laplacian rows, flux rows),
  rho   = mean_i sum_j |w_hat_ij| |y_hat_j|^mu : weighted l1 norm of the weights, the
          scale-free form of eq. (20) of Davydov & Schaback (Minimal numerical
          differentiation formulas), mu = deg.
"""

import math
import os

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from scipy.spatial import cKDTree

torch.set_default_dtype(torch.float64)
HERE = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# Test problems (exact solutions as torch functions of X = (N,2))
# =============================================================================
class TP1:
    """Test Problem 1 of Davydov & Safarpoor (2021): Gamma: y = 2x^3,
    alpha+ = 20 (y > 2x^3), alpha- = 1."""
    name, ap, am = "TP1", 20.0, 1.0

    @staticmethod
    def normal(xg):                               # unit normal on Gamma, points into Omega+
        x = xg[:, 0]
        n = np.stack([-6 * x**2, np.ones_like(x)], 1)
        return n / np.sqrt(1 + 36 * x**4)[:, None]

    @staticmethod
    def up(X):
        s = X[:, 1] - 2 * X[:, 0]**3
        return (s**2 - 30 * s) / 20

    @staticmethod
    def um(X):
        s = X[:, 1] - 2 * X[:, 0]**3
        return s**2 - 30 * s


class TP2:
    """Test Problem 2 of Davydov & Safarpoor (2021): Gamma: circle r = 1/2,
    Omega+ inside, alpha+ = alpha- = 1, u+ = 1, u- = 1 + log(2r)."""
    name, ap, am = "TP2", 1.0, 1.0

    @staticmethod
    def normal(xg):                               # outward normal of the circle
        return xg / np.linalg.norm(xg, axis=1, keepdims=True)

    @staticmethod
    def up(X):
        return torch.ones(X.shape[0]) + 0 * X[:, 0]

    @staticmethod
    def um(X):
        return 1 + torch.log(2 * torch.sqrt(X[:, 0]**2 + X[:, 1]**2))


def derivatives(u, X):
    """u(X), grad u(X) (N,2) and Laplacian u(X) by automatic differentiation.
    u maps (N,2) -> (N,) pointwise."""
    X = X.clone().requires_grad_(True)
    val = u(X)
    (g,) = torch.autograd.grad(val.sum(), X, create_graph=True)
    lap = torch.zeros(X.shape[0])
    if g.requires_grad:                            # otherwise grad u is constant: Lap u = 0
        for d in range(2):
            (h,) = torch.autograd.grad(g[:, d].sum(), X, retain_graph=True, allow_unused=True)
            if h is not None:
                lap = lap + h[:, d]
    return val.detach(), g.detach(), lap.detach()


def random_solution(gen, nterms=4, wmax=5.0):
    """Random smooth function u(X) = sum_j A_j sin(w_j . x + phase_j)."""
    A = torch.randn(nterms, generator=gen)
    W = (2 * torch.rand(nterms, 2, generator=gen) - 1) * wmax
    ph = 2 * math.pi * torch.rand(nterms, generator=gen)
    return lambda X: (A * torch.sin(X @ W.T + ph)).sum(1)


# =============================================================================
# Baseline weights: PHS + polynomials (RBF-FD), in scaled coordinates
# =============================================================================
def monomials(deg):
    ea, eb = [], []
    for t in range(deg + 1):
        for b in range(t + 1):
            ea.append(t - b), eb.append(b)
    return torch.tensor(ea), torch.tensor(eb)


@torch.no_grad()
def phs_weights(Yh, op, nvec, deg, beta, chunk=2000):
    """Scaled RBF-FD weights w_hat (B,n) for L = Laplacian ('lap') or d/dn ('dn') at x_hat = 0,
    and an orthonormal basis Qp (B,n,Q) of range(P). True weights: w_hat / r^k (k = 2 or 1)."""
    ea, eb = monomials(deg)
    Q = len(ea)
    W, QP = [], []
    for i in range(0, Yh.shape[0], chunk):
        Y = Yh[i:i + chunk]
        B, n, _ = Y.shape
        D = torch.cdist(Y, Y, compute_mode="donot_use_mm_for_euclid_dist")
        rr = torch.sqrt((Y**2).sum(2))
        P = Y[:, :, 0:1]**ea * Y[:, :, 1:2]**eb
        rp = torch.zeros(B, Q)
        if op == "lap":
            rk = beta**2 * rr**(beta - 2)
            rp[:, (ea == 2) & (eb == 0)] = 2.0
            rp[:, (ea == 0) & (eb == 2)] = 2.0
        else:
            nv = nvec[i:i + chunk]
            rk = beta * rr**(beta - 2) * (-(Y[:, :, 0] * nv[:, None, 0] + Y[:, :, 1] * nv[:, None, 1]))
            rp[:, (ea == 1) & (eb == 0)] = nv[:, 0:1]
            rp[:, (ea == 0) & (eb == 1)] = nv[:, 1:2]
        M = torch.cat([torch.cat([D**beta, P], 2),
                       torch.cat([P.transpose(1, 2), torch.zeros(B, Q, Q)], 2)], 1)
        sol = torch.linalg.solve(M, torch.cat([rk, rp], 1).unsqueeze(2)).squeeze(2)
        W.append(sol[:, :n])
        QP.append(torch.linalg.qr(P).Q)
    return torch.cat(W), torch.cat(QP)


# =============================================================================
# Geometry: nodes, unknown numbering, stencils, baseline weights
# =============================================================================
class Geometry:
    """Node set X^(m) of a test problem (exported from nodes_TP*.m) with the unknown
    numbering, the stencils of the FD solver and the baseline (PHS r^beta) weights."""

    def __init__(self, problem, m, ntype="uniform", deg=4, beta=5, nloc=None):
        self.problem, self.m, self.ntype, self.deg, self.beta = problem, m, ntype, deg, beta
        Q = (deg + 1) * (deg + 2) // 2
        nloc = 3 * Q if nloc is None else nloc
        d = sio.loadmat(os.path.join(HERE, "nodes", f"{problem.name}_{ntype}_m{m}.mat"))
        f = lambda k: np.asarray(d[k], float).reshape(-1, 2)
        Xip, Xim, Xbp, Xbm, XG = f("Xint_p"), f("Xint_m"), f("Xbd_p"), f("Xbd_m"), f("XG")
        self.h = float(d["h"].squeeze())
        self.Xp, self.Xm = np.vstack([Xip, Xbp, XG]), np.vstack([Xim, Xbm, XG])
        self.nIp, self.nIm, self.nBp, self.nBm, self.nG = len(Xip), len(Xim), len(Xbp), len(Xbm), len(XG)
        self.Np, self.Nm = len(self.Xp), len(self.Xm)
        self.N = self.Np + self.Nm
        self.XG, self.nrm = XG, problem.normal(XG)
        self.iGp = self.nIp + self.nBp + np.arange(self.nG)       # interface nodes in X+ (local)
        self.iGm = self.nIm + self.nBm + np.arange(self.nG)       # interface nodes in X- (local)
        self.interior = torch.tensor(np.r_[np.arange(self.nIp), self.Np + np.arange(self.nIm)])
        # row groups (row order as in the MATLAB solver)
        r0 = self.nIp + self.nIm + self.nBp + self.nBm + self.nG   # first flux row
        self.groups = [
            self._group(0, "lap", self.Xp, np.arange(self.nIp), 0, 0, -problem.ap, nloc),
            self._group(1, "lap", self.Xm, np.arange(self.nIm), self.Np, self.nIp, -problem.am, nloc),
            self._group(2, "dn", self.Xp, self.iGp, 0, r0, +problem.ap, nloc),
            self._group(3, "dn", self.Xm, self.iGm, self.Np, r0, -problem.am, nloc),
        ]
        self.rows_lap = torch.arange(self.nIp + self.nIm)
        self.rows_flux = r0 + torch.arange(self.nG)
        # fixed rows: Dirichlet (u = g) and jump (u+ - u- = phi)
        r = self.nIp + self.nIm
        kb = np.r_[self.nIp + np.arange(self.nBp), self.Np + self.nIm + np.arange(self.nBm)]
        rb = r + np.arange(self.nBp + self.nBm)
        rj = r + self.nBp + self.nBm + np.arange(self.nG)
        self.fix_I = torch.tensor(np.r_[rb, rj, rj])
        self.fix_J = torch.tensor(np.r_[kb, self.iGp, self.Np + self.iGm])
        self.fix_V = torch.tensor(np.r_[np.ones(len(kb)), np.ones(self.nG), -np.ones(self.nG)])

    def _group(self, gid, op, Xs, irows, col0, row0, coef, nloc):
        n = min(nloc, len(Xs))                                     # stencil size (capped)
        _, idx = cKDTree(Xs).query(Xs[irows], k=n)
        Y = Xs[idx] - Xs[irows][:, None, :]
        r = np.sqrt((Y**2).sum(2)).max(1)                          # stencil radius
        g = dict(id=gid, geom=self, op=op, coef=coef, n=n, k=2 if op == "lap" else 1,
                 Yh=torch.tensor(Y / r[:, None, None]), r=torch.tensor(r),
                 cols=torch.tensor(col0 + idx),
                 rows=torch.tensor(row0 + np.arange(len(irows))),
                 nvec=torch.tensor(self.nrm) if op == "dn" else None)
        g["wb"], g["Qp"] = phs_weights(g["Yh"], op, g["nvec"], self.deg, self.beta)
        g["s"] = torch.sqrt((g["wb"]**2).mean(1, keepdim=True))    # rms of the baseline weights
        g["rr"] = torch.sqrt((g["Yh"]**2).sum(2))
        return g


# =============================================================================
# Corrections v (one number per stencil node)
# =============================================================================
class Baseline(torch.nn.Module):
    """No correction: classical RBF-FD (PHS r^beta + polynomials)."""

    def correction(self, g):
        return None


class FreeCorrection(torch.nn.Module):
    """One trainable number per (row, stencil node) of ONE node set, no network.
    Gives the best the loss can reach on that node set; it does not transfer to other node sets."""

    def __init__(self, geom):
        super().__init__()
        self.geom = geom
        self.v = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros_like(g["wb"])) for g in geom.groups])

    def correction(self, g):
        if g["geom"] is not self.geom:
            raise ValueError("FreeCorrection belongs to one node set only")
        return self.v[g["id"]]


class StencilNet(torch.nn.Module):
    """Permutation-equivariant network: stencil -> one number per stencil node.
    Node features: 'lap': (y1, y2, |y|, w_base/s);  'dn': (y.n, y.t, |y|, w_base/s)
    (n = normal, t = tangent). Each node sees its own features and the pooled
    (mean, max) features of the whole stencil. The last layer starts at zero."""

    def __init__(self, width=64):
        super().__init__()
        act = torch.nn.SiLU()
        self.phi = torch.nn.Sequential(torch.nn.Linear(4, width), act, torch.nn.Linear(width, width), act)
        self.psi = torch.nn.Sequential(torch.nn.Linear(3 * width, width), act, torch.nn.Linear(width, width), act)
        self.out = torch.nn.Linear(width, 1)
        torch.nn.init.zeros_(self.out.weight)
        torch.nn.init.zeros_(self.out.bias)

    def forward(self, g):
        Yh, wb = g["Yh"], (g["wb"] / g["s"])[:, :, None]
        if g["op"] == "dn":
            nv = g["nvec"]
            yn = Yh[:, :, 0] * nv[:, None, 0] + Yh[:, :, 1] * nv[:, None, 1]
            yt = -Yh[:, :, 0] * nv[:, None, 1] + Yh[:, :, 1] * nv[:, None, 0]
            feat = torch.cat([yn[:, :, None], yt[:, :, None], g["rr"][:, :, None], wb], 2)
        else:
            feat = torch.cat([Yh, g["rr"][:, :, None], wb], 2)
        z = self.phi(feat)                                         # (B,n,W)
        pool = torch.cat([z.mean(1), z.max(1).values], 1)          # (B,2W)
        z = torch.cat([z, pool[:, None, :].expand(-1, z.shape[1], -1)], 2)
        return self.out(self.psi(z)).squeeze(2)                    # (B,n)


class NetCorrection(torch.nn.Module):
    """Learned correction: one StencilNet for the Laplacian rows, one for the flux rows."""

    def __init__(self, width=64):
        super().__init__()
        self.nets = torch.nn.ModuleDict({"lap": StencilNet(width), "dn": StencilNet(width)})

    def correction(self, g):
        return self.nets[g["op"]](g)


# =============================================================================
# Weights, system matrix
# =============================================================================
def scaled_weights(model, g):
    """w_hat = w_base + s (I - Pi) v : exact on the polynomials for every v."""
    v = model.correction(g)
    if v is None:
        return g["wb"]
    Qp = g["Qp"]
    v = v - torch.einsum("bnq,bq->bn", Qp, torch.einsum("bnq,bn->bq", Qp, v))
    return g["wb"] + g["s"] * v


def rows(model, geom):
    """Triplets (I, J, V) of the derivative rows and rho = mean_i sum_j |w_hat_ij| |y_hat_j|^deg."""
    I, J, V, rho = [], [], [], []
    for g in geom.groups:
        wh = scaled_weights(model, g)
        w = g["coef"] * wh / g["r"][:, None]**g["k"]
        I.append(g["rows"][:, None].expand_as(w).reshape(-1))
        J.append(g["cols"].reshape(-1))
        V.append(w.reshape(-1))
        rho.append((wh.abs() * g["rr"]**geom.deg).sum(1))
    return torch.cat(I), torch.cat(J), torch.cat(V), torch.cat(rho).mean()


def matrix(model, geom):
    """Dense system matrix A (differentiable) and rho."""
    I, J, V, rho = rows(model, geom)
    I, J, V = torch.cat([I, geom.fix_I]), torch.cat([J, geom.fix_J]), torch.cat([V, geom.fix_V])
    return torch.zeros(geom.N, geom.N).index_put((I, J), V, accumulate=True), rho


@torch.no_grad()
def sparse_matrix(model, geom):
    I, J, V, _ = rows(model, geom)
    I, J, V = torch.cat([I, geom.fix_I]), torch.cat([J, geom.fix_J]), torch.cat([V, geom.fix_V])
    return sp.csc_matrix((V.numpy(), (I.numpy(), J.numpy())), shape=(geom.N, geom.N))


# =============================================================================
# Right-hand sides
# =============================================================================
def data(geom, up, um):
    """Exact nodal values U* (N,) and right-hand side b (N,) for the solution (u+, u-)."""
    pr = geom.problem
    vp, gp, lp = derivatives(up, torch.tensor(geom.Xp))
    vm, gm, lm = derivatives(um, torch.tensor(geom.Xm))
    nv = torch.tensor(geom.nrm)
    iGp, iGm = torch.tensor(geom.iGp), torch.tensor(geom.iGm)
    kbp = geom.nIp + torch.arange(geom.nBp)
    kbm = geom.nIm + torch.arange(geom.nBm)
    b = torch.cat([
        -pr.ap * lp[:geom.nIp], -pr.am * lm[:geom.nIm],            # f = -alpha Lap u
        vp[kbp], vm[kbm],                                          # g
        vp[iGp] - vm[iGm],                                         # phi
        pr.ap * (gp[iGp] * nv).sum(1) - pr.am * (gm[iGm] * nv).sum(1),   # psi
    ])
    return torch.cat([vp, vm]), b


def manufactured(geom, nsol, gen):
    """nsol random smooth solutions (independent on the two sides): U* (N,nsol), b (N,nsol)."""
    U, Bm = [], []
    for _ in range(nsol):
        u, b = data(geom, random_solution(gen), random_solution(gen))
        U.append(u), Bm.append(b)
    return torch.stack(U, 1), torch.stack(Bm, 1)


# =============================================================================
# Loss and training
# =============================================================================
class Reference:
    """Baseline quantities of one node set (A_0 factorized, rho_0)."""

    def __init__(self, geom):
        with torch.no_grad():
            self.A0, self.rho0 = matrix(Baseline(), geom)
            self.lu0 = torch.linalg.lu_factor(self.A0)

    def terms(self, geom, U, b):
        E0 = torch.linalg.lu_solve(*self.lu0, b) - U
        tau0 = self.A0 @ U - b
        return ((E0[geom.interior]**2).sum(), (tau0[geom.rows_lap]**2).sum(), (tau0[geom.rows_flux]**2).sum())


def loss_terms(model, geom, ref, U, b):
    """(||E||^2/||E_0||^2,  mean of the tau ratios of the Laplacian and flux rows,  rho/rho_0)."""
    A, rho = matrix(model, geom)
    E = torch.linalg.solve(A, b) - U
    tau = A @ U - b
    e0, tl0, tf0 = ref.terms(geom, U, b)
    rE = (E[geom.interior]**2).sum() / e0
    rT = 0.5 * ((tau[geom.rows_lap]**2).sum() / tl0 + (tau[geom.rows_flux]**2).sum() / tf0)
    return rE, rT, rho / ref.rho0


def train(model, geoms, steps=300, nsol=8, lam_tau=0.1, lam_l1=0.1, lr=2e-3, seed=0, log=25, nval=16):
    """Minimize  ||E||^2/||E_0||^2 + lam_tau * (tau ratio) + lam_l1 * rho/rho_0  over
    manufactured solutions on geoms (new random solutions at every step). Every `log`
    steps the three terms are evaluated on a fixed validation set (nval solutions per
    node set). Returns [(step, [(rE, rT, rRho) per node set])]."""
    gen = torch.Generator().manual_seed(seed)
    vgen = torch.Generator().manual_seed(seed + 12345)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    refs = [Reference(g) for g in geoms]
    val = [manufactured(g, nval, vgen) for g in geoms]
    hist = []
    for it in range(steps + 1):
        if it % log == 0:
            with torch.no_grad():
                v = [tuple(float(t) for t in loss_terms(model, g, r, U, b)) for g, r, (U, b) in zip(geoms, refs, val)]
            print(f"step {it:4d}  validation  E ratio: " + " ".join(f"{t[0]:7.3f}" for t in v)
                  + "   tau ratio: " + " ".join(f"{t[1]:6.3f}" for t in v)
                  + "   rho ratio: " + " ".join(f"{t[2]:5.2f}" for t in v), flush=True)
            hist.append((it, v))
        if it == steps:
            break
        opt.zero_grad()
        loss = 0.0
        for g, r in zip(geoms, refs):
            U, b = manufactured(g, nsol, gen)
            rE, rT, rR = loss_terms(model, g, r, U, b)
            loss = loss + (rE + lam_tau * rT + lam_l1 * rR) / len(geoms)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return hist


# =============================================================================
# Solving a test problem
# =============================================================================
def sigma(A):
    """Stability constant sigma = condest(A')/norm(A,inf): estimate of ||A^{-1}||_inf."""
    lu = spla.splu(A.tocsc())
    op = spla.LinearOperator(A.shape, matvec=lambda v: lu.solve(v, trans="T"),
                             rmatvec=lambda v: lu.solve(v))
    return spla.onenormest(op)


def solve(model, geom, with_sigma=True):
    """Solve the test problem of geom with its exact data; errors over the interior nodes."""
    A = sparse_matrix(model, geom)
    U, b = data(geom, geom.problem.up, geom.problem.um)
    u = spla.spsolve(A, b.numpy())
    e = np.abs(u - U.numpy())
    ei = e[geom.interior.numpy()]
    return dict(h=geom.h, Nint=len(ei), rms=float(np.sqrt((ei**2).mean())), max=float(ei.max()),
                sigma=sigma(A) if with_sigma else float("nan"), u=u, err=e, nnz_row=A.nnz / geom.N)


def table(model, problem, levels, ntype="uniform", deg=4, beta=5, nloc=None, msigma=3):
    """Convergence table: baseline (PHS r^beta) against the model."""
    print(f"{problem.name}, FD, degree {deg}, {ntype} nodes: baseline (PHS r^{beta}) vs learned weights")
    print(f"{'m':>2} {'h':>8} {'Nint':>7} | {'rms base':>10} {'max base':>10} {'sigma':>9} |"
          f" {'rms learn':>10} {'max learn':>10} {'sigma':>9} | {'rms ratio':>9}")
    out = []
    for m in levels:
        geom = Geometry(problem, m, ntype, deg, beta, nloc)
        r0 = solve(Baseline(), geom, with_sigma=m <= msigma)
        r1 = solve(model, geom, with_sigma=m <= msigma)
        print(f"{m:2d} {geom.h:8.5f} {r0['Nint']:7d} | {r0['rms']:10.3e} {r0['max']:10.3e} {r0['sigma']:9.2e} |"
              f" {r1['rms']:10.3e} {r1['max']:10.3e} {r1['sigma']:9.2e} | {r1['rms'] / r0['rms']:9.3f}", flush=True)
        out.append((m, geom.h, r0, r1))
    return out


def phs_table(problem, levels, ntype="uniform", deg=4, betas=(3, 5, 7, 9), nloc=None):
    """Reference: rms error (interior nodes) of classical RBF-FD with PHS r^beta alone."""
    print(f"{problem.name}, FD, degree {deg}, {ntype} nodes: rms error of PHS r^beta alone")
    print(f"{'beta':>5} " + " ".join(f"{'m=' + str(m):>10}" for m in levels))
    out = {}
    for beta in betas:
        out[beta] = [solve(Baseline(), Geometry(problem, m, ntype, deg, beta, nloc), with_sigma=False)["rms"]
                     for m in levels]
        print(f"{beta:5d} " + " ".join(f"{e:10.3e}" for e in out[beta]), flush=True)
    return out
