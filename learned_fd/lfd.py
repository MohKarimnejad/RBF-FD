"""
lfd -- learned meshless finite differences with a per-stencil kernel mixture.

Setting (same discretization as PHS_fd_TP1.m / PHS_fd_TP2.m, solver = 'FD'):
  elliptic interface problem  -div(alpha grad u) = f  in Omega+ and Omega-,
  u = g on dOmega,  [u] = phi,  [alpha du/dn] = psi  on Gamma.
  Unknowns: u+ at X+ = [Xint+; Xbd+; XG], u- at X- = [Xint-; Xbd-; XG].
  Rows: Laplacian rows (Xint+, Xint-), Dirichlet rows, jump rows, flux rows.
  Every derivative row uses one stencil = the n nearest nodes of the same side.

Weights of one stencil (row point x, scaled coordinates y_hat = (y - x)/r):
    K = sum_k a_k K_k,   a_k >= 0, sum_k a_k = 1         (kernel mixture)
    [K(Y,Y) P; P' 0] [w_hat; gamma] = [L K(0,Y); L p(0)]  (RBF-FD saddle-point system)
  P = polynomials of degree <= deg, so every row is exact on these polynomials for
  every choice of a. A small network maps the stencil geometry to a (softmax).
  a = e_k (one kernel only) gives the classical RBF-FD weights of that kernel.

Loss (training on manufactured solutions on the same node sets):
    L = ||E||^2 / ||E_0||^2  +  lam * rho / rho_0
  E   = -A^{-1} tau, tau = A U* - b : nodal error of the learned discretization,
  E_0 : the same for the baseline (PHS r^5 only),
  rho = mean_i sum_j |w_hat_ij| |y_hat_j|^mu  : scale-free form of eq. (20) of
        Davydov & Schaback (Minimal numerical differentiation formulas), mu = deg.
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
# Kernel dictionary (radial, in scaled stencil coordinates)
# =============================================================================
# Each kernel: (name, phi, g, lap) with, for r = ||x - y||,
#   phi(r) = K,   g(r) = K'(r)/r,   lap(r) = K''(r) + K'(r)/r  (Laplacian in 2D).
# PHS signs are chosen so that every kernel is conditionally positive definite
# of order <= deg + 1 for deg >= 3:  r^3, -r^5, r^7.  Then any nonnegative
# combination is conditionally positive definite and the local system is solvable.
# The kernels are rescaled by normalize_dictionary before use.

def _phs(beta, sign):
    return (lambda r: sign * r**beta,
            lambda r: sign * beta * r**(beta - 2),
            lambda r: sign * beta**2 * r**(beta - 2))


def _gauss(eps):
    e2 = eps**2
    return (lambda r: torch.exp(-e2 * r**2),
            lambda r: -2 * e2 * torch.exp(-e2 * r**2),
            lambda r: (-4 * e2 + 4 * e2**2 * r**2) * torch.exp(-e2 * r**2))


def _matern52(eps):
    # K = (1 + s + s^2/3) exp(-s), s = eps r
    return (lambda r: (1 + eps * r + (eps * r)**2 / 3) * torch.exp(-eps * r),
            lambda r: -(eps**2 / 3) * (1 + eps * r) * torch.exp(-eps * r),
            lambda r: -(eps**2 / 3) * (2 + 2 * eps * r - (eps * r)**2) * torch.exp(-eps * r))


def make_dictionary(gauss_eps=(3.0, 5.0, 8.0), matern_eps=(2.0, 6.0)):
    """Kernel dictionary: PHS r^3, -r^5, r^7, Gaussians and Matern-5/2 kernels.
    Shape parameters refer to scaled coordinates (stencil radius = 1)."""
    D = [("phs3",) + _phs(3, +1.0), ("phs5",) + _phs(5, -1.0), ("phs7",) + _phs(7, +1.0)]
    D += [(f"gauss{e:g}",) + _gauss(e) for e in gauss_eps]
    D += [(f"matern{e:g}",) + _matern52(e) for e in matern_eps]
    return D


def monomials(deg):
    ea, eb = [], []
    for t in range(deg + 1):
        for b in range(t + 1):
            ea.append(t - b), eb.append(b)
    return torch.tensor(ea), torch.tensor(eb)


def normalize_dictionary(kernels, deg, nloc=None, nref=200, seed=0):
    """Rescale every kernel so that the mixture coefficients are comparable.
    Only the part of K(Y,Y) on the null space of P' (Z'KZ, Z = orthonormal basis of
    {v : P'v = 0}) determines the weights, and its size differs by orders of magnitude
    between kernels. Each kernel is divided by the median of trace(Z'KZ)/(n-Q) over
    nref random reference stencils (n = nloc points in the unit disc, one at the origin)."""
    ea, eb = monomials(deg)
    Q = len(ea)
    n = 3 * Q if nloc is None else nloc
    gen = torch.Generator().manual_seed(seed)
    rad = torch.sqrt(torch.rand(nref, n, generator=gen))
    ang = 2 * math.pi * torch.rand(nref, n, generator=gen)
    Y = torch.stack([rad * torch.cos(ang), rad * torch.sin(ang)], 2)
    Y[:, 0] = 0.0
    Y = Y / torch.sqrt((Y**2).sum(2)).max(1).values[:, None, None]
    P = Y[:, :, 0:1]**ea * Y[:, :, 1:2]**eb
    Z = torch.linalg.qr(P, mode="complete").Q[:, :, Q:]
    D = torch.cdist(Y, Y, compute_mode="donot_use_mm_for_euclid_dist")
    out = []
    for name, phi, g, lap in kernels:
        tr = torch.einsum("bij,bjk,bki->b", Z.transpose(1, 2), phi(D), Z) / (n - Q)
        c = float(tr.median())
        if c <= 0:
            raise ValueError(f"kernel {name} is not conditionally positive definite with this sign")
        out.append((name, (lambda r, f=phi, c=c: f(r) / c), (lambda r, f=g, c=c: f(r) / c),
                    (lambda r, f=lap, c=c: f(r) / c)))
    return out


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
# Geometry: nodes, unknown numbering, stencils
# =============================================================================
class Geometry:
    """Node set X^(m) of a test problem (exported from nodes_TP*.m) with the
    unknown numbering and the stencils of the FD solver."""

    def __init__(self, problem, m, ntype="uniform", deg=4, nloc=None):
        self.problem, self.m, self.ntype, self.deg = problem, m, ntype, deg
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
        # all unknowns, and the interior ones (errors are measured there, as in the paper)
        self.X = torch.tensor(np.vstack([self.Xp, self.Xm]))
        self.interior = torch.tensor(np.r_[np.arange(self.nIp), self.Np + np.arange(self.nIm)])
        self.side_p = torch.arange(self.N) < self.Np
        # row groups: (operator, rows in A, stencil data); row order as in the MATLAB solver
        r0 = self.nIp + self.nIm + self.nBp + self.nBm + self.nG   # first flux row
        self.groups = [
            self._group("lap", self.Xp, np.arange(self.nIp), 0, 0, -problem.ap, nloc),
            self._group("lap", self.Xm, np.arange(self.nIm), self.Np, self.nIp, -problem.am, nloc),
            self._group("dn", self.Xp, self.iGp, 0, r0, +problem.ap, nloc),
            self._group("dn", self.Xm, self.iGm, self.Np, r0, -problem.am, nloc),
        ]
        # fixed rows: Dirichlet (u = g) and jump (u+ - u- = phi)
        r = self.nIp + self.nIm
        kb = np.r_[self.nIp + np.arange(self.nBp), self.Np + self.nIm + np.arange(self.nBm)]
        rb = r + np.arange(self.nBp + self.nBm)
        rj = r + self.nBp + self.nBm + np.arange(self.nG)
        self.fix_I = torch.tensor(np.r_[rb, rj, rj])
        self.fix_J = torch.tensor(np.r_[kb, self.iGp, self.Np + self.iGm])
        self.fix_V = torch.tensor(np.r_[np.ones(len(kb)), np.ones(self.nG), -np.ones(self.nG)])
        self.rows_lap = torch.arange(self.nIp + self.nIm)
        self.rows_flux = r0 + torch.arange(self.nG)

    def _group(self, op, Xs, irows, col0, row0, coef, nloc):
        n = min(nloc, len(Xs))                                     # stencil size (capped)
        _, idx = cKDTree(Xs).query(Xs[irows], k=n)
        Y = Xs[idx] - Xs[irows][:, None, :]
        r = np.sqrt((Y**2).sum(2)).max(1)                          # stencil radius
        g = dict(op=op, coef=coef, n=n,
                 Yh=torch.tensor(Y / r[:, None, None]), r=torch.tensor(r),
                 cols=torch.tensor(col0 + idx),
                 rows=torch.tensor(row0 + np.arange(len(irows))))
        if op == "dn":
            g["nvec"] = torch.tensor(self.nrm)
        return g


# =============================================================================
# Local weights from a kernel mixture
# =============================================================================
def local_weights(g, a, deg, kernels):
    """Scaled weights w_hat (B,n) of the rows of group g for mixture coefficients a (B,K).
    The true weights are w_hat / r^k (k = 2 for 'lap', 1 for 'dn')."""
    Yh = g["Yh"]
    B, n, _ = Yh.shape
    ea, eb = monomials(deg)
    Q = len(ea)
    D = torch.cdist(Yh, Yh, compute_mode="donot_use_mm_for_euclid_dist")   # (B,n,n)
    rr = torch.sqrt((Yh**2).sum(2))                                # distances to x_hat = 0
    if g["op"] == "dn":
        nv = g["nvec"]
        yn = -(Yh[:, :, 0] * nv[:, None, 0] + Yh[:, :, 1] * nv[:, None, 1])   # (x_hat - y_hat).n
    Kmat = torch.zeros(B, n, n)
    rk = torch.zeros(B, n)
    for k, (_, phi, gk, lap) in enumerate(kernels):
        ak = a[:, k]
        Kmat = Kmat + ak[:, None, None] * phi(D)
        rk = rk + ak[:, None] * (lap(rr) if g["op"] == "lap" else gk(rr) * yn)
    P = Yh[:, :, 0:1]**ea * Yh[:, :, 1:2]**eb                      # (B,n,Q)
    rp = torch.zeros(B, Q)
    if g["op"] == "lap":
        rp[:, (ea == 2) & (eb == 0)] = 2.0
        rp[:, (ea == 0) & (eb == 2)] = 2.0
    else:
        rp[:, (ea == 1) & (eb == 0)] = nv[:, 0:1]
        rp[:, (ea == 0) & (eb == 1)] = nv[:, 1:2]
    M = torch.cat([torch.cat([Kmat, P], 2),
                   torch.cat([P.transpose(1, 2), torch.zeros(B, Q, Q)], 2)], 1)
    sol = torch.linalg.solve(M, torch.cat([rk, rp], 1).unsqueeze(2)).squeeze(2)
    return sol[:, :n]


# =============================================================================
# Network: stencil geometry -> mixture coefficients
# =============================================================================
class StencilNet(torch.nn.Module):
    """Permutation-invariant network (DeepSets): per-node features -> pooled -> a (softmax).
    Features: 'lap': (y1, y2, |y|);  'dn': (y.n, y.t, |y|) with n the normal, t the tangent.
    At initialization a is (almost) the baseline kernel only."""

    def __init__(self, nk, base=1, width=48, bias0=-5.0):
        super().__init__()
        act = torch.nn.SiLU()
        self.phi = torch.nn.Sequential(torch.nn.Linear(3, width), act, torch.nn.Linear(width, width), act)
        self.rho = torch.nn.Sequential(torch.nn.Linear(2 * width, width), act)
        self.out = torch.nn.Linear(width, nk)
        torch.nn.init.zeros_(self.out.weight)
        with torch.no_grad():
            self.out.bias.fill_(bias0)
            self.out.bias[base] = 0.0

    def forward(self, g):
        Yh = g["Yh"]
        rr = torch.sqrt((Yh**2).sum(2, keepdim=True))
        if g["op"] == "dn":
            nv = g["nvec"]
            yn = Yh[:, :, 0] * nv[:, None, 0] + Yh[:, :, 1] * nv[:, None, 1]
            yt = -Yh[:, :, 0] * nv[:, None, 1] + Yh[:, :, 1] * nv[:, None, 0]
            feat = torch.cat([yn[:, :, None], yt[:, :, None], rr], 2)
        else:
            feat = torch.cat([Yh, rr], 2)
        z = self.phi(feat)
        z = torch.cat([z.mean(1), z.max(1).values], 1)
        return torch.softmax(self.out(self.rho(z)), 1)


class MixtureFD(torch.nn.Module):
    """Learned discretization: one StencilNet for the Laplacian rows, one for the flux rows."""

    def __init__(self, deg=4, kernels=None, base="phs5", nloc=None):
        super().__init__()
        self.deg = deg
        kernels = make_dictionary() if kernels is None else kernels
        self.kernels = normalize_dictionary(kernels, deg, nloc)
        self.names = [k[0] for k in self.kernels]
        self.base = self.names.index(base)
        self.nets = torch.nn.ModuleDict({op: StencilNet(len(self.kernels), self.base) for op in ("lap", "dn")})

    def coefficients(self, g, baseline=False):
        if baseline:                                               # one kernel only: classical RBF-FD
            a = torch.zeros(g["Yh"].shape[0], len(self.kernels))
            a[:, self.base] = 1.0
            return a
        return self.nets[g["op"]](g)

    def rows(self, geom, baseline=False):
        """Triplets (I, J, V) of the derivative rows, and the growth term rho (eq. 20, mu = deg)."""
        I, J, V, rho = [], [], [], []
        for g in geom.groups:
            a = self.coefficients(g, baseline)
            wh = local_weights(g, a, self.deg, self.kernels)
            k = 2 if g["op"] == "lap" else 1
            w = g["coef"] * wh / g["r"][:, None]**k
            I.append(g["rows"][:, None].expand_as(w).reshape(-1))
            J.append(g["cols"].reshape(-1))
            V.append(w.reshape(-1))
            rr = torch.sqrt((g["Yh"]**2).sum(2))
            rho.append((wh.abs() * rr**self.deg).sum(1))
        return torch.cat(I), torch.cat(J), torch.cat(V), torch.cat(rho).mean()

    def matrix(self, geom, baseline=False):
        """Dense system matrix A (differentiable) and the growth term."""
        I, J, V, rho = self.rows(geom, baseline)
        I, J, V = torch.cat([I, geom.fix_I]), torch.cat([J, geom.fix_J]), torch.cat([V, geom.fix_V])
        A = torch.zeros(geom.N, geom.N).index_put((I, J), V, accumulate=True)
        return A, rho

    @torch.no_grad()
    def sparse_matrix(self, geom, baseline=False):
        I, J, V, _ = self.rows(geom, baseline)
        I, J, V = torch.cat([I, geom.fix_I]), torch.cat([J, geom.fix_J]), torch.cat([V, geom.fix_V])
        return sp.csc_matrix((V.numpy(), (I.numpy(), J.numpy())), shape=(geom.N, geom.N))

    @torch.no_grad()
    def mixture_map(self, geom):
        """Mixture coefficients a (rows x K) of the Laplacian rows and of the flux rows (+ side)."""
        lap = torch.cat([self.coefficients(geom.groups[0]), self.coefficients(geom.groups[1])])
        return lap.numpy(), self.coefficients(geom.groups[2]).numpy()


# =============================================================================
# Right-hand sides
# =============================================================================
def data(geom, up, um):
    """Exact nodal values U* (N,) and right-hand side b (N,) for the solution (u+, u-)."""
    pr = geom.problem
    Xp, Xm = torch.tensor(geom.Xp), torch.tensor(geom.Xm)
    vp, gp, lp = derivatives(up, Xp)
    vm, gm, lm = derivatives(um, Xm)
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
# Training
# =============================================================================
def train(model, geoms, steps=300, nsol=8, lam=0.1, lr=3e-3, seed=0, log=25, nval=16):
    """Minimize  ||E||^2/||E_0||^2 + lam * rho/rho_0  over manufactured solutions on geoms.
    E_0, rho_0: baseline (one kernel) on the same node set and the same solutions.
    New random solutions are drawn at every step; every `log` steps the error ratio is
    also evaluated on a fixed validation set of nval solutions per node set.
    Returns the history [(step, loss, validation ratio per node set)]."""
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    base, val = [], []
    with torch.no_grad():
        for geom in geoms:
            A0, rho0 = model.matrix(geom, baseline=True)
            base.append((torch.linalg.lu_factor(A0), rho0))
    vgen = torch.Generator().manual_seed(seed + 12345)
    for geom, (lu0, _) in zip(geoms, base):
        U, b = manufactured(geom, nval, vgen)
        E0 = torch.linalg.lu_solve(*lu0, b) - U
        val.append((U, b, (E0[geom.interior]**2).sum()))
    hist = []
    for it in range(steps + 1):
        if it % log == 0:
            with torch.no_grad():
                vr = []
                for geom, (U, b, e0) in zip(geoms, val):
                    A, _ = model.matrix(geom)
                    E = torch.linalg.solve(A, b) - U
                    vr.append(float((E[geom.interior]**2).sum() / e0))
            print(f"step {it:4d}  validation ||E||^2/||E_0||^2: " + "  ".join(f"{r:.3f}" for r in vr))
            hist.append((it, vr))
        if it == steps:
            break
        opt.zero_grad()
        loss = 0.0
        for geom, (lu0, rho0) in zip(geoms, base):
            U, b = manufactured(geom, nsol, gen)
            A, rho = model.matrix(geom)
            E = torch.linalg.solve(A, b) - U
            with torch.no_grad():
                E0 = torch.linalg.lu_solve(*lu0, b) - U
            ratio = (E[geom.interior]**2).sum() / (E0[geom.interior]**2).sum()
            loss = loss + (ratio + lam * rho / rho0) / len(geoms)
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


@torch.no_grad()
def solve(model, geom, baseline=False, with_sigma=True):
    """Solve the test problem of geom with its exact data; errors over the interior nodes."""
    A = model.sparse_matrix(geom, baseline)
    with torch.enable_grad():
        U, b = data(geom, geom.problem.up, geom.problem.um)
    u = spla.spsolve(A, b.numpy())
    e = np.abs(u - U.numpy())
    ei = e[geom.interior.numpy()]
    return dict(h=geom.h, Nint=len(ei), rms=float(np.sqrt((ei**2).mean())), max=float(ei.max()),
                sigma=sigma(A) if with_sigma else float("nan"), u=u, err=e,
                nnz_row=A.nnz / geom.N)


def table(model, problem, levels, ntype="uniform", deg=None, nloc=None, msigma=3):
    """Convergence table: baseline (one kernel) against the learned mixture."""
    deg = model.deg if deg is None else deg
    print(f"{problem.name}, FD, degree {deg}, {ntype} nodes: baseline ({model.names[model.base]}) vs learned mixture")
    print(f"{'m':>2} {'h':>8} {'Nint':>7} | {'rms base':>10} {'max base':>10} {'sigma':>9} |"
          f" {'rms learn':>10} {'max learn':>10} {'sigma':>9} | {'rms ratio':>9}")
    out = []
    for m in levels:
        geom = Geometry(problem, m, ntype, deg, nloc)
        r0 = solve(model, geom, baseline=True, with_sigma=m <= msigma)
        r1 = solve(model, geom, baseline=False, with_sigma=m <= msigma)
        print(f"{m:2d} {geom.h:8.5f} {r0['Nint']:7d} | {r0['rms']:10.3e} {r0['max']:10.3e} {r0['sigma']:9.2e} |"
              f" {r1['rms']:10.3e} {r1['max']:10.3e} {r1['sigma']:9.2e} | {r1['rms'] / r0['rms']:9.3f}")
        out.append((m, geom.h, r0, r1))
    return out


def single_kernel_table(model, problem, levels, ntype="uniform", nloc=None):
    """Reference: rms error (interior nodes) of classical RBF-FD with each dictionary kernel alone."""
    print(f"{problem.name}, FD, degree {model.deg}, {ntype} nodes: rms error with one kernel only")
    print(f"{'kernel':>9} " + " ".join(f"{'m=' + str(m):>10}" for m in levels))
    geoms = [Geometry(problem, m, ntype, model.deg, nloc) for m in levels]
    base, out = model.base, {}
    for k, name in enumerate(model.names):
        model.base = k
        out[name] = [solve(model, g, baseline=True, with_sigma=False)["rms"] for g in geoms]
        print(f"{name:>9} " + " ".join(f"{e:10.3e}" for e in out[name]))
    model.base = base
    return out
