"""Node set X^(m) for Test Problem 1 of Davydov & Safarpoor (2021).
Python translation of nodes_TP1.m (same construction, same output fields)."""

import numpy as np
import contourpy


def haltonseq(npts, ndims):
    """Halton points 1..npts in [0,1]^ndims (as haltonseq.m, bases 2, 3, 5, ...)."""
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    H = np.zeros((npts, ndims))
    for d in range(ndims):
        b = primes[d]
        for k in range(1, npts + 1):
            f, r, i = 1.0, 0.0, k
            while i > 0:
                f /= b
                r += f * (i % b)
                i //= b
            H[k - 1, d] = r
    return H


def nodes_TP1(m, ntype="uniform"):
    """Omega = (-1,1)^2, Gamma: y = 2x^3, Omega+ : y > 2x^3, Omega- : y < 2x^3.
    Returns a dict with Xint_p, Xint_m, Xbd_p, Xbd_m, XG, h, Nint, type."""
    a = 0.25
    h = 0.4 / 2**m
    n1 = round(2 / h) + 1
    t = np.linspace(-1, 1, n1)
    GX, GY = np.meshgrid(t, t)
    G = np.column_stack([GX.ravel(order="F"), GY.ravel(order="F")])

    phi = lambda x, y: y - 2 * x**3
    gam = lambda x, y: (y - 2 * x**3) / np.sqrt(1 + 36 * x**4)
    tend = 2 ** (-1 / 3)

    # Interface nodes: contouring + secant projection
    lines = contourpy.contour_generator(x=t, y=t, z=gam(GX, GY)).lines(0.0)
    V = np.unique(np.vstack(lines), axis=0)
    onbox = (np.abs(np.abs(V[:, 0]) - 1) < 1e-12) | (np.abs(np.abs(V[:, 1]) - 1) < 1e-12)
    V = V[~onbox]
    V = np.array([secant_project(p, gam, h) for p in V])
    V = V[np.abs(V[:, 1]) < 1]
    XG = np.vstack([[-tend, -1.0], V, [tend, 1.0]])
    XG = thin_points(XG, 0.5 * h, [0, len(XG) - 1])

    # Grid points: boundary nodes, interior nodes
    onB = (np.abs(np.abs(G[:, 0]) - 1) < 1e-12) | (np.abs(np.abs(G[:, 1]) - 1) < 1e-12)
    keep = np.abs(gam(G[:, 0], G[:, 1])) >= a * h
    Xbd = G[onB & keep]
    if ntype == "uniform":
        Xint = G[~onB & keep]
    elif ntype == "halton":
        H = 2 * haltonseq((n1 - 1) ** 2, 2) - 1
        dB = 1 - np.abs(H).max(1)
        Xint = H[(dB >= a * h) & (np.abs(gam(H[:, 0], H[:, 1])) >= a * h)]
    else:
        raise ValueError("ntype must be 'uniform' or 'halton'")

    sp = phi(Xint[:, 0], Xint[:, 1]) > 0
    sb = phi(Xbd[:, 0], Xbd[:, 1]) > 0
    return dict(Xint_p=Xint[sp], Xint_m=Xint[~sp], Xbd_p=Xbd[sb], Xbd_m=Xbd[~sb],
                XG=XG, h=h, Nint=len(Xint), type=ntype)


def secant_project(p, gam, h):
    """Move p onto {gam = 0} along the direction of grad(gam) at p (secant method)."""
    x, y = p
    s = np.sqrt(1 + 36 * x**4)
    g = y - 2 * x**3
    gr = np.array([(-6 * x**2 * s - g * 36 * x**3 / s) / s**2, 1 / s])
    d = gr / np.linalg.norm(gr)
    f = lambda s: gam(p[0] + s * d[0], p[1] + s * d[1])
    s0, s1 = 0.0, h**2
    f0, f1 = f(s0), f(s1)
    for _ in range(50):
        if f1 == f0:
            break
        s2 = s1 - f1 * (s1 - s0) / (f1 - f0)
        s0, f0, s1 = s1, f1, s2
        f1 = f(s1)
        if abs(f1) < 1e-15:
            break
    return p + s1 * d


def thin_points(X, dmin, fixed):
    """Remove points until all pairwise distances are >= dmin. At each step the closest
    pair is found and one of its points is removed (never a fixed point; otherwise the one
    closer to its other neighbours)."""
    X = X.copy()
    isfix = np.zeros(len(X), bool)
    isfix[fixed] = True
    while True:
        D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(2))
        np.fill_diagonal(D, np.inf)
        idx = np.argmin(D.ravel(order="F"))
        if D.ravel(order="F")[idx] >= dmin:
            break
        i, j = np.unravel_index(idx, D.shape, order="F")
        if isfix[i]:
            r = j
        elif isfix[j]:
            r = i
        else:
            r = i if np.sort(D[i])[1] <= np.sort(D[j])[1] else j
        X = np.delete(X, r, 0)
        isfix = np.delete(isfix, r)
    return X[np.argsort(X[:, 0], kind="stable")]
