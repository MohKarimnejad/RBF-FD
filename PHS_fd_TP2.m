function out = PHS_fd_TP2(m, beta, deg, nloc, do_sigma, ntype, basis)
%PHS_FD_TP2  RBF-FD (square collocation, PHS + polynomials) for
%  Test Problem 2 of Davydov & Safarpoor (2021):
%
%    -div(alpha grad u) = f  in Omega = (-1,1)^2,   u = g on dOmega,
%    [u] = phi,  [alpha du/dn] = psi  on Gamma: circle r = 1/2,
%    alpha+ = alpha- = 1,  Omega+ : r < 1/2,  Omega- : r > 1/2,
%    u = 1 in Omega+,  u = 1 + log(2r) in Omega-  (f = 0, [u] = 0, [du/dn] = -2,
%    n = outward unit normal of the circle, pointing from Omega+ to Omega-).
%
%  m        : node level, h ~ 0.15/2^m (nodes from nodes_TP2)
%  beta     : odd PHS exponent, phi(r) = r^beta
%  deg      : polynomial degree (deg >= ceil(beta/2)-1)
%  nloc     : stencil size = number of nearest nodes of the same side (default 3*Q)
%  do_sigma : compute the stability constant sigma (default true; extra LU)
%  ntype    : node type, 'uniform' (default, grid as in the paper) or 'halton'
%  basis    : local approximation, 'phs' (default, PHS + polynomials),
%             'gmls' (polynomials only, weighted least squares = GMLS on the stencil,
%             Wendland weight centred at the row point, support 1.5 x max distance) or
%             'pqr' (polynomials only, pQR method of Davydov & Safarpoor, 2021, of order
%             deg+1, with the stencil as initial cloud)
%
%  Unknowns: u+ at X+ = [Xint+; Xbd+; XG] and u- at X- = [Xint-; Xbd-; XG]
%  (two unknowns per interface node). Equations:
%    -alpha+ Lap u+ = f   at Xint+     (RBF-FD rows, stencils of side +)
%    -alpha- Lap u- = f   at Xint-     (RBF-FD rows, stencils of side -)
%     u = g               at Xbd+, Xbd-
%     u+ - u- = phi       at XG
%     alpha+ dn u+ - alpha- dn u- = psi at XG (one-sided RBF-FD rows)
%  Stencils: for each row point x_i, the nloc nearest nodes of its own side
%  (interface nodes belong to both sides), so stencils never cross Gamma.
%  One local saddle-point system per row (no partition of unity).

Q = (deg + 1)*(deg + 2)/2;
if nargin < 4 || isempty(nloc), nloc = 3*Q; end
if nargin < 5, do_sigma = true; end
if nargin < 6 || isempty(ntype), ntype = 'uniform'; end
if nargin < 7 || isempty(basis), basis = 'phs'; end
basis = lower(basis);
if ~ismember(basis, {'phs','gmls','pqr'}), error('basis must be ''phs'', ''gmls'' or ''pqr''.'); end
if strcmp(basis, 'phs')
    if mod(beta, 2) ~= 1, error('beta must be odd.'); end
    if deg < ceil(beta/2) - 1, error('Need deg >= ceil(beta/2)-1.'); end
end

%% Exact solution and data
ap = 1; am = 1;
up    = @(x,y) ones(size(x));
um    = @(x,y) 1 + log(2*sqrt(x.^2 + y.^2));
f     = @(x,y) zeros(size(x));                              % Lap u = 0 on both sides
gradp = @(x,y) [zeros(size(x)), zeros(size(x))];
gradm = @(x,y) [x, y] ./ (x.^2 + y.^2);
nrm   = @(x,y) [x, y] ./ sqrt(x.^2 + y.^2);                 % outward normal of the circle

%% Nodes
nd = nodes_TP2(m, ntype);
h  = nd.h;
Xp = [nd.Xint_p; nd.Xbd_p; nd.XG];   Xm = [nd.Xint_m; nd.Xbd_m; nd.XG];
nIp = size(nd.Xint_p,1); nBp = size(nd.Xbd_p,1);
nIm = size(nd.Xint_m,1); nBm = size(nd.Xbd_m,1);
nG  = size(nd.XG,1);
Np  = size(Xp,1);  Nm = size(Xm,1);  Ntot = Np + Nm;
iGp = nIp + nBp + (1:nG);            % interface nodes in X+ (local)
iGm = nIm + nBm + (1:nG);            % interface nodes in X- (local)

%% PHS kernel, polynomial basis, and the functionals (in scaled coordinates)
ea = []; eb = [];                                 % monomials x^a y^b, a+b <= deg
for tt = 0:deg, for bb = 0:tt, ea(end+1) = tt-bb; eb(end+1) = bb; end, end %#ok<AGROW>
K.beta = beta; K.ea = ea; K.eb = eb; K.Q = Q; K.basis = basis; K.deg = deg;

tic
nsp = min(nloc, Np); nsm = min(nloc, Nm);       % stencil size (capped by side size)

%% Assemble the global system  A u = b
I = {}; J = {}; V = {}; b = zeros(Ntot, 1); row = 0;

% PDE rows, side +
[Ii, Jj, Vv] = fd_rows(Xp, 1:nIp, 'lap', [], K, nsp);
I{end+1} = row + Ii; J{end+1} = Jj; V{end+1} = -ap*Vv;
b(row + (1:nIp)) = f(Xp(1:nIp,1), Xp(1:nIp,2)); row = row + nIp;
% PDE rows, side -
[Ii, Jj, Vv] = fd_rows(Xm, 1:nIm, 'lap', [], K, nsm);
I{end+1} = row + Ii; J{end+1} = Np + Jj; V{end+1} = -am*Vv;
b(row + (1:nIm)) = f(Xm(1:nIm,1), Xm(1:nIm,2)); row = row + nIm;
% Dirichlet rows on dOmega
k = nIp + (1:nBp)';
I{end+1} = row + (1:nBp)'; J{end+1} = k; V{end+1} = ones(nBp,1);
b(row + (1:nBp)) = up(Xp(k,1), Xp(k,2)); row = row + nBp;
k = nIm + (1:nBm)';
I{end+1} = row + (1:nBm)'; J{end+1} = Np + k; V{end+1} = ones(nBm,1);
b(row + (1:nBm)) = um(Xm(k,1), Xm(k,2)); row = row + nBm;
% Jump [u] = u+ - u- = phi on Gamma
xg = nd.XG;
I{end+1} = row + [(1:nG)'; (1:nG)']; J{end+1} = [iGp(:); Np + iGm(:)];
V{end+1} = [ones(nG,1); -ones(nG,1)];
b(row + (1:nG)) = up(xg(:,1), xg(:,2)) - um(xg(:,1), xg(:,2)); row = row + nG;
% Flux jump [alpha du/dn] = alpha+ dn u+ - alpha- dn u- = psi on Gamma
nG_ = nrm(xg(:,1), xg(:,2));
[Ii, Jj, Vv] = fd_rows(Xp, iGp, 'dn', nG_, K, nsp);
I{end+1} = row + Ii; J{end+1} = Jj;      V{end+1} =  ap*Vv;
[Ii, Jj, Vv] = fd_rows(Xm, iGm, 'dn', nG_, K, nsm);
I{end+1} = row + Ii; J{end+1} = Np + Jj; V{end+1} = -am*Vv;
b(row + (1:nG)) = ap*sum(gradp(xg(:,1), xg(:,2)) .* nG_, 2) ...
                - am*sum(gradm(xg(:,1), xg(:,2)) .* nG_, 2);       % psi (exact, = -2)
row = row + nG; %#ok<NASGU>

A = sparse(vertcat(I{:}), vertcat(J{:}), vertcat(V{:}), Ntot, Ntot);
u = A \ b;
t = toc;

%% Errors on interior nodes (as in the paper: rms and max over X_int)
ue = [up(Xp(1:nIp,1), Xp(1:nIp,2)); um(Xm(1:nIm,1), Xm(1:nIm,2))];
uh = [u(1:nIp); u(Np + (1:nIm))];
e  = abs(uh - ue);
out.h = h; out.Nint = nd.Nint; out.Q = Q; out.nloc = nloc;
out.nlocal = [nsp; nsm];                                      % stencil size per side
out.rms = sqrt(mean(e.^2)); out.max = max(e);
if do_sigma
    out.sigma = condest(A') / norm(A, inf);              % stability constant (3.2)
else
    out.sigma = NaN;
end
out.nnz_row = nnz(A) / Ntot;
out.time = t;

%% Growth function of the Laplacian rows (Davydov-Schaback), exponent q = deg:
%  rho_i = sum_j |w_ij| * ||x_j - x_i||^deg,  w = Laplacian weights of row i
%  (the global row divided by -alpha), one value per interior node.
nL = nIp + nIm;
[ri, ci, vi] = find(A(1:nL, :));
Xu  = [Xp; Xm];
Xr  = [Xp(1:nIp,:); Xm(1:nIm,:)];
alr = [ap*ones(nIp,1); am*ones(nIm,1)];
dij = sqrt(sum((Xu(ci,:) - Xr(ri,:)).^2, 2));
out.growth  = accumarray(ri, abs(vi) ./ alr(ri) .* dij.^deg, [nL, 1]);
out.Xgrowth = Xr;

%% Solution and absolute error at all nodes (for plots, as Fig. 9 of the paper)
%  Interface nodes appear once: value u+, error max(|u+ - u|, |u- - u|).
uep = up(Xp(:,1), Xp(:,2)); uem = um(Xm(:,1), Xm(:,2));
ep_ = abs(u(1:Np) - uep);   em_ = abs(u(Np+1:end) - uem);
iBp = nIp + (1:nBp); iBm = nIm + (1:nBm);
out.Xall = [Xp(1:nIp,:); Xm(1:nIm,:); Xp(iBp,:); Xm(iBm,:); xg];
out.uall = [u(1:nIp); u(Np + (1:nIm)); u(iBp); u(Np + iBm); u(iGp)];
out.eall = [ep_(1:nIp); em_(1:nIm); ep_(iBp); em_(iBm); max(ep_(iGp), em_(iGm))];

%% Relative error for plots: |e| divided by max|u| of its own side,
%  e_rel = |e+|/max|u+| on X+,  |e-|/max|u-| on X-;  interface nodes: max of the two.
sp = max(abs(uep));  sm = max(abs(uem));
out.erel = [ep_(1:nIp)/sp; em_(1:nIm)/sm; ep_(iBp)/sp; em_(iBm)/sm; ...
            max(ep_(iGp)/sp, em_(iGm)/sm)];
end

% ============================================================================
function [Ii, Jj, Vv] = fd_rows(X, irows, type, nvec, K, n)
% RBF-FD rows at x = X(irows(k),:): stencil = the n nearest nodes of X (same
% side, so stencils never cross Gamma), scaled coordinates Yh = (y - x)/r
% (r = stencil radius), so x_hat = 0. One local saddle-point system per row:
%   [A P; P' 0] [w_hat; gamma] = [L phi(||0 - Yh_j||); L p(0)],
%   w = w_hat / r^2 (Laplacian)  or  w_hat / r (normal derivative).
b = K.beta; ea = K.ea; eb = K.eb; Q = K.Q;
Lp0 = zeros(Q,1); Lp0(ea==2 & eb==0) = 2; Lp0(ea==0 & eb==2) = 2;   % Lap p at 0
idx = knnsearch(X, X(irows,:), 'K', n);                % n nearest nodes (incl. x itself)
Ic = cell(numel(irows),1); Jc = Ic; Vc = Ic;
for k = 1:numel(irows)
    x  = X(irows(k),:);
    J  = idx(k,:)';
    Yh = X(J,:) - x;
    r  = sqrt(max(Yh(:,1).^2 + Yh(:,2).^2));
    Yh = Yh / r;
    P  = (Yh(:,1).^ea) .* (Yh(:,2).^eb);
    if strcmp(K.basis, 'phs')
        A  = sqrt((Yh(:,1) - Yh(:,1)').^2 + (Yh(:,2) - Yh(:,2)').^2).^b;
    end
    rr = sqrt(Yh(:,1).^2 + Yh(:,2).^2);
    if strcmp(type, 'lap')
        rk = b^2 * rr.^(b-2);                                     % Lap_x r^beta at x_hat = 0
        rp = Lp0;  sc = r^2;
    else
        nn = nvec(k,:);
        rk = b * rr.^(b-2) .* (-Yh(:,1)*nn(1) - Yh(:,2)*nn(2));   % dn_x r^beta (x_hat - y_hat = -Yh)
        rp = zeros(Q,1); rp(ea==1 & eb==0) = nn(1); rp(ea==0 & eb==1) = nn(2);
        sc = r;
    end
    if strcmp(K.basis, 'phs')
        sol = [A, P; P', zeros(Q)] \ [rk; rp];
        al  = sol(1:n);
    else
        if strcmp(K.basis, 'gmls')
            al  = gmls_weights(P, rp, rr);
        else                                          % 'pqr' (P is already shifted/scaled to x)
            al  = pqr_weights(P, rp, rr, K.deg + 1);
        end
    end
    Ic{k} = k*ones(n,1); Jc{k} = J; Vc{k} = al / sc;
end
Ii = vertcat(Ic{:}); Jj = vertcat(Jc{:}); Vv = vertcat(Vc{:});
end

function al = gmls_weights(P, rp, d)
% GMLS weights (n > Q): exact on the polynomials, P' al = rp, minimal weighted 2-norm:
%   al = Th P (P' Th P)^-1 rp,  Th = diag(theta_j),
%   theta_j = Wendland C2 of d_j / (1.5 max d), d_j = distance from the row point x to
%   node j (weight centred at x). Computed by QR of sqrt(Th) P.
rho = d / (1.5*max(d));
th  = max(1 - rho, 0).^4 .* (4*rho + 1);
s = sqrt(th);
[Qm, Rm] = qr(s .* P, 0);
al = s .* (Qm * (Rm' \ rp(:)));
end

function al = pqr_weights(P, rp, d, q)
% pQR weights of Davydov & Safarpoor (2021), Sec. 2 (algorithm "pQR method of order q").
%   P  : basis evaluated at the stencil nodes, monomials shifted to the row point x and
%        scaled by the largest distance (column 1 = constant, all other columns vanish at x)
%   rp : the operator applied to the basis at x (same scaled coordinates)
%   d  : distances ||y_j - x|| (one of them is 0: the row point itself)
%   q  : order (exact for polynomials of degree q-1)
% The equation for the constant is eliminated; the system A w = b (A = basis 2..Q at the
% nodes y_j ~= x) is rescaled as (A Th) v = b, w = Th v, Th = diag(d_j^-q), and solved by
% QR with column pivoting (basic solution, at most rank(A) nonzero weights). The weight of
% x is fixed by exactness for the constant: w_x = rp(1) - sum_j w_j.
rp = rp(:);  n = numel(d);
i0 = find(d <= 1e-12*max(d), 1);
if isempty(i0), error('pqr_weights: the row point must belong to the stencil.'); end
ir = setdiff((1:n)', i0);
A  = P(ir, 2:end)';                               % (Q-1) x (n-1)
b  = rp(2:end);
th = d(ir).^(-q);  th = th / max(th);             % Th (scaled, scale-free)
[Qm, Rm, E] = qr(A .* th', 0);                    % (A Th)(:,E) = Qm Rm
dR = abs(diag(Rm));
r  = sum(dR > 1e-12*dR(1));                       % numerical rank
vt = Rm(1:r,1:r) \ (Qm(:,1:r)' * b);
v  = zeros(numel(ir), 1);  v(E(1:r)) = vt;
al = zeros(n, 1);
al(ir) = th .* v;
al(i0) = rp(1) - sum(al(ir));
al(abs(al) < 1e-12*max(abs(al))) = 0;          % round-off weights: node not selected
end
