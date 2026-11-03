function out = PHS_drbfpu_TP1(m, beta, deg, nloc, do_sigma, ntype)
%DRBFPU_TP1  D-RBF-PU (square collocation, PHS + polynomials) for
%  Test Problem 1 of Davydov & Safarpoor (2021):
%
%    -div(alpha grad u) = f  in Omega = (-1,1)^2,   u = g on dOmega,
%    [u] = phi,  [alpha du/dn] = psi  on Gamma: y = 2x^3,
%    alpha = 20 in Omega+ (y > 2x^3),  alpha = 1 in Omega- (y < 2x^3).
%
%  m        : node level, h = 0.4/2^m (nodes from nodes_TP1)
%  beta     : odd PHS exponent, phi(r) = r^beta
%  deg      : polynomial degree (deg >= ceil(beta/2)-1)
%  nloc     : target number of nodes per patch (default 3*Q)
%  do_sigma : compute the stability constant sigma (default true; extra LU)
%  ntype    : node type, 'uniform' (default, grid as in the paper) or 'halton'
%
%  Unknowns: u+ at X+ = [Xint+; Xbd+; XG] and u- at X- = [Xint-; Xbd-; XG]
%  (two unknowns per interface node). Equations:
%    -alpha+ Lap u+ = f   at Xint+     (D-RBF-PU rows, patches of side +)
%    -alpha- Lap u- = f   at Xint-     (D-RBF-PU rows, patches of side -)
%     u = g               at Xbd+, Xbd-
%     u+ - u- = phi       at XG
%     alpha+ dn u+ - alpha- dn u- = psi at XG (one-sided D-RBF-PU rows)
%  Patches: a separate set of patches for each side; each patch only uses the
%  nodes of its own side (interface nodes belong to both sides). A patch with
%  fewer than nmin = 2Q nodes is enlarged (radius x 1.25) until it has nmin.

Q = (deg + 1)*(deg + 2)/2;
if nargin < 4 || isempty(nloc), nloc = 3*Q; end
if nargin < 5, do_sigma = true; end
if nargin < 6 || isempty(ntype), ntype = 'uniform'; end
if mod(beta, 2) ~= 1, error('beta must be odd.'); end
if deg < ceil(beta/2) - 1, error('Need deg >= ceil(beta/2)-1.'); end
nmin = 2*Q;

%% Exact solution and data
ap = 20; am = 1;
s     = @(x,y) y - 2*x.^3;
up    = @(x,y) (s(x,y).^2 - 30*s(x,y)) / 20;
um    = @(x,y)  s(x,y).^2 - 30*s(x,y);
lapq  = @(x,y) 2*(36*x.^4 + 1) - 12*x.*(2*s(x,y) - 30);   % Lap(s^2 - 30 s)
f     = @(x,y) -lapq(x,y);                                  % f = -alpha Lap u (both sides)
gradq = @(x,y) [(2*s(x,y) - 30).*(-6*x.^2), (2*s(x,y) - 30)];  % grad(s^2 - 30 s)
nrm   = @(x) [-6*x.^2, ones(size(x))] ./ sqrt(1 + 36*x.^4);    % unit normal, points into Omega+

%% Nodes
nd = nodes_TP1(m, ntype);
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
K.beta = beta; K.ea = ea; K.eb = eb; K.Q = Q;

%% Patches (base radius from nloc: pi R^2 / h^2 = nloc; centre spacing = R)
R0  = h * sqrt(nloc/pi);
npc = ceil(2/R0) + 1;  R0 = 2/(npc - 1);
[C1, C2] = meshgrid(linspace(-1, 1, npc)); PC = [C1(:), C2(:)];

tic
Sp = side_patches(Xp, [1:nIp, iGp], PC, R0, nmin, K);
Sm = side_patches(Xm, [1:nIm, iGm], PC, R0, nmin, K);

%% Assemble the global system  A u = b
I = {}; J = {}; V = {}; b = zeros(Ntot, 1); row = 0;

% PDE rows, side +
[Ii, Jj, Vv] = pu_rows(Sp, Xp, 1:nIp, 'lap', [], K);
I{end+1} = row + Ii; J{end+1} = Jj; V{end+1} = -ap*Vv;
b(row + (1:nIp)) = f(Xp(1:nIp,1), Xp(1:nIp,2)); row = row + nIp;
% PDE rows, side -
[Ii, Jj, Vv] = pu_rows(Sm, Xm, 1:nIm, 'lap', [], K);
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
nG_ = nrm(xg(:,1));
[Ii, Jj, Vv] = pu_rows(Sp, Xp, iGp, 'dn', nG_, K);
I{end+1} = row + Ii; J{end+1} = Jj;      V{end+1} =  ap*Vv;
[Ii, Jj, Vv] = pu_rows(Sm, Xm, iGm, 'dn', nG_, K);
I{end+1} = row + Ii; J{end+1} = Np + Jj; V{end+1} = -am*Vv;
gq = gradq(xg(:,1), xg(:,2));
b(row + (1:nG)) = ap*sum((gq/20) .* nG_, 2) - am*sum(gq .* nG_, 2);   % psi (exact, = 0)
row = row + nG; %#ok<NASGU>

A = sparse(vertcat(I{:}), vertcat(J{:}), vertcat(V{:}), Ntot, Ntot);
u = A \ b;
t = toc;

%% Truncation error tau = A u_exact - b (one entry per row)
%  Row order: Lap rows of Xint+, Lap rows of Xint-, Dirichlet rows (tau = 0),
%  jump rows [u] (tau = 0), flux rows [alpha du/dn] at XG.
uex = [up(Xp(:,1), Xp(:,2)); um(Xm(:,1), Xm(:,2))];
tau = A*uex - b;

%% Errors on interior nodes (as in the paper: rms and max over X_int)
ue = [up(Xp(1:nIp,1), Xp(1:nIp,2)); um(Xm(1:nIm,1), Xm(1:nIm,2))];
uh = [u(1:nIp); u(Np + (1:nIm))];
e  = abs(uh - ue);
out.h = h; out.Nint = nd.Nint; out.Q = Q; out.nloc = nloc; out.R0 = R0;
out.npatch = [numel(Sp.J), numel(Sm.J)];
out.nlocal = [cellfun(@numel, Sp.J); cellfun(@numel, Sm.J)];   % nodes per patch
out.rms = sqrt(mean(e.^2)); out.max = max(e);
if do_sigma
    out.sigma = condest(A') / norm(A, inf);              % stability constant (3.2)
else
    out.sigma = NaN;
end
out.nnz_row = nnz(A) / Ntot;
out.time = t;

%% Solution and absolute error at all nodes (for plots, as Fig. 9 of the paper)
%  Interface nodes appear once: value u+, error max(|u+ - u|, |u- - u|).
uep = up(Xp(:,1), Xp(:,2)); uem = um(Xm(:,1), Xm(:,2));
ep_ = abs(u(1:Np) - uep);   em_ = abs(u(Np+1:end) - uem);
iBp = nIp + (1:nBp); iBm = nIm + (1:nBm);
out.Xall = [Xp(1:nIp,:); Xm(1:nIm,:); Xp(iBp,:); Xm(iBm,:); xg];
out.uall = [u(1:nIp); u(Np + (1:nIm)); u(iBp); u(Np + iBm); u(iGp)];
out.eall = [ep_(1:nIp); em_(1:nIm); ep_(iBp); em_(iBm); max(ep_(iGp), em_(iGm))];

%% Normalized error alpha*|e| (u+- = q/alpha+-, so both sides have the same scale)
%  Interface nodes: max(alpha+ |e+|, alpha- |e-|).
out.enorm = [ap*ep_(1:nIp); am*em_(1:nIm); ap*ep_(iBp); am*em_(iBm); ...
             max(ap*ep_(iGp), am*em_(iGm))];

%% Truncation error map
%  out.tau_lap at out.Xtau_lap : |tau| of the Laplacian rows (Xint+ then Xint-)
%  out.tau_dn  at out.Xtau_dn  : |tau| of the flux-jump rows (interface nodes)
out.Xtau_lap = [Xp(1:nIp,:); Xm(1:nIm,:)];
out.tau_lap  = abs(tau(1:nIp+nIm));
out.Xtau_dn  = xg;
out.tau_dn   = abs(tau(end-nG+1:end));
end

% ============================================================================
function S = side_patches(X, ieval, PC, R0, nmin, K)
% Patches of one side: keep centres whose disc (radius R0) contains an
% evaluation node of this side; local nodes = all nodes of this side in the
% disc; enlarge the radius until the patch has at least nmin nodes.
% Each patch stores the LU factors of its scaled saddle-point matrix.
Xe = X(ieval, :);
keep = false(size(PC,1),1);
for l = 1:size(PC,1)
    keep(l) = any((Xe(:,1) - PC(l,1)).^2 + (Xe(:,2) - PC(l,2)).^2 < R0^2);
end
S.C = PC(keep, :);
nP = size(S.C,1); S.J = cell(nP,1); S.F = cell(nP,1); S.R = zeros(nP,1);
for l = 1:nP
    r  = R0;
    d2 = (X(:,1) - S.C(l,1)).^2 + (X(:,2) - S.C(l,2)).^2;
    while nnz(d2 < r^2) < nmin && nnz(d2 < r^2) < size(X,1)
        r = 1.25*r;                                       % stops once all side nodes are in
    end
    J  = find(d2 < r^2);
    if numel(J) <= K.Q
        error('Patch with %d nodes; need more than Q = %d (level too coarse for this degree).', numel(J), K.Q);
    end
    Yh = (X(J,:) - S.C(l,:)) / r;                         % scaled coordinates
    A  = sqrt((Yh(:,1) - Yh(:,1)').^2 + (Yh(:,2) - Yh(:,2)').^2).^K.beta;
    P  = (Yh(:,1).^K.ea) .* (Yh(:,2).^K.eb);
    [Lf, Uf, pf] = lu([A, P; P', zeros(K.Q)], 'vector');
    S.J{l} = J; S.R(l) = r; S.F{l} = {Lf, Uf, pf, Yh};
end
end

function [Ii, Jj, Vv] = pu_rows(S, X, irows, type, nvec, K)
% D-RBF-PU rows at x = X(irows(k),:), k = 1..numel(irows):
%   row_k = sum_l w_l(x) * alpha^l,  M_l [alpha_hat; gamma] = [L phi; L p] (scaled),
%   alpha^l = alpha_hat / R_l^2 (Laplacian)  or  alpha_hat / R_l (normal derivative).
% type = 'lap' (Laplacian) or 'dn' (derivative along nvec(k,:)).
b = K.beta; ea = K.ea; eb = K.eb;
wend = @(r) max(1 - r, 0).^4 .* (4*r + 1);
Ic = cell(numel(irows),1); Jc = Ic; Vc = Ic;
for k = 1:numel(irows)
    x  = X(irows(k),:);
    d  = sqrt((S.C(:,1) - x(1)).^2 + (S.C(:,2) - x(2)).^2);
    pl = find(d < S.R);
    w  = wend(d(pl) ./ S.R(pl)); w = w / sum(w);           % Shepard weights
    Ii = []; Jj = []; Vv = [];
    for q = 1:numel(pl)
        l  = pl(q); J = S.J{l}; r = S.R(l);
        [Lf, Uf, pf, Yh] = S.F{l}{:};
        xh = (x - S.C(l,:)) / r;
        dx = xh(1) - Yh(:,1); dy = xh(2) - Yh(:,2); rr = sqrt(dx.^2 + dy.^2);
        if strcmp(type, 'lap')
            rk = b^2 * rr.^(b-2);                                     % Lap r^beta (d = 2)
            rp = (ea.*(ea-1)) .* xh(1).^max(ea-2,0) .* xh(2).^eb + ...
                 (eb.*(eb-1)) .* xh(1).^ea .* xh(2).^max(eb-2,0);
            sc = r^2;
        else
            n  = nvec(k,:);
            rk = b * rr.^(b-2) .* (dx*n(1) + dy*n(2));                % dn r^beta
            rp = n(1) * ea .* xh(1).^max(ea-1,0) .* xh(2).^eb + ...
                 n(2) * eb .* xh(1).^ea .* xh(2).^max(eb-1,0);
            sc = r;
        end
        rhs = [rk; rp(:)];
        sol = Uf \ (Lf \ rhs(pf));
        al  = sol(1:numel(J)) / sc;
        Ii = [Ii; k*ones(numel(J),1)]; Jj = [Jj; J]; Vv = [Vv; w(q)*al]; %#ok<AGROW>
    end
    Ic{k} = Ii; Jc{k} = Jj; Vc{k} = Vv;
end
Ii = vertcat(Ic{:}); Jj = vertcat(Jc{:}); Vv = vertcat(Vc{:});
end