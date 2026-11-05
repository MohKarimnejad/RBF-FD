function out = PHS_drbfpu_TP2(m, beta, deg, nloc, do_sigma, ntype, basis)
%DRBFPU_TP2  D-RBF-PU (square collocation, PHS + polynomials) for
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
%  nloc     : target number of nodes per patch (default 3*Q)
%  do_sigma : compute the stability constant sigma (default true; extra LU)
%  ntype    : node type, 'uniform' (default, grid as in the paper) or 'halton'
%  basis    : local approximation, 'phs' (default, PHS + polynomials),
%             'ls' (polynomials only, unweighted least squares) or
%             'wls' (polynomials only, weighted least squares = GMLS,
%             Wendland weight centred at the row point, support 1.5 x max distance)
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
if nargin < 7 || isempty(basis), basis = 'phs'; end
basis = lower(basis);
if ~ismember(basis, {'phs','ls','wls'}), error('basis must be ''phs'', ''ls'' or ''wls''.'); end
if strcmp(basis, 'phs')
    if mod(beta, 2) ~= 1, error('beta must be odd.'); end
    if deg < ceil(beta/2) - 1, error('Need deg >= ceil(beta/2)-1.'); end
end
nmin = 2*Q;

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
K.beta = beta; K.ea = ea; K.eb = eb; K.Q = Q; K.basis = basis;

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
nG_ = nrm(xg(:,1), xg(:,2));
[Ii, Jj, Vv] = pu_rows(Sp, Xp, iGp, 'dn', nG_, K);
I{end+1} = row + Ii; J{end+1} = Jj;      V{end+1} =  ap*Vv;
[Ii, Jj, Vv] = pu_rows(Sm, Xm, iGm, 'dn', nG_, K);
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
    P  = (Yh(:,1).^K.ea) .* (Yh(:,2).^K.eb);
    if strcmp(K.basis, 'phs')
        A  = sqrt((Yh(:,1) - Yh(:,1)').^2 + (Yh(:,2) - Yh(:,2)').^2).^K.beta;
        [Lf, Uf, pf] = lu([A, P; P', zeros(K.Q)], 'vector');
    else
        Lf = []; Uf = []; pf = [];                        % polynomial-only: solved per row
    end
    S.J{l} = J; S.R(l) = r; S.F{l} = {Lf, Uf, pf, Yh, P};
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
        [Lf, Uf, pf, Yh, P] = S.F{l}{:};
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
        if strcmp(K.basis, 'phs')
            rhs = [rk; rp(:)];
            sol = Uf \ (Lf \ rhs(pf));
            al  = sol(1:numel(J)) / sc;
        else
            al  = poly_weights(P, rp, rr, K.basis) / sc;
        end
        Ii = [Ii; k*ones(numel(J),1)]; Jj = [Jj; J]; Vv = [Vv; w(q)*al]; %#ok<AGROW>
    end
    Ic{k} = Ii; Jc{k} = Jj; Vc{k} = Vv;
end
Ii = vertcat(Ic{:}); Jj = vertcat(Jc{:}); Vv = vertcat(Vc{:});
end

function al = poly_weights(P, rp, d, basis)
% Polynomial-only weights (n > Q): exact on the polynomials, P' al = rp.
%   'ls'  : al = P (P'P)^-1 rp            (unweighted least squares, minimal 2-norm)
%   'wls' : al = Th P (P' Th P)^-1 rp      (weighted least squares = GMLS),
%           Th = diag(theta_j), theta_j = Wendland C2 of d_j / (1.5 max d),
%           d_j = distance from the row point x to node j (weights centred at x).
if strcmp(basis, 'ls')
    th = ones(size(d));
else
    rho = d / (1.5*max(d));
    th  = max(1 - rho, 0).^4 .* (4*rho + 1);
end
s = sqrt(th);
[Qm, Rm] = qr(s .* P, 0);
al = s .* (Qm * (Rm' \ rp(:)));
end
