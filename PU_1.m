function out = PU_1(m, deg, nloc, ep, nu, method, do_sigma, ntype)
%PU_1  Direct PU with local GMLS fits for Test Problem 1 of Davydov & Safarpoor (2021):
%
%    -div(alpha grad u) = f  in Omega = (-1,1)^2,   u = g on dOmega,
%    [u] = phi,  [alpha du/dn] = psi  on Gamma: y = 2x^3,
%    alpha = 20 in Omega+ (y > 2x^3),  alpha = 1 in Omega- (y < 2x^3).
%
%  m        : node level, h = 0.4/2^m (nodes from nodes_TP1)
%  deg      : polynomial degree of the GMLS fit (exact on P_deg)
%  nloc     : target number of nodes per patch (default 3*Q)
%  ep       : shape parameter of the Matern weight
%  nu       : Matern order, 0.5, 1.5 or 2.5
%  method   : 'mls'      : patches with the nodes of the own side only (do not cross Gamma),
%                          weight theta_j = Matern(ep * ||x_hat - y_hat_j||)
%             'mls-vsdk' : patches with the nodes of both sides (cross Gamma),
%                          weight theta_j = Matern(ep * d_psi(x, y_j)),
%                          d_psi^2 = ||x_hat - y_hat_j||^2 + (psi(x) - psi(y_j))^2
%  do_sigma : compute the stability constant sigma (default true; extra LU)
%  ntype    : node type, 'uniform' (default, grid as in the paper) or 'halton'
%
%  Unknowns (as in the paper): u+ at X+ = [Xint+; Xbd+; XG], u- at X- = [Xint-; Xbd-; XG].
%  Equations:
%    -alpha+ Lap u+ = f   at Xint+
%    -alpha- Lap u- = f   at Xint-
%     u = g               at Xbd+, Xbd-
%     u+ - u- = phi       at XG
%     alpha+ dn u+ - alpha- dn u- = psi at XG
%  A row approximating u^s (s = + or -) at x:
%    row = sum_l w_l(x) * al^l,  w_l = Shepard weights (Wendland C2 of ||x - c_l|| / R_l),
%    al^l = GMLS weights at x on the nodes of patch l: Th P (P' Th P)^-1 L p(x),
%  with the weight theta_j centred at x, in patch-scaled coordinates y_hat = (y - c_l)/R_l.
%  A node of Omega+ enters with u+, a node of Omega- with u-, an interface node with u^s.
%  Scale function (mls-vsdk): psi = 1 on Omega+, 0 on Omega-; psi = psi^s (1 for u+,
%  0 for u-) at the row point and at the interface nodes.
%  Patches: centres on a uniform grid, a separate set for u+ and for u- (centres whose disc
%  contains a row point of that side); a patch with fewer than nmin = 2Q nodes is enlarged
%  (radius x 1.25) until it has nmin.

Q = (deg + 1)*(deg + 2)/2;
if nargin < 3 || isempty(nloc), nloc = 3*Q; end
if nargin < 6 || isempty(method), method = 'mls'; end
if nargin < 7, do_sigma = true; end
if nargin < 8 || isempty(ntype), ntype = 'uniform'; end
method = lower(method);
if ~ismember(method, {'mls','mls-vsdk'}), error('method must be ''mls'' or ''mls-vsdk''.'); end
if ~ismember(nu, [0.5 1.5 2.5]), error('nu must be 0.5, 1.5 or 2.5.'); end
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

%% All distinct nodes (each interface node once): side, columns, candidate patch nodes
%  side: +1 (Omega+), -1 (Omega-), 0 (Gamma)
%  col{1} / col{2}: column of the unknown used in a row for u+ / u-
Xa   = [Xp(1:nIp+nBp,:); Xm(1:nIm+nBm,:); nd.XG];
side = [ones(nIp+nBp,1); -ones(nIm+nBm,1); zeros(nG,1)];
cpm  = [(1:nIp+nBp)'; Np + (1:nIm+nBm)'];
col  = {[cpm; iGp(:)], [cpm; Np + iGm(:)]};
if strcmp(method, 'mls')
    cand = {find(side >= 0), find(side <= 0)};          % own side + interface
else
    cand = {(1:size(Xa,1))', (1:size(Xa,1))'};          % both sides
end
psis = [1, 0];                                          % psi^s for u+ / u-
xg   = nd.XG;  nG_ = nrm(xg(:,1));
Xev  = {[Xp(1:nIp,:); xg], [Xm(1:nIm,:); xg]};          % row points of u+ / u-

%% Polynomial basis and weight
ea = []; eb = [];                                 % monomials x^a y^b, a+b <= deg
for tt = 0:deg, for bb = 0:tt, ea(end+1) = tt-bb; eb(end+1) = bb; end, end %#ok<AGROW>
K.ea = ea; K.eb = eb; K.Q = Q; K.ep = ep; K.nu = nu; K.vsdk = strcmp(method, 'mls-vsdk');

%% Patches (base radius from nloc: pi R^2 / h^2 = nloc; centre spacing = R)
R0  = h * sqrt(nloc/pi);
npc = ceil(2/R0) + 1;  R0 = 2/(npc - 1);
[C1, C2] = meshgrid(linspace(-1, 1, npc)); PC = [C1(:), C2(:)];

tic
Sp = make_patches(Xa, cand{1}, Xev{1}, PC, R0, nmin, K);
Sm = make_patches(Xa, cand{2}, Xev{2}, PC, R0, nmin, K);

%% Assemble the global system  A u = b
I = {}; J = {}; V = {}; b = zeros(Ntot, 1); row = 0;

% PDE rows, side +
[Ii, Jj, Vv] = pu_rows(Sp, Xa, side, col{1}, Xp(1:nIp,:), psis(1), 'lap', [], K);
I{end+1} = row + Ii; J{end+1} = Jj; V{end+1} = -ap*Vv;
b(row + (1:nIp)) = f(Xp(1:nIp,1), Xp(1:nIp,2)); row = row + nIp;
% PDE rows, side -
[Ii, Jj, Vv] = pu_rows(Sm, Xa, side, col{2}, Xm(1:nIm,:), psis(2), 'lap', [], K);
I{end+1} = row + Ii; J{end+1} = Jj; V{end+1} = -am*Vv;
b(row + (1:nIm)) = f(Xm(1:nIm,1), Xm(1:nIm,2)); row = row + nIm;
% Dirichlet rows on dOmega
k = nIp + (1:nBp)';
I{end+1} = row + (1:nBp)'; J{end+1} = k; V{end+1} = ones(nBp,1);
b(row + (1:nBp)) = up(Xp(k,1), Xp(k,2)); row = row + nBp;
k = nIm + (1:nBm)';
I{end+1} = row + (1:nBm)'; J{end+1} = Np + k; V{end+1} = ones(nBm,1);
b(row + (1:nBm)) = um(Xm(k,1), Xm(k,2)); row = row + nBm;
% Jump [u] = u+ - u- = phi on Gamma
I{end+1} = row + [(1:nG)'; (1:nG)']; J{end+1} = [iGp(:); Np + iGm(:)];
V{end+1} = [ones(nG,1); -ones(nG,1)];
b(row + (1:nG)) = up(xg(:,1), xg(:,2)) - um(xg(:,1), xg(:,2)); row = row + nG;
% Flux jump [alpha du/dn] = alpha+ dn u+ - alpha- dn u- = psi on Gamma
[Ii, Jj, Vv] = pu_rows(Sp, Xa, side, col{1}, xg, psis(1), 'dn', nG_, K);
I{end+1} = row + Ii; J{end+1} = Jj; V{end+1} =  ap*Vv;
[Ii, Jj, Vv] = pu_rows(Sm, Xa, side, col{2}, xg, psis(2), 'dn', nG_, K);
I{end+1} = row + Ii; J{end+1} = Jj; V{end+1} = -am*Vv;
gq = gradq(xg(:,1), xg(:,2));
b(row + (1:nG)) = ap*sum((gq/20) .* nG_, 2) - am*sum(gq .* nG_, 2);   % psi (exact, = 0)
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
function S = make_patches(Xa, cand, Xev, PC, R0, nmin, K)
% Patches for one side (u+ or u-): keep the centres whose disc (radius R0) contains a
% row point Xev of that side; local nodes = the candidate nodes Xa(cand,:) in the disc;
% enlarge the radius until the patch has at least nmin nodes.
Xc = Xa(cand,:);
keep = false(size(PC,1),1);
for l = 1:size(PC,1)
    keep(l) = any((Xev(:,1) - PC(l,1)).^2 + (Xev(:,2) - PC(l,2)).^2 < R0^2);
end
S.C = PC(keep, :);
nP = size(S.C,1); S.J = cell(nP,1); S.R = zeros(nP,1);
for l = 1:nP
    r  = R0;
    d2 = (Xc(:,1) - S.C(l,1)).^2 + (Xc(:,2) - S.C(l,2)).^2;
    while nnz(d2 < r^2) < nmin && nnz(d2 < r^2) < size(Xc,1)
        r = 1.25*r;                                       % stops once all candidates are in
    end
    Jl = cand(d2 < r^2);
    if numel(Jl) <= K.Q
        error('Patch with %d nodes; need more than Q = %d (level too coarse for this degree).', numel(Jl), K.Q);
    end
    S.J{l} = Jl; S.R(l) = r;
end
end

function [Ii, Jj, Vv] = pu_rows(S, Xa, side, col, Xrows, psis, type, nvec, K)
% Direct PU rows approximating u^s at the row points Xrows (psis = psi^s):
%   row_k = sum_l w_l(x) * al^l,  w_l = Shepard weights (Wendland C2 of ||x - c_l|| / R_l),
%   al^l  = GMLS weights at x on the nodes of patch l (patch-scaled coordinates),
%           divided by R_l^2 (Laplacian) or R_l (normal derivative); columns col(J).
% GMLS weight theta_j = Matern(ep * d_j), centred at x:
%   mls      : d_j = ||x_hat - y_hat_j||
%   mls-vsdk : d_j^2 = ||x_hat - y_hat_j||^2 + (psis - psi_j)^2,
%              psi_j = 1 (Omega+), 0 (Omega-), psis (interface node).
ea = K.ea; eb = K.eb;
wend = @(r) max(1 - r, 0).^4 .* (4*r + 1);
psia = double(side > 0);  psia(side == 0) = psis;                     % psi at all nodes
Ic = cell(size(Xrows,1),1); Jc = Ic; Vc = Ic;
for k = 1:size(Xrows,1)
    x  = Xrows(k,:);
    d  = sqrt((S.C(:,1) - x(1)).^2 + (S.C(:,2) - x(2)).^2);
    pl = find(d < S.R);
    w  = wend(d(pl) ./ S.R(pl)); w = w / sum(w);           % Shepard weights
    Ii = []; Jj = []; Vv = [];
    for q = 1:numel(pl)
        l  = pl(q); Jl = S.J{l}; r = S.R(l);
        Yh = (Xa(Jl,:) - S.C(l,:)) / r;                     % patch-scaled coordinates
        xh = (x - S.C(l,:)) / r;
        P  = (Yh(:,1).^ea) .* (Yh(:,2).^eb);
        if strcmp(type, 'lap')
            rp = (ea.*(ea-1)) .* xh(1).^max(ea-2,0) .* xh(2).^eb + ...
                 (eb.*(eb-1)) .* xh(1).^ea .* xh(2).^max(eb-2,0);
            sc = r^2;
        else
            n  = nvec(k,:);
            rp = n(1) * ea .* xh(1).^max(ea-1,0) .* xh(2).^eb + ...
                 n(2) * eb .* xh(1).^ea .* xh(2).^max(eb-1,0);
            sc = r;
        end
        d2 = (xh(1) - Yh(:,1)).^2 + (xh(2) - Yh(:,2)).^2;
        if K.vsdk, d2 = d2 + (psis - psia(Jl)).^2; end
        al = gmls_weights(P, rp, matern(K.ep * sqrt(d2), K.nu)) / sc;
        Ii = [Ii; k*ones(numel(Jl),1)]; Jj = [Jj; col(Jl)]; Vv = [Vv; w(q)*al]; %#ok<AGROW>
    end
    Ic{k} = Ii; Jc{k} = Jj; Vc{k} = Vv;
end
Ii = vertcat(Ic{:}); Jj = vertcat(Jc{:}); Vv = vertcat(Vc{:});
end

function al = gmls_weights(P, rp, th)
% GMLS weights: exact on the polynomials, P' al = rp, minimal weighted 2-norm:
%   al = Th P (P' Th P)^-1 rp,  Th = diag(th). Computed by QR of sqrt(Th) P.
sq = sqrt(th);
[Qm, Rm] = qr(sq .* P, 0);
al = sq .* (Qm * (Rm' \ rp(:)));
end

function th = matern(t, nu)
% Matern radial function of order nu (nu = 0.5, 1.5, 2.5), argument t = ep * d.
switch nu
    case 0.5, th = exp(-t);
    case 1.5, th = (1 + t) .* exp(-t);
    case 2.5, th = (1 + t + t.^2/3) .* exp(-t);
end
end
