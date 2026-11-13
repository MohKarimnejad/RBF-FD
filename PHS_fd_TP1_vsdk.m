function out = PHS_fd_TP1_vsdk(m, deg, nloc, ep, nu, do_sigma, ntype)
%PHS_FD_TP1_VSDK  GMLS finite differences with a variably scaled discontinuous
%  (Matern) weight and stencils that cross the interface, for
%  Test Problem 1 of Davydov & Safarpoor (2021):
%
%    -div(alpha grad u) = f  in Omega = (-1,1)^2,   u = g on dOmega,
%    [u] = phi,  [alpha du/dn] = psi  on Gamma: y = 2x^3,
%    alpha = 20 in Omega+ (y > 2x^3),  alpha = 1 in Omega- (y < 2x^3).
%
%  m        : node level, h = 0.4/2^m (nodes from nodes_TP1)
%  deg      : polynomial degree of the GMLS fit (exact on P_deg)
%  nloc     : stencil size = number of nearest nodes of BOTH sides (default 3*Q)
%  ep       : shape parameter of the Matern weight
%  nu       : Matern smoothness, 0.5, 1.5 or 2.5
%  do_sigma : compute the stability constant sigma (default true; extra LU)
%  ntype    : node type, 'uniform' (default, grid as in the paper) or 'halton'
%
%  Unknowns and equations: as in PHS_fd_TP1 (Davydov & Safarpoor):
%  u+ at X+ = [Xint+; Xbd+; XG] and u- at X- = [Xint-; Xbd-; XG],
%    -alpha+ Lap u+ = f   at Xint+
%    -alpha- Lap u- = f   at Xint-
%     u = g               at Xbd+, Xbd-
%     u+ - u- = phi       at XG
%     alpha+ dn u+ - alpha- dn u- = psi at XG
%
%  Stencils: for a row approximating u^s (s = + or -) at x, the nloc nearest nodes of
%  Xint+ u Xbd+ u Xint- u Xbd- u XG (they cross Gamma). A node of Omega+ enters with
%  its unknown u+, a node of Omega- with u-, an interface node with u^s.
%
%  Weights: GMLS, al = Th P (P' Th P)^-1 L p(x), exact on P_deg, with
%    theta_j = Matern_nu(ep * d_psi(x, x_j)),
%    d_psi^2 = ||x_hat - y_hat_j||^2 + (psi(x) - psi(x_j))^2,
%  in scaled coordinates y_hat = (y - x)/r (r = stencil radius), scale function
%    psi = 1 on Omega+, psi = 0 on Omega-,
%  and, for the row point and the interface nodes, psi = psi^s (1 for u+, 0 for u-):
%  nodes of the side being approximated get the larger weights.

Q = (deg + 1)*(deg + 2)/2;
if nargin < 3 || isempty(nloc), nloc = 3*Q; end
if nargin < 6, do_sigma = true; end
if nargin < 7 || isempty(ntype), ntype = 'uniform'; end
if ~ismember(nu, [0.5 1.5 2.5]), error('nu must be 0.5, 1.5 or 2.5.'); end

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

%% All distinct nodes (each interface node once), their side and their columns
%  side: +1 (Omega+), -1 (Omega-), 0 (Gamma)
%  colp / colm: column of the unknown used in a stencil for u+ / u-
Xa   = [Xp(1:nIp+nBp,:); Xm(1:nIm+nBm,:); nd.XG];
side = [ones(nIp+nBp,1); -ones(nIm+nBm,1); zeros(nG,1)];
cpm  = [(1:nIp+nBp)'; Np + (1:nIm+nBm)'];
colp = [cpm; iGp(:)];
colm = [cpm; Np + iGm(:)];

%% Polynomial basis and weight
ea = []; eb = [];                                 % monomials x^a y^b, a+b <= deg
for tt = 0:deg, for bb = 0:tt, ea(end+1) = tt-bb; eb(end+1) = bb; end, end %#ok<AGROW>
K.ea = ea; K.eb = eb; K.Q = Q; K.ep = ep; K.nu = nu;

tic
ns = min(nloc, size(Xa,1));                       % stencil size

%% Assemble the global system  A u = b
I = {}; J = {}; V = {}; b = zeros(Ntot, 1); row = 0;

% PDE rows, side +
[Ii, Jj, Vv] = vsdk_rows(Xa, side, colp, Xp(1:nIp,:), 1, 'lap', [], K, ns);
I{end+1} = row + Ii; J{end+1} = Jj; V{end+1} = -ap*Vv;
b(row + (1:nIp)) = f(Xp(1:nIp,1), Xp(1:nIp,2)); row = row + nIp;
% PDE rows, side -
[Ii, Jj, Vv] = vsdk_rows(Xa, side, colm, Xm(1:nIm,:), 0, 'lap', [], K, ns);
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
xg = nd.XG;
I{end+1} = row + [(1:nG)'; (1:nG)']; J{end+1} = [iGp(:); Np + iGm(:)];
V{end+1} = [ones(nG,1); -ones(nG,1)];
b(row + (1:nG)) = up(xg(:,1), xg(:,2)) - um(xg(:,1), xg(:,2)); row = row + nG;
% Flux jump [alpha du/dn] = alpha+ dn u+ - alpha- dn u- = psi on Gamma
nG_ = nrm(xg(:,1));
[Ii, Jj, Vv] = vsdk_rows(Xa, side, colp, xg, 1, 'dn', nG_, K, ns);
I{end+1} = row + Ii; J{end+1} = Jj; V{end+1} =  ap*Vv;
[Ii, Jj, Vv] = vsdk_rows(Xa, side, colm, xg, 0, 'dn', nG_, K, ns);
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
out.h = h; out.Nint = nd.Nint; out.Q = Q; out.nloc = nloc;
out.nlocal = ns;                                              % stencil size
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
function [Ii, Jj, Vv] = vsdk_rows(Xa, side, col, Xrows, psis, type, nvec, K, n)
% GMLS rows approximating u^s at the row points Xrows (psis = psi^s: 1 for u+, 0 for u-).
% Stencil = the n nearest nodes of Xa (both sides), scaled coordinates Yh = (y - x)/r
% (r = stencil radius), so x_hat = 0. Weight theta_j = Matern(ep * d_psi), with
%   d_psi^2 = ||Yh_j||^2 + (psis - psi_j)^2,
%   psi_j = 1 (Omega+), 0 (Omega-), psis (interface node).
% Rows: w = al / r^2 (Laplacian) or al / r (normal derivative); columns col(J).
ea = K.ea; eb = K.eb; Q = K.Q;
Lp0 = zeros(Q,1); Lp0(ea==2 & eb==0) = 2; Lp0(ea==0 & eb==2) = 2;   % Lap p at 0
psia = double(side > 0);  psia(side == 0) = psis;                     % psi at all nodes
idx = knnsearch(Xa, Xrows, 'K', n);              % n nearest nodes (incl. x itself)
Ic = cell(size(Xrows,1),1); Jc = Ic; Vc = Ic;
for k = 1:size(Xrows,1)
    x  = Xrows(k,:);
    Js = idx(k,:)';
    Yh = Xa(Js,:) - x;
    r  = sqrt(max(Yh(:,1).^2 + Yh(:,2).^2));
    Yh = Yh / r;
    P  = (Yh(:,1).^ea) .* (Yh(:,2).^eb);
    if strcmp(type, 'lap')
        rp = Lp0;  sc = r^2;
    else
        nn = nvec(k,:);
        rp = zeros(Q,1); rp(ea==1 & eb==0) = nn(1); rp(ea==0 & eb==1) = nn(2);
        sc = r;
    end
    dpsi = sqrt(Yh(:,1).^2 + Yh(:,2).^2 + (psis - psia(Js)).^2);
    th   = matern(K.ep * dpsi, K.nu);
    sq   = sqrt(th);
    [Qm, Rm] = qr(sq .* P, 0);
    al   = sq .* (Qm * (Rm' \ rp));
    Ic{k} = k*ones(n,1); Jc{k} = col(Js); Vc{k} = al / sc;
end
Ii = vertcat(Ic{:}); Jj = vertcat(Jc{:}); Vv = vertcat(Vc{:});
end

function th = matern(t, nu)
% Matern radial function of order nu (nu = 0.5, 1.5, 2.5), argument t = ep * d.
switch nu
    case 0.5, th = exp(-t);
    case 1.5, th = (1 + t) .* exp(-t);
    case 2.5, th = (1 + t + t.^2/3) .* exp(-t);
end
end
