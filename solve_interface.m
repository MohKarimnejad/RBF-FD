function out = solve_interface(pb, m, method, prm, do_sigma)
%SOLVE_INTERFACE  Interface problem pb (from problem_TP) on the node set X^(m), with
%  the discretization of Davydov & Safarpoor (2021): unknowns u+ at X+ = [Xint+; Xbd+; XG]
%  and u- at X- = [Xint-; Xbd-; XG] (two unknowns per interface node), equations
%    -alpha+ Lap u+ = f   at Xint+
%    -alpha- Lap u- = f   at Xint-
%     u = g               at Xbd+, Xbd-
%     u+ - u- = phi       at XG
%     alpha+ dn u+ - alpha- dn u- = psi at XG.
%  Every stencil or patch uses the nodes of its own side only (no crossing of Gamma).
%
%  method : 'pu'       D-RBF-PU, PHS r^beta + polynomials of degree deg, patches of
%                      nloc nodes (target), Shepard weights (Wendland C2), direct PU
%           'fd'       RBF-FD, PHS r^beta + polynomials, stencil = nloc nearest nodes
%           'pqr'      pQR method of Davydov & Safarpoor of order deg+1, initial cloud =
%                      nloc nearest nodes
%           'gmls'     GMLS (weighted least squares, exact on P_deg) on the nloc nearest
%                      nodes, weight theta_j = Matern_nu(ep * ||y_hat_j||)
%           (y_hat = (y - x)/r, r = stencil radius)
%  prm    : struct with fields deg, nloc, beta, ep, nu, ntype, itype
%  do_sigma : compute sigma = condest(A')/norm(A,inf) = ||A^-1||_inf (eq. (3.2))
%
%  out: h, N (all nodes, each interface node once), Nint, max (error over all unknowns:
%       interior and boundary nodes, and u+ and u- at the interface nodes), sigma,
%       nnz_row, time.

deg = prm.deg;  Q = (deg + 1)*(deg + 2)/2;
nloc = prm.nloc;
method = lower(method);
if ~ismember(method, {'pu','fd','pqr','gmls'})
    error('method must be ''pu'', ''fd'', ''pqr'' or ''gmls''.');
end
if ismember(method, {'pu','fd'})
    if mod(prm.beta, 2) ~= 1, error('beta must be odd.'); end
    if deg < ceil(prm.beta/2) - 1, error('Need deg >= ceil(beta/2)-1.'); end
end

%% Nodes
nd = pb.nodes(m, prm.ntype, prm.itype);
Xp = [nd.Xint_p; nd.Xbd_p; nd.XG];   Xm = [nd.Xint_m; nd.Xbd_m; nd.XG];
nIp = size(nd.Xint_p,1); nBp = size(nd.Xbd_p,1);
nIm = size(nd.Xint_m,1); nBm = size(nd.Xbd_m,1);
nG  = size(nd.XG,1);
Np  = size(Xp,1);  Nm = size(Xm,1);  Ntot = Np + Nm;
iGp = nIp + nBp + (1:nG);            % interface nodes in X+ (local)
iGm = nIm + nBm + (1:nG);            % interface nodes in X- (local)
xg  = nd.XG;
nG_ = pb.nrm(xg(:,1), xg(:,2));

%% Local approximation
ea = []; eb = [];                                 % monomials x^a y^b, a+b <= deg
for tt = 0:deg, for bb = 0:tt, ea(end+1) = tt-bb; eb(end+1) = bb; end, end %#ok<AGROW>
K.method = method; K.beta = prm.beta; K.ea = ea; K.eb = eb; K.Q = Q; K.deg = deg;
K.ep = prm.ep; K.nu = prm.nu;

tic
if strcmp(method, 'pu')
    R0  = nd.h * sqrt(nloc/pi);                   % patch radius: pi R^2 / h^2 = nloc
    npc = ceil(2/R0) + 1;  R0 = 2/(npc - 1);      % centre spacing = R0
    [C1, C2] = meshgrid(linspace(-1, 1, npc)); PC = [C1(:), C2(:)];
    Sp = side_patches(Xp, [1:nIp, iGp], PC, R0, 2*Q, K);
    Sm = side_patches(Xm, [1:nIm, iGm], PC, R0, 2*Q, K);
    rows = @(side, irows, type, nv) pu_rows(side, irows, type, nv, K);
    sideP = {Sp, Xp};  sideM = {Sm, Xm};
else
    nsp = min(nloc, Np);  nsm = min(nloc, Nm);
    rows = @(side, irows, type, nv) fd_rows(side, irows, type, nv, K);
    sideP = {Xp, nsp};  sideM = {Xm, nsm};
end

%% Assemble the global system  A u = b
I = {}; J = {}; V = {}; b = zeros(Ntot, 1); row = 0;
% PDE rows
[Ii, Jj, Vv] = rows(sideP, 1:nIp, 'lap', []);
I{end+1} = row + Ii; J{end+1} = Jj; V{end+1} = -pb.ap*Vv;
b(row + (1:nIp)) = pb.fp(Xp(1:nIp,1), Xp(1:nIp,2)); row = row + nIp;
[Ii, Jj, Vv] = rows(sideM, 1:nIm, 'lap', []);
I{end+1} = row + Ii; J{end+1} = Np + Jj; V{end+1} = -pb.am*Vv;
b(row + (1:nIm)) = pb.fm(Xm(1:nIm,1), Xm(1:nIm,2)); row = row + nIm;
% Dirichlet rows on dOmega
k = nIp + (1:nBp)';
I{end+1} = row + (1:nBp)'; J{end+1} = k; V{end+1} = ones(nBp,1);
b(row + (1:nBp)) = pb.up(Xp(k,1), Xp(k,2)); row = row + nBp;
k = nIm + (1:nBm)';
I{end+1} = row + (1:nBm)'; J{end+1} = Np + k; V{end+1} = ones(nBm,1);
b(row + (1:nBm)) = pb.um(Xm(k,1), Xm(k,2)); row = row + nBm;
% Jump [u] = u+ - u- = phi on Gamma
I{end+1} = row + [(1:nG)'; (1:nG)']; J{end+1} = [iGp(:); Np + iGm(:)];
V{end+1} = [ones(nG,1); -ones(nG,1)];
b(row + (1:nG)) = pb.up(xg(:,1), xg(:,2)) - pb.um(xg(:,1), xg(:,2)); row = row + nG;
% Flux jump [alpha du/dn] = alpha+ dn u+ - alpha- dn u- = psi on Gamma
[Ii, Jj, Vv] = rows(sideP, iGp, 'dn', nG_);
I{end+1} = row + Ii; J{end+1} = Jj;      V{end+1} =  pb.ap*Vv;
[Ii, Jj, Vv] = rows(sideM, iGm, 'dn', nG_);
I{end+1} = row + Ii; J{end+1} = Np + Jj; V{end+1} = -pb.am*Vv;
b(row + (1:nG)) = pb.ap*sum(pb.gradp(xg(:,1), xg(:,2)) .* nG_, 2) ...
                - pb.am*sum(pb.gradm(xg(:,1), xg(:,2)) .* nG_, 2);

A = sparse(vertcat(I{:}), vertcat(J{:}), vertcat(V{:}), Ntot, Ntot);
u = A \ b;
t = toc;

%% Error over all unknowns: u+ at X+ and u- at X- (interface nodes: both u+ and u-)
ue = [pb.up(Xp(:,1), Xp(:,2)); pb.um(Xm(:,1), Xm(:,2))];
e  = abs(u - ue);
out.h = nd.h; out.Nint = nd.Nint; out.N = nIp + nIm + nBp + nBm + nG;
out.max = max(e);
if nargin >= 5 && do_sigma
    out.sigma = condest(A') / norm(A, inf);
else
    out.sigma = NaN;
end
out.nnz_row = nnz(A) / Ntot;
out.time = t;
end

% ============================================================================
function [Ii, Jj, Vv] = fd_rows(side, irows, type, nvec, K)
% Stencil rows at x = X(irows(k),:): stencil = the n nearest nodes of X (own side),
% scaled coordinates Yh = (y - x)/r (r = stencil radius), so x_hat = 0.
%   fd       : [A P; P' 0] [w_hat; gamma] = [L phi(||0 - Yh_j||); L p(0)]
%   pqr      : pQR of order deg+1
%   gmls     : w_hat = Th P (P' Th P)^-1 L p(0),  theta_j = Matern(ep ||Yh_j||)
% w = w_hat / r^2 (Laplacian) or w_hat / r (normal derivative).
[X, n] = side{:};
b = K.beta; ea = K.ea; eb = K.eb; Q = K.Q;
Lp0 = zeros(Q,1); Lp0(ea==2 & eb==0) = 2; Lp0(ea==0 & eb==2) = 2;   % Lap p at 0
idx = knnsearch(X, X(irows,:), 'K', n);              % n nearest nodes (incl. x itself)
Ic = cell(numel(irows),1); Jc = Ic; Vc = Ic;
for k = 1:numel(irows)
    x  = X(irows(k),:);
    J  = idx(k,:)';
    Yh = X(J,:) - x;
    r  = sqrt(max(Yh(:,1).^2 + Yh(:,2).^2));
    Yh = Yh / r;
    P  = (Yh(:,1).^ea) .* (Yh(:,2).^eb);
    rr = sqrt(Yh(:,1).^2 + Yh(:,2).^2);
    if strcmp(type, 'lap')
        rk = b^2 * rr.^(b-2);                                     % Lap_x r^beta at x_hat = 0
        rp = Lp0;  sc = r^2;
    else
        nn = nvec(k,:);
        rk = b * rr.^(b-2) .* (-Yh(:,1)*nn(1) - Yh(:,2)*nn(2));   % dn_x r^beta
        rp = zeros(Q,1); rp(ea==1 & eb==0) = nn(1); rp(ea==0 & eb==1) = nn(2);
        sc = r;
    end
    switch K.method
        case 'fd'
            A   = sqrt((Yh(:,1) - Yh(:,1)').^2 + (Yh(:,2) - Yh(:,2)').^2).^b;
            sol = [A, P; P', zeros(Q)] \ [rk; rp];
            al  = sol(1:n);
        case 'pqr'
            al  = pqr_weights(P, rp, rr, K.deg + 1);
        case 'gmls'
            al  = gmls_weights(P, rp, matern(K.ep * rr, K.nu));
    end
    Ic{k} = k*ones(n,1); Jc{k} = J; Vc{k} = al / sc;
end
Ii = vertcat(Ic{:}); Jj = vertcat(Jc{:}); Vv = vertcat(Vc{:});
end

function S = side_patches(X, ieval, PC, R0, nmin, K)
% Patches of one side: keep centres whose disc (radius R0) contains a row point of this
% side; local nodes = the nodes of this side in the disc; enlarge the radius (x 1.25)
% until the patch has at least nmin nodes. Each patch stores the LU factors of its scaled
% PHS + polynomial saddle-point matrix.
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
        r = 1.25*r;
    end
    J  = find(d2 < r^2);
    if numel(J) <= K.Q
        error('Patch with %d nodes; need more than Q = %d.', numel(J), K.Q);
    end
    Yh = (X(J,:) - S.C(l,:)) / r;
    P  = (Yh(:,1).^K.ea) .* (Yh(:,2).^K.eb);
    A  = sqrt((Yh(:,1) - Yh(:,1)').^2 + (Yh(:,2) - Yh(:,2)').^2).^K.beta;
    [Lf, Uf, pf] = lu([A, P; P', zeros(K.Q)], 'vector');
    S.J{l} = J; S.R(l) = r; S.F{l} = {Lf, Uf, pf, Yh};
end
end

function [Ii, Jj, Vv] = pu_rows(side, irows, type, nvec, K)
% D-RBF-PU rows at x = X(irows(k),:):
%   row_k = sum_l w_l(x) * alpha^l,  M_l [alpha_hat; gamma] = [L phi; L p] (scaled),
%   alpha^l = alpha_hat / R_l^2 (Laplacian) or alpha_hat / R_l (normal derivative).
[S, X] = side{:};
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
            rk = b^2 * rr.^(b-2);
            rp = (ea.*(ea-1)) .* xh(1).^max(ea-2,0) .* xh(2).^eb + ...
                 (eb.*(eb-1)) .* xh(1).^ea .* xh(2).^max(eb-2,0);
            sc = r^2;
        else
            nn = nvec(k,:);
            rk = b * rr.^(b-2) .* (dx*nn(1) + dy*nn(2));
            rp = nn(1) * ea .* xh(1).^max(ea-1,0) .* xh(2).^eb + ...
                 nn(2) * eb .* xh(1).^ea .* xh(2).^max(eb-1,0);
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

function al = gmls_weights(P, rp, th)
% GMLS weights: exact on the polynomials, P' al = rp, minimal weighted 2-norm:
%   al = Th P (P' Th P)^-1 rp,  Th = diag(th). Computed by QR of sqrt(Th) P.
sq = sqrt(th);
[Qm, Rm] = qr(sq .* P, 0);
al = sq .* (Qm * (Rm' \ rp(:)));
end

function al = pqr_weights(P, rp, d, q)
% pQR weights of Davydov & Safarpoor (2021), Sec. 2 (algorithm "pQR method of order q").
%   P  : monomials in coordinates shifted to the row point x and scaled by the largest
%        distance (column 1 = constant, all other columns vanish at x)
%   rp : the operator applied to the basis at x;  d : distances to x;  q : order
% The constant is eliminated; (A Th) v = b with Th = diag(d_j^-q) is solved by QR with
% column pivoting (basic solution); w = Th v; w_x = rp(1) - sum_j w_j.
rp = rp(:);  n = numel(d);
i0 = find(d <= 1e-12*max(d), 1);
if isempty(i0), error('pqr_weights: the row point must belong to the stencil.'); end
ir = setdiff((1:n)', i0);
A  = P(ir, 2:end)';
b  = rp(2:end);
th = d(ir).^(-q);  th = th / max(th);
[Qm, Rm, E] = qr(A .* th', 0);
dR = abs(diag(Rm));
r  = sum(dR > 1e-12*dR(1));
vt = Rm(1:r,1:r) \ (Qm(:,1:r)' * b);
v  = zeros(numel(ir), 1);  v(E(1:r)) = vt;
al = zeros(n, 1);
al(ir) = th .* v;
al(i0) = rp(1) - sum(al(ir));
al(abs(al) < 1e-12*max(abs(al))) = 0;
end

function th = matern(t, nu)
% Matern radial function of order nu (nu = 0.5, 1.5, 2.5), argument t = ep * d.
switch nu
    case 0.5, th = exp(-t);
    case 1.5, th = (1 + t) .* exp(-t);
    case 2.5, th = (1 + t + t.^2/3) .* exp(-t);
    otherwise, error('nu must be 0.5, 1.5 or 2.5.');
end
end
