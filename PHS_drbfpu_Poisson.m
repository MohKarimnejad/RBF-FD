function out = PHS_drbfpu_Poisson(m, beta, deg, nloc, do_sigma, ntype)
%PHS_DRBFPU_POISSON  D-RBF-PU (square collocation, PHS + polynomials) for
%
%    -Lap u = f  in Omega = (0,1)^2,   u = g on dOmega,
%    u(x,y) = sin(2 pi x) cos(2 pi y) + exp(x + y/2).
%
%  m        : node level, h = 0.2/2^m (nodes from nodes_Poisson)
%  beta     : odd PHS exponent, phi(r) = r^beta
%  deg      : polynomial degree (deg >= ceil(beta/2)-1)
%  nloc     : target number of nodes per patch (default 3*Q)
%  do_sigma : compute the stability constant sigma (default true; extra LU)
%  ntype    : node type, 'uniform' (default) or 'halton'
%
%  Unknowns: u at X = [Xint; Xbd]. Equations:
%    -Lap u = f   at Xint   (D-RBF-PU rows)
%     u = g       at Xbd
%  Patches: centres on a uniform grid; a patch with fewer than nmin = 2Q nodes
%  is enlarged (radius x 1.25) until it has nmin.

Q = (deg + 1)*(deg + 2)/2;
if nargin < 4 || isempty(nloc), nloc = 3*Q; end
if nargin < 5, do_sigma = true; end
if nargin < 6 || isempty(ntype), ntype = 'uniform'; end
if mod(beta, 2) ~= 1, error('beta must be odd.'); end
if deg < ceil(beta/2) - 1, error('Need deg >= ceil(beta/2)-1.'); end
nmin = 2*Q;

%% Exact solution and data
uex = @(x,y) sin(2*pi*x).*cos(2*pi*y) + exp(x + 0.5*y);
f   = @(x,y) 8*pi^2*sin(2*pi*x).*cos(2*pi*y) - 1.25*exp(x + 0.5*y);   % f = -Lap u

%% Nodes
nd = nodes_Poisson(m, ntype);
h  = nd.h;
X  = [nd.Xint; nd.Xbd];
nI = size(nd.Xint,1); nB = size(nd.Xbd,1); N = nI + nB;

%% PHS kernel and polynomial basis (in scaled coordinates)
ea = []; eb = [];                                 % monomials x^a y^b, a+b <= deg
for tt = 0:deg, for bb = 0:tt, ea(end+1) = tt-bb; eb(end+1) = bb; end, end %#ok<AGROW>
K.beta = beta; K.ea = ea; K.eb = eb; K.Q = Q;

%% Patches (base radius from nloc: pi R^2 / h^2 = nloc; centre spacing = R)
R0  = h * sqrt(nloc/pi);
npc = ceil(1/R0) + 1;  R0 = 1/(npc - 1);
[C1, C2] = meshgrid(linspace(0, 1, npc)); PC = [C1(:), C2(:)];

tic
S = make_patches(X, 1:nI, PC, R0, nmin, K);

%% Assemble the global system  A u = b
[Ii, Jj, Vv] = pu_rows(S, X, 1:nI, K);
A = sparse([Ii; nI + (1:nB)'], [Jj; nI + (1:nB)'], [-Vv; ones(nB,1)], N, N);
b = [f(X(1:nI,1), X(1:nI,2)); uex(X(nI+1:end,1), X(nI+1:end,2))];
u = A \ b;
t = toc;

%% Errors on interior nodes (rms and max over X_int)
ue = uex(X(:,1), X(:,2));
e  = abs(u(1:nI) - ue(1:nI));
out.h = h; out.Nint = nd.Nint; out.Q = Q; out.nloc = nloc; out.R0 = R0;
out.npatch = numel(S.J);
out.nlocal = cellfun(@numel, S.J);                     % nodes per patch
out.rms = sqrt(mean(e.^2)); out.max = max(e);
if do_sigma
    out.sigma = condest(A') / norm(A, inf);            % stability constant
else
    out.sigma = NaN;
end
out.nnz_row = nnz(A) / N;
out.time = t;

%% Solution and absolute error at all nodes (for plots)
out.Xall = X;
out.uall = u;
out.eall = abs(u - ue);
end

% ============================================================================
function S = make_patches(X, ieval, PC, R0, nmin, K)
% Keep centres whose disc (radius R0) contains an evaluation node; local
% nodes = all nodes in the disc; enlarge the radius until the patch has at
% least nmin nodes. Each patch stores the LU factors of its scaled
% saddle-point matrix.
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
        error('Patch with %d nodes; need more than Q = %d (level too coarse for this degree).', numel(J), K.Q);
    end
    Yh = (X(J,:) - S.C(l,:)) / r;                         % scaled coordinates
    A  = sqrt((Yh(:,1) - Yh(:,1)').^2 + (Yh(:,2) - Yh(:,2)').^2).^K.beta;
    P  = (Yh(:,1).^K.ea) .* (Yh(:,2).^K.eb);
    [Lf, Uf, pf] = lu([A, P; P', zeros(K.Q)], 'vector');
    S.J{l} = J; S.R(l) = r; S.F{l} = {Lf, Uf, pf, Yh};
end
end

function [Ii, Jj, Vv] = pu_rows(S, X, irows, K)
% D-RBF-PU Laplacian rows at x = X(irows(k),:):
%   row_k = sum_l w_l(x) * alpha^l,  M_l [alpha_hat; gamma] = [Lap phi; Lap p] (scaled),
%   alpha^l = alpha_hat / R_l^2.
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
        rr = sqrt((xh(1) - Yh(:,1)).^2 + (xh(2) - Yh(:,2)).^2);
        rk = b^2 * rr.^(b-2);                                         % Lap r^beta (d = 2)
        rp = (ea.*(ea-1)) .* xh(1).^max(ea-2,0) .* xh(2).^eb + ...
             (eb.*(eb-1)) .* xh(1).^ea .* xh(2).^max(eb-2,0);
        rhs = [rk; rp(:)];
        sol = Uf \ (Lf \ rhs(pf));
        al  = sol(1:numel(J)) / r^2;
        Ii = [Ii; k*ones(numel(J),1)]; Jj = [Jj; J]; Vv = [Vv; w(q)*al]; %#ok<AGROW>
    end
    Ic{k} = Ii; Jc{k} = Jj; Vc{k} = Vv;
end
Ii = vertcat(Ic{:}); Jj = vertcat(Jc{:}); Vv = vertcat(Vc{:});
end
