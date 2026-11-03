function out = PHS_fd_Poisson(m, beta, deg, nloc, do_sigma, ntype)
%PHS_FD_POISSON  RBF-FD (square collocation, PHS + polynomials) for
%
%    -Lap u = f  in Omega = (0,1)^2,   u = g on dOmega,
%    u(x,y) = sin(2 pi x) cos(2 pi y) + exp(x + y/2).
%
%  m        : node level, h = 0.2/2^m (nodes from nodes_Poisson)
%  beta     : odd PHS exponent, phi(r) = r^beta
%  deg      : polynomial degree (deg >= ceil(beta/2)-1)
%  nloc     : stencil size = number of nearest nodes (default 3*Q)
%  do_sigma : compute the stability constant sigma (default true; extra LU)
%  ntype    : node type, 'uniform' (default) or 'halton'
%
%  Unknowns: u at X = [Xint; Xbd]. Equations:
%    -Lap u = f   at Xint   (RBF-FD rows)
%     u = g       at Xbd
%  Stencils: for each row point x_i, its nloc nearest nodes.
%  One local saddle-point system per row (no partition of unity).

Q = (deg + 1)*(deg + 2)/2;
if nargin < 4 || isempty(nloc), nloc = 3*Q; end
if nargin < 5, do_sigma = true; end
if nargin < 6 || isempty(ntype), ntype = 'uniform'; end
if mod(beta, 2) ~= 1, error('beta must be odd.'); end
if deg < ceil(beta/2) - 1, error('Need deg >= ceil(beta/2)-1.'); end

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

tic
ns = min(nloc, N);                                % stencil size (capped by N)

%% Assemble the global system  A u = b
[Ii, Jj, Vv] = fd_rows(X, 1:nI, K, ns);
A = sparse([Ii; nI + (1:nB)'], [Jj; nI + (1:nB)'], [-Vv; ones(nB,1)], N, N);
b = [f(X(1:nI,1), X(1:nI,2)); uex(X(nI+1:end,1), X(nI+1:end,2))];
u = A \ b;
t = toc;

%% Errors on interior nodes (rms and max over X_int)
ue = uex(X(:,1), X(:,2));
e  = abs(u(1:nI) - ue(1:nI));
out.h = h; out.Nint = nd.Nint; out.Q = Q; out.nloc = nloc;
out.nlocal = ns;                                   % stencil size
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
function [Ii, Jj, Vv] = fd_rows(X, irows, K, n)
% RBF-FD Laplacian rows at x = X(irows(k),:): stencil = the n nearest nodes
% of X, scaled coordinates Yh = (y - x)/r (r = stencil radius), so x_hat = 0:
%   [A P; P' 0] [w_hat; gamma] = [Lap phi(||0 - Yh_j||); Lap p(0)],
%   w = w_hat / r^2.
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
    A  = sqrt((Yh(:,1) - Yh(:,1)').^2 + (Yh(:,2) - Yh(:,2)').^2).^b;
    P  = (Yh(:,1).^ea) .* (Yh(:,2).^eb);
    rr = sqrt(Yh(:,1).^2 + Yh(:,2).^2);
    rk = b^2 * rr.^(b-2);                                  % Lap_x r^beta at x_hat = 0
    sol = [A, P; P', zeros(Q)] \ [rk; Lp0];
    Ic{k} = k*ones(n,1); Jc{k} = J; Vc{k} = sol(1:n) / r^2;
end
Ii = vertcat(Ic{:}); Jj = vertcat(Jc{:}); Vv = vertcat(Vc{:});
end
