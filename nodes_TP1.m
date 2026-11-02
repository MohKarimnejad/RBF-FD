function nd = nodes_TP1(m, ntype)
%NODES_TP1  Node set X^(m) for Test Problem 1 of Davydov & Safarpoor (2021),
%  Appl. Numer. Math. 161, 489-509:
%    Omega = (-1,1)^2,  Gamma: y = 2x^3,  Omega+ : y > 2x^3,  Omega- : y < 2x^3.
%
%  Construction (Sections 2 and 3 of the paper):
%   - Cartesian grid h*Z^2, h = 0.4/2^m (aligned with the square, so the
%     boundary nodes come from the same grid).
%   - Interior nodes: grid points inside Omega+ or Omega- with |gamma| >= a*h
%     (a = 0.25), where gamma is the first-order signed distance to Gamma used
%     in the paper. This reproduces N_int of Table 1 exactly for m = 0,...,6.
%     Grid points on dOmega are boundary nodes (same criterion near Gamma).
%   - Interface nodes: contouring of gamma(x,y) = (y-2x^3)/sqrt(1+36x^4) on the
%     grid, vertices moved onto Gamma by the secant method along grad(gamma)
%     (second initial point at distance h^2), then thinned so that all
%     interface nodes are at least 0.5h apart.
%
%  ntype = 'uniform' (default) : interior nodes are the grid points (paper).
%  ntype = 'halton'            : interior nodes are Halton points in the square
%     with the same density as the grid (n^2 points, n = 2/h), removed if they
%     are within a*h of dOmega or of Gamma (|gamma| < a*h). Boundary and
%     interface nodes are constructed exactly as in the uniform case.
%
%  Output struct nd:
%   nd.Xint_p, nd.Xint_m : interior nodes in Omega+ / Omega-
%   nd.Xbd_p,  nd.Xbd_m  : boundary nodes (dOmega \ Gamma) on the + / - side
%   nd.XG                : interface nodes (Gamma, incl. Gamma cap dOmega)
%   nd.h, nd.Nint

if nargin < 2 || isempty(ntype), ntype = 'uniform'; end
a  = 0.25;
h  = 0.4 / 2^m;
n1 = round(2/h) + 1;                       % grid points per side
t  = linspace(-1, 1, n1);
[GX, GY] = meshgrid(t);
G  = [GX(:), GY(:)];

phi   = @(x,y) y - 2*x.^3;                 % level set (sign gives the side)
gam   = @(x,y) (y - 2*x.^3) ./ sqrt(1 + 36*x.^4);
tend  = 2^(-1/3);                          % Gamma meets y = +-1 at x = +-tend

%% Interface nodes: contouring + secant projection
C  = contourc(t, t, gam(GX, GY), [0 0]);
V  = [];
k  = 1;
while k < size(C, 2)                       % collect all contour vertices
    nv = C(2, k);
    V  = [V; C(:, k+1:k+nv)'];             %#ok<AGROW>
    k  = k + nv + 1;
end
V = unique(V, 'rows');
onBox = abs(abs(V(:,1)) - 1) < 1e-12 | abs(abs(V(:,2)) - 1) < 1e-12;
V = V(~onBox, :);                          % end points are added exactly below

for j = 1:size(V, 1)
    V(j,:) = secant_project(V(j,:), gam, h);
end
V = V(abs(V(:,2)) < 1, :);                 % keep points of Gamma inside Omega
XG = [-tend, -1; V; tend, 1];              % exact end points Gamma cap dOmega
XG = thin_points(XG, 0.5*h, [1, size(XG,1)]);

%% Grid points: boundary nodes (both cases), interior nodes (uniform case)
onB  = abs(abs(G(:,1)) - 1) < 1e-12 | abs(abs(G(:,2)) - 1) < 1e-12;
keep = abs(gam(G(:,1), G(:,2))) >= a*h;
Xbd  = G( onB & keep, :);

switch lower(ntype)
    case 'uniform'
        Xint = G(~onB & keep, :);
    case 'halton'
        H    = 2*haltonseq((n1 - 1)^2, 2) - 1;            % same density as the grid
        dB   = 1 - max(abs(H), [], 2);                    % distance to dOmega
        Xint = H(dB >= a*h & abs(gam(H(:,1), H(:,2))) >= a*h, :);
    otherwise
        error('ntype must be ''uniform'' or ''halton''.');
end

sp = phi(Xint(:,1), Xint(:,2)) > 0;
nd.Xint_p = Xint( sp, :);   nd.Xint_m = Xint(~sp, :);
sb = phi(Xbd(:,1), Xbd(:,2)) > 0;
nd.Xbd_p  = Xbd( sb, :);    nd.Xbd_m  = Xbd(~sb, :);
nd.XG   = XG;
nd.h    = h;
nd.Nint = size(Xint, 1);
nd.type = lower(ntype);
end

% ============================================================================
function p = secant_project(p, gam, h)
% Move p onto {gam = 0} along the direction of grad(gam) at p (secant method).
x = p(1); y = p(2); s = sqrt(1 + 36*x^4);
g = y - 2*x^3;
gr = [(-6*x^2*s - g*36*x^3/s) / s^2, 1/s];
d  = gr / norm(gr);
f  = @(s) gam(p(1) + s*d(1), p(2) + s*d(2));
s0 = 0; s1 = h^2; f0 = f(s0); f1 = f(s1);
for it = 1:50
    if f1 == f0, break; end
    s2 = s1 - f1*(s1 - s0)/(f1 - f0);
    s0 = s1; f0 = f1; s1 = s2; f1 = f(s1);
    if abs(f1) < 1e-15, break; end
end
p = p + s1*d;
end

function X = thin_points(X, dmin, fixed)
% Remove points until all pairwise distances are >= dmin. At each step the
% closest pair is found and one of its points is removed (never a fixed point;
% otherwise the one closer to its other neighbours).
isfix = false(size(X,1),1); isfix(fixed) = true;
while true
    D = sqrt((X(:,1) - X(:,1)').^2 + (X(:,2) - X(:,2)').^2);
    D(1:size(X,1)+1:end) = inf;
    [dm, idx] = min(D(:));
    if dm >= dmin, break; end
    [i, j] = ind2sub(size(D), idx);
    if isfix(i), r = j;
    elseif isfix(j), r = i;
    else
        Di = sort(D(i,:)); Dj = sort(D(j,:));
        if Di(2) <= Dj(2), r = i; else, r = j; end
    end
    X(r,:) = []; isfix(r) = [];
end
[~, o] = sort(X(:,1)); X = X(o,:);
end