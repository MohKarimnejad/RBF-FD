function nd = nodes_TP3(m, ntype)
%NODES_TP3  Node set X^(m) for Test Problem 3 of Davydov & Safarpoor (2021),
%  Appl. Numer. Math. 161, 489-509:
%    Omega = (-1,1)^2,  Gamma: gamma(r,theta) = 0.5 + 0.1 sin(4 theta + pi) - r = 0,
%    Omega+ : gamma > 0 (inside the flower),  Omega- : gamma < 0 (outside).
%
%  Construction (Sections 2 and 3 of the paper):
%   - Cartesian grid h*Z^2, h = 0.2/2^m (aligned with the square, so the
%     boundary nodes come from the same grid).
%   - Interior nodes: grid points inside Omega+ or Omega- with |gs| >= a*h
%     (a = 0.25), gs = gamma/|grad gamma| = first-order signed distance to Gamma.
%   - Interface nodes (isosurface method): contouring of gs on the grid, vertices
%     moved onto Gamma by the secant method along grad(gamma) (second initial
%     point at distance h^2), then thinned so that all interface nodes are at
%     least 0.5h apart.
%  In |grad gamma| = sqrt(1 + (0.4 cos(4 theta)/r)^2), r is replaced by max(r, 0.2):
%  this only changes gs near the origin (far from Gamma, where 0.4 <= r <= 0.6) and
%  avoids gs = 0 at r = 0.
%
%  ntype = 'uniform' (default) : interior nodes are the grid points (paper).
%  ntype = 'halton'            : interior nodes are Halton points in the square
%     with the same density as the grid (n^2 points, n = 2/h), removed if they
%     are within a*h of dOmega or of Gamma (|gs| < a*h). Boundary and
%     interface nodes are constructed exactly as in the uniform case.
%
%  Output struct nd: Xint_p, Xint_m, Xbd_p, Xbd_m, XG, h, Nint (as nodes_TP1).

if nargin < 2 || isempty(ntype), ntype = 'uniform'; end
a  = 0.25;
h  = 0.2 / 2^m;
n1 = round(2/h) + 1;                       % grid points per side
t  = linspace(-1, 1, n1);
[GX, GY] = meshgrid(t);
G  = [GX(:), GY(:)];

%% Interface nodes: contouring + secant projection
C  = contourc(t, t, gs3(GX, GY), [0 0]);
V  = [];
k  = 1;
while k < size(C, 2)                       % collect all contour vertices
    nv = C(2, k);
    V  = [V; C(:, k+1:k+nv)'];             %#ok<AGROW>
    k  = k + nv + 1;
end
V = unique(V, 'rows');
for j = 1:size(V, 1)
    V(j,:) = secant_project(V(j,:), h);
end
XG = thin_points(V, 0.5*h);

%% Grid points: boundary nodes, interior nodes
onB  = abs(abs(G(:,1)) - 1) < 1e-12 | abs(abs(G(:,2)) - 1) < 1e-12;
keep = abs(gs3(G(:,1), G(:,2))) >= a*h;
switch lower(ntype)
    case 'uniform'
        Xint = G(~onB & keep, :);
    case 'halton'
        H    = 2*haltonseq((n1 - 1)^2, 2) - 1;            % same density as the grid
        dB   = 1 - max(abs(H), [], 2);                    % distance to dOmega
        Xint = H(dB >= a*h & abs(gs3(H(:,1), H(:,2))) >= a*h, :);
    otherwise
        error('ntype must be ''uniform'' or ''halton''.');
end

sp = gs3(Xint(:,1), Xint(:,2)) > 0;
nd.Xint_p = Xint( sp, :);   nd.Xint_m = Xint(~sp, :);
nd.Xbd_p  = zeros(0, 2);                 % dOmega lies entirely in Omega-
nd.Xbd_m  = G(onB, :);
nd.XG   = XG;
nd.h    = h;
nd.Nint = size(Xint, 1);
nd.type = lower(ntype);
end

% ============================================================================
function g = gam3(x, y)
% gamma = 0.5 + 0.1 sin(4 theta + pi) - r
g = 0.5 + 0.1*sin(4*atan2(y, x) + pi) - sqrt(x.^2 + y.^2);
end

function gr = grad_gam3(x, y)
% Cartesian gradient of gamma: d/dr = -1, d/dtheta = 0.4 cos(4 theta + pi)
r  = sqrt(x.^2 + y.^2);  th = atan2(y, x);
gt = 0.4*cos(4*th + pi);
gr = [-x./r - gt.*y./r.^2, -y./r + gt.*x./r.^2];
end

function g = gs3(x, y)
% first-order signed distance gamma/|grad gamma| (r -> max(r, 0.2) in |grad gamma|)
r  = max(sqrt(x.^2 + y.^2), 0.2);
gt = 0.4*cos(4*atan2(y, x) + pi);
g  = gam3(x, y) ./ sqrt(1 + (gt./r).^2);
end

function p = secant_project(p, h)
% Move p onto {gamma = 0} along the direction of grad(gamma) at p (secant method).
gr = grad_gam3(p(1), p(2));
d  = gr / norm(gr);
f  = @(s) gam3(p(1) + s*d(1), p(2) + s*d(2));
s0 = 0; s1 = h^2; f0 = f(s0); f1 = f(s1);
for it = 1:50
    if f1 == f0, break; end
    s2 = s1 - f1*(s1 - s0)/(f1 - f0);
    s0 = s1; f0 = f1; s1 = s2; f1 = f(s1);
    if abs(f1) < 1e-15, break; end
end
p = p + s1*d;
end

function X = thin_points(X, dmin)
% Remove points until all pairwise distances are >= dmin. At each step the
% closest pair is found and one of its points is removed (the one closer to
% its other neighbours).
while true
    D = sqrt((X(:,1) - X(:,1)').^2 + (X(:,2) - X(:,2)').^2);
    D(1:size(X,1)+1:end) = inf;
    [dm, idx] = min(D(:));
    if dm >= dmin, break; end
    [i, j] = ind2sub(size(D), idx);
    Di = sort(D(i,:)); Dj = sort(D(j,:));
    if Di(2) <= Dj(2), X(i,:) = []; else, X(j,:) = []; end
end
[~, o] = sort(atan2(X(:,2), X(:,1))); X = X(o,:);
end
