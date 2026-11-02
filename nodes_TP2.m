function nd = nodes_TP2(m, ntype)
%NODES_TP2  Node set X^(m) for Test Problem 2 of Davydov & Safarpoor (2021):
%    Omega = (-1,1)^2,  Gamma: circle of radius 1/2 centred at the origin,
%    Omega+ : r < 1/2 (inside),  Omega- : r > 1/2 (outside).
%
%  Construction (Sections 2 and 3 of the paper):
%   - Cartesian grid with h = 0.4/2^m replaced by h0 = 0.15/2^m, adjusted to
%     h = 2/round(2/h0) so that the grid is aligned with the square and the
%     boundary nodes come from the same grid.
%   - Interior nodes: grid points inside Omega+ or Omega- with |r - 1/2| >= a*h
%     (a = 0.25); for the circle, r - 1/2 is the exact signed distance.
%   - Interface nodes (projection method): for every node of X+_int at distance
%     at most h from Gamma, its nearest point on Gamma; then thinned so that all
%     interface nodes are at least 0.5h apart.
%
%  ntype = 'uniform' (default) : interior nodes are the grid points (paper).
%  ntype = 'halton'            : interior nodes are Halton points in the square
%     with the same density as the grid (n^2 points, n = 2/h), removed if they
%     are within a*h of dOmega or of Gamma. Boundary nodes and the projection
%     construction of the interface nodes are the same as in the uniform case.
%
%  Output struct nd: Xint_p, Xint_m, Xbd_p, Xbd_m, XG, h, Nint (as nodes_TP1).

if nargin < 2 || isempty(ntype), ntype = 'uniform'; end
a  = 0.25;
h0 = 0.15 / 2^m;
n  = round(2/h0);
h  = 2/n;
t  = linspace(-1, 1, n+1);
[GX, GY] = meshgrid(t);
G  = [GX(:), GY(:)];
r  = sqrt(G(:,1).^2 + G(:,2).^2);

onB  = abs(abs(G(:,1)) - 1) < 1e-12 | abs(abs(G(:,2)) - 1) < 1e-12;
switch lower(ntype)
    case 'uniform'
        keep = abs(r - 0.5) >= a*h;
        Xint = G(~onB & keep, :);
    case 'halton'
        H    = 2*haltonseq(n^2, 2) - 1;                   % same density as the grid
        dB   = 1 - max(abs(H), [], 2);                    % distance to dOmega
        rH   = sqrt(H(:,1).^2 + H(:,2).^2);
        Xint = H(dB >= a*h & abs(rH - 0.5) >= a*h, :);
    otherwise
        error('ntype must be ''uniform'' or ''halton''.');
end
rint = sqrt(Xint(:,1).^2 + Xint(:,2).^2);
nd.Xint_p = Xint(rint < 0.5, :);
nd.Xint_m = Xint(rint > 0.5, :);
nd.Xbd_p  = zeros(0, 2);                 % dOmega lies entirely in Omega-
nd.Xbd_m  = G(onB, :);

% Interface nodes: projection of the X+_int nodes within distance h of Gamma
rp = sqrt(nd.Xint_p(:,1).^2 + nd.Xint_p(:,2).^2);
P  = nd.Xint_p(0.5 - rp <= h, :);
XG = 0.5 * P ./ sqrt(P(:,1).^2 + P(:,2).^2);
XG = thin_points(XG, 0.5*h);

nd.XG   = XG;
nd.h    = h;
nd.Nint = size(Xint, 1);
nd.type = lower(ntype);
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