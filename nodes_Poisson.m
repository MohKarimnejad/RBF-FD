function nd = nodes_Poisson(m, ntype)
%NODES_POISSON  Node set X^(m) for the Poisson problem on Omega = (0,1)^2.
%
%  Construction (same rules as nodes_TP1):
%   - Cartesian grid h*Z^2, h = 0.2/2^m (aligned with the square, so the
%     boundary nodes come from the same grid; same number of grid points per
%     side as in TP1).
%   - Boundary nodes: the grid points on dOmega (in both cases).
%
%  ntype = 'uniform' (default) : interior nodes are the interior grid points.
%  ntype = 'halton'            : interior nodes are Halton points in the square
%     with the same density as the grid (n^2 points, n = 1/h), removed if they
%     are within a*h of dOmega (a = 0.25).
%
%  Output struct nd:
%   nd.Xint : interior nodes,  nd.Xbd : boundary nodes,
%   nd.h, nd.Nint, nd.type

if nargin < 2 || isempty(ntype), ntype = 'uniform'; end
a  = 0.25;
h  = 0.2 / 2^m;
n1 = round(1/h) + 1;                       % grid points per side
t  = linspace(0, 1, n1);
[GX, GY] = meshgrid(t);
G  = [GX(:), GY(:)];

tol = 1e-12;
onB = G(:,1) < tol | G(:,1) > 1 - tol | G(:,2) < tol | G(:,2) > 1 - tol;
Xbd = G(onB, :);

switch lower(ntype)
    case 'uniform'
        Xint = G(~onB, :);
    case 'halton'
        H    = haltonseq((n1 - 1)^2, 2);                  % same density as the grid
        dB   = min(min(H, 1 - H), [], 2);                 % distance to dOmega
        Xint = H(dB >= a*h, :);
    otherwise
        error('ntype must be ''uniform'' or ''halton''.');
end

nd.Xint = Xint;
nd.Xbd  = Xbd;
nd.h    = h;
nd.Nint = size(Xint, 1);
nd.type = lower(ntype);
end
