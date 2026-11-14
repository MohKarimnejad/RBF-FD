function pb = problem_TP(k)
%PROBLEM_TP  Data of Test Problems 1-3 of Davydov & Safarpoor (2021),
%  Appl. Numer. Math. 161, 489-509:
%
%    -alpha div grad u = f  in Omega+ and Omega-,  Omega = (-1,1)^2,   u = g on dOmega,
%    [u] = u+ - u- = phi,  [alpha du/dn] = alpha+ du+/dn - alpha- du-/dn = psi  on Gamma.
%
%  Output struct pb:
%    pb.name          : 'TP1', 'TP2' or 'TP3'
%    pb.ap, pb.am     : alpha+ and alpha-
%    pb.up, pb.um     : exact solution on Omega+ / Omega-          @(x,y), column vectors
%    pb.fp, pb.fm     : right-hand side f = -alpha Lap u on Omega+ / Omega-
%    pb.gradp, pb.gradm : gradient of u+ / u-                        @(x,y) -> [ux, uy]
%    pb.nrm           : unit normal on Gamma (the same for both sides)  @(x,y) -> [nx, ny]
%    pb.nodes         : @(m, ntype, itype) node set X^(m) (nodes_TP1/2/3)
%  The normal is the one used in the paper's jump condition; phi and psi are computed
%  from the exact solution with this normal.

switch k
    case 1
        % Gamma: y = 2x^3,  Omega+ : y > 2x^3,  alpha+ = 20, alpha- = 1,
        % u+ = (s^2 - 30 s)/20,  u- = s^2 - 30 s,  s = y - 2x^3 (u and alpha du/dn continuous)
        pb.name = 'TP1';
        pb.ap = 20; pb.am = 1;
        s       = @(x,y) y - 2*x.^3;
        lapq    = @(x,y) 2*(36*x.^4 + 1) - 12*x.*(2*s(x,y) - 30);       % Lap(s^2 - 30 s)
        gradq   = @(x,y) [(2*s(x,y) - 30).*(-6*x.^2), (2*s(x,y) - 30)];  % grad(s^2 - 30 s)
        pb.up    = @(x,y) (s(x,y).^2 - 30*s(x,y)) / 20;
        pb.um    = @(x,y)  s(x,y).^2 - 30*s(x,y);
        pb.fp    = @(x,y) -lapq(x,y);
        pb.fm    = @(x,y) -lapq(x,y);
        pb.gradp = @(x,y) gradq(x,y) / 20;
        pb.gradm = @(x,y) gradq(x,y);
        pb.nrm   = @(x,y) [-6*x.^2, ones(size(x))] ./ sqrt(1 + 36*x.^4);   % into Omega+
        pb.nodes = @(m, ntype, itype) nodes_TP1(m, ntype);
    case 2
        % Gamma: circle r = 1/2,  Omega+ : r < 1/2,  alpha+ = alpha- = 1,
        % u+ = 1,  u- = 1 + log(2r)  (f = 0, [u] = 0, [du/dn] = -2)
        pb.name = 'TP2';
        pb.ap = 1; pb.am = 1;
        pb.up    = @(x,y) ones(size(x));
        pb.um    = @(x,y) 1 + log(2*sqrt(x.^2 + y.^2));
        pb.fp    = @(x,y) zeros(size(x));
        pb.fm    = @(x,y) zeros(size(x));
        pb.gradp = @(x,y) [zeros(size(x)), zeros(size(x))];
        pb.gradm = @(x,y) [x, y] ./ (x.^2 + y.^2);
        pb.nrm   = @(x,y) [x, y] ./ sqrt(x.^2 + y.^2);                      % outward
        pb.nodes = @(m, ntype, itype) nodes_TP2(m, ntype, itype);
    case 3
        % Gamma: gamma(r,theta) = 0.5 + 0.1 sin(4 theta + pi) - r = 0,  Omega+ : gamma > 0,
        % alpha+ = alpha- = 1,  u+ = r^4,  u- = r^2 sin(theta)  (discontinuous)
        pb.name = 'TP3';
        pb.ap = 1; pb.am = 1;
        rr    = @(x,y) sqrt(x.^2 + y.^2);
        gt    = @(x,y) 0.4 * cos(4*atan2(y, x) + pi);                       % d gamma / d theta
        gam   = @(x,y) 0.5 + 0.1*sin(4*atan2(y, x) + pi) - rr(x,y);
        ggam  = @(x,y) [-x./rr(x,y) - gt(x,y).*y./rr(x,y).^2, -y./rr(x,y) + gt(x,y).*x./rr(x,y).^2];
        pb.up    = @(x,y) rr(x,y).^4;
        pb.um    = @(x,y) y .* rr(x,y);
        pb.fp    = @(x,y) -16 * rr(x,y).^2;
        pb.fm    = @(x,y) -3 * y ./ rr(x,y);
        pb.gradp = @(x,y) 4 * rr(x,y).^2 .* [x, y];
        pb.gradm = @(x,y) [x.*y ./ rr(x,y), rr(x,y) + y.^2 ./ rr(x,y)];
        pb.nrm   = @(x,y) -ggam(x,y) ./ sqrt(sum(ggam(x,y).^2, 2));         % Omega+ -> Omega-
        pb.nodes = @(m, ntype, itype) nodes_TP3(m, ntype);
    otherwise
        error('k must be 1, 2 or 3.');
end
end
