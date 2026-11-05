clc; clear; close all
%% D-RBF-PU or RBF-FD (PHS + polynomials) for Test Problem 2 of Davydov & Safarpoor (2021).
%  Errors as in the paper: rms and max over the interior nodes X_int;
%  sigma = condest(A')/norm(A,inf) as in (3.2).

solver = 'FD';           % solver: 'PU' (D-RBF-PU) or 'FD' (RBF-FD)
basis  = 'phs';          % local basis: 'phs' (PHS + polynomials), 'ls' (polynomials, least squares)
                         %              or 'wls' (polynomials, weighted least squares = GMLS)
beta   = 5;              % PHS exponent, phi(r) = r^beta
deg    = 6;              % polynomial degree  (pQR order q = deg + 1)
Q      = (deg+1)*(deg+2)/2;
nloc   = 3*Q;            % nodes per patch (PU) / stencil size (FD)
ntype  = 'uniform';      % node type: 'uniform' (grid, as in the paper) or 'halton'
mlist  = 0:5;            % node levels X^(m), h ~ 0.15/2^m
msigma = 3;              % compute sigma only up to this level (extra LU, memory)
mplot  = [3 4 5];          % levels used for the solution / error surfaces

switch upper(solver)
    case 'PU', solve = @PHS_drbfpu_TP2; sname = 'D-RBF-PU';
    case 'FD', solve = @PHS_fd_TP2;     sname = 'RBF-FD';
    otherwise, error('solver must be ''PU'' or ''FD''.');
end
sname = sprintf('%s [%s]', sname, basis);

fprintf('%s, PHS r^%d, degree %d (Q = %d), nloc = %d, %s nodes\n', sname, beta, deg, Q, nloc, ntype);
fprintf('%3s %8s %7s %10s %10s %10s %8s %9s %8s\n', 'm', 'h', 'Nint', 'rms', 'max', ...
    'sigma', 'nnz/row', 'min nloc', 'time[s]');
res = nan(numel(mlist), 4);
for k = 1:numel(mlist)
    m = mlist(k);
    try
        o = solve(m, beta, deg, nloc, m <= msigma, ntype, basis);
    catch err
        fprintf('%3d  failed: %s\n', m, err.message); continue
    end
    res(k,:) = [o.Nint, o.h, o.rms, o.max];
    fprintf('%3d %8.5f %7d %10.3e %10.3e %10.2e %8.1f %9d %8.1f\n', m, o.h, o.Nint, ...
        o.rms, o.max, o.sigma, o.nnz_row, min(o.nlocal), o.time);
    if ismember(m, mplot), op{m+1} = o; end
end

%% Convergence order of the max error (as in Table 4 of the paper):
%  slope of the least-squares line fitted to (ln h_m, ln max_m)
ok = ~isnan(res(:,1));
p  = polyfit(log(res(ok,2)), log(res(ok,4)), 1);
fprintf('Estimated convergence order (max): %.1f\n', p(1));

%% rms error vs N_int (as in Fig. 10 of the paper)
figure
loglog(res(:,1), res(:,3), 'o-', 'LineWidth', 1.2); hold on
i0 = find(ok, 1);
loglog(res(:,1), res(i0,3)*(res(:,1)/res(i0,1)).^(-1), 'r--')    % h^2 = O(N^-1)
loglog(res(:,1), res(i0,3)*(res(:,1)/res(i0,1)).^(-2), 'r-.')    % h^4 = O(N^-2)
grid on; xlabel('N_{int}'); ylabel('rms error')
legend(sname, 'h^2', 'h^4', 'Location', 'southwest')
title(sprintf('%s, Test Problem 2, PHS r^%d, degree %d, nloc = %d', sname, beta, deg, nloc))

%% Numerical solution and absolute error (as in Fig. 11 of the paper: -u is plotted)
for m = mplot
    o   = op{m+1};
    tri = delaunay(o.Xall(:,1), o.Xall(:,2));
    figure('Position', [100 100 1100 420])
    subplot(1,2,1)
    trisurf(tri, o.Xall(:,1), o.Xall(:,2), -o.uall, 'EdgeColor', 'none')
    title('Numerical solution (-u)'); view(-37.5, 30); axis tight
    subplot(1,2,2)
    trisurf(tri, o.Xall(:,1), o.Xall(:,2), o.eall, 'EdgeColor', 'none')
    title('Absolute error'); view(-37.5, 30); axis tight
    sgtitle(sprintf('%s, Test Problem 2, X^{(%d)} (h = %.4g), PHS r^%d, degree %d', m, o.h, beta, deg))
end
