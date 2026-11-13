clc; clear; close all
%% GMLS-FD with a variably scaled discontinuous Matern weight, stencils crossing Gamma,
%  for Test Problem 1 of Davydov & Safarpoor (2021). Unknowns and equations as in the paper.
%  Errors: rms and max over the interior nodes X_int; sigma = condest(A')/norm(A,inf).

deg    = 4;              % polynomial degree of the GMLS fit
Q      = (deg+1)*(deg+2)/2;
nloc   = 3*Q;            % stencil size (nearest nodes of both sides)
ep     = 1;              % shape parameter of the Matern weight
nu     = 2.5;            % Matern order: 0.5, 1.5 or 2.5
ntype  = 'uniform';      % node type: 'uniform' (grid, as in the paper) or 'halton'
mlist  = 0:5;            % node levels X^(m), h = 0.4/2^m
msigma = 4;              % compute sigma only up to this level (extra LU, memory)
mplot  = [3,4,5];        % levels used for the solution / error surfaces
sname  = sprintf('FD [gmls-vsdk, Matern %.1f, \\epsilon = %g]', nu, ep);

fprintf('%s, degree %d (Q = %d), nloc = %d, %s nodes\n', sname, deg, Q, nloc, ntype);
fprintf('%3s %8s %7s %10s %10s %10s %8s %8s\n', 'm', 'h', 'Nint', 'rms', 'max', ...
    'sigma', 'nnz/row', 'time[s]');
res = nan(numel(mlist), 4);
for k = 1:numel(mlist)
    m = mlist(k);
    try
        o = PHS_fd_TP1_vsdk(m, deg, nloc, ep, nu, m <= msigma, ntype);
    catch err
        fprintf('%3d  failed: %s\n', m, err.message); continue
    end
    res(k,:) = [o.Nint, o.h, o.rms, o.max];
    fprintf('%3d %8.5f %7d %10.3e %10.3e %10.2e %8.1f %8.1f\n', m, o.h, o.Nint, ...
        o.rms, o.max, o.sigma, o.nnz_row, o.time);
    if ismember(m, mplot), op{m+1} = o; end
end

%% rms error vs N_int (as in Figs. 5-6 of the paper)
figure
loglog(res(:,1), res(:,3), 'o-', 'LineWidth', 1.2); hold on
ok = find(~isnan(res(:,1)), 1);
loglog(res(:,1), res(ok,3)*(res(:,1)/res(ok,1)).^(-1), 'r--')    % h^2 = O(N^-1)
loglog(res(:,1), res(ok,3)*(res(:,1)/res(ok,1)).^(-2), 'r-.')    % h^4 = O(N^-2)
grid on; xlabel('N_{int}'); ylabel('rms error')
legend(sname, 'h^2', 'h^4', 'Location', 'southwest')
title(sprintf('Test Problem 1, degree %d, nloc = %d', deg, nloc))

%% Numerical solution, relative error and growth function (as in Fig. 9 of the paper)
for m = mplot
    o   = op{m+1};
    tri = delaunay(o.Xall(:,1), o.Xall(:,2));
    figure('Position', [100 100 1500 420])
    subplot(1,3,1)
    trisurf(tri, o.Xall(:,1), o.Xall(:,2), o.uall, 'EdgeColor', 'none')
    title('Numerical solution'); view(-37.5, 30); axis tight
    subplot(1,3,2)
    trisurf(tri, o.Xall(:,1), o.Xall(:,2), o.erel, 'EdgeColor', 'none')
    title('Relative error |e|/max|u| (per side)'); view(-37.5, 30); axis tight
    subplot(1,3,3)
    scatter(o.Xgrowth(:,1), o.Xgrowth(:,2), 6, log10(o.growth), 'filled')
    axis equal tight; box on; colorbar
    title(sprintf('log_{10} growth function (q = %d)', deg))
    sgtitle(sprintf('Test problem 1, FD gmls-vsdk, %s nodes, level %d, poly degree %d', ntype, m, deg))
end
