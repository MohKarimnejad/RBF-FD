clc; clear; close all
%% Test Problem 2 of Davydov & Safarpoor (2021): GMLS finite differences (FD_2)
%  with the MLS or the MLS-VSDK weight.
%  Errors as in the paper: rms and max over the interior nodes X_int;
%  sigma = condest(A')/norm(A,inf) as in (3.2).

method = 'mls-vsk';      % 'mls' (Matern weight) or 'mls-vsk' (variably scaled Matern weight,
                         % scale function psi = lam*dh*tanh(gamma/dh), dh = delta*h,
                         % gamma = signed distance to Gamma);
                         % stencils of the own side only in both cases
lam    = 1;              % amplitude of the scale function (mls-vsk only)
delta  = 2;              % width of the band around Gamma in units of h (mls-vsk only)
deg    = 6;              % polynomial degree of the GMLS fit
Q      = (deg+1)*(deg+2)/2;
nloc   = 3*Q;            % stencil size
ep     = 1;              % shape parameter of the Matern weight
nu     = 2.5;            % Matern order: 0.5, 1.5 or 2.5
ntype  = 'uniform';      % node type: 'uniform' (grid, as in the paper) or 'halton'
mlist  = 0:4;            % node levels X^(m), h ~ 0.15/2^m
msigma = 4;              % compute sigma only up to this level (extra LU, memory)
mplot  = [2,3,4];        % levels used for the solution / error surfaces

sname = sprintf('FD [%s]', method);

fprintf('%s, Matern %.1f, ep = %g, lam = %g, delta = %g, degree %d (Q = %d), nloc = %d, %s nodes\n', ...
    sname, nu, ep, lam, delta, deg, Q, nloc, ntype);
fprintf('%3s %8s %7s %10s %10s %10s %8s %9s %8s\n', 'm', 'h', 'Nint', 'rms', 'max', ...
    'sigma', 'nnz/row', 'min nloc', 'time[s]');
res = nan(numel(mlist), 4);
for k = 1:numel(mlist)
    m = mlist(k);
    try
        o = FD_2(m, deg, nloc, ep, nu, method, lam, delta, m <= msigma, ntype);
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
ok = find(~isnan(res(:,1)), 1);
loglog(res(:,1), res(ok,3)*(res(:,1)/res(ok,1)).^(-1), 'r--')    % h^2 = O(N^-1)
loglog(res(:,1), res(ok,3)*(res(:,1)/res(ok,1)).^(-2), 'r-.')    % h^4 = O(N^-2)
grid on; xlabel('N_{int}'); ylabel('rms error')
legend(sname, 'h^2', 'h^4', 'Location', 'southwest')
title(sprintf('%s, Test Problem 2, degree %d, nloc = %d', sname, deg, nloc))

%% Numerical solution, relative error and growth function (as in Fig. 11 of the paper: -u is plotted)
for m = mplot
    o   = op{m+1};
    tri = delaunay(o.Xall(:,1), o.Xall(:,2));
    figure('Position', [100 100 1900 420])
    subplot(1,4,1)
    trisurf(tri, o.Xall(:,1), o.Xall(:,2), -o.uall, 'EdgeColor', 'none')
    title('Numerical solution (-u)'); view(-37.5, 30); axis tight
    subplot(1,4,2)
    trisurf(tri, o.Xall(:,1), o.Xall(:,2), o.erel, 'EdgeColor', 'none')
    title('Relative error |e|/max|u| (per side)'); view(-37.5, 30); axis tight
    subplot(1,4,3)
    scatter(o.Xgrowth(:,1), o.Xgrowth(:,2), 6, log10(o.growth), 'filled')
    axis equal tight; box on; colorbar
    title(sprintf('log_{10} \\rho, Laplacian rows (q = %d)', deg+1))
    subplot(1,4,4)
    scatter(o.XG(:,1), o.XG(:,2), 12, log10(max(o.growth_dn, [], 2)), 'filled')
    axis equal; xlim([-1 1]); ylim([-1 1]); box on; colorbar
    title(sprintf('log_{10} max(\\rho_+, \\rho_-), normal-derivative rows (q = %d)', deg+1))
    sgtitle(sprintf('Test problem 2, %s, %s nodes, level %d, poly degree %d', sname, ntype, m, deg))
end
