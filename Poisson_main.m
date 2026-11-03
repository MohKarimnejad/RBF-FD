clc; clear; close all
%% D-RBF-PU or RBF-FD (PHS + polynomials) for the Poisson problem
%    -Lap u = f in (0,1)^2,  u = g on the boundary,
%    u(x,y) = sin(2 pi x) cos(2 pi y) + exp(x + y/2).
%  Errors: rms and max over the interior nodes X_int;
%  sigma = condest(A')/norm(A,inf).

solver = 'FD';           % solver: 'PU' (D-RBF-PU) or 'FD' (RBF-FD)
beta   = 5;              % PHS exponent, phi(r) = r^beta
deg    = 4;              % polynomial degree
Q      = (deg+1)*(deg+2)/2;
nloc   = 3*Q;            % nodes per patch (PU) / stencil size (FD)
ntype  = 'uniform';      % node type: 'uniform' (grid) or 'halton'
mlist  = 0:5;            % node levels X^(m), h = 0.2/2^m
msigma = 4;              % compute sigma only up to this level (extra LU, memory)
mplot  = [3 4 5];        % levels used for the solution / error surfaces

switch upper(solver)
    case 'PU', solve = @PHS_drbfpu_Poisson; sname = 'D-RBF-PU';
    case 'FD', solve = @PHS_fd_Poisson;     sname = 'RBF-FD';
    otherwise, error('solver must be ''PU'' or ''FD''.');
end

fprintf('%s, PHS r^%d, degree %d (Q = %d), nloc = %d, %s nodes\n', sname, beta, deg, Q, nloc, ntype);
fprintf('%3s %8s %7s %10s %10s %10s %8s %9s %8s\n', 'm', 'h', 'Nint', 'rms', 'max', ...
    'sigma', 'nnz/row', 'min nloc', 'time[s]');
res = nan(numel(mlist), 4);
for k = 1:numel(mlist)
    m = mlist(k);
    try
        o = solve(m, beta, deg, nloc, m <= msigma, ntype);
    catch err
        fprintf('%3d  failed: %s\n', m, err.message); continue
    end
    res(k,:) = [o.Nint, o.h, o.rms, o.max];
    fprintf('%3d %8.5f %7d %10.3e %10.3e %10.2e %8.1f %9d %8.1f\n', m, o.h, o.Nint, ...
        o.rms, o.max, o.sigma, o.nnz_row, min(o.nlocal), o.time);
    if ismember(m, mplot), op{m+1} = o; end
end

%% Convergence order of the max error:
%  slope of the least-squares line fitted to (ln h_m, ln max_m)
ok = ~isnan(res(:,1));
p  = polyfit(log(res(ok,2)), log(res(ok,4)), 1);
fprintf('Estimated convergence order (max): %.1f\n', p(1));

%% rms error vs N_int
figure
loglog(res(:,1), res(:,3), 'o-', 'LineWidth', 1.2); hold on
i0 = find(ok, 1);
loglog(res(:,1), res(i0,3)*(res(:,1)/res(i0,1)).^(-1), 'r--')    % h^2 = O(N^-1)
loglog(res(:,1), res(i0,3)*(res(:,1)/res(i0,1)).^(-2), 'r-.')    % h^4 = O(N^-2)
grid on; xlabel('N_{int}'); ylabel('rms error')
legend(sname, 'h^2', 'h^4', 'Location', 'southwest')
title(sprintf('%s, Poisson, PHS r^%d, degree %d, nloc = %d', sname, beta, deg, nloc))

%% Numerical solution and absolute error
for m = mplot
    o   = op{m+1};
    tri = delaunay(o.Xall(:,1), o.Xall(:,2));
    figure('Position', [100 100 1100 420])
    subplot(1,2,1)
    trisurf(tri, o.Xall(:,1), o.Xall(:,2), o.uall, 'EdgeColor', 'none')
    title('Numerical solution'); view(-37.5, 30); axis tight
    subplot(1,2,2)
    trisurf(tri, o.Xall(:,1), o.Xall(:,2), o.eall, 'EdgeColor', 'none')
    title('Absolute error'); view(-37.5, 30); axis tight
    sgtitle(sprintf('%s, %s nodes, level %d', upper(solver), ntype, m))
end
