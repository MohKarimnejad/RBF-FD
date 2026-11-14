clc; clear; close all
%% Comparison of D-RBF-PU, RBF-FD, pQR and GMLS on Test Problems 1-3 of
%  Davydov & Safarpoor (2021). For every problem, polynomial degree and node level:
%  max error over all unknowns (interior, boundary, and u+ and u- at the interface nodes),
%  and sigma = condest(A')/norm(A,inf).
%  Output (folder outdir):
%    results_TP<k>.mat   : all numbers (struct R)
%    TP<k>_deg<d>.tex     : LaTeX table per degree d: max error and sigma per method
%                          (see write_tables; make_tables redoes them from the .mat files)
%    TP<k>.csv           : all numbers in long format
%    TP<k>_conv.fig/.png : max error vs N (all nodes), one panel per degree

%% Parameters
probs   = [1 2 3];                                   % test problems
methods = {'pu', 'fd', 'pqr', 'gmls'};    % see solve_interface
mnames  = {'D-RBF-PU', 'RBF-FD', 'pQR', 'GMLS'};
degs    = 2:6;               % polynomial degrees
mlist   = 0:4;               % node levels
msigma  = 4;                 % compute sigma only up to this level
nlocf   = @(Q) 3*Q;          % stencil size (FD, pQR, GMLS) / nodes per patch (PU), Q = dim P_deg
prm.beta  = 5;               % PHS exponent (D-RBF-PU, RBF-FD)
prm.ep    = 10;              % Matern shape parameter (GMLS)
prm.nu    = 0.5;             % Matern order: 0.5, 1.5 or 2.5
prm.ntype = 'uniform';       % 'uniform' (grid, as in the paper) or 'halton'
prm.itype = 'contour';       % TP2 interface nodes: 'contour' or 'projection'
outdir  = 'results';

if ~exist(outdir, 'dir'), mkdir(outdir); end
nd_ = numel(degs); nm_ = numel(mlist); nM = numel(methods);

for p = probs
    pb = problem_TP(p);
    R.problem = pb.name; R.methods = mnames; R.degs = degs; R.mlist = mlist; R.prm = prm;
    R.nloc = arrayfun(@(d) nlocf((d+1)*(d+2)/2), degs);
    R.max = nan(nd_, nm_, nM); R.sigma = R.max; R.time = R.max;
    R.N = nan(nd_, nm_); R.Nint = R.N; R.h = R.N;
    fprintf('\n===== %s, %s nodes, beta = %d, ep = %g, nu = %g =====\n', ...
        pb.name, prm.ntype, prm.beta, prm.ep, prm.nu);
    for id = 1:nd_
        prm.deg  = degs(id);
        prm.nloc = R.nloc(id);
        for im = 1:nm_
            m = mlist(im);
            for iM = 1:nM
                try
                    o = solve_interface(pb, m, methods{iM}, prm, m <= msigma);
                catch err
                    fprintf('deg %d, m %d, %-9s failed: %s\n', degs(id), m, mnames{iM}, err.message);
                    continue
                end
                R.max(id,im,iM) = o.max;
                R.sigma(id,im,iM) = o.sigma;  R.time(id,im,iM) = o.time;
                R.N(id,im) = o.N;  R.Nint(id,im) = o.Nint;  R.h(id,im) = o.h;
                fprintf('deg %d, m %d, %-9s  N %7d  max %9.2e  sigma %9.2e  (%.1f s)\n', ...
                    degs(id), m, mnames{iM}, o.N, o.max, o.sigma, o.time);
            end
        end
    end
    save(fullfile(outdir, sprintf('results_%s.mat', pb.name)), 'R');
    write_tables(R, outdir);
    write_csv(R, fullfile(outdir, sprintf('%s.csv', pb.name)));
    plot_conv(R, fullfile(outdir, sprintf('%s_conv', pb.name)));
end

% ============================================================================
function write_csv(R, fname)
% Long format: one line per (degree, level, method).
[id, im, iM] = ndgrid(1:numel(R.degs), 1:numel(R.mlist), 1:numel(R.methods));
id = id(:); im = im(:); iM = iM(:);
T = table(repmat({R.problem}, numel(id), 1), R.degs(id)', R.mlist(im)', ...
    R.h(sub2ind(size(R.h), id, im)), R.N(sub2ind(size(R.N), id, im)), ...
    R.Nint(sub2ind(size(R.Nint), id, im)), R.methods(iM)', R.max(:), R.sigma(:), R.time(:), ...
    'VariableNames', {'problem','deg','m','h','N','Nint','method','max','sigma','time'});
writetable(T, fname);
end

function plot_conv(R, fbase)
% max error vs N (all nodes), one panel per degree, one line per method.
mk = {'o-', 's-', 'd-', '^-', 'v-', 'x-'};
nd_ = numel(R.degs);
fig = figure('Position', [50 100 330*nd_ 360]);
tiledlayout(1, nd_, 'TileSpacing', 'compact');
for id = 1:nd_
    nexttile
    for iM = 1:numel(R.methods)
        loglog(R.N(id,:), R.max(id,:,iM), mk{iM}, 'LineWidth', 1.2, 'MarkerSize', 5); hold on
    end
    grid on; xlabel('N');
    if id == 1, ylabel('max error'); end
    title(sprintf('degree %d', R.degs(id)));
end
legend(R.methods, 'Location', 'southwest');
sgtitle(sprintf('%s, %s nodes: max error vs N', R.problem, R.prm.ntype));
savefig(fig, [fbase, '.fig']);
exportgraphics(fig, [fbase, '.png'], 'Resolution', 200);
end
