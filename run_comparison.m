clc; clear; close all
%% Comparison of D-RBF-PU, RBF-FD, pQR, GMLS and GMLS-VSK on Test Problems 1-3 of
%  Davydov & Safarpoor (2021). For every problem, polynomial degree and node level:
%  rms and max error over the interior nodes, and sigma = condest(A')/norm(A,inf).
%  Output (folder outdir):
%    results_TP<k>.mat   : all numbers (struct R)
%    TP<k>_<q>_deg<d>.tex : LaTeX table (booktabs), one per quantity q = rms, max, sigma
%                          and degree d (see write_tables; make_tables redoes them from
%                          the .mat files without re-running)
%    TP<k>.csv           : all numbers in long format
%    TP<k>_conv.fig/.png : rms error vs N_int, one panel per degree

%% Parameters
probs   = [1 2 3];                                   % test problems
methods = {'pu', 'fd', 'pqr', 'gmls', 'gmls-vsk'};    % see solve_interface
mnames  = {'D-RBF-PU', 'RBF-FD', 'pQR', 'GMLS', 'GMLS-VSK'};
degs    = 2:6;               % polynomial degrees
mlist   = 0:4;               % node levels
msigma  = 4;                 % compute sigma only up to this level
nlocf   = @(Q) 3*Q;          % stencil size (FD, pQR, GMLS) / nodes per patch (PU), Q = dim P_deg
prm.beta  = 5;               % PHS exponent (D-RBF-PU, RBF-FD)
prm.ep    = 10;              % Matern shape parameter (GMLS, GMLS-VSK)
prm.nu    = 0.5;             % Matern order: 0.5, 1.5 or 2.5
prm.lam   = 1;               % amplitude of the scale function (GMLS-VSK)
prm.ntype = 'uniform';       % 'uniform' (grid, as in the paper) or 'halton'
prm.itype = 'contour';       % TP2 interface nodes: 'contour' or 'projection'
outdir  = 'results';

if ~exist(outdir, 'dir'), mkdir(outdir); end
nd_ = numel(degs); nm_ = numel(mlist); nM = numel(methods);

for p = probs
    pb = problem_TP(p);
    R.problem = pb.name; R.methods = mnames; R.degs = degs; R.mlist = mlist; R.prm = prm;
    R.nloc = arrayfun(@(d) nlocf((d+1)*(d+2)/2), degs);
    R.rms = nan(nd_, nm_, nM); R.max = R.rms; R.sigma = R.rms; R.time = R.rms;
    R.Nint = nan(nd_, nm_); R.h = nan(nd_, nm_);
    fprintf('\n===== %s, %s nodes, beta = %d, ep = %g, nu = %g, lam = %g =====\n', ...
        pb.name, prm.ntype, prm.beta, prm.ep, prm.nu, prm.lam);
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
                R.rms(id,im,iM) = o.rms;  R.max(id,im,iM) = o.max;
                R.sigma(id,im,iM) = o.sigma;  R.time(id,im,iM) = o.time;
                R.Nint(id,im) = o.Nint;  R.h(id,im) = o.h;
                fprintf('deg %d, m %d, %-9s  rms %9.2e  max %9.2e  sigma %9.2e  (%.1f s)\n', ...
                    degs(id), m, mnames{iM}, o.rms, o.max, o.sigma, o.time);
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
    R.h(sub2ind(size(R.h), id, im)), R.Nint(sub2ind(size(R.Nint), id, im)), ...
    R.methods(iM)', R.rms(:), R.max(:), R.sigma(:), R.time(:), ...
    'VariableNames', {'problem','deg','m','h','Nint','method','rms','max','sigma','time'});
writetable(T, fname);
end

function plot_conv(R, fbase)
% rms error vs N_int, one panel per degree, one line per method.
mk = {'o-', 's-', 'd-', '^-', 'v-'};
nd_ = numel(R.degs);
fig = figure('Position', [50 100 330*nd_ 360]);
tiledlayout(1, nd_, 'TileSpacing', 'compact');
for id = 1:nd_
    nexttile
    for iM = 1:numel(R.methods)
        loglog(R.Nint(id,:), R.rms(id,:,iM), mk{iM}, 'LineWidth', 1.2, 'MarkerSize', 5); hold on
    end
    grid on; xlabel('N_{int}');
    if id == 1, ylabel('rms error'); end
    title(sprintf('degree %d', R.degs(id)));
end
legend(R.methods, 'Location', 'southwest');
sgtitle(sprintf('%s: rms error vs N_{int}', R.problem));
savefig(fig, [fbase, '.fig']);
exportgraphics(fig, [fbase, '.png'], 'Resolution', 200);
end
