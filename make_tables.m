clc; clear
%% Rebuild the LaTeX tables from the saved results (no re-running):
%  for every results/results_TP<k>.mat, one table per quantity (rms, max, sigma) and
%  polynomial degree: results/TP<k>_<q>_deg<d>.tex (see write_tables).

outdir = 'results';
drop   = {'GMLS-VSK'};          % methods left out of the tables
F = dir(fullfile(outdir, 'results_TP*.mat'));
for k = 1:numel(F)
    S = load(fullfile(outdir, F(k).name), 'R');
    keep = ~ismember(S.R.methods, drop);
    S.R.methods = S.R.methods(keep);
    for q = {'rms', 'max', 'sigma', 'time'}
        S.R.(q{1}) = S.R.(q{1})(:,:,keep);
    end
    write_tables(S.R, outdir);
    fprintf('%s: tables written for degrees %s\n', S.R.problem, mat2str(S.R.degs));
end
