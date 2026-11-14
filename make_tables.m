clc; clear
%% Rebuild the LaTeX tables from the saved results (no re-running):
%  for every results/results_TP<k>.mat, per polynomial degree d:
%  results/TP<k>_deg<d>.tex (rms, max error and sigma).

outdir = 'results';
F = dir(fullfile(outdir, 'results_TP*.mat'));
for k = 1:numel(F)
    S = load(fullfile(outdir, F(k).name), 'R');
    write_tables(S.R, outdir);
    fprintf('%s: tables written for degrees %s\n', S.R.problem, mat2str(S.R.degs));
end
