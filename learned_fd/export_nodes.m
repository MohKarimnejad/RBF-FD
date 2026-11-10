%% Export the node sets of Test Problems 1 and 2 for the Python code (learned_fd/lfd.py).
%  Run in MATLAB or Octave:  run('learned_fd/export_nodes.m')
%  Writes learned_fd/nodes/TP<k>_<ntype>_m<level>.mat with the fields of nodes_TP<k>.
here = fileparts(mfilename('fullpath'));
addpath(fileparts(here));                 % repository root (nodes_TP1, nodes_TP2, haltonseq)
levels = {0:5, 0:4};                      % levels for TP1, TP2
for tp = 1:2
    for ntype = {'uniform', 'halton'}
        for m = levels{tp}
            if tp == 1, nd = nodes_TP1(m, ntype{1}); else, nd = nodes_TP2(m, ntype{1}); end
            Xint_p = nd.Xint_p; Xint_m = nd.Xint_m; Xbd_p = nd.Xbd_p; Xbd_m = nd.Xbd_m;
            XG = nd.XG; h = nd.h;
            file = fullfile(here, 'nodes', sprintf('TP%d_%s_m%d.mat', tp, ntype{1}, m));
            save(file, 'Xint_p', 'Xint_m', 'Xbd_p', 'Xbd_m', 'XG', 'h', '-v7');
        end
    end
end
