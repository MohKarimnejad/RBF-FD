clc; clear; close all
%% Compare the Laplacian weights of FD [pqr] and D-PU [phs] on Test Problem 2.
%  Both methods are exact for polynomials of degree deg on the same nodes, so the
%  difference d_i = w_i^PU - w_i^pQR (row i, zero-padded to the union of supports)
%  annihilates polynomials. Per interior node x_i:
%    delta_i = ||d_i||_{1,mu} / ||w_i^pQR||_{1,mu},  ||v||_{1,mu} = sum_j |v_j| ||x_j - x_i||^mu
%              (relative weighted difference, Davydov-Schaback seminorm, mu = deg)
%    tau_i   = sum_j w_ij u(x_j) - Lap u(x_i)       (local truncation error of each method)
%    dtau_i  = |tau_i^PU - tau_i^pQR| = |sum_j d_ij u(x_j)|

beta   = 5;              % PHS exponent for D-PU [phs]
deg    = 3;              % polynomial degree (both methods)
Q      = (deg+1)*(deg+2)/2;
nloc   = 2*Q;            % stencil size (FD) / nodes per patch (PU)
ntype  = 'halton';       % node type: 'uniform' or 'halton'
mlist  = 2:5;            % node levels
mplot  = [3 4 5];        % levels with maps
mu     = deg;            % exponent of the weighted seminorm (as in the growth function)

fprintf('FD [pqr] vs D-PU [phs r^%d], degree %d (Q = %d), nloc = %d, %s nodes, mu = %d\n', ...
    beta, deg, Q, nloc, ntype, mu);
fprintf('%3s %8s %10s %10s %10s %11s %11s %11s\n', 'm', 'h', 'med delta', 'max delta', ...
    'max dtau', 'max|tau|FD', 'max|tau|PU', 'rms|tau|PU/FD');
for m = mlist
    oF = PHS_fd_TP2(m, beta, deg, nloc, false, ntype, 'pqr');
    oP = PHS_drbfpu_TP2(m, beta, deg, nloc, false, ntype, 'phs');

    D  = oP.Lw - oF.Lw;                              % difference of the weight rows
    nL = size(D, 1);
    [r, c, w] = find(D);                             % ||d_i||_{1,mu}
    sD = accumarray(r, abs(w) .* sqrt(sum((oF.Xu(c,:) - oF.Xr(r,:)).^2, 2)).^mu, [nL, 1]);
    [r, c, w] = find(oF.Lw);                         % ||w_i^pQR||_{1,mu}
    sF = accumarray(r, abs(w) .* sqrt(sum((oF.Xu(c,:) - oF.Xr(r,:)).^2, 2)).^mu, [nL, 1]);
    delta = sD ./ sF;

    tauF = oF.Lw*oF.uex - oF.lapu;
    tauP = oP.Lw*oP.uex - oP.lapu;
    dtau = abs(tauP - tauF);

    fprintf('%3d %8.5f %10.3e %10.3e %10.3e %11.3e %11.3e %11.3f\n', m, oF.h, median(delta), ...
        max(delta), max(dtau), max(abs(tauF)), max(abs(tauP)), ...
        sqrt(mean(tauP.^2))/sqrt(mean(tauF.^2)));

    if ismember(m, mplot)
        X = oF.Xr;
        figure('Position', [100 100 1100 900])
        subplot(2,2,1)
        scatter(X(:,1), X(:,2), 6, log10(delta), 'filled'); axis equal tight; box on; colorbar
        title('log_{10} \delta_i = ||w^{PU}-w^{pQR}||_{1,\mu} / ||w^{pQR}||_{1,\mu}')
        subplot(2,2,2)
        scatter(X(:,1), X(:,2), 6, log10(dtau), 'filled'); axis equal tight; box on; colorbar
        title('log_{10} |\tau^{PU} - \tau^{pQR}|')
        subplot(2,2,3)
        scatter(X(:,1), X(:,2), 6, log10(abs(tauF)), 'filled'); axis equal tight; box on; colorbar
        title('log_{10} |\tau^{pQR}|')
        subplot(2,2,4)
        scatter(X(:,1), X(:,2), 6, log10(abs(tauP)), 'filled'); axis equal tight; box on; colorbar
        title('log_{10} |\tau^{PU}|')
        sgtitle(sprintf('Test problem 2, FD [pqr] vs D-PU [phs], %s nodes, level %d, degree %d', ...
            ntype, m, deg))
    end
end

