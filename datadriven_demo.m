%% DATADRIVEN_DEMO
% Demonstrates two data-driven / model-reduction techniques on a
% synthetic "flow-like" spatiotemporal dataset -- no simulation software
% needed, just a combination of traveling waves plus noise (mimicking a
% simple fluid/structural oscillation).
%   1) DMD (Dynamic Mode Decomposition): extract dominant spatial
%      patterns and their oscillation frequencies/growth rates directly
%      from snapshot data.
%   2) POD-based Reduced Order Model: compress a high-dimensional
%      "simulation" down to a handful of modes and show it still
%      reconstructs the data well.
%
% Requires: base MATLAB only

clear; clc; close all;

%% ---- Generate synthetic spatiotemporal data ("toy fluid flow") ----
% Two traveling waves at different spatial wavenumbers and temporal
% frequencies, superimposed, plus noise -- mimics a flow field with two
% dominant coherent structures (e.g., vortex shedding at 2 frequencies).
nx = 200;                          % spatial points
nt = 400;                          % time snapshots
x = linspace(0, 20, nx)';
t = linspace(0, 20, nt);
dt = t(2) - t(1);

mode1 = sech(x - 5) .* cos(2*pi*0.5*t);      % slow mode, freq 0.5 Hz
mode2 = 0.6*sech(x - 12) .* sin(2*pi*1.4*t); % faster mode, freq 1.4 Hz
noise = 0.02*randn(nx, nt);

Xdata = mode1 + mode2 + noise;     % nx x nt "snapshot matrix"

figure('Name','Synthetic Flow Data','Position',[100 100 700 350]);
imagesc(t, x, Xdata); axis xy;
xlabel('time'); ylabel('space'); title('Synthetic Spatiotemporal Data (our "flow field")');
colorbar; colormap(turbo);

%% ================= PART 1: DYNAMIC MODE DECOMPOSITION =================
% Standard exact-DMD algorithm (Schmid 2010) via SVD.
X1 = Xdata(:, 1:end-1);
X2 = Xdata(:, 2:end);

r = 6;   % truncation rank -- keep top r singular values
[U, S, V] = svd(X1, 'econ');
Ur = U(:, 1:r); Sr = S(1:r,1:r); Vr = V(:, 1:r);

Atilde = Ur' * X2 * Vr / Sr;             % low-rank operator
[W, D] = eig(Atilde);                     % eigenvalues = DMD growth/freq
Phi = X2 * Vr / Sr * W;                   % DMD modes (in full space)

omega = log(diag(D)) / dt;                % continuous-time eigenvalues
freqHz = imag(omega) / (2*pi);
growthRate = real(omega);

fprintf('DMD found %d modes. Top modes by frequency:\n', r);
[~, order] = sort(abs(freqHz));
for k = order'
    fprintf('  mode %d: freq = %6.3f Hz, growth rate = %7.4f\n', ...
        k, freqHz(k), growthRate(k));
end

figure('Name','DMD Modes','Position',[100 500 1000 400]);
subplot(1,2,1);
plot(real(diag(D)), imag(diag(D)), 'bo', 'MarkerSize', 8, 'LineWidth', 1.5);
hold on;
th = linspace(0,2*pi,100); plot(cos(th), sin(th), 'k--'); % unit circle
axis equal; grid on;
xlabel('Real'); ylabel('Imag');
title('DMD Eigenvalues (unit circle = neutral/stable oscillation)');

subplot(1,2,2);
[~, dominantIdx] = sort(abs(freqHz));
plot(x, real(Phi(:, dominantIdx(end))), 'LineWidth', 2); hold on;
plot(x, real(Phi(:, dominantIdx(end-1))), 'LineWidth', 2);
xlabel('space'); ylabel('mode shape');
legend('Fastest oscillating mode', '2nd fastest mode');
title('Extracted DMD Spatial Mode Shapes');

%% ---- Discussion prompts (DMD) ----
% 1) We built the data from exactly 2 traveling waves at 0.5 Hz and
%    1.4 Hz. Do the top DMD frequencies recover those numbers?
% 2) Eigenvalues sitting ON the unit circle mean "steady oscillation."
%    What would eigenvalues INSIDE the circle mean physically? Outside?
% 3) DMD needed NO governing equations -- just snapshots of data. Where
%    in your own CFD/experimental work could this replace needing a
%    full physics model?

%% ================= PART 2: POD-BASED REDUCED ORDER MODEL =================
% POD (Proper Orthogonal Decomposition) is essentially PCA applied to
% spatiotemporal snapshot data. We show that a handful of modes can
% reconstruct the full nx x nt dataset almost perfectly.
[Upod, Spod, Vpod] = svd(Xdata, 'econ');
singVals = diag(Spod);
energyCaptured = cumsum(singVals.^2) / sum(singVals.^2) * 100;

figure('Name','POD Reduced Order Model','Position',[1150 100 900 400]);
subplot(1,2,1);
plot(energyCaptured(1:20), 'o-', 'LineWidth', 2);
xlabel('Number of POD modes kept'); ylabel('% Energy Captured');
title('How Few Modes Do We Need?');
grid on; yline(99, 'r--', '99%');

% Reconstruct using only 4 modes (a "reduced order model")
rROM = 4;
Xrom = Upod(:,1:rROM) * Spod(1:rROM,1:rROM) * Vpod(:,1:rROM)';
reconError = norm(Xdata - Xrom, 'fro') / norm(Xdata, 'fro') * 100;

subplot(1,2,2);
imagesc(t, x, Xrom); axis xy;
xlabel('time'); ylabel('space');
title(sprintf('Reconstruction with only %d modes (%.2f%% relative error)', ...
    rROM, reconError));
colorbar; colormap(turbo);

fprintf('\nFull data: %d x %d = %d numbers.\n', nx, nt, nx*nt);
fprintf('ROM with %d modes: needs only %d numbers to reconstruct nearly perfectly.\n', ...
    rROM, rROM*(nx+nt+1));

%% ---- Discussion prompts (POD/ROM) ----
% 1) Compare the imagesc plot here to the very first "Synthetic Flow
%    Data" figure -- can you tell them apart using only 4 modes?
% 2) A full CFD simulation might have millions of grid points. If it
%    only needs ~10 POD modes to capture 99% of the energy, what does
%    that mean for running that model in REAL TIME (e.g., in a digital
%    twin or a control system)?
% 3) POD/ROM and PCA (from dimreduction_demo.m) use the exact same math
%    (SVD). What's different about how we're USING it here?
