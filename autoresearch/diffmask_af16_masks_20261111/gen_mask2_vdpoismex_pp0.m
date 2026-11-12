% gen_mask2_vdpoismex_pp0.m -- mask 2 of the redone diff-mask study, from Michael Lustig's ESPIRiT toolbox (2026-11-11).
% Generator: vdPoisMex of /data0/congcong/code/ESPIRiT/utils (the compiled mex the house masks come from, not rebuilt).
% Call: the house one, vdPoisMex(256,176,256,176, accelx, 4, 24, 0, pp), with ONE setting changed: pp = 0 instead of 1.
% pp is the polynomial order of the variable density; the toolbox's own help says "0 = uniform". So the calibration
% block is fully sampled (calib 24 -> 24 x 16 on this plane) and the rest of the plane is sampled at one density
% everywhere, instead of a density that rises towards the centre.
% Budget: exactly TARGET = 2964 of 45056 points (T1 15.20x), the September diff-mask study's budget. A draw's point
% count is random, so, as for the AF24 house mask:
%   stage A: accelx 3.90:0.10:4.80, one draw each, least-squares fit n = c + a/accelx -> accelx* = a/(TARGET - c)
%   stage B: draws cycling through accelx* + (-0.030:0.005:0.030) until the first draw with exactly TARGET points
%            (at most 240 draws; if none hits, the first draw closest to TARGET is taken and reported as such).
% The draw is chosen by its point count alone and is not trimmed. vdPoisMex seeds from time(0), hence pause(1.1) before
% every call; the mask cannot be regenerated bit for bit, the saved file is the record.
% Output directory: environment variable MASK2_OUT. Run: matlab -batch "run('<this file>')"
addpath('/data0/congcong/code/ESPIRiT/utils');
outdir = getenv('MASK2_OUT');
assert(~isempty(outdir), 'MASK2_OUT is not set');
TARGET = 2964;
accely = 4; calib = 24; ellipse = 0; pp = 0;
logf = fopen(fullfile(outdir, 'mask2_vdpoismex_pp0_draws.txt'), 'w');
k = 0;
stageA = 3.90:0.10:4.80;
nA = zeros(size(stageA));
best = struct('n', -1, 'mask', [], 'accelx', NaN, 'k', 0, 'stage', '');
for i = 1:numel(stageA)
    k = k + 1;
    [nA(i), m] = one_draw(logf, k, 'A', stageA(i), accely, calib, ellipse, pp, TARGET);
end
coef = [ones(numel(stageA), 1), 1 ./ stageA(:)] \ nA(:);
accel_star = coef(2) / (TARGET - coef(1));
msg = sprintf('stage A fit: n = %.1f + %.1f / accelx  ->  accelx* = %.4f for %d points', coef(1), coef(2), accel_star, TARGET);
fprintf('%s\n', msg); fprintf(logf, '%s\n', msg);
offs = -0.030:0.005:0.030;
hit = false;
for i = 1:240
    k = k + 1;
    ax = accel_star + offs(mod(i - 1, numel(offs)) + 1);
    [n, m] = one_draw(logf, k, 'B', ax, accely, calib, ellipse, pp, TARGET);
    if best.n < 0 || abs(n - TARGET) < abs(best.n - TARGET)
        best = struct('n', n, 'mask', m, 'accelx', ax, 'k', k, 'stage', 'B');
    end
    if n == TARGET, hit = true; break; end
end
mask = best.mask; accelx = best.accelx; draw = best.k; n_draws = k; exact_hit = hit; n_points = best.n;
save(fullfile(outdir, 'mask_vdpoismex_pp0_256x176_af16_preview.mat'), 'mask', 'accelx', 'accely', 'calib', 'ellipse', ...
     'pp', 'draw', 'n_draws', 'exact_hit', 'n_points', 'accel_star', '-v7');
msg = sprintf('MASK2_DONE draw %03d of %d, accelx %.4f, nnz %d, exact %d', draw, n_draws, accelx, n_points, exact_hit);
fprintf('%s\n', msg); fprintf(logf, '%s\n', msg);
fclose(logf);

function [n, mask] = one_draw(logf, k, stage, accelx, accely, calib, ellipse, pp, TARGET)
    pause(1.1);                                   % fresh time(0) seed for every call
    mask = vdPoisMex(256, 176, 256, 176, accelx, accely, calib, ellipse, pp);
    mask = single(mask ~= 0);
    n = nnz(mask);
    tag = '';
    if n == TARGET, tag = '  <- exactly the target'; end
    msg = sprintf('draw %03d stage %s accelx %.4f nnz %d AF_T1 %.3f%s', k, stage, accelx, n, numel(mask) / n, tag);
    fprintf('%s\n', msg); fprintf(logf, '%s\n', msg);
end
