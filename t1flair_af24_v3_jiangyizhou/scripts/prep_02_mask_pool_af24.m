% prep_02_mask_pool_af24.m -- a pool of house-convention vd-Poisson draws for the jiangyizhou AF24 topic (2026-09-14).
% AF = PE-plane points / sampled points that carry that contrast's data (user decision 2026-09-14). T1's acquired
% support is the whole (ky,kz) = (256,176) plane, so T1 at exactly 24x needs 1877 of 45056 points (45056/1877 = 24.004).
% FLAIR reads the same mask; its vendor shutter leaves the corners unacquired, so FLAIR lands above 24x.
% Generator and settings are those of the AF12 / AF16 / AF20 masks: vdPoisMex(256,176,256,176, accelx, 4, 24, 0, 1).
% Only accelx differs. A draw's point count is random (the four AF20 draws at accelx 5 have 2400-2427 points), so the
% pool is drawn around the accelx that gives 1877 points on average, and prep_03_pick_mask_af24.py takes a draw with
% exactly 1877 points, untrimmed. Nothing is chosen here.
%   stage A: accelx 6.40:0.10:7.00, 2 draws each, least-squares fit n = c + a/accelx -> accelx* = a/(1877 - c)
%   stage B: 120 draws cycling through accelx* + (-0.030:0.005:0.030)
% vdPoisMex reseeds from time(0) at one-second resolution (two calls inside the same second return the same pattern,
% verified 2026-09-12), hence pause(1.1) before every call. An existing pool is never added to or overwritten.
% Run: matlab -batch "run('<this file>')"
addpath('/data0/congcong/code/ESPIRiT/utils');
R = '/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF24_v3_jiangyizhou';
outdir = fullfile(R, 'data', 'mask_pool_af24');
if exist(outdir, 'dir') && ~isempty(dir(fullfile(outdir, 'draw_*.mat')))
    error('a pool already exists in %s; refusing to add to or overwrite it', outdir);
end
if ~exist(outdir, 'dir'), mkdir(outdir); end
TARGET = 1877;
accely = 4; calib = 24; ellipse = 0; pp = 1;
k = 0;
stageA = repelem(6.40:0.10:7.00, 2);
nA = zeros(size(stageA));
for i = 1:numel(stageA)
    k = k + 1;
    nA(i) = one_draw(outdir, k, 'A', stageA(i), accely, calib, ellipse, pp, TARGET);
end
coef = [ones(numel(stageA), 1), 1 ./ stageA(:)] \ nA(:);
accel_star = coef(2) / (TARGET - coef(1));
fprintf('stage A fit: n = %.1f + %.1f / accelx  ->  accelx* = %.4f for %d points\n', coef(1), coef(2), accel_star, TARGET);
offs = -0.030:0.005:0.030;
stageB = repmat(accel_star + offs, 1, ceil(120 / numel(offs)));
stageB = stageB(1:120);
for i = 1:numel(stageB)
    k = k + 1;
    one_draw(outdir, k, 'B', stageB(i), accely, calib, ellipse, pp, TARGET);
end
fprintf('POOL_DONE %d draws in %s\n', k, outdir);

function n = one_draw(outdir, k, stage, accelx, accely, calib, ellipse, pp, TARGET)
    pause(1.1);                                   % fresh time(0) seed for every call
    mask = vdPoisMex(256, 176, 256, 176, accelx, accely, calib, ellipse, pp);
    mask = single(mask ~= 0);
    n = nnz(mask);
    fn = fullfile(outdir, sprintf('draw_%03d.mat', k));
    save(fn, 'mask', 'accelx', 'accely', 'calib', 'ellipse', 'pp', 'stage', '-v7');
    hit = '';
    if n == TARGET, hit = '  <- exactly the target'; end
    fprintf('draw %03d stage %s accelx %.4f nnz %d AF_T1 %.3f%s\n', k, stage, accelx, n, numel(mask) / n, hit);
end
