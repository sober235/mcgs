% prep_mask_draws.m -- pool of house vd-Poisson draws on the (PE, SPE) = (304, 104) plane for the
% pediatric MTP AF-12 study (spec 2026-09-12-16yo-mtp-stage2-af12-design.md, section 4).
% Nothing is chosen here: scripts/prep_mask.py picks the draw whose count inside the acquired
% support first reaches the budget 2635 and trims it to exactly that.
% Isotropic accelx = accely (unlike the 256x176 adult topics' fixed accely = 4): this plane has only
% 104 partitions and a fixed 4x along it would starve the short axis.
% vdPoisMex reseeds from time(0) at ONE-SECOND resolution, hence the pause before every draw.
addpath('/data0/congcong/code/ESPIRiT/utils');
outdir = '/data0/congcong/code/GS/Results_20260908/Peds_TSC_MTP_AF12_retro/data/mask_draws';
if ~exist(outdir, 'dir'), mkdir(outdir); end
calib = 24; ellipse = 0; pp = 1;
accel_list = 2.6:0.05:3.6;
ndraw = 5;
k = 0;
for a = accel_list
    for d = 1:ndraw
        k = k + 1;
        pause(1.1);
        mask = vdPoisMex(304, 104, 304, 104, a, a, calib, ellipse, pp);
        mask = single(mask ~= 0);
        accelx = a; accely = a;
        n = nnz(mask);
        fn = fullfile(outdir, sprintf('draw_%03d.mat', k));
        save(fn, 'mask', 'accelx', 'accely', 'calib', 'ellipse', 'pp', '-v7');
        fprintf('draw %03d accel %.2f nnz %d samp %.4f -> %s\n', k, a, n, mean(mask(:)), fn);
    end
end
exit
