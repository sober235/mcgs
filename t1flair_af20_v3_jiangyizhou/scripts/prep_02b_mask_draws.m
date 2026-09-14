% Three more AF=20 draws of the house vd-Poisson mask, for the mask-draw variance of the MC-GS margin.
% Same call as prep_02_mask.m. vdPoisMex seeds from time(0) on every call, so the draws are separated by
% pause(2) and checked to differ. The registered mask (mask_vdpois_256x176_af20.mat, draw "a") is NOT touched.
addpath('/data0/congcong/code/ESPIRiT/utils');
outdir = '/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF20_v3_jiangyizhou/data';
accely = 4; calib = 24; ellipse = 0; pp = 1; af = 20; accelx = af/accely;
tags = {'b', 'c', 'd'};
masks = {};
for i = 1:numel(tags)
    pause(2);
    mask = single(vdPoisMex(256, 176, 256, 176, accelx, accely, calib, ellipse, pp) ~= 0);
    masks{i} = mask;
    fn = fullfile(outdir, sprintf('mask_vdpois_256x176_af20_d%s.mat', tags{i}));
    assert(~isfile(fn), 'refusing to overwrite %s', fn);
    save(fn, 'mask', '-v7');
    fprintf('draw %s: nnz %d AF_eff %.4f -> %s\n', tags{i}, nnz(mask), numel(mask)/nnz(mask), fn);
end
for i = 1:numel(tags)
    for j = i+1:numel(tags)
        assert(any(masks{i}(:) ~= masks{j}(:)), 'draws %s and %s are identical', tags{i}, tags{j});
    end
end
fprintf('all draws distinct\n');
exit
