function run_loraks_probe(infile, outprefix, ranks, R, workers, max_iter, variant, slice_sel, lambda)
% run_loraks_probe -- rank sweep of P-LORAKS or AC-LORAKS (LORAKS v2 toolbox) on an exported
% slice subset; one parfor over (rank, slice) pairs; one output file per rank (layout of run_ploraks).
%   variant   'P' (P_LORAKS, iterative nonconvex rank penalty, strict DC) or 'AC' (AC_LORAKS,
%             nullspace from the fully sampled centre, linear solve)
%   slice_sel optional 1-based indices into the exported slices (default: all)
addpath('/data0/congcong/code/LORAKS_v2');
if nargin < 4 || isempty(R), R = 2; end
if nargin < 5 || isempty(workers), workers = 24; end
if nargin < 6 || isempty(max_iter), max_iter = 30; end
if nargin < 7 || isempty(variant), variant = 'P'; end
kre = h5read(infile, '/kre'); kim = h5read(infile, '/kim');
k = permute(complex(kre, kim), [4 3 2 1]); clear kre kim;         % nx x ky x kz x coil
mask = permute(h5read(infile, '/mask'), [2 1]);
if nargin < 8 || isempty(slice_sel), slice_sel = 1:size(k, 1); end
if nargin < 9 || isempty(lambda), lambda = 1e-3; end
k = k(slice_sel, :, :, :);
S = [];
if strcmp(variant, 'SENSE')
    sre = h5read(infile, '/sre'); sim = h5read(infile, '/sim'); S = permute(complex(sre, sim), [4 3 2 1]); clear sre sim; S = S(slice_sel, :, :, :);
end
nx = size(k, 1); nr = numel(ranks);
jobs = cell(nr * nx, 1); tj = zeros(nr * nx, 1);
p = gcp('nocreate'); if isempty(p) || p.NumWorkers ~= workers, delete(gcp('nocreate')); p = parpool('local', workers); end
fprintf('%s-LORAKS probe: ranks %s, %d slices, R %d, max_iter %d, workers %d\n', variant, mat2str(ranks), nx, R, max_iter, workers);
t0 = tic;
parfor j = 1:nr * nx
    ir = floor((j - 1) / nx) + 1; i = mod(j - 1, nx) + 1;
    kd = squeeze(k(i, :, :, :)); t1 = tic;
    if strcmp(variant, 'AC')
        jobs{j} = AC_LORAKS(kd, mask, ranks(ir), R, 'S', 0, [], 1e-3, max_iter);
    elseif strcmp(variant, 'SENSE')
        jobs{j} = SENSE_LORAKS(kd, mask, squeeze(S(i, :, :, :)), ranks(ir), lambda, R, 'S', 4, 1e-3, max_iter);   % returns the image
    else
        jobs{j} = P_LORAKS(kd, mask, ranks(ir), R, 'S', 0, 4, 1e-3, max_iter);
    end
    tj(j) = toc(t1);
end
fprintf('all done in %.0f s; per-slice time median %.0f s, max %.0f s\n', toc(t0), median(tj), max(tj));
for ir = 1:nr
    outfile = sprintf('%s_r%d.h5', outprefix, ranks(ir));
    if exist(outfile, 'file'), delete(outfile); end
    if strcmp(variant, 'SENSE')
        x = zeros([nx size(k, 2) size(k, 3)], 'like', k);
        for i = 1:nx, x(i, :, :) = reshape(jobs{(ir - 1) * nx + i}, [1 size(k, 2) size(k, 3)]); end
        x = permute(x, [3 2 1]);
        h5create(outfile, '/xre', size(x), 'Datatype', 'single'); h5write(outfile, '/xre', single(real(x)));
        h5create(outfile, '/xim', size(x), 'Datatype', 'single'); h5write(outfile, '/xim', single(imag(x)));
        h5create(outfile, '/rank', 1); h5write(outfile, '/rank', ranks(ir)); h5create(outfile, '/R', 1); h5write(outfile, '/R', R); h5create(outfile, '/lambda', 1); h5write(outfile, '/lambda', lambda);
        fprintf('wrote %s\n', outfile); continue
    end
    rec = zeros(size(k), 'like', k);
    for i = 1:nx, rec(i, :, :, :) = reshape(jobs{(ir - 1) * nx + i}, [1 size(jobs{(ir - 1) * nx + i})]); end
    rec = permute(rec, [4 3 2 1]);
    h5create(outfile, '/rre', size(rec), 'Datatype', 'single'); h5write(outfile, '/rre', single(real(rec)));
    h5create(outfile, '/rim', size(rec), 'Datatype', 'single'); h5write(outfile, '/rim', single(imag(rec)));
    h5create(outfile, '/rank', 1); h5write(outfile, '/rank', ranks(ir));
    h5create(outfile, '/R', 1); h5write(outfile, '/R', R);
    h5create(outfile, '/slices_sel', numel(slice_sel)); h5write(outfile, '/slices_sel', slice_sel);
    fprintf('wrote %s\n', outfile);
end
end
