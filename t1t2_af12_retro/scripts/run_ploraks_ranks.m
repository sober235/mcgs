function run_ploraks_ranks(infile, outprefix, ranks, R, workers, max_iter)
% run_ploraks_ranks -- rank sweep of P-LORAKS on an exported slice subset: one parfor over
% (rank, slice) pairs, one output file per rank (same layout as run_ploraks).
addpath('/data0/congcong/code/LORAKS_v2');
if nargin < 4 || isempty(R), R = 2; end
if nargin < 5 || isempty(workers), workers = 32; end
if nargin < 6 || isempty(max_iter), max_iter = 30; end
kre = h5read(infile, '/kre'); kim = h5read(infile, '/kim');
k = permute(complex(kre, kim), [4 3 2 1]); clear kre kim;         % nx x ky x kz x coil
mask = permute(h5read(infile, '/mask'), [2 1]);
nx = size(k, 1); nr = numel(ranks);
jobs = cell(nr * nx, 1); tj = zeros(nr * nx, 1);
p = gcp('nocreate'); if isempty(p) || p.NumWorkers ~= workers, delete(gcp('nocreate')); p = parpool('local', workers); end
fprintf('rank sweep: ranks %s, %d slices, R %d, max_iter %d, workers %d\n', mat2str(ranks), nx, R, max_iter, workers);
t0 = tic;
parfor j = 1:nr * nx
    ir = floor((j - 1) / nx) + 1; i = mod(j - 1, nx) + 1;
    kd = squeeze(k(i, :, :, :)); t1 = tic;
    jobs{j} = P_LORAKS(kd, mask, ranks(ir), R, 'S', 0, 4, 1e-3, max_iter);
    tj(j) = toc(t1);
end
fprintf('all done in %.0f s; per-slice time median %.0f s, max %.0f s\n', toc(t0), median(tj), max(tj));
for ir = 1:nr
    rec = zeros(size(k), 'like', k);
    for i = 1:nx, rec(i, :, :, :) = reshape(jobs{(ir - 1) * nx + i}, [1 size(jobs{(ir - 1) * nx + i})]); end
    rec = permute(rec, [4 3 2 1]);
    outfile = sprintf('%s_r%d.h5', outprefix, ranks(ir));
    if exist(outfile, 'file'), delete(outfile); end
    h5create(outfile, '/rre', size(rec), 'Datatype', 'single'); h5write(outfile, '/rre', single(real(rec)));
    h5create(outfile, '/rim', size(rec), 'Datatype', 'single'); h5write(outfile, '/rim', single(imag(rec)));
    h5create(outfile, '/rank', 1); h5write(outfile, '/rank', ranks(ir));
    h5create(outfile, '/R', 1); h5write(outfile, '/R', R);
    fprintf('wrote %s\n', outfile);
end
end
