function run_ploraks(infile, outfile, rank, R, workers, max_iter)
% run_ploraks -- P-LORAKS (Haldar & Zhuo 2016; LORAKS v2 toolbox) on every exported readout
% slice, in parallel. Each slice is an independent 2-D multi-coil problem (ky x kz x coil).
% infile : HDF5 from ploraks_export.py (kre, kim: nx x ky x kz x coil in MATLAB order after
%          transposition; mask: ky x kz)
% outfile: HDF5 with the reconstructed multi-coil k-space (rre, rim), same layout as the input.
addpath('/data0/congcong/code/LORAKS_v2');
if nargin < 4 || isempty(R), R = 3; end
if nargin < 5 || isempty(workers), workers = 32; end
if nargin < 6 || isempty(max_iter), max_iter = 50; end
kre = h5read(infile, '/kre'); kim = h5read(infile, '/kim');      % h5read returns dims reversed: coil x kz x ky x nx
k = complex(kre, kim); clear kre kim;
k = permute(k, [4 3 2 1]);                                        % -> nx x ky x kz x coil
mask = h5read(infile, '/mask'); mask = permute(mask, [2 1]);      % -> ky x kz
nx = size(k, 1);
fprintf('run_ploraks: %d slices, k %s, rank %d, R %d, workers %d\n', nx, mat2str(size(k)), rank, R, workers);
rec = zeros(size(k), 'like', k);
p = gcp('nocreate'); if isempty(p) || p.NumWorkers ~= workers, delete(gcp('nocreate')); p = parpool('local', workers); end
t0 = tic;
parfor i = 1:nx
    kd = squeeze(k(i, :, :, :));                                  % ky x kz x coil
    if max(abs(kd(:))) == 0, rec(i, :, :, :) = kd; continue; end
    r = P_LORAKS(kd, mask, rank, R, 'S', 0, 4, 1e-3, max_iter);   % strict data consistency, alg 4 (FFT, approx.)
    rec(i, :, :, :) = reshape(r, [1 size(r)]);
end
fprintf('done in %.0f s\n', toc(t0));
rec = permute(rec, [4 3 2 1]);                                    % back to coil x kz x ky x nx for h5write
if exist(outfile, 'file'), delete(outfile); end
h5create(outfile, '/rre', size(rec), 'Datatype', 'single'); h5write(outfile, '/rre', single(real(rec)));
h5create(outfile, '/rim', size(rec), 'Datatype', 'single'); h5write(outfile, '/rim', single(imag(rec)));
h5create(outfile, '/rank', 1); h5write(outfile, '/rank', rank);
h5create(outfile, '/R', 1); h5write(outfile, '/R', R);
fprintf('wrote %s\n', outfile);
end
