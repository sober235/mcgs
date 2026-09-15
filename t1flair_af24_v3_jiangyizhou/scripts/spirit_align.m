% spirit_align.m -- reference L1-SPIRiT on exported slices, for scripts/spirit_align_check.py.
% Runs Lustig's original toolbox, nothing else, and saves what the GPU port must reproduce.
% usage: matlab -batch "infile='...'; outfile='...'; run('.../spirit_align.m')"
E = '/data0/congcong/code/ESPIRiT';
addpath(E); addpath(fullfile(E, 'utils')); addpath(fullfile(E, 'SPIRiT_code'));

load(infile);                        % kCalib, DATA (sx,sy,nc), kSize, CalibTyk, nIter, wavWeight
nc = size(DATA, 3);
kernel = calibSPIRiT(kCalib, double(kSize), nc, double(CalibTyk));
GOP = SPIRiT(kernel, 'fft', [size(DATA, 1), size(DATA, 2)]);
gx = GOP * DATA;                     % (G - I) applied to the measured data
res = pocsSPIRiT(DATA, GOP, double(nIter), DATA, double(wavWeight), 0);
save(outfile, 'kernel', 'gx', 'res', '-v7.3');
