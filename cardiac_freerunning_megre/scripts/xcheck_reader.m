function xcheck_reader(datfile, idxfile, outfile)
% Spec 5.1: read the imaging lines listed (1-based, acquisition order) in idxfile with the authors'
% reader (mapVBVD_JH, unsorted, flagIgnoreSeg) and save them for a bitwise comparison.
addpath('/data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/third_party/ReadDataAndTrajectory_FreeRunningFatFractionHeart/Code_Dependencies');
idx = readmatrix(idxfile);
twix = mapVBVD_JH(datfile);
if iscell(twix), twix = twix{end}; end
img = twix.image;
img.flagIgnoreSeg = true;
fprintf('image NAcq = %d, NCol = %d, NCha = %d\n', img.NAcq, img.NCol, img.NCha);
lines = img.unsorted(idx(:).');           % [NCol, NCha, numel(idx)] single complex
save(outfile, 'lines', '-v7');
fprintf('saved lines %s\n', mat2str(size(lines)));
end
