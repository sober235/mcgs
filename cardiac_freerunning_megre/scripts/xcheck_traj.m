function xcheck_traj(outfile)
% Spec 5.3: the authors' trajectory for this protocol (N 224, 22 segments, 1000 shots, self-navigation).
addpath('/data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/third_party/ReadDataAndTrajectory_FreeRunningFatFractionHeart/Code_Dependencies');
[kx, ky, kz] = computePhyllotaxis(224, 22, 1000, false);
save(outfile, 'kx', 'ky', 'kz', '-v7');
fprintf('saved kx %s\n', mat2str(size(kx)));
end
