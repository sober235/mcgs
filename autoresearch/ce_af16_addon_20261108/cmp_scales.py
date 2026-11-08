import sys, torch, numpy as np
R = '/data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_ce_threeaxis_20261107/runs/%s/s42/training/fixed_count_relocation/checkpoints/latest.pth'
a = torch.load(R % 'TO02_lambda_nl_plain_1_on_tn15', map_location='cpu', weights_only=True)['model_state']
b = torch.load(R % 'TP02_scale_lr_30', map_location='cpu', weights_only=True)['model_state']
assert a['positions'].shape == b['positions'].shape
d = (b['scales'] - a['scales']).double()                      # log-scale change per local axis
w = (b['density_real'].double()**2 + b['density_imag'].double()**2)
w = w / w.sum()
mx = d.abs().max(1).values
print('points', d.shape[0], 'pos max |dx| (norm units)', float((b['positions']-a['positions']).abs().max()))
print('|dlog s| quantiles (any axis) q50 q90 q99 q999:', [round(float(torch.quantile(mx[::7], q)), 4) for q in (.5, .9, .99, .999)])
print('mean dlog s (local axes):', [round(float(x), 5) for x in d.mean(0)])
print('weighted mean dlog s:', [round(float((w[:, None]*d).sum(0)[i]), 5) for i in range(3)])
print('fraction shrunk >10%% on some axis: %.4f  grown >10%%: %.4f' % (float((d < -0.105).any(1).double().mean()), float((d > 0.0953).any(1).double().mean())))
print('weight share of points that changed >10%%: %.4f' % float(w[(mx > 0.1)].sum()))
ra, rb = a['rotations'].double(), b['rotations'].double()
ra = ra / ra.norm(dim=1, keepdim=True); rb = rb / rb.norm(dim=1, keepdim=True)
ang = 2 * torch.acos((ra*rb).sum(1).abs().clamp(max=1))
print('rotation change deg q50 q99:', [round(float(torch.quantile(ang[::7], q))*57.3, 3) for q in (.5, .99)])
print('anisotropy max/min scale q50 q90 (TO02, TP02):', [round(float(torch.quantile((x.max(1).values-x.min(1).values)[::7], q)), 4) for x in (a['scales'], b['scales']) for q in (.5, .9)])
