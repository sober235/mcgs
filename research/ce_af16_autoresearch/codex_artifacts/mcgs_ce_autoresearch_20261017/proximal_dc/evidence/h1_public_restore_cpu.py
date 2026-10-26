"""Read-only actual H14500 public-restore witness, CPU only; no rendering/data fit."""
import importlib.util
import json
from pathlib import Path
import sys
import time
import types
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_px_restore_witness_runner',ROOT/'runner.py')
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
import torch
import numpy as np
started=time.perf_counter();definition=json.loads((ROOT/'evidence/h1_definition_v1.json').read_text())
before_files={p:m.sha256(p) for p in definition['files']}
assert before_files==definition['files']
checkpoint=torch.load(definition['checkpoint']['path'],map_location='cpu',weights_only=True,mmap=True)
module=m.source_module('gaussian.joint_gaussian_model','gaussian/joint_gaussian_model.py');state=checkpoint['model_state']
shape=np.load(definition['scans']['pre']['canonical_array']['path'],mmap_mode='r',allow_pickle=False).shape
model=module.JointGaussianModel3D(3600000,shape,'cpu',initial_positions=state['positions'],
    initial_densities_plain=torch.complex(state['density_real'],state['density_imag']),
    initial_densities_contrast=torch.complex(state['density_real_contrast'],state['density_imag_contrast']),
    initial_scales=torch.exp(state['scales']))
optimizer=torch.optim.Adam(model.get_optimizable_params(lr_position=.0003,lr_scale=.002,lr_rotation=.0003,lr_density=.015))
scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=.9995)
trainer=types.SimpleNamespace(model=model,optimizer=optimizer,scheduler=scheduler,config={'CPU_witness_only':True},train=lambda:None)
restore=m.restore_export_only(trainer,checkpoint,definition['frequency_adam_recipe'],definition['checkpoint'])
restored=m.snapshot(trainer)
for key in ('parameter_sha256','adam','model_sha256','geometry_sha256','density_sha256'):assert restored[key]==definition['expected_model_state'][key],key
metadata=m.load(ROOT.parent/'frequency_adam/overrides.py','_px_restore_witness_metadata')._snapshot(optimizer,scheduler)
assert metadata==definition['expected_optimizer_actual']
with m.export_only(trainer) as calls:pass
after=m.snapshot(trainer)
assert after==restored
assert {p:m.sha256(p) for p in definition['files']}==before_files
record=dict(status='actual_H14500_public_restore_CPU_witness',runtime=m.numerical_runtime(),model_definition_sha256=m.sha256(ROOT/'evidence/h1_definition_v1.json'),
    checkpoint=definition['checkpoint'],iteration=4500,source_model_and_moment_hashes_match=True,optimizer_actual=metadata,
    parameter_identities_preserved=restore['parameter_identities_before']==restore['parameter_identities_after'],export_only_state_unchanged=True,
    forbidden_calls=calls,protected_original_files_unchanged=len(before_files),gaussian_renders=0,training_updates=0,clinical_GPU_or_score=False,
    elapsed_seconds=time.perf_counter()-started)
path=ROOT/'evidence/h1_public_restore_cpu_v1.json'
with path.open('x') as stream:json.dump(record,stream,indent=2,allow_nan=False);stream.write('\n')
print(json.dumps(record,indent=2))
