"""Independent CPU probes. All writes are under a fresh /tmp directory."""
import ast, importlib.util, json, os, sys, tempfile, types
from pathlib import Path
from unittest.mock import patch
import torch
P=Path('/home/congcongliu/codex_artifacts/mcgs_ce_autoresearch_20261017/proximal_dc')
def load(p,n):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
m=load(P/'runner.py','independent_probe_runner');t=load(P/'tests/test_runner.py','independent_probe_fixtures')
trainer=t.cpu_export_fixture();source=m.source_module('trainers.joint_trainer','trainers/joint_trainer.py')
def ds(_):
 return types.SimpleNamespace(volume_shape=(6,6,6),kspace_undersampled_complex=torch.zeros((2,6,6,6),dtype=torch.complex64),sensitivity_maps=torch.ones((2,6,6,6),dtype=torch.complex64),mask=torch.ones((6,6,6)))
trainer.config={'data':{'plain':{},'contrast':{}}};trainer.device=torch.device('cpu');seen=[]
old=source.ifft3c
def observe(x): seen.append(len(x));return old(x)
with patch.object(source,'_load_dataset',ds),patch.object(source,'ifft3c',observe):source.JointTrainer._setup_data(trainer)
controls,canonical=m.canonical_control(trainer,backend_kind='cpu_test')
sup=m.supervisor();zeros=dict.fromkeys(sup.PHYSICS_COUNTERS,0)
scans={s:{'n_coils':2,'work':zeros} for s in ('pre','post')}
total=m.total_work(canonical,scans);sup.validate_work_total({'scans':scans,'work_total':total},'canonical')
print(json.dumps({'probe':'constructor_work','actual_constructor_ifft_batches':len(seen),'actual_constructor_coil_ifft_transforms':sum(seen),'actual_export_ifft_batches':canonical['canonical_ifft_batches'],'reported_total_ifft_batches':total['total_ifft_batches'],'actual_combined_ifft_batches':len(seen)+canonical['canonical_ifft_batches'],'reported_total_coil_ifft_transforms':total['total_coil_ifft_transforms'],'actual_combined_coil_ifft_transforms':sum(seen)+canonical['canonical_coil_ifft_transforms'],'parent_validation_accepted':True}))
root=Path(tempfile.mkdtemp(prefix='proximal_review_failure_round1_'));output=root/'outputs';output.mkdir();exp='synthetic_failure_only';review=root/'review_unused.json';model=root/'model_unused.json';launchpath=root/'launch_unused.json'
I={'accepted_h1':{'definition_manifest':{},'definition':{}}};hashes={};tokens=['--stage','prox','--exp',exp,'--output',str(output),'--launch-manifest',str(launchpath),'--review-gate',str(review),'--model-definition',str(model)]
reserved=types.SimpleNamespace(stage='prox',exp=exp,review_gate=review,model_definition=model)
launch={'command':[str(m.GSTMI),'-X','utf8','-B','-u',str(root/'runner.py'),*tokens],'parent_pid':os.getppid(),'initial_visibility':os.environ.get('CUDA_VISIBLE_DEVICES'),'control_bindings':{}}
fake=types.SimpleNamespace(source_hashes=lambda:hashes,snapshot_invariants=lambda *a:I,validate_launch=lambda *a:(launch,reserved),stage_profile=lambda *a:{},sha256=lambda *a:'unused',control_bindings=lambda *a:{})
real_work={}
ph_actual=m.physics()
def fail_in_actual_physics(*args):
 ph=ph_actual;g=torch.Generator().manual_seed(187);shape=(4,4,4);csm=torch.randn((2,*shape),generator=g,dtype=torch.complex64);data=torch.randn(csm.shape,generator=g,dtype=torch.complex64);raw=torch.randn(shape,generator=g,dtype=torch.complex64);op=ph.SenseOperator(csm,data,torch.ones(shape));normal=op.normal;calls=0
 def fail_after_normal(x,**kw):
  nonlocal calls
  v=normal(x,**kw);calls+=1
  if calls==2:raise RuntimeError('deliberate independent CPU fault after actual second normal')
  return v
 op.normal=fail_after_normal
 try:ph.proximal_dc(op,raw)
 except Exception as e:real_work.update(e.proximal_work);raise
with patch.object(m,'ROOT',root),patch.object(m,'supervisor',lambda:fake),patch.object(m,'clinical_stage',fail_in_actual_physics),patch.object(sys,'argv',[str(root/'runner.py'),*tokens]),patch.object(torch.cuda,'is_available',lambda:True),patch.object(torch.cuda,'device_count',lambda:1):
 try:m.main(tokens)
 except RuntimeError as error:err=error
 else:raise AssertionError('fault was swallowed')
failure=json.loads((output/exp/'failure.json').read_text())
verify=root/'verification';verify.mkdir();args=types.SimpleNamespace(reservation_token='synthetic_review_token',exp=exp,stage='prox')
(verify/f'{exp}_selection.json').write_text(json.dumps({'parent_pid':os.getpid(),'process_start':sup.process_start(),'reservation_token':args.reservation_token}))
with patch.object(sup,'ROOT',root),patch.object(sup,'VERIFY',verify):sup.record_failure(args,err)
parent=json.loads((verify/f'{exp}_failure.json').read_text())
print(json.dumps({'probe':'exception_work_loss','temp_root':str(root),'actual_attached_work':real_work,'child_failure_keys':sorted(failure),'child_elapsed_seconds_recorded':'elapsed_seconds' in failure,'child_provenance_exists':(output/exp/'provenance.json').exists(),'parent_child_observation':parent['child_observation'],'parent_failure_keys':sorted(parent)}))
