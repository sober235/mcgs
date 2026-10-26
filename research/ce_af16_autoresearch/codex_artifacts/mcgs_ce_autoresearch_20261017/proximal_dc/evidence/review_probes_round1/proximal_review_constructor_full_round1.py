from pathlib import Path
import importlib.util,sys,tempfile,types,json
from unittest.mock import patch
import numpy as np,h5py,torch
P=Path('/home/congcongliu/codex_artifacts/mcgs_ce_autoresearch_20261017/proximal_dc')
def load(p,n):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
m=load(P/'runner.py','independent_full_constructor_runner');t=load(P/'tests/test_runner.py','independent_full_constructor_fixture')
trainer=t.cpu_export_fixture();source=m.source_module('trainers.joint_trainer','trainers/joint_trainer.py');dataset=sys.modules['data.dataset']
root=Path(tempfile.mkdtemp(prefix='proximal_review_constructor_round1_'));rng=np.random.default_rng(328)
for name,key,arr in [('data','kspace',(rng.normal(size=(2,6,6,6))+1j*rng.normal(size=(2,6,6,6))).astype(np.complex64)),('map','map',np.ones((2,6,6,6),dtype=np.complex64)),('mask','mask',np.ones((6,6,6),dtype=np.float32))]:
 with h5py.File(root/f'{name}.h5','w') as f:f[key]=arr
cfg=dict(data_path=str(root/'data.h5'),csm_path=str(root/'map.h5'),mask_path=str(root/'mask.h5'),presampled=False,normalize_kspace=True,normalization_percentile=99.9,mask_type='external',skip_readout_check=True,readout_axis=0,phase_axes=[1,2],use_acs=False)
trainer.config={'data':{'plain':dict(cfg),'contrast':dict(cfg)}};trainer.device=torch.device('cpu');observed=[]
def observe(fn,domain,kind):
 def f(x,*a,**kw):observed.append(dict(domain=domain,kind=kind,ndim=x.ndim,volumes=(len(x) if x.ndim==4 else 1)));return fn(x,*a,**kw)
 return f
with patch.object(dataset,'fft3c',observe(dataset.fft3c,'dataset','fft')),patch.object(dataset,'ifft3c',observe(dataset.ifft3c,'dataset','ifft')),patch.object(source,'ifft3c',observe(source.ifft3c,'trainer','ifft')):source.JointTrainer._setup_data(trainer)
controls,canon=m.canonical_control(trainer,backend_kind='cpu_test');sup=m.supervisor();zeros=dict.fromkeys(sup.PHYSICS_COUNTERS,0);scans={s:{'n_coils':2,'work':zeros} for s in ('pre','post')};total=m.total_work(canon,scans);sup.validate_work_total({'scans':scans,'work_total':total},'canonical')
print(json.dumps(dict(probe='full_presampled_false_constructor',temp_root=str(root),observed_constructor=observed,constructor_fft_batches=sum(v['kind']=='fft' for v in observed),constructor_ifft_batches=sum(v['kind']=='ifft' for v in observed),constructor_fft_volumes=sum(v['volumes'] for v in observed if v['kind']=='fft'),constructor_ifft_volumes=sum(v['volumes'] for v in observed if v['kind']=='ifft'),reported_total_fft_batches=total['total_fft_batches'],reported_total_ifft_batches=total['total_ifft_batches'],reported_coil_fft_transforms=total['total_coil_fft_transforms'],reported_coil_ifft_transforms=total['total_coil_ifft_transforms'],parent_validation_accepted=True),indent=2))
