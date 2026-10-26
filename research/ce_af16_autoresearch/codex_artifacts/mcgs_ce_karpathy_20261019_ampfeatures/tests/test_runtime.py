import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch


def runtime():
    path = Path(__file__).parents[1] / 'runtime.py'
    if not path.exists():
        pytest.fail('acquired-only runtime is not implemented')
    spec = importlib.util.spec_from_file_location('campaign_runtime_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unacquired_values_cannot_change_training_data():
    module = runtime()
    mask = torch.zeros(3, 4, 5); mask[:, 1, :] = 1
    k = torch.randn(2, 3, 4, 5, dtype=torch.complex64)
    modified = k + (1-mask)[None] * (100+23j)
    actual = module.normalized_acquired(k, mask, 183.53231811523438)
    assert torch.equal(actual, module.normalized_acquired(modified, mask, 183.53231811523438))
    assert torch.equal(actual, (k / 183.53231811523438) * mask[None])


@pytest.mark.parametrize('scale', [0, -1, float('nan'), float('inf')])
def test_invalid_frozen_scale_is_rejected(scale):
    with pytest.raises(ValueError):
        runtime().normalized_acquired(torch.ones(2, 3, 4, 5, dtype=torch.complex64), torch.ones(3,4,5), scale)


def test_nonbinary_mask_is_rejected():
    with pytest.raises(ValueError):
        runtime().normalized_acquired(torch.ones(2,3,4,5,dtype=torch.complex64), torch.full((3,4,5),.5), 2)


def test_zero_parity_is_complex_and_strict_about_schema():
    module = runtime()
    ref = np.ones((3,4,5), np.complex64) * (1+2j)
    assert module.zero_parity(ref.copy(), ref)['passed']
    assert not module.zero_parity(ref * np.complex64(1j), ref)['passed']
    with pytest.raises(ValueError): module.zero_parity(ref.real, ref)
    with pytest.raises(ValueError): module.zero_parity(ref.transpose(2,1,0), ref)
    bad = ref.copy(); bad.flat[0] = np.nan
    with pytest.raises(ValueError): module.zero_parity(bad, ref)


def test_actual_source_loader_projects_before_ifft_and_exposes_no_gt(tmp_path,monkeypatch):
    import sys
    import h5py
    source=Path('/home/congcongliu/codex_artifacts/mcgs_ce_autoresearch_20261017/nonlocal_autoresearch/trainer/paired_gs')
    monkeypatch.syspath_prepend(str(source))
    from data.dataset import MRIDataset
    from data.transforms import ifft3c
    raw=torch.randn(2,8,8,8,dtype=torch.complex64)
    mask=torch.zeros(8,8,8);mask[:,::2,::2]=1
    csm=torch.full((2,8,8,8),complex(2**-.5),dtype=torch.complex64)
    paths=[tmp_path/f'{name}.h5' for name in ['kspace','mask','csm']]
    for path,key,value in zip(paths,['kspace','mask','csm'],[raw,mask,csm]):
        with h5py.File(path,'w') as f: f[key]=value.numpy()
    cls=runtime().acquired_dataset_class(MRIDataset,{str(paths[0]):7.0})
    ds=cls(data_path=str(paths[0]),mask_path=str(paths[1]),csm_path=str(paths[2]),
           acceleration_factor=4,mask_type='external',use_acs=False,readout_axis=0,phase_axes=(1,2),presampled=False)
    expected=(raw/7.)*mask[None]
    assert torch.equal(ds.kspace_undersampled_complex,expected)
    assert torch.equal(ds.zero_filled_complex,(csm.conj()*ifft3c(expected)).sum(0))
    assert not hasattr(ds,'ground_truth_image') and ds.kspace_full is None
    assert ds.acquired_only_evidence['ground_truth_constructed'] is False
    assert ds.acquired_only_evidence['outside_mask_nonzero']==0


def test_frequency_observer_uses_actual_g2_clock_without_fictitious_prior_calls():
    m=runtime()
    state={'completed_steps':0,'current_iteration':None,'step_scans':[],
           'curriculum_iteration':None,'loss_calls_total':0}
    with pytest.raises(ValueError): m.begin_frequency_step(state,4500,4001)
    m.begin_frequency_step(state,4500,4501)
    state.update(step_scans=['pre','post'],curriculum_iteration=4501,loss_calls_total=2)
    m.finish_frequency_step(state,4500,4501)
    assert state['completed_steps']==1 and state['loss_calls_total']==2
    m.begin_frequency_step(state,4500,4502)
    state.update(step_scans=['pre'],curriculum_iteration=4502,loss_calls_total=3)
    with pytest.raises(ValueError): m.finish_frequency_step(state,4500,4502)
