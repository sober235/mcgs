import importlib.util
from pathlib import Path
import copy
import pytest


def module():
    path = Path(__file__).parents[1] / 'runner.py'
    if not path.exists(): pytest.fail('new G2 numerical runner is not implemented')
    spec = importlib.util.spec_from_file_location('new_runner_test', path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def config():
    return {'training': {'resume_offset':4000, 'max_iterations':4500, 'seed':42},
            'output': {'output_dir':'old','exp_name':'old'}, 'loss':{'tv_weight':.1},
            'adaptive_control':{}, 'gaussian': {'initial_num_points':3600000},
            'data': {c:{'acceleration_factor':16,'presampled':False} for c in ['plain','contrast']}}


def test_new_runner_preserves_loss_and_data_but_declares_new_clock(tmp_path):
    m = module(); source = config(); before = copy.deepcopy(source)
    out = m.build_config(source,tmp_path,2)
    assert source == before
    assert out['loss'] == source['loss'] and out['data'] == source['data']
    assert out['training']['resume_offset'] == 4500
    assert out['training']['max_iterations'] == 4502
    assert out['output']['output_dir'] == str(tmp_path/'training')


def test_runner_rejects_changed_acquisition_and_adaptive_control(tmp_path):
    m=module(); source=config(); source['data']['plain']['acceleration_factor']=12
    with pytest.raises(ValueError): m.build_config(source,tmp_path,2)
    source=config(); source['adaptive_control']={'enable':True}
    with pytest.raises(ValueError): m.build_config(source,tmp_path,2)


def test_g2_recipe_is_validated_before_model_restore():
    m=module()
    m.validate_g2_recipe({'iter':4500,'frequency_adam_recipe':{'frequency_t':1.,'geometry_factor':2.}})
    with pytest.raises(ValueError): m.validate_g2_recipe({'iter':4000,'frequency_adam_recipe':{'frequency_t':1.,'geometry_factor':2.}})
    with pytest.raises(ValueError): m.validate_g2_recipe({'iter':4500,'frequency_adam_recipe':{'frequency_t':0.,'geometry_factor':2.}})


def test_runner_builds_selected_paired_field_without_mutating_frozen_model():
    import torch
    from types import SimpleNamespace
    m = module()
    positions = torch.arange(15, dtype=torch.float64).reshape(5, 3)
    bases = {'plain': torch.tensor([1j, 2j, 3j, 4j, 5j], dtype=torch.complex128),
             'contrast': torch.tensor([5j, 1j, 3j, 2j, 4j], dtype=torch.complex128)}
    frozen = SimpleNamespace(positions=positions, get_density=bases.__getitem__)
    launch = dict(mode='shared', feature_mode='paired_anchor', bands=4, width=64, gain_bound=.25)
    value = m.build_residual_field(frozen, launch)
    assert value.feature_mode == 'paired_anchor'
    assert value.anchor_projection.weight.shape == (64, 2)
    assert torch.equal(value.anchor_medians, torch.tensor([3., 3.], dtype=torch.float64))
    for channel, base in bases.items():
        assert torch.equal(value.get_density(channel), base)
    assert torch.equal(positions, torch.arange(15, dtype=torch.float64).reshape(5, 3))
    assert m.build_residual_field(frozen, {**launch, 'mode':'baseline', 'feature_mode':'coordinates'}) is None


@pytest.mark.parametrize('change', [{'feature_mode':None}, {'feature_mode':'wrong'},
                                  {'mode':'explicit'}, {'mode':'baseline'}])
def test_runner_cannot_ignore_missing_or_incompatible_feature_mode(change):
    m = module()
    launch = dict(mode='shared', feature_mode='paired_anchor', bands=4, width=64, gain_bound=.25)
    launch.update(change)
    with pytest.raises(ValueError, match='feature|shared'):
        m.build_residual_field(None, launch)
