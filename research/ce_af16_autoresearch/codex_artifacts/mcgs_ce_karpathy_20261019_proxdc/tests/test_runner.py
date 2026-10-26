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
