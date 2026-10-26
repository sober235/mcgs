"""Actual frozen C loss/builder exercise for the isolated frequency-Adam hook."""
import ast
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from test_route import module

SOURCE=Path(__file__).resolve().parents[2]/"nonlocal_autoresearch/trainer/paired_gs/trainers/joint_trainer.py"


def source_trainer():
    """Use verbatim source loss/builder/curriculum, with tiny measured tensors."""
    r=module("runner");r.load_resume_protocol()
    from data.transforms import fft3c
    tree=ast.parse(SOURCE.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="JointTrainer")
    methods=[copy.deepcopy(n) for n in cls.body if isinstance(n,ast.FunctionDef)
             and n.name in ("_build_freq_weight","_freq_weight_curriculum","_kspace_loss")]
    assert len(methods)==3
    tiny=ast.Module(body=[ast.ImportFrom(module="__future__",names=[ast.alias(name="annotations")],level=0),
                          ast.ClassDef(name="SourceFixture",bases=[],keywords=[],body=methods,decorator_list=[])],type_ignores=[])
    env={"torch":torch,"fft3c":fft3c}
    exec(compile(ast.fix_missing_locations(tiny),str(SOURCE),"exec"),env)
    t=env["SourceFixture"]();t.volume_shape=(9,9,9);t.device=torch.device("cpu")
    t.config={"loss":{"kspace_freq_adaptive":True,"kspace_freq_curriculum":False,
        "kspace_center_fraction":.05,"kspace_center_weight":.3,"kspace_mid_weight":1.,
        "kspace_high_weight":2.8,"kspace_high_weight_plain":2.2}}
    gen=torch.Generator().manual_seed(74)
    for tag in ("p","c"):
        mask=(torch.rand(t.volume_shape,generator=gen)>.75).float();mask[4,4,4]=1.
        fit=mask.clone();fit[0,0,0]=1.
        setattr(t,"mask_"+tag,mask);setattr(t,"mask_fit_"+tag,fit)
        setattr(t,"csm_fit_"+tag,torch.randn(2,*t.volume_shape,dtype=torch.complex128,generator=gen))
        setattr(t,"kspace_fit_"+tag,torch.randn(2,*t.volume_shape,dtype=torch.complex128,generator=gen))
        setattr(t,"_dc_norm_"+tag,13.5 if tag=="p" else 17.25)
    t._render=lambda channel: torch.zeros(t.volume_shape,dtype=torch.complex128)
    return t


def actual_loss_pair(trainer,r,iteration=4001):
    r.begin_frequency_step(trainer,iteration)
    trainer._static_freq_w=trainer._build_freq_weight()
    trainer._static_freq_w_plain=trainer._build_freq_weight(high_override=2.2)
    post=trainer._freq_weight_curriculum(iteration,4500)
    images=[];losses=[]
    for scan,tag,weight in (("pre","p",trainer._static_freq_w_plain),("post","c",post)):
        image=torch.linspace(.1,1.7,729).reshape(9,9,9).to(torch.complex128).requires_grad_(True)
        loss=trainer._kspace_loss(image,getattr(trainer,"csm_fit_"+tag),getattr(trainer,"kspace_fit_"+tag),
                                  getattr(trainer,"mask_fit_"+tag),weight,norm=getattr(trainer,"_dc_norm_"+tag))
        images.append(image);losses.append(loss)
    return images,losses,r.frequency_step_evidence(trainer,iteration)


@pytest.mark.parametrize("t",[0.,1.])
def test_actual_source_training_loss_and_gradient_use_installed_weights(t,tmp_path):
    r=module("runner");trainer=source_trainer()
    original_loss=trainer._kspace_loss
    originals={"pre":trainer._build_freq_weight(high_override=2.2),"post":trainer._build_freq_weight()}
    record=r.install_frequency_weights(trainer,t,tmp_path)
    images,losses,use=actual_loss_pair(trainer,r)
    assert use["loss_call_scans"]==["pre","post"] and use["loss_calls_total"]==2
    assert use["installed_before_first_forward"] is use["actual_trainer_weights_match"] is True
    for index,(scan,tag) in enumerate((("pre","p"),("post","c"))):
        weight=originals[scan]
        acquired=getattr(trainer,"mask_"+tag).bool()
        expected=weight if t==0. else torch.full_like(weight,float(weight[acquired].double().square().mean().sqrt()))
        control=images[index].detach().clone().requires_grad_(True)
        value=original_loss(control,getattr(trainer,"csm_fit_"+tag),getattr(trainer,"kspace_fit_"+tag),
                            getattr(trainer,"mask_fit_"+tag),expected,norm=getattr(trainer,"_dc_norm_"+tag))
        torch.testing.assert_close(losses[index],value,rtol=0,atol=0)
        left,=torch.autograd.grad(losses[index],images[index]);right,=torch.autograd.grad(value,control)
        torch.testing.assert_close(left,right,rtol=0,atol=0)
        if t==0.:
            state=trainer._frequency_adam_state
            assert state["effective_weights"][scan] is state["source_weights"][scan]
    assert record["frequency_array_check"]["passed"] is True
    final=r.frequency_record(trainer,1.)
    assert final["final_weights_check"]["passed"] is True
    assert final["final_frequency_use"]==use and final["completed_steps"]==1


@pytest.mark.parametrize("defect",["missing","reverted","mutated","norm","curriculum"])
def test_actual_loss_refuses_missing_reverted_or_mutated_weights(tmp_path,defect):
    r=module("runner");t=source_trainer();r.install_frequency_weights(t,1.,tmp_path)
    r.begin_frequency_step(t,4001)
    if defect=="curriculum":
        t.config["loss"]["kspace_freq_curriculum"]=True
        with pytest.raises(ValueError,match="curriculum"):t._freq_weight_curriculum(4001,4500)
        return
    t._static_freq_w=t._build_freq_weight();t._static_freq_w_plain=t._build_freq_weight(high_override=2.2)
    t._freq_weight_curriculum(4001,4500)
    weight=t._static_freq_w_plain;norm=t._dc_norm_p
    if defect=="missing":weight=None
    elif defect=="reverted":weight=t._frequency_adam_state["source_weights"]["pre"]
    elif defect=="mutated":weight.add_(1.)
    else:norm=norm+1.
    with pytest.raises(ValueError,match="weight|norm|source"):
        t._kspace_loss(torch.ones(9,9,9,dtype=torch.complex128),t.csm_fit_p,t.kspace_fit_p,t.mask_fit_p,weight,norm=norm)


def test_saved_formula_and_original_mask_are_independently_checked(tmp_path):
    r=module("runner");t=source_trainer();record=r.install_frequency_weights(t,1.,tmp_path)
    evidence=record["frequency_evidence"]
    descriptor=evidence["scans"]["pre"]["acquired_mask"]
    path=Path(descriptor["path"]);changed=np.load(path);changed.flat[0]=1.-changed.flat[0]
    np.save(path,changed,allow_pickle=False)
    descriptor["file_sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
    descriptor["byte_sha256"]=hashlib.sha256(changed.tobytes()).hexdigest()
    (tmp_path/"frequency_weights/manifest.json").write_text(json.dumps(evidence))
    with pytest.raises(ValueError,match="mask|source"):
        r.validate_saved_frequency_evidence(tmp_path,evidence,1.,t)


def test_declared_adam_protocol_distinguishes_t0_noop_and_t1_zero():
    s=module("run_gpu")
    assert s.stage_profile("audit0",1.,1.)["frequency_t"]==1.
    assert s.stage_profile("noop2",1.,0.)["frequency_t"]==0.
    for factor in (1.,2.):assert s.stage_profile("trial500",factor,1.)["steps"]==500
    for stage,factor,t in (("audit0",1.,0.),("noop2",1.,1.),("trial500",4.,1.),("trial500",1.,0.)):
        with pytest.raises(ValueError):s.stage_profile(stage,factor,t)


@pytest.mark.parametrize('defect',[None,'pending_q','multiple_q','cg80','strict_global','strict_q'])
def test_formal_recipe_requires_completed_Q_and_defers_any_strict_win(tmp_path,monkeypatch,defect):
    s=module('run_gpu');root=tmp_path/'H';root.mkdir();monkeypatch.setattr(s,'ROOT',root)
    summary=tmp_path/'completed_summary.json'
    q={'name':'explicit_cpu_fixture','method_variant':'inner_precision_mm','cg_steps':96,
       'true_residual_diagnostics':True,'strict_success':False}
    history={'status':'completed_real_trial_snapshot','inner_precision_family':{
        'frequency_t':1.,'cg_steps':96,'true_residual_diagnostics':True,'completed_candidates':[q['name']]},'trials':[q]}
    if defect=='pending_q':history.pop('inner_precision_family')
    elif defect=='multiple_q':history['inner_precision_family']['completed_candidates'].append('unmeasured_fixture')
    elif defect=='cg80':history['inner_precision_family']['cg_steps']=80
    elif defect=='strict_q':q['strict_success']=True
    summary.write_text(json.dumps(history));inc=tmp_path/'development_incumbent.json'
    incumbent={'score':-.123,'strict_success':defect=='strict_global','validated_summary':str(summary),
               'validated_summary_sha256':s.sha256(summary)}
    inc.write_text(json.dumps(incumbent));hashes={'fixture':'sourcehash'};invariants={'fixture':'inputhash'}
    recipe={'status':'selected','selected_by':'root','code_sha256':hashes,'invariants':invariants,
        'geometry_factors':[1.,2.],'frequency_t':1.,'source_iteration':4000,'steps':500,
        'local_factor':1.,'nlambda':0.,'sigma_rel':.015,'density_lr_factor':1.,'betas':[.9,.999],
        'minimum_score_improvement':1e-4,'incumbent':{'path':str(inc),'sha256':s.sha256(inc),'score':incumbent['score']},
        'completed_prior_summary':{'path':str(summary),'sha256':s.sha256(summary)}}
    selected=root/'recipe.json';selected.write_text(json.dumps(recipe))
    args=s.parser().parse_args(['--stage','trial500','--exp','not_run','--geometry-factor','1',
        '--review-gate','unused','--zero-audit-gate','unused','--noop-gate','unused','--selected-recipe',str(selected)])
    monkeypatch.setattr(s,'validate_review',lambda *a:None);monkeypatch.setattr(s,'validate_observed_gate',lambda *a:None)
    if defect is None:s.validate_stage_gates(args,hashes,invariants)
    else:
        with pytest.raises(ValueError,match='completed|strict|Q'):
            s.validate_stage_gates(args,hashes,invariants)


@pytest.mark.parametrize('early_forward',[False,True])
def test_actual_factory_installs_weights_before_first_render_and_restores_class(tmp_path,monkeypatch,early_forward):
    from types import SimpleNamespace
    r=module('runner');s=r.supervisor();folder=tmp_path/'trainer/paired_gs/gaussian/cuda_ext';folder.mkdir(parents=True)
    for name,value in (('tile_voxelize_cuda.cpp',b'cpp'),('tile_voxelize_cuda_kernel.cu',b'cuda')):(folder/name).write_bytes(value)
    pieces=[hashlib.md5((folder/name).read_bytes()).hexdigest()[:8] for name in ('tile_voxelize_cuda.cpp','tile_voxelize_cuda_kernel.cu')]
    binary=folder/'build'/('tile_voxelize_cuda_'+'_'.join(pieces)+'.so');binary.parent.mkdir();binary.write_bytes(b'CPU fixture only')
    tile_path=folder.parent/'tile_voxelizer.py';tile_path.write_text('# CPU fixture only\n')
    tile=SimpleNamespace(__file__=str(tile_path),_cuda_ext=None,_cuda_load_error=None)
    extension=SimpleNamespace(__file__=str(binary),forward=lambda:None,backward=lambda:None)
    events=[];fixture=source_trainer()
    class Factory(type(fixture)):
        def __init__(self,cfg,device=None):
            self.__dict__.update({k:v for k,v in fixture.__dict__.items() if k!='_render'})
            self.config=cfg;self.voxelizer=SimpleNamespace(backend='tile_cuda',strict_cuda=True,use_cuda=True,use_cupy=False)
            self.writer=None
            if early_forward:self._render('plain')
        def _render(self,channel):
            events.append(channel);return torch.zeros(self.volume_shape,dtype=torch.complex128)
    original_render=Factory._render
    trainer_module=SimpleNamespace(__file__=str(tmp_path/'trainer/paired_gs/trainers/joint_trainer.py'),JointTrainer=Factory)
    monkeypatch.setattr(r,'load_file',lambda path,name:extension if Path(path)==binary else module('runner').load_file(path,name))
    monkeypatch.setattr(r.importlib,'import_module',lambda name:tile if name=='gaussian.tile_voxelizer' else trainer_module)
    record,restore=r.install_strict_backend(SimpleNamespace(TRAINER=tmp_path/'trainer'),
        {'prebuilt_extension':{'path':str(binary),'sha256':s.sha256(binary)}},frequency_t=1.,run_dir=tmp_path/'run')
    try:
        cfg={'mode':{'strict_cuda':True},'loss':copy.deepcopy(fixture.config['loss'])}
        if early_forward:
            with pytest.raises(ValueError,match='before frequency'):trainer_module.JointTrainer(cfg)
            assert not events
        else:
            trainer=trainer_module.JointTrainer(cfg);assert not events
            trainer._render('plain');assert events==['plain']
        assert Factory._render is original_render
    finally:restore()
    assert trainer_module.JointTrainer is Factory
