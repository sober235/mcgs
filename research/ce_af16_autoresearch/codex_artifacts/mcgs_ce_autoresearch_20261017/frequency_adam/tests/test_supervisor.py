"""CPU-only post-child proof and exit-status regression cases."""
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_route import module


@pytest.mark.parametrize('mutation',[None,'code','input','checkpoint','binary','control','nonzero','interrupt','partial'])
def test_post_child_proofs_and_failures_preserve_actual_exit_and_never_publish_false_completion(tmp_path,monkeypatch,mutation):
    s=module('run_gpu');root=tmp_path/'geometry';verify=root/'verification';verify.mkdir(parents=True)
    monkeypatch.setattr(s,'ROOT',root);monkeypatch.setattr(s,'VERIFY',verify)
    monkeypatch.setattr(s,'CLAIM_LOCK',tmp_path/'claim.lock')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','7');monkeypatch.setattr(s,'initial_visibility',lambda:'7')
    monkeypatch.setattr(s,'parent_runtime',lambda:{'python':'stdlib-only-test-parent','torch_imported':False})
    hashes={'one':'1'*64,'two':'2'*64,'three':'3'*64}
    invariant={'source_checkpoint_sha256':'f'*64,'input':'unchanged','checkpoint':'unchanged','binary':'unchanged'}
    state={'child_finished':False}
    def code_hashes():return {**hashes,'changed':'yes'} if state['child_finished'] and mutation=='code' else hashes
    def invariants():
        result=copy.deepcopy(invariant)
        if state['child_finished'] and mutation in ('input','checkpoint','binary'):result[mutation]='changed'
        return result
    monkeypatch.setattr(s,'code_hashes',code_hashes);monkeypatch.setattr(s,'snapshot_invariants',invariants)
    selected={'gpu':7,'free':True}
    campaign=SimpleNamespace(gpu_snapshot=lambda:[],select_gpu=lambda *args:dict(selected))
    monkeypatch.setattr(s,'load_campaign',lambda:campaign)
    review=verify/'review.json';review.write_text(json.dumps({'status':'pass','round':1,
        'actionable_findings':0,'unresolved_findings':0,'code_sha256':hashes,'invariants':invariant}))
    fd=os.open(s.CLAIM_LOCK,os.O_CREAT|os.O_RDWR,0o600)
    argv=['--stage','audit0','--exp','owned','--review-gate',str(review),'--reserved-gpu','7',
          '--reservation-token','token','--claim-fd',str(fd)]
    args=s.parser().parse_args(argv)
    s.write_new(verify/'owned_selection.json',{'parent_pid':os.getpid(),'process_start':s.process_start(),
                'reservation_token':'token','request':s.request(args),'resource':selected,'code_sha256':hashes,'invariants':invariant,
                'control_bindings':s.control_bindings(args)})
    observed_command={}
    class Child:
        pid=654321;returncode=None
        def __init__(self,command,env,stdout,stderr):
            observed_command.update(command=command,env=env)
            run=root/'outputs/owned';run.mkdir(parents=True)
            checkpoint=run/'latest.pth';checkpoint.write_bytes(b'ordinary-checkpoint-fixture')
            provenance={'status':'complete','source_iteration':4000,'end_iteration':4000,'forward_backward_evaluations':0,
                        'source_checkpoint_sha256':'f'*64,'torch_version':'2.7.1+cu118','cuda_version':'11.8'}
            metrics={k:provenance[k] for k in ('source_iteration','end_iteration','forward_backward_evaluations','source_checkpoint_sha256')}
            metrics['final_checkpoint']=str(checkpoint)
            geometry={'status':'complete','code_sha256':hashes,'runtime':{'torch_version':'2.7.1+cu118'}}
            # This deliberately fake child models GPU metadata, not a GPU run.
            # The frequency arrays and independent NumPy check are real CPU work.
            from test_frequency_adam import source_trainer
            r=module('runner');trainer=source_trainer();r.install_frequency_weights(trainer,1.,run)
            frequency=r.frequency_record(trainer,1.)
            for row in frequency['frequency_evidence']['scans'].values():
                for name in ('source_weight','effective_weight','acquired_mask'):row[name]['device']='cuda:0'
            (run/'frequency_weights/manifest.json').write_text(json.dumps(frequency['frequency_evidence']))
            (run/'frequency_provenance.json').write_text(json.dumps(frequency))
            validation={'status':'complete','passed':True,'stage':'audit0','geometry_factor':1.,'code_sha256':hashes,'invariants':invariant,
                        'checks':{'ordinary_adam_checkpoint':True,'wavelet_replay':True,'actual_optimizer_rates':True,
                                  'actual_frequency_weights':True,'independent_frequency_arrays':True},'zero_audit':{'passed':True},
                        'frequency_files':s.validate_frequency_files(run,frequency['frequency_evidence'],1.)}
            for name,value in [('provenance.json',provenance),('metrics.json',metrics),('geometry_provenance.json',geometry),('validation.json',validation)]:
                value.update(method_variant='frequency_adam',frequency_t=1.,geometry_factor=1.)
                if name!='validation.json':value['frequency_adam']=frequency
                (run/name).write_text(json.dumps(value))
            for name in ('actual_config.yaml','step_trace.jsonl','source_to_copy_manifest.json','source_to_copy.diff','candidate_pre_complex.npy','candidate_post_complex.npy'):
                (run/name).write_bytes(b'fixture')
            if mutation=='partial':(run/'provenance.json').write_text('{')
        def poll(self):return self.returncode
        def wait(self):
            self.returncode=17 if mutation in ('nonzero','interrupt') else 0;state['child_finished']=True
            if mutation=='control':review.write_text('{"status":"mutated after child"}')
            if mutation=='interrupt':raise KeyboardInterrupt('actual wait interruption')
            return self.returncode
    monkeypatch.setattr(s.subprocess,'Popen',Child)
    if mutation is None:
        s.main(argv)
        completed=s.read(verify/'owned_completed.json');assert completed['returncode']==0
        assert completed['passed'] is True
        proof=s.read(verify/'owned_post_run_invariants.json');assert proof['before']==proof['after']
    else:
        with pytest.raises(KeyboardInterrupt if mutation=='interrupt' else Exception):s.main(argv)
        assert not (verify/'owned_completed.json').exists()
        failure=s.read(verify/'owned_failure.json')
        assert failure['returncode']==(17 if mutation in ('nonzero','interrupt') else 0)
        if mutation=='partial':assert 'JSONDecodeError' in failure['artifact_read_error']
        if mutation in ('code','input','checkpoint','binary','control'):
            assert s.read(verify/'owned_post_run_invariants.json')['status']=='failed'
    assert observed_command['command'][0]==str(s.PYTHON)
    assert observed_command['env']['CUDA_VISIBLE_DEVICES']=='7'
    assert observed_command['env']['CUDA_CACHE_DISABLE']=='1'


def test_complete_source_binding_accepts_metadata_outside_production_subset_and_rejects_changes(tmp_path,monkeypatch):
    """Historical production103 excludes3 metadata docs, not their full hashes."""
    import hashlib
    s=module('run_gpu');base=tmp_path/'C';original=tmp_path/'original'
    monkeypatch.setattr(s,'BASE',base);monkeypatch.setattr(s,'SOURCE_TRAINER',original)
    monkeypatch.setattr(s,'SOURCE_DEFINITION',tmp_path/'sources.json')
    monkeypatch.setattr(s,'INPUT_DEFINITION',tmp_path/'inputs.json')
    common=['SOURCE.txt','SOURCE_AR.txt','arc/program.md','paired_gs/model.py',
            'paired_gs/gaussian/cuda_ext/tile_voxelize_cuda.cpp',
            'paired_gs/gaussian/cuda_ext/tile_voxelize_cuda_kernel.cu']
    for root in (base/'trainer',original):
        for name in common:
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(name)
    added=base/'trainer/paired_gs/new_helper.py';added.write_text('# new C helper\n')
    production={str(base/'trainer'/name):s.sha256(base/'trainer'/name) for name in common if name not in common[:3]}
    production[str(added)]=s.sha256(added)
    for index in range(99):
        path=base/f'production_{index}.py';path.write_text(str(index));production[str(path)]=s.sha256(path)
    assert len(production)==103
    gated=dict(list(production.items())[:13])
    s.SOURCE_DEFINITION.write_text(json.dumps({'gated_code_sha256':gated,'production_code_sha256':production}))
    inputs={}
    for index in range(35):
        path=tmp_path/f'input_{index}';path.write_text(str(index));inputs[str(path)]=s.sha256(path)
    manifest={'source_checkpoint':str(tmp_path/'input_0'),'source_config':str(tmp_path/'input_1'),'files':inputs}
    s.INPUT_DEFINITION.write_text(json.dumps(manifest))
    folder=base/'trainer/paired_gs/gaussian/cuda_ext'
    pieces=[hashlib.md5((folder/name).read_bytes()).hexdigest()[:8] for name in ('tile_voxelize_cuda.cpp','tile_voxelize_cuda_kernel.cu')]
    binary=folder/'build'/('tile_voxelize_cuda_'+'_'.join(pieces)+'.so');binary.parent.mkdir();binary.write_bytes(b'prebuilt')
    references=[]
    for label in ('C','O'):
        ref=tmp_path/f'reference_{label}';ref.mkdir();references.append(ref);checkpoint=ref/'latest.pth';checkpoint.write_bytes(b'checkpoint')
        (ref/'metrics.json').write_text(json.dumps({'source_iteration':4000,'end_iteration':4002,
            'forward_backward_evaluations':2,'final_checkpoint':str(checkpoint)}))
        (ref/'provenance.json').write_text('{"status":"complete"}')
        for name in ('step_trace.jsonl','candidate_pre_complex.npy','candidate_post_complex.npy'):(ref/name).write_bytes(b'control')
    monkeypatch.setattr(s,'NOOP_REFERENCES',tuple(references))
    binding=s.snapshot_invariants();definition=binding['source_definition']
    assert definition['copy_files_outside_production_subset']==sorted(common[:3])
    assert definition['missing_original_sources']==['paired_gs/new_helper.py']
    assert all(isinstance(value,str) and len(value)==64 for value in definition['original_sha256'].values())
    assert all(name in definition['copy_sha256'] for name in common[:3])
    review=tmp_path/'review.json';hashes={'one':'a','two':'b','three':'c'}
    review.write_text(json.dumps({'status':'pass','round':1,'actionable_findings':0,'unresolved_findings':0,
                                  'code_sha256':hashes,'invariants':binding}))
    unknown=base/'trainer/paired_gs/unknown.py';unknown.write_text('# unexpected new production source\n')
    changed=s.snapshot_invariants()
    with pytest.raises(ValueError,match='bindings'):s.validate_review(review,hashes,changed)
    production_file=Path(next(iter(production)));production_file.write_text('changed')
    with pytest.raises(ValueError,match='frozen C'):s.snapshot_invariants()
