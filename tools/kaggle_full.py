"""Run the remaining lab entirely inside one Kaggle GPU session."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from solution.engine import write_json
from solution.reporting import locate
from tools.check_rubric import audit


def measure_e0_resources():
    """Measure a labelled validation probe without repeating E0 test inference."""
    import torch
    from scripts.labcommon import read_jsonl
    from scripts.prepare_data import _find_root, prepare
    from solution.engine import generate, release_memory
    from solution.modeling import load_base
    meta=prepare(_find_root('/kaggle/input'),ROOT/'data')
    entries=read_jsonl(ROOT/'data/splits/val.jsonl')[:4]
    model,processor=load_base()
    torch.cuda.reset_peak_memory_stats()
    _,seconds=generate(model,processor,entries,ROOT/'work/e0_resource_val.jsonl')
    measured={'dataset_version':meta['dataset_version'],'split':'val','n_images':len(entries),'image_ids':[e['image_id'] for e in entries],'batch_size':4,'max_new_tokens':512,'peak_vram_gb':torch.cuda.max_memory_allocated()/2**30,'inference_minutes':seconds/60,'scope':'Four validation images; E0 test predictions are reused without new test inference.'}
    write_json(ROOT/'report/evidence/e0_resource_probe.json',measured)
    del model,processor
    release_memory()


def main():
    if not Path('/kaggle/input').is_dir():
        raise SystemExit('This full workflow only runs on Kaggle.')
    # Reuse the already executed E0 exactly. No second baseline test inference.
    for relative in ('report/evidence/e0.json','report/evidence/preflight.json','work/e0.jsonl'):
        source=locate(relative)
        destination=ROOT/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        if source.resolve()!=destination.resolve():
            shutil.copy2(source,destination)
    evidence=json.loads((ROOT/'report/evidence/e0.json').read_text(encoding='utf-8'))
    assert evidence['missing_predictions']==0 and evidence['split']=='test'
    measure_e0_resources()
    completed=['e0']
    phases=['e1','e2','e3','e4','e5','report']
    try:
        measured_e1=locate('report/evidence/e1.json')
    except FileNotFoundError:
        measured_e1=None
    if measured_e1 is not None:
        e1=json.loads(measured_e1.read_text(encoding='utf-8'))
        assert e1['dataset_version']==evidence['dataset_version'] and e1['missing_predictions']==0 and e1['split']=='test'
        for relative in ('report/evidence/e1.json','work/e1.jsonl'):
            source=locate(relative)
            destination=ROOT/relative
            destination.parent.mkdir(parents=True,exist_ok=True)
            if source.resolve()!=destination.resolve(): shutil.copy2(source,destination)
        completed.append('e1')
        phases.remove('e1')
    state={'status':'running','completed':completed,'reused_evidence':completed.copy(),'started_at':time.time()}
    for phase in phases:
        state['phase']=phase
        write_json(ROOT/'report/evidence/cloud_workflow.json',state)
        print(f'FULL WORKFLOW: starting {phase}',flush=True)
        try:
            subprocess.run([sys.executable,'-u','-m','solution.run','--phase',phase],cwd=ROOT,check=True)
        except subprocess.CalledProcessError:
            state.update(status='failed',failed_phase=phase)
            write_json(ROOT/'report/evidence/cloud_workflow.json',state)
            raise
        state['completed'].append(phase)
        write_json(ROOT/'report/evidence/cloud_workflow.json',state)
    checks=audit(ROOT)
    text='\n'.join(f'[{status}] {name} — {detail}' for status,name,detail in checks)
    (ROOT/'submission-check.txt').write_text(text,encoding='utf-8')
    print(text,flush=True)
    failures=[name for status,name,_ in checks if status=='FAIL']
    if failures:
        state.update(status='failed',rubric_failures=failures)
        write_json(ROOT/'report/evidence/cloud_workflow.json',state)
        raise RuntimeError(f'Actual rubric conditions not met: {failures}')
    state.update(status='complete',finished_at=time.time())
    write_json(ROOT/'report/evidence/cloud_workflow.json',state)
    print('FULL WORKFLOW COMPLETE — all measured rubric checks passed.',flush=True)


if __name__=='__main__':
    main()
