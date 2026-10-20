"""Check completed supplement artifacts, original protocol and frozen outputs."""
import sys,json,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.supplement import OUT,DATA,RESULT,csvrows
from tools.astra_filter_comparison import preflight,DEFAULT_PLAN
from fgs.io import read_json,read_jsonl,file_hash,write_json
from fgs.data import assert_disjoint

def verify():
    plan=preflight(DEFAULT_PLAN)
    for phase in ['stats','local','transformer','finish']:assert read_json(OUT/(phase+'_completion.json'))['status']=='completed'
    local=csvrows(OUT/'local_summary.csv');assert len(local)==72
    transformer=csvrows(OUT/'transformer/summary.csv');assert len(transformer)==12
    models=[]
    for family,expected in [('learning',48),('grouped',12),('permutation',12),('transformer',12)]:
        files=list((OUT/family).rglob('model.joblib'));assert len(files)==expected,(family,len(files))
        for p in files:
            completion=read_json(p.parent/'completion.json');assert completion['model_sha256']==file_hash(p)
            assert completion['reload_verified']
            predictions=csvrows(p.parent/'test_predictions.csv')
            m=completion['metrics']['test'];assert len(predictions)==m['n']
            assert abs(sum(r['label']==r['prediction'] for r in predictions)/len(predictions)-m['accuracy'])<1e-10
            models.append(dict(path=str(p.relative_to(ROOT)),sha256=file_hash(p)))
    for role in ['question','answer']:
        folder=OUT/'grouped_data'/role;splits={s:read_jsonl(folder/(s+'.jsonl')) for s in ['train','validation','test']};assert_disjoint(splits)
        audit=read_json(folder/'audit.json')
        for s in splits:assert audit['splits'][s]['sha256']==file_hash(folder/(s+'.jsonl'))
    for source in read_json(RESULT/'provenance.json')['sources'].values():
        source=Path(source['path']);receipt=read_json(source/'receipt.json')
        for name,h in receipt['files'].items():assert file_hash(source/name)==h
    old=read_jsonl(RESULT/'responses.jsonl');new=read_jsonl(OUT/'transformer_pipeline/responses.jsonl')
    assert len(old)==3600 and len(new)==1440
    allrows=read_jsonl(OUT/'extended_comparison/responses.jsonl');assert allrows==old+new
    assert len({(r['system'],r['id'],r['replicate']) for r in allrows})==5040
    assert len({r['id'] for r in allrows})==240 and len({r['system'] for r in allrows})==21
    assert not any(r['status']=='provider_or_filter_error' for r in allrows)
    assert all(e['cache_hit'] for r in new for e in r['api_events'])
    assert read_json(OUT/'transformer_pipeline/summary.json')['api_accounting']['network_calls']==0
    for path,n in [('paired_comparisons.csv',26),('transformer_classifier_comparisons.csv',8),('transformer_pipeline_comparisons.csv',14)]:
        rows=csvrows(OUT/'statistics'/path);assert len(rows)==n
        for r in rows:
            assert 0<float(r['cluster_signflip_p'])<=1
            assert float(r['ci_low'])<=float(r['ci_high'])
    spec=read_json(OUT/'optional_api/protocol.json')
    for name,h in spec['files'].items():assert file_hash(OUT/'optional_api'/name)==h
    files={str(p.relative_to(ROOT)):file_hash(p) for p in OUT.rglob('*') if p.is_file() and p.suffix in {'.json','.csv','.md','.py','.txt','.jsonl'} and 'hf_cache' not in p.parts and p.name!='verification.json'}
    write_json(OUT/'verification.json',dict(status='passed',models_verified=len(models),local_jobs=72,transformer_heads=12,original_protocol_files=len(plan['files']),
        original_outputs_unchanged=True,grouped_splits_disjoint=True,extended_records=5040,systems=21,questions=240,new_paid_api_calls=0,
        automated_tests=dict(passed=18,command='python -m unittest tests.test_filter_comparison tests.test_astra_direct tests.test_astra_comparison tests.test_supplement -v',
                             note='Run with PYTHONUTF8=1. First legacy test read failed under Windows cp950; UTF-8 rerun passed.'),
        models=models,artifact_hashes=files))
    print('Verified: 84 models, original receipts unchanged, 5040 rows/240 questions, zero new API calls.')

if __name__=='__main__':verify()
