"""Post-hoc revision supplement. Original inputs/models are read-only.

Run phases separately; completed jobs are checkpointed with source hashes.
No network API generation in this tool. Transformer download is explicit.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse, csv, json, sys, time
from pathlib import Path
from collections import defaultdict
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import joblib
from scipy.stats import binomtest
from fgs.io import read_json, read_jsonl, write_json, write_jsonl, write_csv, file_hash, environment
from fgs.data import normalize, clean_group, split_rows, assert_disjoint
from fgs.metrics import classification
from fgs.models import TextClassifier

DATA = ROOT / 'data/random_80_10_10_20261010'
RESULT = ROOT / 'runs/astra_comparison_completed_20261013'
OUT = ROOT / 'runs/supplement_20261013'
SEEDS = [42,43,44]
MODEL = 'BAAI/bge-small-zh-v1.5'

def csvrows(p):
    with Path(p).open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))

def holm(ps):
    order = np.argsort(ps); result = np.ones(len(ps)); previous = 0.
    for rank, i in enumerate(order):
        previous = max(previous, min(1., (len(ps)-rank)*ps[i])); result[i] = previous
    return result.tolist()

def paired(a,b,groups,seed=42,repeats=10000):
    a,b = np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    if a.shape != b.shape or len(a)!=len(groups) or not len(a): raise ValueError('Unpaired input')
    d=a-b; unique=sorted(set(groups)); ix=[np.flatnonzero(np.asarray(groups)==g) for g in unique]
    sums=np.array([d[i].sum() for i in ix]); counts=np.array([len(i) for i in ix])
    rng=np.random.default_rng(seed)
    draws=rng.integers(0,len(ix),(repeats,len(ix)))
    boot=sums[draws].sum(1)/counts[draws].sum(1)
    null=(rng.choice([-1,1],size=(repeats,len(ix)))*sums).sum(1)/len(d)
    p=(1+int((np.abs(null)>=abs(d.mean())-1e-12).sum()))/(repeats+1)
    wins=int((d>0).sum()); losses=int((d<0).sum())
    return dict(n=len(d),groups=len(ix),mean_a=float(a.mean()),mean_b=float(b.mean()),
                difference=float(d.mean()),ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)),
                cluster_signflip_p=p,wins=wins,losses=losses,ties=int((d==0).sum()),
                paired_dz=float(d.mean()/d.std(ddof=1)) if len(d)>1 and d.std(ddof=1)>0 else None,
                mcnemar_exact_p=binomtest(wins,wins+losses,.5).pvalue if wins+losses and set(d).issubset({-1.,0.,1.}) else None)

def stats():
    from fgs.evaluation import lexical_scores
    from tools.astra_filter_comparison import preflight, DEFAULT_PLAN
    preflight(DEFAULT_PLAN)
    provenance=read_json(RESULT/'provenance.json')
    for source in provenance['sources'].values():
        assert file_hash(Path(source['path'])/'responses.jsonl')==source['sha256']
    rows=read_jsonl(RESULT/'responses.jsonl'); items=read_jsonl(ROOT/'data/astra_direct_20261012/test_items.jsonl')
    ids=sorted(r['id'] for r in items); meta={r['id']:r for r in items}
    table={}; details=[]
    for r in rows:
        accepted=r['status']=='accepted'
        overlap=lexical_scores(r['answer'],r['reference']) if accepted and r['reference'] else {'character_f1':0.,'rouge_l_character_f1':0.}
        row=dict(id=r['id'],system=r['system'],group_id=meta[r['id']]['group_id'],category=r['category'],
                 synthetic=r['synthetic'],status=r['status'],coverage=int(accepted),
                 action_accuracy=int(accepted==(r['expected_action']=='answer')),**overlap)
        table[r['system'],r['id']]=row; details.append(row)
    assert len(table)==3600 and len(ids)==240
    write_csv(OUT/'statistics/item_scores.csv',details)
    contrasts=[('tfidf__full','word2vec__full'),('domain_b1','generic_b1'),('domain_refinement_only','domain_b1')]
    for profile in ['word2vec','tfidf']:
        contrasts += [(profile+'__full',s) for s in ['domain_b1','domain_refinement_only',profile+'__b2',profile+'__no_query',profile+'__no_answer']]
    comparisons=[]
    for metric in ['action_accuracy','rouge_l_character_f1']:
        selected=[i for i in ids if metric=='action_accuracy' or meta[i]['category']=='related']
        for a,b in contrasts:
            comparison=paired([table[a,i][metric] for i in selected],[table[b,i][metric] for i in selected],[meta[i]['group_id'] for i in selected])
            comparisons.append(dict(metric=metric,a=a,b=b,**comparison))
    for r,p in zip(comparisons,holm([r['cluster_signflip_p'] for r in comparisons])): r['holm_p_26_tests']=p
    write_csv(OUT/'statistics/paired_comparisons.csv',comparisons)
    summaries=[]
    for system in sorted({r['system'] for r in rows}):
        for subset in ['all','related','unrelated','synthetic','legacy_origin']:
            keep=[i for i in ids if subset=='all' or meta[i]['category']==subset or (subset=='synthetic' and meta[i]['synthetic']) or (subset=='legacy_origin' and not meta[i]['synthetic'])]
            if not keep: continue
            a=np.array([table[system,i]['action_accuracy'] for i in keep]); ci=paired(a,np.zeros(len(a)),[meta[i]['group_id'] for i in keep])
            summaries.append(dict(system=system,subset=subset,n=len(keep),action_accuracy=float(a.mean()),ci_low=ci['ci_low'],ci_high=ci['ci_high'],
                                  coverage=float(np.mean([table[system,i]['coverage'] for i in keep]))))
    write_csv(OUT/'statistics/subgroups.csv',summaries)
    cost=[]; distinct={}
    cache=ROOT/'runs/astra_filter_comparison_shared_cache'
    for system in sorted({r['system'] for r in rows}):
        events={e['cache_key']:e for r in rows if r['system']==system for e in r['api_events']}
        prompt=completion=cached=0
        for k in events:
            data=read_json(cache/(k+'.json')); usage=data['usage']
            prompt+=usage['prompt_tokens']; completion+=usage['completion_tokens']; cached+=usage.get('prompt_tokens_details',{}).get('cached_tokens',0)
            distinct[k]=usage
        cost.append(dict(system=system,unique_calls=len(events),prompt_tokens=prompt,completion_tokens=completion,cached_prompt_tokens=cached,
                         uncached_list_price_equivalent_usd=(prompt*10+completion*50)/1e6))
    write_csv(OUT/'statistics/cost_by_system.csv',cost)
    write_json(OUT/'statistics/accounting.json',dict(unique_formal_successful_calls=len(distinct),prompt_tokens=sum(x['prompt_tokens'] for x in distinct.values()),
        completion_tokens=sum(x['completion_tokens'] for x in distinct.values()),notes=['Per-system costs are standalone token-use equivalents; do not sum shared calls as actual billed usage.',
        'Rates are the prior verified USD10 input/USD50 output per million, not a live invoice. Cached input discounts not applied; cached token count provided.',
        'Pilot, failed requests, other jobs, tax excluded. API network latency not comparable across shared-cache variants.']))
    write_json(OUT/'statistics/protocol.json',dict(post_hoc=True,bootstrap_repeats=10000,seed=42,unit='question, clustered by recorded source/template group',
        comparisons=contrasts,holm_family='all 26 action/lexical tests',limitations=['Not externally independent data.','Lexical scores are not factual correctness.',
        'Action accuracy measures pipeline acceptance status, not semantic refusals inside accepted LLM text.','Boundary-perfect bootstrap intervals are conditional empirical intervals, not proof of zero population error.',
        'McNemar p is unadjusted item-independence sensitivity only; primary p uses group sign flips.']))
    print('statistics completed',flush=True)

def grouped_data(role):
    folder=OUT/'grouped_data'/role
    if (folder/'audit.json').exists(): return folder
    original=[r for s in ['train','validation','test'] for r in read_jsonl(DATA/role/(s+'.jsonl'))]
    # Expand all group links before connected-component closure, including merged provenance.
    expanded=[dict(text=r['text'],label=r['label'],group_id=g,source=r['id']) for r in original for g in r.get('group_ids',[r['group_id']])]
    clean,excluded,near=clean_group(expanded,.9)
    splits=split_rows(clean,42,test_fraction=.2,validation_fraction=.2); assert_disjoint(splits)
    for s,rs in splits.items(): write_jsonl(folder/(s+'.jsonl'),rs)
    write_jsonl(folder/'excluded.jsonl',excluded)
    write_csv(folder/'near_edges.csv',near,['id_a','id_b','jaccard','label_conflict'])
    write_json(folder/'audit.json',dict(source_rows=len(original),unique_rows=len(clean),excluded_rows=len(excluded),near_edges=len(near),
        splits={s:dict(n=len(rs),positive=sum(r['label'] for r in rs),groups=len({r['group_id'] for r in rs}),sha256=file_hash(folder/(s+'.jsonl'))) for s,rs in splits.items()},
        note='Post-hoc 60/20/20 regrouping of existing corpus using all known source/template links and exact/near text. Classification-only sensitivity, NOT new external data; question/answer splits separate.'))
    return folder

def fit_job(data,folder,kind,seed,fraction=1.,permuted=False,embedding=None,revision=None):
    done=folder/'completion.json'
    hashes={s:file_hash(data/(s+'.jsonl')) for s in ['train','validation','test']}
    if done.exists():
        record=read_json(done)
        assert record['hashes']==hashes
        return record
    from fgs.experiments import training_subset
    splits={s:read_jsonl(data/(s+'.jsonl')) for s in hashes}
    train=training_subset(splits['train'],fraction,seed); val=splits['validation']
    y=np.array([r['label'] for r in train]); vy=[r['label'] for r in val]
    if permuted: y=np.random.default_rng(seed).permutation(y)
    best=None; tuning=[]; started=time.perf_counter()
    candidates=[.001,.01] if kind=='word2vec' else [.1,1.,10.]
    for hp in candidates:
        m=TextClassifier(kind=kind,seed=seed,learning_rate=hp if kind=='word2vec' else .001,C=hp if kind!='word2vec' else 1.,
                         cache_dir=str(OUT/'.tokenizer_cache'),embedding_model=embedding,embedding_revision=revision,local_files_only=bool(embedding))
        m.fit([r['text'] for r in train],y,[r['text'] for r in val],vy)
        vm=classification(vy,m.probabilities([r['text'] for r in val])); score=(vm['macro_f1'],-vm['log_loss'])
        tuning.append(dict(hyperparameter=hp,validation_macro_f1=vm['macro_f1'],validation_log_loss=vm['log_loss']))
        if best is None or score>best[0]: best=(score,m,hp)
    m=best[1]; folder.mkdir(parents=True,exist_ok=True); joblib.dump(m,folder/'model.joblib')
    fit_seconds=time.perf_counter()-started; metrics={}; timing={}
    for s,rs in dict(splits,train=train).items():
        labels=y if s=='train' and permuted else [r['label'] for r in rs]
        start=time.perf_counter(); p=m.probabilities([r['text'] for r in rs]); timing[s]=time.perf_counter()-start
        metrics[s]=classification(labels,p)
        write_csv(folder/(s+'_predictions.csv'),[dict(id=r['id'],group_id=r['group_id'],label=int(label),probability=float(prob),prediction=int(prob>=.5)) for r,label,prob in zip(rs,labels,p)])
    reload=joblib.load(folder/'model.joblib'); check=reload.probabilities([r['text'] for r in splits['test'][:10]])
    assert np.allclose(check,m.probabilities([r['text'] for r in splits['test'][:10]]),atol=1e-7)
    record=dict(kind=kind,seed=seed,train_fraction=fraction,train_n=len(train),permuted=permuted,selected_hp=best[2],
                fit_seconds=fit_seconds,model_bytes=(folder/'model.joblib').stat().st_size,metrics=metrics,inference_seconds=timing,hashes=hashes,
                model_sha256=file_hash(folder/'model.joblib'),reload_verified=True)
    write_csv(folder/'validation_search.csv',tuning); write_json(done,record)
    print(str(folder.relative_to(OUT)), 'test F1',round(metrics['test']['macro_f1'],4),flush=True)
    return record

def local():
    results=[]
    for role in ['question','answer']:
        grouped=grouped_data(role)
        for seed in SEEDS:
            for kind in ['tfidf','word2vec']:
                jobs=[('grouped',grouped,1.,False)]+[('learning',DATA/role,f,False) for f in [.25,.5,.75,1.]]+[('permutation',DATA/role,1.,True)]
                for name,data,f,perm in jobs:
                    r=fit_job(data,OUT/name/role/f'{kind}_seed{seed}_f{f}',kind,seed,f,perm)
                    results.append(dict(experiment=name,role=role,kind=kind,seed=seed,fraction=f,train_n=r['train_n'],
                        train_macro_f1=r['metrics']['train']['macro_f1'],validation_macro_f1=r['metrics']['validation']['macro_f1'],test_macro_f1=r['metrics']['test']['macro_f1'],
                        test_accuracy=r['metrics']['test']['accuracy'],fit_seconds=r['fit_seconds']))
                    write_csv(OUT/'local_summary.csv',results)

def transformer():
    import torch
    torch.set_num_threads(4)
    from huggingface_hub import HfApi,snapshot_download
    from sentence_transformers import SentenceTransformer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    folder=OUT/'transformer'; folder.mkdir(parents=True,exist_ok=True)
    spec=folder/'encoder.json'
    revision=read_json(spec)['revision'] if spec.exists() else HfApi().model_info(MODEL).sha
    write_json(spec,dict(model=MODEL,revision=revision,selection='Chosen for Chinese support and local CPU feasibility, before supplementary scores; not latest/SOTA claim.',
                         pooling='model default CLS',normalize=True,max_length=512,head='StandardScaler + LogisticRegression',Cs=[.1,1.,10.]))
    model_path=snapshot_download(MODEL,revision=revision,cache_dir=str(OUT/'hf_cache'),allow_patterns=['*.json','*.txt','*.safetensors','1_Pooling/*'])
    encoder=SentenceTransformer(model_path,device='cpu',trust_remote_code=False,local_files_only=True)
    encoder.max_seq_length=512
    enc_files={str(p.relative_to(model_path)):file_hash(p) for p in Path(model_path).rglob('*') if p.is_file() and '.cache' not in p.parts}
    write_json(folder/'encoder_files.json',dict(files=enc_files,total_file_bytes=sum(p.stat().st_size for p in Path(model_path).rglob('*') if p.is_file())))
    summary=[]
    for role in ['question','answer']:
        for design,data in [('random',DATA/role),('grouped',grouped_data(role))]:
            splits={s:read_jsonl(data/(s+'.jsonl')) for s in ['train','validation','test']}; features={}; encode_times={}
            base=folder/design/role; base.mkdir(parents=True,exist_ok=True)
            for s,rs in splits.items():
                cache=base/(s+'_embeddings.npz'); h=file_hash(data/(s+'.jsonl'))
                if cache.exists():
                    z=np.load(cache); assert str(z['input_hash'])==h and str(z['revision'])==revision
                    features[s]=z['features']; encode_times[s]=float(z['seconds'])
                else:
                    start=time.perf_counter(); features[s]=encoder.encode([r['text'] for r in rs],normalize_embeddings=True,batch_size=32,show_progress_bar=False)
                    encode_times[s]=time.perf_counter()-start
                    np.savez_compressed(cache,features=features[s],input_hash=h,revision=revision,seconds=encode_times[s])
                print('encoded',design,role,s,len(rs),flush=True)
            scaler=StandardScaler().fit(features['train']); X={s:scaler.transform(x) for s,x in features.items()}; y={s:np.array([r['label'] for r in rs]) for s,rs in splits.items()}
            for seed in SEEDS:
                out=base/f'seed{seed}'; out.mkdir(exist_ok=True); trials=[]; best=None; start=time.perf_counter()
                for c in [.1,1.,10.]:
                    head=LogisticRegression(C=c,max_iter=2000,random_state=seed).fit(X['train'],y['train'])
                    vm=classification(y['validation'],head.predict_proba(X['validation'])[:,1]); score=(vm['macro_f1'],-vm['log_loss'])
                    trials.append(dict(C=c,validation_macro_f1=vm['macro_f1'],validation_log_loss=vm['log_loss']))
                    if best is None or score>best[0]: best=(score,head,c)
                model=TextClassifier(kind='transformer',seed=seed,C=best[2],embedding_model=model_path,local_files_only=True)
                model.scaler=scaler; model.classifier=best[1]; joblib.dump(model,out/'model.joblib')
                metrics={}
                for s,rs in splits.items():
                    p=best[1].predict_proba(X[s])[:,1]; metrics[s]=classification(y[s],p)
                    write_csv(out/(s+'_predictions.csv'),[dict(id=r['id'],group_id=r['group_id'],label=r['label'],probability=float(prob),prediction=int(prob>=.5)) for r,prob in zip(rs,p)])
                loaded=joblib.load(out/'model.joblib'); loaded.encoder=encoder
                assert np.allclose(loaded.probabilities([r['text'] for r in splits['test'][:8]]),best[1].predict_proba(X['test'][:8])[:,1],atol=1e-5)
                write_csv(out/'validation_search.csv',trials)
                write_json(out/'completion.json',dict(metrics=metrics,revision=revision,selected_C=best[2],head_fit_seconds=time.perf_counter()-start,encode_seconds=encode_times,
                     hashes={s:file_hash(data/(s+'.jsonl')) for s in splits},reload_verified=True,model_sha256=file_hash(out/'model.joblib')))
                summary.append(dict(design=design,role=role,seed=seed,C=best[2],train_f1=metrics['train']['macro_f1'],validation_f1=metrics['validation']['macro_f1'],test_f1=metrics['test']['macro_f1'],test_accuracy=metrics['test']['accuracy']))
                write_csv(folder/'summary.csv',summary)
                print('transformer',design,role,seed,metrics['test']['macro_f1'],flush=True)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('phase',choices=['stats','local','transformer']); args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if not (OUT/'protocol.json').exists():
        write_json(OUT/'protocol.json',dict(status='post_hoc_supplement',seeds=SEEDS,fractions=[.25,.5,.75,1.],threshold=.5,
        learning_rates=[.001,.01],Cs=[.1,1.,10.],epochs=300,patience=25,selection='validation macro-F1 then log_loss',
        grouping='known source/template connected components plus normalized exact/near duplicates, role-specific 60/20/20',
        limitations=['Existing test inspected before design; all analyses supplementary.', 'Group-heldout is re-split internal corpus, not new external validation.',
        'Permutation uses shuffled train labels but true validation labels; only a diagnostic, not a formal permutation p-value.',
        'Single modern encoder does not establish superiority over all Transformers.'],source_hashes={str(p.relative_to(ROOT)):file_hash(p) for p in DATA.rglob('*.jsonl')}))
    write_json(OUT/(args.phase+'_environment.json'),environment())
    globals()[args.phase]()
    write_json(OUT/(args.phase+'_completion.json'),dict(status='completed',tool_sha256=file_hash(__file__)))

if __name__=='__main__': main()
