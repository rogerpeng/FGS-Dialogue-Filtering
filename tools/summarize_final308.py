"""Read-only snapshot and reproducible exploratory summary of frozen ratings."""
import csv, hashlib, hmac, itertools, json, math, sqlite3, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT.parent/'投稿版_第三次修改/本機統整_20261017'
BASE=ROOT/'runs/short_seven_rating_20261014'
AXES=['appropriateness','clarity','helpfulness']
SYSTEMS=['domain_b1','word2vec__b2','word2vec__full','generic_b1','domain_refinement_only','tfidf__full','transformer__full']
def dump(name,x): (OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def csvout(name,rows):
    if not rows:return
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sign_p(d):
    pos=int((d>0).sum());neg=int((d<0).sum());n=pos+neg
    return min(1.,2*sum(math.comb(n,k) for k in range(min(pos,neg)+1))/2**n) if n else 1.
def holm(rows,key,out):
    order=sorted(range(len(rows)),key=lambda i:rows[i][key]);v=0.
    for j,i in enumerate(order):
        v=max(v,min(1.,rows[i][key]*(len(rows)-j)));rows[i][out]=v

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    snap=OUT/'PRIVATE_rating_snapshot.sqlite3'
    if not snap.exists():
        with sqlite3.connect('file:'+str(BASE/'comparison_ratings.sqlite3')+'?mode=ro',uri=True) as src,sqlite3.connect(snap) as dst:src.backup(dst)
    c=sqlite3.connect('file:'+str(snap)+'?mode=ro',uri=True)
    scope=json.loads((BASE/'final_scope_308.json').read_text())
    key=(BASE/'PRIVATE_token_secret.txt').read_text().encode()
    mapping={};answers={};bad_hash=[]
    for name,sha in scope['answer_sha256'].items():
        p=BASE/'completed'/name
        if hashlib.sha256(p.read_bytes()).hexdigest()!=sha:bad_hash.append(name)
        for r in json.loads(p.read_text()):
            token=hmac.new(key,(r['id']+'\0'+r['system']).encode(),hashlib.sha256).hexdigest()
            mapping[token]=(r['id'],r['system']);answers[(r['id'],r['system'])]=r
    assert not bad_hash
    records={};raw=[]
    for person,token,payload,version,at in c.execute('select participant,token,payload,version,updated_at from ratings'):
        assert token in mapping
        qid,s=mapping[token];p=json.loads(payload)
        assert not p.get('skipped') and all(type(p.get(a)) is int and 1<=p[a]<=5 for a in AXES)
        assert (qid,s) not in records
        records[qid,s]=p
        raw.append(dict(question_id=qid,system=s,**{a:p[a] for a in AXES},**{f:p.get(f) for f in ['fabricated_quote','impersonation','harmful_or_biased']},version=version,updated_at=at))
    ids=scope['question_ids'];assert len(records)==len(ids)*len(SYSTEMS)==2156
    assert set(records)==set(answers)
    bank={r['id']:r for r in json.loads((ROOT.parent/'星雲大師相關提問_500題_20261014/questions.json').read_text())}
    # Metadata clusters only; broad legacy bucket deliberately remains one group.
    clusters=[bank[i].get('source_url') or bank[i].get('source_title') or 'unknown' for i in ids]
    names=sorted(set(clusters));ci={x:i for i,x in enumerate(names)};g=np.array([ci[x] for x in clusters]);G=len(names)
    y=np.array([[[records[i,s][a] for a in AXES] for s in SYSTEMS] for i in ids],dtype=float)
    rng=np.random.default_rng(20261017);B=20000
    # Resample complete question vectors so systems and axes stay paired.
    weights=rng.multinomial(len(ids),np.ones(len(ids))/len(ids),size=B)
    means=weights@y.reshape(len(ids),-1)/len(ids)
    sums=np.array([y[g==j].sum(axis=0).ravel() for j in range(G)]);sizes=np.bincount(g)
    cw=rng.multinomial(G,np.ones(G)/G,size=B);cm=(cw@sums)/(cw@sizes)[:,None]
    summary=[]
    for si,s in enumerate(SYSTEMS):
        rs=[answers[i,s] for i in ids];lens=np.array([sum(not z.isspace() for z in r['answer']) for r in rs]);st=Counter(r['status'] for r in rs)
        row=dict(system=s,n=308,accepted=st.get('accepted',0),other_statuses=json.dumps({k:v for k,v in st.items() if k!='accepted'}),mean_three_axes=float(y[:,si,:].mean()),length_mean=float(lens.mean()),length_median=float(np.median(lens)),length_p90=float(np.quantile(lens,.9)),outside_20_60=int(((lens<20)|(lens>60)).sum()))
        for ai,a in enumerate(AXES):
            v=y[:,si,ai];j=si*3+ai
            row.update({a+'_mean':float(v.mean()),a+'_sd':float(v.std(ddof=1)),a+'_median':float(np.median(v)),a+'_five':int((v==5).sum()),a+'_one':int((v==1).sum()),a+'_ci_low':float(np.quantile(means[:,j],.025)),a+'_ci_high':float(np.quantile(means[:,j],.975))})
        summary.append(row)
    pairs=[]
    for si,sj in itertools.combinations(range(7),2):
        for ai,a in enumerate(AXES):
            d=y[:,si,ai]-y[:,sj,ai];dd=means[:,si*3+ai]-means[:,sj*3+ai];dc=cm[:,si*3+ai]-cm[:,sj*3+ai]
            pairs.append(dict(system_a=SYSTEMS[si],system_b=SYSTEMS[sj],axis=a,n=308,mean_difference=float(d.mean()),median_difference=float(np.median(d)),wins=int((d>0).sum()),ties=int((d==0).sum()),losses=int((d<0).sum()),probability_superiority_ties_half=float(((d>0).sum()+.5*(d==0).sum())/308),item_ci_low=float(np.quantile(dd,.025)),item_ci_high=float(np.quantile(dd,.975)),metadata_cluster_ci_low=float(np.quantile(dc,.025)),metadata_cluster_ci_high=float(np.quantile(dc,.975)),exact_sign_p=sign_p(d)))
    holm(pairs,'exact_sign_p','holm_p_63')
    # Same question and identical answer should not be mistaken for independent generations.
    duplicates=[]
    for si,sj in itertools.combinations(range(7),2):
        same=[j for j,i in enumerate(ids) if answers[i,SYSTEMS[si]]['answer']==answers[i,SYSTEMS[sj]]['answer']]
        duplicates.append(dict(system_a=SYSTEMS[si],system_b=SYSTEMS[sj],identical_answers=len(same),different_rating_vectors=sum(not np.array_equal(y[j,si],y[j,sj]) for j in same)))
    overlap=[]
    from rating_app.comparison import normalize
    splitsets={}
    for split in ['train','validation','test']:
        splitsets[split]={normalize(json.loads(line)['text']) for line in (ROOT/f'data/random_80_10_10_20261010/question/{split}.jsonl').read_text().splitlines() if line.strip()}
    for i in ids:
        q=bank[i];overlap.append(dict(question_id=i,source_cluster=ci[q.get('source_url') or q.get('source_title') or 'unknown'],origin=q.get('origin'),from_original_240=q.get('from_original_240'),**{s:normalize(q['question']) in texts for s,texts in splitsets.items()}))
    failures=[]
    for i in ids:
        for s in SYSTEMS:
            r=answers[i,s]
            if r['status']!='accepted': failures.append(dict(question_id=i,system=s,question=r['question'],answer=r['answer'],status=r['status'],**{a:records[i,s][a] for a in AXES}))
    subsets=[]
    for s in SYSTEMS:
        for label,selected in [('accepted_only_selection_biased',[i for i in ids if answers[i,s]['status']=='accepted']),('no_exact_normalized_question_overlap',[r['question_id'] for r in overlap if not any(r[k] for k in splitsets)]),('from_original_240',[r['question_id'] for r in overlap if r['from_original_240']]),('not_from_original_240',[r['question_id'] for r in overlap if not r['from_original_240']])]:
            subsets.append(dict(system=s,subset=label,n=len(selected),**{a+'_mean':float(np.mean([records[i,s][a] for i in selected])) for a in AXES}))
    frozen=[dict(question_id=i,system=s,question=answers[i,s]['question'],answer=answers[i,s]['answer'],status=answers[i,s]['status'],**{a:records[i,s][a] for a in AXES}) for i in ids for s in SYSTEMS]
    audit=dict(question_count=308,systems=SYSTEMS,answer_records=2156,rating_records=len(records),axis_scores=len(records)*3,shared_login_count=1,human_raters_reported_by_author=5,rater_assignment_recoverable=False,skipped=0,missing_axes=0,answer_hashes_verified=len(scope['answer_sha256']),source_metadata_clusters=G,largest_cluster=max(Counter(clusters).values()),five_score_fraction=float((y==5).mean()),risk_flags={f:dict(Counter(p.get(f) for p in records.values())) for f in ['fabricated_quote','impersonation','harmful_or_biased']},question_overlap={s:sum(r[s] for r in overlap) for s in splitsets},original_240_overlap=sum(r['from_original_240'] for r in overlap),origins=dict(Counter(r['origin'] for r in overlap)),dates=c.execute('select min(updated_at),max(updated_at) from ratings').fetchone(),snapshot_sha256=hashlib.sha256(snap.read_bytes()).hexdigest(),bootstrap_repetitions=B,seed=20261017,limitations=['Five people shared one account, according to author. No per-rating rater identities; cannot fit rater effects or compute inter-rater reliability.','Question bootstrap and exact sign tests are exploratory conditional sensitivities, not confirmatory population inference.','Metadata source clusters are incomplete provenance proxies and do not recover rater clusters.','Holm family includes all 21 pairs times three axes (63 tests); no post-hoc composite significance test.','All questions are domain-related and overlap existing development data; no external safety or specificity inference.'])
    csvout('01_各組人評與長度.csv',summary);csvout('02_同題配對比較_探索性.csv',pairs);csvout('03_相同答案評分一致性.csv',duplicates);csvout('04_題庫重合稽核.csv',overlap);csvout('05_流程拒答案例.csv',failures);csvout('06_逐筆回答與分數.csv',frozen)
    csvout('07_分層描述與僅放行敏感度.csv',subsets)
    dump('audit.json',audit);dump('summary.json',summary);dump('paired.json',pairs)
    print(json.dumps(audit,ensure_ascii=False,indent=2));print(json.dumps([{k:v for k,v in r.items() if k in ['system','mean_three_axes','accepted','appropriateness_mean','clarity_mean','helpfulness_mean']} for r in summary],ensure_ascii=False,indent=2))
if __name__=='__main__':main()
