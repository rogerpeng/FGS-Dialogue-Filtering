"""Prepare short-answer protocol and length audit without any API request."""
import json,hashlib,statistics,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if not (ROOT/'fgs').exists():ROOT=Path('D:/FGS-LLM/revision3_code')
sys.path.insert(0,str(ROOT))
from fgs.api import GENERATION_PROMPT,REFINEMENT_PROMPT

RULE='回答形式：使用繁體中文，直接回答核心問題，以1至3句、20至80個字元（包含標點、不含空白）為目標；必要時可少於20字。不要標題、條列、重述問題、長篇前言或額外例子。簡短但不要省略必要的不確定性、拒答原因或專業求助提醒。不得為縮短內容而捏造或改變原意。'

def count(text):return sum(not c.isspace() for c in text)
def percentile(values,q):
    x=sorted(values);pos=(len(x)-1)*q;i=int(pos);j=math.ceil(pos)
    return x[i]+(x[j]-x[i])*(pos-i)
def stats(values):return dict(n=len(values),mean=statistics.mean(values),p50=percentile(values,.5),p75=percentile(values,.75),p90=percentile(values,.9),p95=percentile(values,.95),max=max(values),over80=sum(x>80 for x in values))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def lines(path):return [json.loads(s) for s in path.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def prepare():
    out=ROOT/'data/final_short_answer_20261014'
    if out.exists():raise ValueError('Version already exists; choose a new dated version, never overwrite.')
    source=ROOT/'data/random_80_10_10_20261010/answer/train.jsonl'
    old=ROOT/'runs/supplement_20261013/extended_comparison/responses.jsonl'
    bank=ROOT.parent/'星雲大師相關提問_500題_20261014/questions.json'
    protected=[source,old,bank,ROOT/'fgs/api.py',ROOT/'fgs/pipeline.py',ROOT/'configs/astra.direct.json',ROOT/'configs/astra.generic.json']
    for profile in ['word2vec','tfidf']:
        for role in ['question','answer']:protected.append(ROOT/f'runs/random_80_10_10_20261010/{role}_train/{profile}_seed42/model.joblib')
    for role in ['question','answer']:protected.append(ROOT/f'runs/supplement_20261013/transformer/random/{role}/seed42/model.joblib')
    before={str(p):sha(p) for p in protected}
    training=lines(source);previous=lines(old);questions=read(bank)
    audit={'length_unit':'Unicode non-whitespace characters, including punctuation; not tokens','selection_data':'related answer training set only; generated lengths descriptive only','actual_training_rows':len(training),'training_related':stats([count(r['text']) for r in training if r['label']==1]),'training_unrelated':stats([count(r['text']) for r in training if r['label']==0])}
    for name in ['generic_b1','domain_b1','domain_refinement_only']:
        audit[name]=stats([count(r['answer']) for r in previous if r['system']==name and r['category']=='related'])
    out.mkdir(parents=True)
    dump(out/'length_audit.json',audit)
    generic=read(ROOT/'configs/astra.generic.json')
    prompts=dict(domain_generation=GENERATION_PROMPT+'\n'+RULE,generic_generation=generic['generation_prompt']+'\n'+RULE,refinement=REFINEMENT_PROMPT+'\n'+RULE,length_rule=RULE)
    dump(out/'prompts.json',prompts)
    cfg=read(ROOT/'configs/astra.direct.json');cfg['max_calls']=1500
    dump(out/'api_settings.json',cfg)
    (out/'questions.jsonl').write_text(''.join(json.dumps(dict(id=r['id'],question=r['question'],category='related',expected_action='answer',synthetic=r.get('question_synthetic'),original_id=r['original_id'],source_url=r.get('source_url'),source_title=r.get('source_title'),review_status='pending_final_review'),ensure_ascii=False)+'\n' for r in questions),encoding='utf-8')
    protocol=dict(version='final_short_answer_20261014',status='prepared_not_run',live_authorized=False,budget_usd=None,model='gpt-6-astra',reasoning_effort='low',question_count=500,systems=sorted({r['system'] for r in previous}),replicates=1,shared_draws=['generic_generation','domain_generation','domain_refinement'],max_generation_attempts=1,max_refinements=1,planned_calls=1500,answer_records=10500,target_characters=[20,80],target_sentences=[1,3],hard_truncation=False,automatic_length_retries=False,compliance_reporting='record actual length and outside-target status; keep full original output and do not select only compliant successes',token_cap=4096,token_cap_note='Includes reasoning budget; retained to avoid incomplete outputs. Character target is a prompt requirement, not a guaranteed API limit.',cache_policy='Separate short-answer cache and versioned nonce. No old long-answer output reuse.',primary_comparison=['tfidf__full','word2vec__full'],limits=['Post-hoc revision after observing long-answer mismatch; not preregistered.','500 candidates include train/validation overlap; final question list pending review.','All-positive candidate set cannot estimate false acceptance on unrelated questions.','Identical length prompts do not guarantee matched actual lengths; report distribution per system.','Input/output topic filters are not doctrinal correctness tests.'],protected_original_files=before)
    dump(out/'protocol.json',protocol)
    dump(out/'lock.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()})
    assert before=={str(p):sha(p) for p in protected}
    estimate=dict(status='estimate_only_no_calls',new_calls=1500,old_response_reuse=0,reference_full_length_mean_usd=500*.074849,reference_full_length_p90_scenario_usd=500*.09642,short_answer_cost_note='No short pilot yet. These are historical long-answer reference scenarios, not a validated short-answer forecast. Reasoning tokens may prevent proportional savings.',output_token_cap_component_usd=1500*4096*50/1e6,pricing_url='https://developers.openai.com/api/docs/models/gpt-6-astra',price_date='2026-10-14',input_usd_per_million=10,output_usd_per_million=50)
    dump(out/'cost_reference.json',estimate)
    # Lock also covers the estimate; the lock does not hash itself.
    dump(out/'lock.json',{p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='lock.json'})
    print(json.dumps(dict(folder=str(out),audit=audit,estimate=estimate,old_files_unchanged=True),ensure_ascii=False))
if __name__=='__main__':prepare()
