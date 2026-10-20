"""Offline read-only verification of the Table 1 public research-data export."""
from pathlib import Path
import collections,csv,hashlib,json,sqlite3,sys
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'research_data'
def require(ok,message):
    if not ok:raise ValueError(message)
def jsonl(name):return [json.loads(x) for x in (D/name).read_text(encoding='utf-8').splitlines() if x.strip()]
def verify():
    manifest=D/'DATA_MANIFEST.json'
    if manifest.exists():
        for x in json.loads(manifest.read_text(encoding='utf-8'))['files']:
            require(hashlib.sha256((D/x['path']).read_bytes()).hexdigest()==x['sha256'],'Data hash mismatch: '+x['path'])
    qa=jsonl('classifier_pairs_2400.jsonl');texts=jsonl('classifier_text_partitions.jsonl');bq=jsonl('balanced_questions_240.jsonl');bo=jsonl('balanced_outputs_5040.jsonl');sq=jsonl('short_questions_308.jsonl');so=jsonl('short_outputs_2156.jsonl')
    with (D/'short_ratings_2156.csv').open(encoding='utf-8-sig',newline='') as f:ratings=list(csv.DictReader(f))
    require(len(qa)==2400 and collections.Counter(x['label'] for x in qa)=={0:1200,1:1200},'Classifier pair counts')
    require(collections.Counter(x['partition'] for x in qa)=={'train':1920,'validation':240,'test':240},'Original pair split')
    expected={('random','question'):[1828,240,240],('random','answer'):[1414,226,228],('grouped','question'):[1352,457,445],('grouped','answer'):[1016,332,336]}
    for (protocol,task),counts in expected.items():
        seen=set();seen_texts=set();seen_groups=set()
        for part,n in zip(['train','validation','test'],counts):
            rows=[x for x in texts if x['protocol']==protocol and x['task']==task and x['partition']==part]
            require(len(rows)==n,f'Partition count {protocol}/{task}/{part}')
            require(len({x['id'] for x in rows})==n,'Duplicate task IDs')
            if protocol=='grouped':
                groups={x['group_id'] for x in rows};values={x['text'] for x in rows};ids={x['id'] for x in rows}
                require(not (seen & ids or seen_texts & values or seen_groups & groups),'Grouped partition overlap')
                seen|=ids;seen_texts|=values;seen_groups|=groups
    require(len(bq)==240 and len(bo)==5040,'Balanced cardinality')
    require(collections.Counter(x['category'] for x in bq)=={'related':120,'unrelated':120},'Balanced labels')
    require(len({(x['id'],x['system']) for x in bo})==5040,'Balanced output duplicates')
    require(all(v==240 for v in collections.Counter(x['system'] for x in bo).values()) and len({x['system'] for x in bo})==21,'21 balanced configurations')
    require(len(sq)==308 and len(so)==2156 and len(ratings)==2156,'Short-answer cardinality')
    outputs={(x['id'],x['system']):x for x in so};scores={(x['question_id'],x['system']):tuple(int(x[a]) for a in ['appropriateness','clarity','helpfulness']) for x in ratings}
    require(len(outputs)==2156 and outputs.keys()==scores.keys(),'Short answer/rating alignment')
    require(all(1<=a<=5 for v in scores.values() for a in v),'Rating range')
    systems=sorted({x['system'] for x in so});ids={x['id'] for x in sq}
    require(len(systems)==7 and set(outputs)=={(i,s) for i in ids for s in systems},'Complete 308 x 7 grid')
    common={i for i in ids if all(outputs[i,s]['status']=='accepted' for s in systems)};require(len(common)==258,'Common acceptance count')
    expected_short={'word2vec__b2':(282,4.866,4.540,4.886),'word2vec__full':(269,4.862,4.386,4.875),'generic_b1':(308,4.847,4.847,4.913),'tfidf__full':(292,4.837,4.637,4.875),'domain_b1':(308,4.833,4.833,4.884),'transformer__full':(302,4.828,4.753,4.875),'domain_refinement_only':(308,4.821,4.821,4.876)}
    table45={}
    for s,(n,acc,allmean,commonmean) in expected_short.items():
        accepted=[i for i in ids if outputs[i,s]['status']=='accepted']
        avg=lambda chosen:sum(sum(scores[i,s]) for i in chosen)/(3*len(chosen))
        vals=(len(accepted),round(avg(accepted),3),round(avg(ids),3),round(avg(common),3));require(vals==(n,acc,allmean,commonmean),'Tables 4/5 mismatch: '+s)
        table45[s]={'accepted':n,'interception_percent':round((308-n)/308*100,2),'accepted_mean':vals[1],'all_mean':vals[2],'common_mean':vals[3]}
    table7={}
    for s,n in [('tfidf__full',240),('transformer__full',238),('word2vec__full',237),('domain_b1',120)]:
        rows=[x for x in bo if x['system']==s]
        correct=sum((x['status']=='accepted')==(x['category']=='related') for x in rows);require(correct==n,'Table 7 mismatch: '+s);table7[s]=dict(correct=correct,n=240,conformity=round(correct/240,4))
    with sqlite3.connect((D/'table1_research_data.sqlite3').resolve().as_uri()+'?mode=ro',uri=True) as c:
        require(c.execute('PRAGMA integrity_check').fetchone()[0]=='ok','SQLite integrity');require(not c.execute('PRAGMA foreign_key_check').fetchall(),'SQLite foreign keys')
        counts={t:c.execute('SELECT COUNT(*) FROM '+t).fetchone()[0] for t in ['classifier_pairs','classifier_texts','balanced_questions','balanced_outputs','short_questions','short_outputs','short_ratings']}
        require(counts==dict(classifier_pairs=2400,classifier_texts=len(texts),balanced_questions=240,balanced_outputs=5040,short_questions=308,short_outputs=2156,short_ratings=2156),'SQL counts')
        require({tuple(x) for x in c.execute('SELECT question_id,system,appropriateness,clarity,helpfulness FROM short_ratings')}=={(i,s,*v) for (i,s),v in scores.items()},'SQL/CSV ratings')
        require(set(c.execute('SELECT question_id,system,answer,status FROM short_outputs'))=={(x['id'],x['system'],x['answer'],x['status']) for x in so},'SQL/JSON short outputs')
        require(set(c.execute('SELECT question_id,system,answer,status FROM balanced_outputs'))=={(x['id'],x['system'],x['answer'],x['status']) for x in bo},'SQL/JSON balanced outputs')
        require(set(c.execute('SELECT id,question,answer,label,partition FROM classifier_pairs'))=={(x['id'],x['question'],x['answer'],x['label'],x['partition']) for x in qa},'SQL/JSON corpus')
        require(set(c.execute('SELECT protocol,task,partition,id,text,label,group_id FROM classifier_texts'))=={(x['protocol'],x['task'],x['partition'],x['id'],x['text'],x['label'],x['group_id']) for x in texts},'SQL/JSON task partitions')
    return dict(status='passed',counts=counts,axis_scores=6468,common_accepted_questions=258,tables_4_5=table45,table_7=table7,api_calls=0,training_runs=0)
if __name__=='__main__':
    try:print(json.dumps(verify(),ensure_ascii=False,indent=2))
    except Exception as e:print('VERIFICATION FAILED: '+str(e),file=sys.stderr);sys.exit(1)
