"""Invented four-question seven-system demo. No API or real research records."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from rating_app.study import create_study,SYSTEMS
out=ROOT/'studies/demo7'
if out.exists():raise SystemExit('Demo already exists; serve it without regenerating local codes.')
out.mkdir(parents=True)
bank=[];rows=[]
seven=['domain_b1','word2vec__b2','word2vec__full','generic_b1','domain_refinement_only','tfidf__full','transformer__full']
questions=['如何尊重不同意見？','遇到挫折如何整理心情？','如何整理桌面？','如何安排散步時間？']
for n,q in enumerate(questions,1):
    q='【虛構程式示範】'+q
    bank.append(dict(id=f'DEMO{n:03}',question=q))
    for j,s in enumerate(seven,1):
        rows.append(dict(id=f'DEMO{n:03}',system=s,replicate=0,category='related' if n<=2 else 'unrelated',status='accepted',question=q,answer=f'【虛構示範回答{j}，非研究資料】先理解情境，再用清楚且尊重的方式處理。'))
(out/'bank.json').write_text(json.dumps(bank,ensure_ascii=False,indent=2),encoding='utf-8')
(out/'responses.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows),encoding='utf-8')
auth=out/'auth_input.jsonl';auth.write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows if r['system'] in SYSTEMS),encoding='utf-8')
create_study(auth,out/'auth',participants=4,items_per_person=2,pool_size=4)
print('Four fictional questions, seven configurations; no API calls. Codes: studies/demo7/auth/')
