"""Combine complete domain ablations with the generic baseline for blind ratings."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fgs.io import read_json,read_jsonl,write_json,write_jsonl,new_output,file_hash
from fgs.evaluation import summarize_responses
p=argparse.ArgumentParser();p.add_argument('--domain',required=True);p.add_argument('--generic',required=True);p.add_argument('--out',required=True);a=p.parse_args()
domain,generic=Path(a.domain),Path(a.generic)
c1,c2=read_json(domain/'config.json'),read_json(generic/'config.json')
for k in ['generator_model','refiner_model','reasoning_effort','max_completion_tokens','temperature','max_generation_attempts','max_refinements','items_sha256']:
 if c1.get(k)!=c2.get(k):raise ValueError('Baseline settings/data do not match: '+k)
d,g=read_jsonl(domain/'responses.jsonl'),read_jsonl(generic/'responses.jsonl')
expected=set(c1['variants'])
keys={(r['id'],r['replicate']) for r in g}
if len(g)!=len(keys) or any(r['system']!='b1' for r in g):raise ValueError('Expected unique generic b1 rows')
if {(r['id'],r['replicate']) for r in d}!=keys:raise ValueError('Different evaluation items')
for key in keys:
 rows=[r for r in d if (r['id'],r['replicate'])==key]
 if len(rows)!=len(expected) or {r['system'] for r in rows}!=expected:raise ValueError('Incomplete domain variants')
if any(r['status']=='provider_or_filter_error' for r in d+g):raise ValueError('Resolve incomplete/provider-error runs before study comparison')
for r in g:r['original_system']='b1';r['system']='generic_b1';r['baseline_note']='Same Astra and settings; generic system prompt'
combined=d+g;out=new_output(a.out);write_jsonl(out/'responses.jsonl',combined);write_json(out/'summary.json',summarize_responses(combined))
write_json(out/'provenance.json',{'domain_sha256':file_hash(domain/'responses.jsonl'),'generic_sha256':file_hash(generic/'responses.jsonl'),'items':len(keys),'systems':len(expected)+1,'records':len(combined),'new_api_calls':0,'note':'Generic b1 renamed to generic_b1 to retain distinct blind-rating identities.'})
print(out)
