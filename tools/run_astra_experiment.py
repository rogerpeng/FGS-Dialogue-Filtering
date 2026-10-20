"""Run the prepared direct-inference protocol. --live explicitly enables paid calls."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fgs.evaluation import generation_experiment
p=argparse.ArgumentParser();p.add_argument('--phase',choices=['pilot','test','generic'],default='pilot');p.add_argument('--out',required=True);p.add_argument('--live',action='store_true');p.add_argument('--cache-dir',default=str(ROOT/'runs/astra_direct_shared_cache'));a=p.parse_args()
data=ROOT/'data/astra_direct_20261012';plan=json.loads((data/'protocol.json').read_text())
def verify(path,expected):
 if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('Frozen protocol input changed: '+str(path))
for key,h in plan['code_hashes'].items():verify(ROOT/key,h)
for key,h in plan['model_hashes'].items():verify(ROOT/plan['classifier_models'][key],h)
for key,h in plan['data_hashes'].items():verify(data/key,h)
for key,h in plan['active_config_hashes'].items():verify(ROOT/'configs'/key,h)
phase=a.phase
items=data/('pilot_validation_items.jsonl' if phase=='pilot' else 'test_items.jsonl')
config=ROOT/'configs'/({'pilot':'astra.pilot.json','test':'astra.direct.json','generic':'astra.generic.json'}[phase])
variants=['b1','refinement_only','full'] if phase=='pilot' else ['b1'] if phase=='generic' else plan['variants']
config_data=json.loads(config.read_text())
if any(config_data[k]!='gpt-6-astra' for k in ('generator_model','refiner_model')):raise ValueError('Direct protocol requires Astra base model for both roles')
if not a.live:
 print(json.dumps({'status':'preflight_passed_no_api_calls','phase':phase,'items':plan['pilot_n'] if phase=='pilot' else plan['test_n'],
                   'variants':variants,'config':str(config),'output':str(Path(a.out).resolve()),
                   'model':'gpt-6-astra','note':'Set API key environment locally, then add --live to enable paid calls.'},ensure_ascii=False,indent=2))
 sys.exit(0)
required=plan['pilot_n']*2 if phase=='pilot' else plan['generic_baseline_call_upper_bound'] if phase=='generic' else plan['domain_call_upper_bound']
if config_data['max_calls']<required:raise ValueError('Configured call budget below complete protocol upper bound')
report=generation_experiment(items,config,Path(a.out),ROOT/plan['classifier_models']['question'],ROOT/plan['classifier_models']['answer'],variants=variants,replicates=1,live=a.live,cache_dir=a.cache_dir)
print(json.dumps(report,ensure_ascii=False,indent=2))
