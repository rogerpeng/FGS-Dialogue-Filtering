import json,hashlib,shutil
from pathlib import Path
from collections import Counter
ROOT=Path('/media/oba/4b23197b-42f0-4db1-be07-a3cdeeb23d6e/llm論文/FGS-LLM')
SOURCE=ROOT/'重建問答資料_20261010_各1200筆/資料切分_隨機80_10_10_20261010'
OUT=ROOT/'revision3_code/data/random_80_10_10_20261010'
if OUT.exists():raise FileExistsError('Preserve existing converted data')
OUT.mkdir(parents=True)
summary={}
for kind in ['question','answer']:
    folder=OUT/kind;folder.mkdir();info={'split_policy':'stratified_random_by_qa_row','seed':42,'task':kind+'_filter','source_split_manifest':str(SOURCE/'split_manifest.json'),'source_split_manifest_sha256':hashlib.sha256((SOURCE/'split_manifest.json').read_bytes()).hexdigest(),'overlap_audit':str(SOURCE/'overlap_audit.json'),'splits':{}}
    for split in ['train','validation','test']:
        src=SOURCE/split/(kind+'_classification.json');raw=json.loads(src.read_text());converted=[]
        for r in raw:
            assert r['review_status']=='approved' and r['group_ids']
            converted.append(dict(r,group_id=r['group_ids'][0],task=kind+'_filter',source_id='reviewed_random_split_20261010'))
        dest=folder/(split+'.jsonl');dest.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in converted))
        assert [r['id'] for r in converted]==[r['id'] for r in raw]
        info['splits'][split]={'count':len(converted),'labels':dict(Counter(r['label'] for r in converted)),'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
    (folder/'manifest.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n');summary[kind]=info['splits']
(OUT/'conversion_report.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
(OUT/'README.md').write_text('''# 分類訓練輸入

沿用已確認的 80/10/10 分層隨機問答切分，亂數種子 42；沒有重新分配資料。問題與回答各有 train.jsonl、validation.jsonl、test.jsonl 及 manifest.json。

同一原始問答 id 不跨集合，分類文字在各集合內已去重。但文字、来源或模板可能跨集合，必須使用 --allow-random-overlap 明確指定。預設程式仍拒絕跨集合文字／群組重疊。本資料不稱為來源獨立的測試集。

這批分類器輸入是 text、label、id、group_id 等欄位。模型只用 train 學習 Word2Vec、縮放與分類參數，用 validation 選設定和門檻。試跑不加 --evaluate-test；測試資料留到設定確定後。
'''.replace('来源','來源'))
print(json.dumps({k:{s:v['count'] for s,v in x.items()} for k,x in summary.items()},ensure_ascii=False))
