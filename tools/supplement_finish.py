"""Complete cached BGE pipeline, comparisons, efficiency checks and Markdown report."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import sys, time, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from tools.supplement import OUT, DATA, RESULT, csvrows, paired, holm, SEEDS
from fgs.io import read_json,read_jsonl,write_json,write_jsonl,write_csv,file_hash
from fgs.metrics import classification
import numpy as np
import joblib

def bge_pipeline():
    from sentence_transformers import SentenceTransformer
    import torch
    torch.set_num_threads(4)
    from tools.astra_filter_comparison import execute_rows
    folder=OUT/'transformer_pipeline'
    if (folder/'completion.json').exists(): return
    filters={'transformer':{}}
    for role in ['question','answer']:
        m=joblib.load(OUT/f'transformer/random/{role}/seed42/model.joblib')
        if role=='question': encoder=SentenceTransformer(m.embedding_model,device='cpu',local_files_only=True,trust_remote_code=False)
        m.encoder=encoder
        class MemoFilter:
            def __init__(self,model): self.model=model; self.cache={}
            def accepts(self,text):
                if text not in self.cache:self.cache[text]=self.model.accepts(text)
                return self.cache[text]
        filters['transformer'][role]=MemoFilter(m)
    definitions=[('transformer__'+name,'transformer',name) for name in ['query_only','answer_only','b2','no_answer','no_query','full']]
    folder.mkdir(exist_ok=True)
    summary=execute_rows(read_jsonl(ROOT/'data/astra_direct_20261012/test_items.jsonl'),read_json(ROOT/'configs/astra.direct.json'),definitions,filters,folder,
                         ROOT/'runs/astra_filter_comparison_shared_cache',False)
    assert summary['api_accounting']['network_calls']==0
    rows=read_jsonl(folder/'responses.jsonl'); assert len(rows)==1440
    write_csv(folder/'systems.csv',summary['systems'])
    write_json(folder/'completion.json',dict(status='completed',records=1440,new_api_calls=0,sha256=file_hash(folder/'responses.jsonl'),
        note='Post-hoc BGE filter extension using original shared Astra drafts and refinements, not independent LLM draws. Original 15 systems unchanged.'))
    print('BGE pipeline complete: 1440 rows, 0 paid calls',flush=True)

def classifier_comparisons():
    output=[]
    for design in ['random','grouped']:
        for role in ['question','answer']:
            a=csvrows(OUT/f'transformer/{design}/{role}/seed42/test_predictions.csv')
            for kind in ['tfidf','word2vec']:
                path=ROOT/f'runs/random_80_10_10_20261010/{role}_test/{kind}_seed42/test_predictions.csv' if design=='random' else OUT/f'grouped/{role}/{kind}_seed42_f1.0/test_predictions.csv'
                b={r['id']:r for r in csvrows(path)}; assert set(b)=={r['id'] for r in a}
                ca=[int(r['label'])==int(r['prediction']) for r in a]
                cb=[int(b[r['id']]['label'])==int(b[r['id']]['prediction']) for r in a]
                output.append(dict(design=design,role=role,a='BGE_LR',b=kind,**paired(ca,cb,[r['group_id'] for r in a])))
    for r,p in zip(output,holm([r['cluster_signflip_p'] for r in output])):r['holm_p_8_tests']=p
    write_csv(OUT/'statistics/transformer_classifier_comparisons.csv',output)
    # BGE pipeline contrasts are a separate post-hoc family, never revise original family.
    original=read_jsonl(RESULT/'responses.jsonl'); extra=read_jsonl(OUT/'transformer_pipeline/responses.jsonl')
    from fgs.evaluation import summarize_responses
    expanded=OUT/'extended_comparison';write_jsonl(expanded/'responses.jsonl',original+extra)
    combined=summarize_responses(original+extra);write_json(expanded/'summary.json',combined);write_csv(expanded/'systems.csv',combined['systems'])
    write_json(expanded/'provenance.json',dict(items=240,systems=21,records=5040,new_api_calls=0,
        original_sha256=file_hash(RESULT/'responses.jsonl'),extension_sha256=file_hash(OUT/'transformer_pipeline/responses.jsonl'),
        note='21 systems on 240 shared questions; BGE is a post-hoc extension, not 5040 independent samples.'))
    candidates=[dict(id=r['id'],system=r['system'],question=r['question'],answer=r['answer'],category=r['category'],status=r['status'],
         automatic_flag='false_reject' if r['expected_action']=='answer' else 'false_accept',expert_notes='') for r in original+extra
         if (r['status']=='accepted')!=(r['expected_action']=='answer')]
    write_csv(OUT/'statistics/policy_failure_candidates.csv',candidates)
    from fgs.evaluation import lexical_scores
    items={r['id']:r for r in read_jsonl(ROOT/'data/astra_direct_20261012/test_items.jsonl')}
    lookup={(r['system'],r['id']):r for r in original+extra}; results=[]
    for metric in ['action_accuracy','rouge_l_character_f1']:
        ids=sorted(i for i,r in items.items() if metric=='action_accuracy' or r['category']=='related')
        def value(s,i):
            r=lookup[s,i]; accept=r['status']=='accepted'
            if metric=='action_accuracy':return int(accept==(r['expected_action']=='answer'))
            return lexical_scores(r['answer'],r['reference'])[metric] if accept else 0.
        for b in ['tfidf__full','word2vec__full','domain_b1','domain_refinement_only','transformer__b2','transformer__no_query','transformer__no_answer']:
            results.append(dict(metric=metric,a='transformer__full',b=b,**paired([value('transformer__full',i) for i in ids],[value(b,i) for i in ids],[items[i]['group_id'] for i in ids])))
    for r,p in zip(results,holm([r['cluster_signflip_p'] for r in results])):r['holm_p_14_tests']=p
    write_csv(OUT/'statistics/transformer_pipeline_comparisons.csv',results)

def benchmark():
    import torch
    torch.set_num_threads(1)
    from threadpoolctl import threadpool_limits
    from sentence_transformers import SentenceTransformer
    records=[]
    for role in ['question','answer']:
        rows=read_jsonl(DATA/role/'test.jsonl'); texts=[r['text'] for r in rows]
        paths={kind:ROOT/f'runs/random_80_10_10_20261010/{role}_train/{kind}_seed42/model.joblib' for kind in ['tfidf','word2vec']}
        paths['transformer']=OUT/f'transformer/random/{role}/seed42/model.joblib'
        for kind,path in paths.items():
            m=joblib.load(path);m.cache_dir=str(OUT/'.tokenizer_cache')
            extra_bytes=0; truncation=None
            if kind=='transformer':
                m.encoder=SentenceTransformer(m.embedding_model,device='cpu',local_files_only=True,trust_remote_code=False)
                extra_bytes=read_json(OUT/'transformer/encoder_files.json')['total_file_bytes']
                lengths=[len(m.encoder.tokenizer(t,truncation=False)['input_ids']) for t in texts]
                truncation=sum(n>m.encoder.max_seq_length for n in lengths)
            with threadpool_limits(limits=1):
                m.probabilities(texts[:8]); durations=[]
                for repeat in range(3):
                    start=time.perf_counter(); p=m.probabilities(texts);durations.append(time.perf_counter()-start)
            records.append(dict(role=role,kind=kind,n=len(texts),median_batch_ms_per_item=float(np.median(durations)*1000/len(texts)),
                                min_seconds=min(durations),max_seconds=max(durations),head_or_local_model_bytes=path.stat().st_size,
                                external_encoder_file_bytes=extra_bytes,total_deployment_bytes=path.stat().st_size+extra_bytes,truncated_items=truncation))
            print('benchmark',role,kind,flush=True)
    write_csv(OUT/'efficiency.csv',records)
    write_json(OUT/'efficiency_protocol.json',dict(device='CPU',threads=1,warmup=8,repeats=3,metric='end-to-end batched text-to-probability; not single-query serving latency',
       caveats=['Encoder weights included in Transformer size, shared once in two-role deployment.','Cache-based LLM wall times excluded.','Hardware/other workloads affect times.']))

def plots_and_report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from collections import defaultdict
    local=csvrows(OUT/'local_summary.csv'); trans=csvrows(OUT/'transformer/summary.csv')
    figures=OUT/'figures';figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for ax,role in zip(axes,['question','answer']):
        for kind in ['tfidf','word2vec']:
            means=[];sd=[]
            for f in [.25,.5,.75,1.]:
                values=[float(r['test_macro_f1']) for r in local if r['experiment']=='learning' and r['role']==role and r['kind']==kind and float(r['fraction'])==f]
                assert len(values)==3;means.append(np.mean(values));sd.append(np.std(values,ddof=1))
            ax.errorbar([25,50,75,100],means,yerr=sd,marker='o',capsize=3,label=kind)
        ax.set(title=role,xlabel='Training group fraction (%)',ylabel='Test macro-F1');ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Internal random holdout: mean +/- seed SD (not CI)');fig.tight_layout()
    fig.savefig(figures/'learning_curves.png',dpi=180);fig.savefig(figures/'learning_curves.pdf');plt.close(fig)
    aggregates=[]
    for experiment in ['grouped','permutation']:
        for role in ['question','answer']:
            for kind in ['tfidf','word2vec']:
                vals=[float(r['test_macro_f1']) for r in local if r['experiment']==experiment and r['role']==role and r['kind']==kind]
                aggregates.append(dict(experiment=experiment,role=role,kind=kind,mean=float(np.mean(vals)),seed_sd=float(np.std(vals,ddof=1))))
    for design in ['random','grouped']:
        for role in ['question','answer']:
            vals=[float(r['test_f1']) for r in trans if r['design']==design and r['role']==role]
            aggregates.append(dict(experiment=design,role=role,kind='BGE_LR',mean=float(np.mean(vals)),seed_sd=float(np.std(vals,ddof=1))))
    write_csv(OUT/'aggregate_results.csv',aggregates)
    comparison=csvrows(OUT/'statistics/paired_comparisons.csv')[0]
    systems=csvrows(RESULT/'systems.csv')+csvrows(OUT/'transformer_pipeline/systems.csv')
    core=[r for r in systems if r['system'] in ['generic_b1','domain_b1','domain_refinement_only','word2vec__full','tfidf__full','transformer__full']]
    lines=['# 第三次修改：補充實驗完成紀錄','',
      '本報告由實際輸出自動彙整。原始 15 組／240 題／3,600 筆保持不變；本次增加本機實驗及 BGE 過濾流程。所有新增分析屬事後補充，並非原先預註冊。','',
      '## 完成與未完成','',
      '|項目|狀態|證據位置（相對 revision3_code）|','|---|---|---|',
      '|原 15 組正式推論與消融|完成，3,600 筆|runs/astra_comparison_completed_20261013|',
      '|配對統計、95% CI、效應量、Holm 校正|完成，26 項原系統比較|runs/supplement_20261013/statistics|',
      '|中文 Transformer 基線|BGE＋LR，兩任務×兩切分×三種子＝12 個分類頭|runs/supplement_20261013/transformer|',
      '|BGE 完整流程與消融|6 組×240 題＝1,440 筆；既有快取，零新 API 呼叫|runs/supplement_20261013/transformer_pipeline|',
      '|來源／模板分組敏感度|兩任務×兩方法×三種子＝12 個模型；另有 BGE|runs/supplement_20261013/grouped|',
      '|學習曲線|25/50/75/100% 訓練群組，兩任務×兩方法×三種子＝48 個模型|runs/supplement_20261013/learning|',
      '|打亂訓練標籤負對照|兩任務×兩方法×三種子＝12 個模型|runs/supplement_20261013/permutation|',
      '|成本與本機效率|完成 token／呼叫彙整、模型大小、CPU 批次推論耗時|statistics/cost_by_system.csv；efficiency.csv|',
      '|全新外部資料驗證|未完成；分組切分不能替代新來源資料|需要獨立資料與標註|',
      '|新增風險挑戰與重複生成|依使用者決定，本次跳過；未執行，不列為後續必跑項目|optional_api/execution_decision.json|',
      '|人工／專家評估、論文與回覆信|未完成|新生成回答需新的評分|','',
      '## 主要流程結果','',
      '**下表是系統有無放行的政策符合率，不是答案正確率或安全率。** 通用／領域基線的 accepted 回答內可能仍包含文字拒答，需另行語意標註才能比較真正拒答品質。','',
      '|系統|題數|流程政策符合率|放行率|','|---|---:|---:|---:|']
    for r in core:lines.append(f"|{r['system']}|{r['n']}|{float(r['action_accuracy']):.4f}|{float(r['coverage']):.4f}|")
    lines += ['',f"原 TF-IDF full 對 Word2Vec full 的政策符合率差為 {float(comparison['difference'])*100:.2f} 個百分點；群組 bootstrap 95% CI [{float(comparison['ci_low'])*100:.2f}, {float(comparison['ci_high'])*100:.2f}]，26 項 Holm 校正 p={float(comparison['holm_p_26_tests']):.4f}。不能把這個小幅差異寫成已證明顯著優越。",'',
      '字元重疊／ROUGE-L 只在有參考答案的相關題上比較；拒答記零、不挑選成功回答。此指標無法證明教義忠實性。','',
      'BGE在原始分類test達到滿分，仍在完整生成流程拒絕了2題相關問題的修訂回答（full為238/240）。分類測試使用的答案分布與實際LLM回答不同，因此不能把分類滿分等同整套系統完美。原Word2Vec full為237/240、TF-IDF full為240/240。','',
      '## 補充分類結果','',
      '|實驗|任務|方法|test macro-F1 平均|seed SD|','|---|---|---|---:|---:|']
    for r in aggregates:lines.append(f"|{r['experiment']}|{r['role']}|{r['kind']}|{r['mean']:.4f}|{r['seed_sd']:.4f}|")
    lines += ['', 'random＝原始隨機切分；grouped＝現有語料重新分組切分；permutation＝只打亂訓練標籤的診斷。不同切分的 test 題目／數量不同，不能把兩表差值當成配對效果。三種子共用同一 test，seed SD 不是泛化信賴區間。','',
      '來源分組下，答案分類的TF-IDF仍高於這次測試的BGE與Word2Vec。不能再以「Word2Vec一定較佳」作為研究前提；比較結果與效率權衡才是目前有證據支持的說法。','',
      '## 效率與成本','',
      '|任務|方法|CPU批次每題中位耗時(ms)|部署檔案大小(MiB)|','|---|---|---:|---:|']
    for r in csvrows(OUT/'efficiency.csv'):
        lines.append(f"|{r['role']}|{r['kind']}|{float(r['median_batch_ms_per_item']):.3f}|{int(r['total_deployment_bytes'])/1048576:.2f}|")
    accounting=read_json(OUT/'statistics/accounting.json')
    equivalent=(accounting['prompt_tokens']*10+accounting['completion_tokens']*50)/1e6
    lines += ['',f"原正式實驗依cache key去重，共{accounting['unique_formal_successful_calls']}個成功API呼叫，input {accounting['prompt_tokens']:,}、output {accounting['completion_tokens']:,} tokens；按先前查核的標準牌價估算約US${equivalent:.2f}，不是平台帳單。本次補充執行新增API費用為零。",'',
      'CPU採單執行緒、暖機後三次整批推論中位數，不等於線上單題延遲。BGE大小已含外部encoder；問題與答案共用同一encoder部署時只需計算一次。硬體紀錄見hardware.json。','',
      '## 方法與解讀限制','',
      '- 原本隨機切分完全保留。新分組實驗先以所有已知 group_ids、正規化相同文字、字元三連組 Jaccard≥0.9 建立連通群組，再以 seed42 建立約60/20/20；問題與答案各自切分，只用於分類敏感度，不能組成來源獨立的完整生成流程。',
      '- 所有超參數依 validation macro-F1、再依 log loss 選擇；threshold固定0.5。Word2Vec新實驗從該次 train 重訓詞向量，MLP最多300 epochs、patience25；不修改原Linux模型。Windows重訓的差異與資料比例差異須分開解讀。',
      '- 學習曲線抽樣單位為訓練群組，實際列數見 local_summary.csv，比例不是精確的資料列百分比；各seed使用巢狀群組子集。',
      '- Windows重新擬合TF-IDF的答案詞彙表，與原保存模型各有2,776個不同詞彙（各保留30,000個特徵，C均為10）；見 tfidf_refit_audit.json。兩次fit不保證跨平台完全相同，原因尚未單獨隔離；新學習曲線100%端點採本次fit，原正式流程仍採原保存模型，不能混用或偷偷替換。',
      '- BGE模型為固定預訓練中文編碼器，不做LLM或Transformer微調。使用原始文字、不加檢索指令、模型預設CLS pooling、正規化向量、train-only scaler與LR。固定revision與權重hash見 transformer/encoder.json 和 encoder_files.json。',
      '- BGE選型基於中文及CPU可行性，不宣稱最新或最強。512 token截斷數見效率表；預訓練資料污染不能排除。官方模型說明：https://huggingface.co/BAAI/bge-small-zh-v1.5',
      '- 新群組測試仍來自同一已看過結果的語料；未標出的語意／來源關係仍可能重疊。不能改稱全新外部驗證。',
      '- 統計主要以來源／模板群組 bootstrap 10,000次及群組符號置換10,000次；差值為效果量。原26項、BGE分類8項、BGE流程14項分開列為補充比較家族，未主張跨全部家族控制同一FWER。',
      '- 全對或全錯的經驗bootstrap可能產生退化區間，並不表示母體沒有不確定性。McNemar只作忽略群組依賴的敏感度，不作主要證據。',
      '- 負對照仍以真實validation選模，因此不是正式隨機置換檢定；只作洩漏／學習行為診斷。',
      '- 成本以共享cache key去重。各方法獨立部署的token成本不能相加當成這次實付；歷史失敗、pilot、稅及其他工作不含。LLM快取時間不作速度排名。',
      '- 本次不啟用新的安全guard；主題相關不代表安全。新增安全機制須以新協定公平套用基線再測。','',
      '## 主要檔案與重現方式','',
      '工作目錄：`D:\\FGS-LLM\\revision3_code`。以下命令使用既有 checkpoint；原始資料／結果不覆寫。','',
      '```powershell',
      'cd D:\\FGS-LLM\\revision3_code',
      '.\\.venv_windows\\Scripts\\python.exe tools/supplement.py stats',
      '.\\.venv_windows\\Scripts\\python.exe tools/supplement.py local',
      '.\\.venv_windows\\Scripts\\python.exe tools/supplement.py transformer',
      '.\\.venv_windows\\Scripts\\python.exe tools/supplement_finish.py',
      '```','',
      '本次已跑完的項目不用再跑。結果：`runs/supplement_20261013/`；圖：`figures/learning_curves.png`、`.pdf`；配對逐筆分數：`statistics/item_scores.csv`；分類預測及超參數搜尋保存於各模型目錄。','',
      '整合原15組與新增BGE六組的查閱版：`extended_comparison/`（21組×240題＝5,040筆，仍只有240題）。原3,600筆檔案不改。`statistics/policy_failure_candidates.csv`列出政策誤放行／誤拒絕候選，尚未作專家解讀。','',
      '程式：`tools/supplement.py`、`tools/supplement_finish.py`、`tools/verify_supplement.py`；測試：`tests/test_supplement.py`；環境與驗證檔均在 `runs/supplement_20261013/` 下：`supplement_requirements.txt`、`verification.json`。','',
      '## 本次跳過新增 API 實驗','',
      '後續所有需要API Key的工作一律跳過，包括新增生成、重複抽樣、風險挑戰、LLM評審／分類、雲端embedding、微調與外部資料的API推論。適用所有供應商，不限於OpenAI或是否收費。不得要求輸入Key、使用既有Key或因快取缺漏改呼叫API；快取不足時將該項標為跳過。只繼續不需Key的本機處理、現有快取分析、人工評估與論文整理。','',
      '依使用者決定，本次跳過新增的重複生成與風險挑戰API測試，不再要求輸入API Key或安排新呼叫。先前完成的正式推論與快取評測保留。相關腳本與題目只作歷史準備紀錄，不代表已完成實驗。','',
      '研究限制仍需保留：未執行多次生成抽樣與新增風險挑戰，因此不主張已驗證生成隨機性穩定或通過此類安全測試。既有回答仍可交由人工／專家評估。','',
      '## 下一步順序','',
      '1. 先看本文件及配對／分類表，選定論文主張；不以test結果回頭改超參數。',
      '2. 不再新增API呼叫；沿用目前已完成的回答及本機結果，進行分析與人工評估。',
      '3. 若補來源獨立的新題目，僅做不需Key的本機分類評估；需要API的完整生成評估跳過。本次群組結果不能替代新外部資料驗證。',
      '4. 完成新回答的盲評、宗教專家忠實性與風險標註，以及文字拒答判讀；再做人評统计。',
      '5. 更新論文、圖表、Algorithm與逐條審稿回覆，明確揭露微調改為推論時修訂及本次補充設計。','']
    (ROOT/'docs/SUPPLEMENT_COMPLETED_20261013.md').write_text('\n'.join(lines),encoding='utf-8')

if __name__=='__main__':
    for phase in ['stats','local','transformer']:assert (OUT/(phase+'_completion.json')).exists(),phase+' incomplete'
    bge_pipeline();classifier_comparisons();benchmark();plots_and_report()
    write_json(OUT/'finish_completion.json',dict(status='completed',tool_sha256=file_hash(__file__)))
