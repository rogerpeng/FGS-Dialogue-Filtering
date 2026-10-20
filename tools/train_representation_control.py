"""Supplementary fixed-representation LR control. Never fit Word2Vec/scaler.

Reuses each previously selected Word2Vec representation and fixed random split.
This is post-hoc relative to earlier test inspection, not a new unseen holdout.
"""
import argparse
import copy
import csv
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from fgs.io import read_json, read_jsonl, write_json, write_jsonl, write_csv, file_hash, environment
from fgs.metrics import classification


def run(out):
    out = Path(out)
    if out.exists(): raise ValueError('Use a new output folder')
    specs = []
    for role in ['question', 'answer']:
        data = ROOT / f'data/random_80_10_10_20261010/{role}'
        original = ROOT / f'runs/random_80_10_10_20261010/{role}_train'
        hashes = read_json(original / 'data_hashes.json')
        for split in ['train', 'validation']:
            if file_hash(data / f'{split}.jsonl') != hashes[split]:
                raise ValueError('Training source changed')
        for seed in [42, 43, 44]:
            source = original / f'word2vec_seed{seed}/model.joblib'
            specs.append({'role': role, 'seed': seed, 'data': str(data), 'source': str(source),
                          'source_sha256': file_hash(source),
                          'data_hashes': {s: file_hash(data / f'{s}.jsonl') for s in ['train', 'validation', 'test']}})
    out.mkdir(parents=True)
    write_json(out / 'protocol.json', {'Cs': [.1, 1., 10.], 'threshold': .5, 'seeds': [42,43,44], 'specs': specs,
               'selection': 'validation macro-F1 then negative log_loss; no test-driven selection',
               'fitted': 'LogisticRegression head only; fixed existing Word2Vec and StandardScaler',
               'limitations': ['Earlier Word2Vec representation selected with an MLP validation search.',
                              'Random text/source overlap remains; post-hoc supplemental analysis.',
                              'TF-IDF versus Word2Vec differs in tokenization, scaling and representation, not embedding alone.']})
    write_json(out / 'environment.json', environment())
    summary, checks = [], []
    for spec in specs:
        data, source = Path(spec['data']), Path(spec['source'])
        model = joblib.load(source)
        model.cache_dir = str((out / '.tokenizer_cache').resolve())
        vectors_hash = joblib.hash(model.word_vectors)
        scaler_hash = joblib.hash(model.scaler)
        train, val = [read_jsonl(data / f'{s}.jsonl') for s in ['train', 'validation']]
        tx = model._features([r['text'] for r in train], fit=False)
        vx = model._features([r['text'] for r in val], fit=False)
        y, vy = [r['label'] for r in train], [r['label'] for r in val]
        trials, best, best_score, best_c = [], None, None, None
        start = time.perf_counter()
        for c in [.1, 1., 10.]:
            head = LogisticRegression(C=c, max_iter=2000, random_state=spec['seed']).fit(tx, y)
            p = head.predict_proba(vx)[:, list(head.classes_).index(1)]
            metrics = classification(vy, p, .5)
            score = (metrics['macro_f1'], -metrics['log_loss'])
            trials.append({'C': c, 'validation_macro_f1': metrics['macro_f1'], 'validation_log_loss': metrics['log_loss']})
            if best_score is None or score > best_score:
                best, best_score, best_c = head, score, c
        model.classifier = best
        model.kind = 'word2vec_linear'
        model.neural = False
        model.C = best_c
        model.threshold = .5
        model.history, model.best_epoch = [], None
        dest = out / spec['role'] / f'word2vec_linear_seed{spec["seed"]}'
        dest.mkdir(parents=True)
        joblib.dump(model, dest / 'model.joblib')
        write_csv(dest / 'validation_search.csv', trials)
        write_json(dest / 'selection.json', {'C': best_c, 'threshold': .5, 'head_fit_seconds': time.perf_counter()-start,
                     'test_used_for_selection': False, 'source_model_sha256': spec['source_sha256']})
        # Test opened only after choosing and saving the head.
        all_metrics = {}
        for split, rows in [('train', train), ('validation', val), ('test', read_jsonl(data / 'test.jsonl'))]:
            probabilities = model.probabilities([r['text'] for r in rows])
            metrics = classification([r['label'] for r in rows], probabilities, .5)
            all_metrics[split] = metrics
            write_csv(dest / f'{split}_predictions.csv', [dict(id=r['id'], group_id=r['group_id'], text=r['text'],
                      label=r['label'], probability=float(p), prediction=int(p >= .5)) for r,p in zip(rows,probabilities)])
            summary.append({'task':spec['role'], 'model':'word2vec_linear', 'seed':spec['seed'], 'split':split,
                            'selected_C':best_c, 'n':len(rows), 'accuracy':metrics['accuracy'], 'macro_f1':metrics['macro_f1']})
        write_json(dest / 'metrics.json', {'kind':model.kind, 'seed':spec['seed'], 'threshold':.5, 'metrics':all_metrics})
        unchanged = (vectors_hash == joblib.hash(model.word_vectors) and scaler_hash == joblib.hash(model.scaler)
                     and file_hash(source) == spec['source_sha256'])
        loaded = joblib.load(dest / 'model.joblib')
        reload_ok = np.allclose(loaded.probabilities([r['text'] for r in val]), model.probabilities([r['text'] for r in val]), atol=0, rtol=0)
        checks.append({'role':spec['role'], 'seed':spec['seed'], 'vectors_scaler_original_unchanged':unchanged,
                       'saved_head_reload_identical':bool(reload_ok)})
        if not unchanged or not reload_ok: raise ValueError('Representation/reload integrity check failed')
    write_csv(out / 'summary.csv', summary)
    write_json(out / 'checks.json', checks)
    aggregates=[]
    for role in ['question','answer']:
        for kind in ['word2vec','tfidf','word2vec_linear']:
            values=[]
            for seed in [42,43,44]:
                p = out / role / f'{kind}_seed{seed}/metrics.json' if kind=='word2vec_linear' else ROOT / f'runs/random_80_10_10_20261010/{role}_test/{kind}_seed{seed}/metrics.json'
                values.append(read_json(p)['metrics']['test']['macro_f1'])
            aggregates.append({'task':role,'model':kind,'test_macro_f1_mean':float(np.mean(values)),
                               'test_macro_f1_seed_sd':float(np.std(values,ddof=1))})
    write_csv(out / 'comparison.csv', aggregates)
    write_json(out / 'completion.json', {'status':'completed','heads_trained':6,'word2vec_training_calls':0,
                'original_models_overwritten':0,'api_calls':0,'checks_passed':all(x['vectors_scaler_original_unchanged'] and x['saved_head_reload_identical'] for x in checks)})
    print(aggregates)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True,type=Path)
    run(parser.parse_args().out)
