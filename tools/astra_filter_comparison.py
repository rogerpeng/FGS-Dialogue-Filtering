"""Versioned comparison of saved filter backends; never retrain or silently resplit.

prepare/preflight/verify-windows are offline. run needs --live or an existing
complete cache. All common domain draws are shared; mock output is not evidence.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fgs.io import file_hash, read_json, read_jsonl, write_json, write_jsonl, write_csv, environment

DEFAULT_PLAN = ROOT / 'data/astra_filter_comparison_20261013/protocol.json'
COMMON = [('domain_b1', None, 'b1'), ('domain_refinement_only', None, 'refinement_only')]
FILTERED = ['query_only', 'answer_only', 'b2', 'no_answer', 'no_query', 'full']
PROFILES = ['word2vec', 'tfidf']


def systems(phase):
    if phase == 'generic':
        return [('generic_b1', None, 'b1')]
    variants = ['full'] if phase == 'pilot' else FILTERED
    return COMMON + [(f'{profile}__{v}', profile, v) for profile in PROFILES for v in variants]


def fresh(path):
    path = Path(path)
    if path.exists():
        raise ValueError(f'Output already exists; choose a new path: {path}')
    path.mkdir(parents=True)
    return path


def prepare(path=DEFAULT_PLAN):
    old_path = ROOT / 'data/astra_direct_20261012/protocol.json'
    old = read_json(old_path)
    files = {str(old_path.relative_to(ROOT)): file_hash(old_path)}
    for key, expected in old['code_hashes'].items():
        if file_hash(ROOT / key) != expected:
            raise ValueError('Original frozen code changed: ' + key)
    for key, expected in old['data_hashes'].items():
        if file_hash(old_path.parent / key) != expected:
            raise ValueError('Original data changed: ' + key)
    for key, expected in old['active_config_hashes'].items():
        if file_hash(ROOT / 'configs' / key) != expected:
            raise ValueError('Original config changed: ' + key)
    for role, expected in old['model_hashes'].items():
        if file_hash(ROOT / old['classifier_models'][role]) != expected:
            raise ValueError('Original model changed: ' + role)
    models = {}
    for profile in PROFILES:
        models[profile] = {}
        for role in ['question', 'answer']:
            relative = f'runs/random_80_10_10_20261010/{role}_train/{profile}_seed42/model.joblib'
            models[profile][role] = relative
            files[relative] = file_hash(ROOT / relative)
            metrics_path = (ROOT / relative).with_name('metrics.json')
            metrics = read_json(metrics_path)
            if metrics['kind'] != profile or metrics['seed'] != 42 or metrics['threshold'] != .5:
                raise ValueError('Unexpected saved model metadata')
            files[str(metrics_path.relative_to(ROOT))] = file_hash(metrics_path)
    phases = {}
    for phase in ['pilot', 'test', 'generic']:
        item_path = old_path.parent / ('pilot_validation_items.jsonl' if phase == 'pilot' else 'test_items.jsonl')
        config = ROOT / 'configs' / {'pilot': 'astra.pilot.json', 'test': 'astra.direct.json', 'generic': 'astra.generic.json'}[phase]
        items = read_jsonl(item_path)
        n = 2 if phase == 'pilot' else 240
        if len(items) != n or len({r['id'] for r in items}) != n:
            raise ValueError('Unexpected evaluation items')
        files[str(item_path.relative_to(ROOT))] = file_hash(item_path)
        files[str(config.relative_to(ROOT))] = file_hash(config)
        phases[phase] = {'items': str(item_path.relative_to(ROOT)), 'config': str(config.relative_to(ROOT)),
                         'n': n, 'systems': systems(phase), 'new_call_upper_bound': n * (1 if phase == 'generic' else 2)}
    for p in sorted((ROOT / 'fgs').glob('*.py')) + [Path(__file__)]:
        files[str(p.relative_to(ROOT))] = file_hash(p)
    plan = {'schema_version': 1, 'date': '2026-10-13', 'status': 'prepared_no_live_results',
            'classifier_models': models, 'phases': phases, 'files': files,
            'seed': 42, 'threshold': .5, 'replicates': 1,
            'primary_contrast': ['tfidf__full', 'word2vec__full'],
            'feature_only_claim_allowed': False,
            'comparison': 'Word2Vec+MLP versus character TF-IDF+LogisticRegression; both representation and head differ.',
            'selection_note': 'Comparison added after observing internal classifier test results; not preregistered feature selection.',
            'limits': ['Internal random split retains text/source overlap.', 'Relevance is not fidelity or safety.',
                       'No independent guard or RAG.', 'Same drafts/refinements shared across profiles.',
                       '15 systems including generic; 3600 records are only 240 questions.',
                       'New blinded human assessment and source-independent evaluation still required.']}
    fresh(Path(path).parent)
    write_json(path, plan)
    write_json(Path(path).with_name('lock.json'), {'protocol_sha256': file_hash(path)})
    return preflight(path)


def preflight(path=DEFAULT_PLAN):
    path = Path(path)
    if file_hash(path) != read_json(path.with_name('lock.json'))['protocol_sha256']:
        raise ValueError('Protocol changed; prepare a new version')
    plan = read_json(path)
    if plan['schema_version'] != 1:
        raise ValueError('Unknown protocol')
    for relative, expected in plan['files'].items():
        if file_hash(ROOT / relative) != expected:
            raise ValueError('Frozen input changed: ' + relative)
    for phase, spec in plan['phases'].items():
        if spec['systems'] != [list(s) for s in systems(phase)]:
            raise ValueError('Protocol system definitions differ')
        cfg = read_json(ROOT / spec['config'])
        if any(cfg[k] != 'gpt-6-astra' for k in ['generator_model', 'refiner_model']):
            raise ValueError('Unexpected LLM model')
        if cfg['max_calls'] < spec['new_call_upper_bound']:
            raise ValueError('Insufficient call cap')
        if cfg.get('guard_model') or cfg['max_generation_attempts'] != 1 or cfg['max_refinements'] != 1:
            raise ValueError('Shared-draw protocol requires no guard and one attempt')
    return plan


def load_filters(plan, cache):
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    filters = {}
    for profile, paths in plan['classifier_models'].items():
        filters[profile] = {}
        for role, relative in paths.items():
            model = joblib.load(ROOT / relative)
            expected = MLPClassifier if profile == 'word2vec' else LogisticRegression
            if model.kind != profile or model.seed != 42 or model.threshold != .5 or not isinstance(model.classifier, expected):
                raise ValueError('Loaded model does not match protocol')
            # Linux's saved tokenizer cache path is not portable. Change only
            # this in-memory runtime path; never rewrite fitted model files.
            model.cache_dir = str(Path(cache).resolve())
            filters[profile][role] = model
    return filters


def execute_rows(items, cfg, definitions, filters, out, cache, live=False):
    from fgs.api import APIBackend, ChatClient
    from fgs.pipeline import DialoguePipeline, VARIANTS
    from fgs.evaluation import summarize_responses
    client = ChatClient(cfg, cache, live)
    backend = APIBackend(client, cfg['generator_model'], cfg['refiner_model'])
    rows = []
    for item in items:
        for system, profile, variant in definitions:
            pair = filters.get(profile, {})
            pipeline = DialoguePipeline(backend, pair.get('question'), pair.get('answer'),
                                        max_generation_attempts=1, max_refinements=1)
            start = len(client.events)
            row = pipeline.run(item['question'], VARIANTS[variant], item['id'], 0)
            row.update(system=system, filter_profile=profile, original_variant=variant,
                       api_events=client.events[start:], reference=item.get('reference', ''),
                       category=item.get('category', 'unspecified'), expected_action=item.get('expected_action', 'unlabeled'),
                       synthetic=bool(item.get('synthetic', False)))
            rows.append(row)
            write_jsonl(out / 'responses.jsonl', rows)
            if row['status'] == 'provider_or_filter_error' or client.fatal_error is not None:
                write_json(out / 'failure.json', {'status': 'incomplete', 'records_saved': len(rows),
                           'error': client.fatal_error, 'trace': row['trace'], 'network_calls': client.calls})
                raise RuntimeError('Stopped on provider/filter error; inspect failure.json')
    summary = summarize_responses(rows)
    summary['api_accounting'] = {'network_calls': client.calls, 'cache_hits': client.cache_hits,
                                  'logical_calls': client.logical_calls, 'reported_new_call_tokens': client.billed_usage}
    summary['latency_note'] = 'Shared-cache wall times are not independent model latency measurements.'
    write_json(out / 'summary.json', summary)
    return summary


def run(path, phase, out, cache, live=False):
    plan = preflight(path)
    spec = plan['phases'][phase]
    out = fresh(out)
    cfg = read_json(ROOT / spec['config'])
    filters = load_filters(plan, out / '.tokenizer_cache') if phase != 'generic' else {}
    write_json(out / 'config.json', {'protocol_sha256': file_hash(path), 'phase': phase, 'api': cfg,
                                     'systems': spec['systems'], 'items_sha256': file_hash(ROOT / spec['items']),
                                     'classifier_models': plan['classifier_models']})
    write_json(out / 'environment.json', environment())
    summary = execute_rows(read_jsonl(ROOT / spec['items']), cfg, spec['systems'], filters, out, cache, live)
    receipt = {'status': 'completed', 'phase': phase, 'protocol_sha256': file_hash(path),
               'records': spec['n'] * len(spec['systems']),
               'files': {name: file_hash(out / name) for name in ['config.json', 'environment.json', 'responses.jsonl', 'summary.json']}}
    write_json(out / 'receipt.json', receipt)
    return summary


def combine(path, domain, generic, out):
    from fgs.evaluation import summarize_responses
    plan = preflight(path)
    combined, sources = [], {}
    for phase, folder in [('test', Path(domain)), ('generic', Path(generic))]:
        receipt = read_json(folder / 'receipt.json')
        if receipt['status'] != 'completed' or receipt['phase'] != phase or receipt['protocol_sha256'] != file_hash(path):
            raise ValueError('Incomplete run or different protocol/phase')
        for name, expected in receipt['files'].items():
            if file_hash(folder / name) != expected:
                raise ValueError('Completed run changed: ' + name)
        spec = plan['phases'][phase]
        items = {r['id']: r for r in read_jsonl(ROOT / spec['items'])}
        rows = read_jsonl(folder / 'responses.jsonl')
        expected_keys = {(i, 0, s[0]) for i in items for s in spec['systems']}
        if len(rows) != len(expected_keys) or {(r['id'], r['replicate'], r['system']) for r in rows} != expected_keys:
            raise ValueError('Incomplete or duplicated systems/items')
        for r in rows:
            if r['status'] == 'provider_or_filter_error' or r['question'] != items[r['id']]['question']:
                raise ValueError('Invalid response or mismatched question')
        combined.extend(rows)
        sources[phase] = {'path': str(folder.resolve()), 'sha256': file_hash(folder / 'responses.jsonl')}
    out = fresh(out)
    write_jsonl(out / 'responses.jsonl', combined)
    summary = summarize_responses(combined)
    write_json(out / 'summary.json', summary)
    write_csv(out / 'systems.csv', summary['systems'])
    write_json(out / 'provenance.json', {'protocol_sha256': file_hash(path), 'sources': sources,
                'items': plan['phases']['test']['n'], 'systems': 15, 'records': len(combined), 'new_api_calls': 0})
    return {'output': str(out), 'records': len(combined), 'systems': 15}


def verify_windows(path, out):
    """Real model reload/prediction parity plus mocked HTTP, with no paid calls."""
    import numpy as np
    import joblib
    from unittest.mock import patch
    plan = preflight(path)
    out = fresh(out)
    parity = []
    with tempfile.TemporaryDirectory(prefix='fgs_dual_verify_') as tmp:
        tmp = Path(tmp)
        for role in ['question', 'answer']:
            for profile in PROFILES:
                for seed in [42, 43, 44]:
                    source = ROOT / f'runs/random_80_10_10_20261010/{role}_train/{profile}_seed{seed}'
                    model = joblib.load(source / 'model.joblib')
                    model.cache_dir = str(tmp / 'tokenizer')
                    for split in ['train', 'validation', 'test']:
                        folder = source if split != 'test' else ROOT / f'runs/random_80_10_10_20261010/{role}_test/{profile}_seed{seed}'
                        with (folder / f'{split}_predictions.csv').open(encoding='utf-8-sig', newline='') as f:
                            saved = list(csv.DictReader(f))
                        probabilities = model.probabilities([r['text'] for r in saved])
                        delta = float(np.max(np.abs(probabilities - np.array([float(r['probability']) for r in saved]))))
                        mismatches = sum(int(p >= model.threshold) != int(r['prediction']) for p, r in zip(probabilities, saved))
                        parity.append({'role': role, 'profile': profile, 'seed': seed, 'split': split,
                                       'n': len(saved), 'label_mismatches': mismatches, 'max_probability_delta': delta})
        write_json(out / 'prediction_parity.json', parity)
        if any(r['label_mismatches'] or r['max_probability_delta'] > 1e-6 for r in parity):
            raise ValueError('Cross-platform prediction mismatch; do not overwrite or retrain originals')

        class Response:
            headers = {'x-request-id': 'offline-only'}
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self):
                return json.dumps({'model': 'gpt-6-astra', 'choices': [{'finish_reason': 'stop',
                      'message': {'content': '可以先平靜心情，理解彼此的需要，再以尊重的態度溝通。'}}],
                      'usage': {'prompt_tokens': 10, 'completion_tokens': 20, 'total_tokens': 30}}).encode()

        spec = plan['phases']['pilot']
        items = read_jsonl(ROOT / spec['items'])
        filters = load_filters(plan, tmp / 'tokenizer')
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'offline-test-placeholder'}), patch('urllib.request.urlopen', return_value=Response()) as network:
            mock_out = fresh(tmp / 'pilot')
            summary = execute_rows(items, read_json(ROOT / spec['config']), systems('pilot'), filters, mock_out, tmp / 'cache', True)
            count = network.call_count
            if count != 4 or len(read_jsonl(mock_out / 'responses.jsonl')) != 8:
                raise AssertionError('Unexpected shared-cache pilot count')
            for call in network.call_args_list:
                payload = json.loads(call.args[0].data)
                if payload['model'] != 'gpt-6-astra' or payload['reasoning_effort'] != 'low' or 'temperature' in payload:
                    raise AssertionError('Wrong API payload')
        verification = {'status': 'passed', 'real_api_calls': 0, 'mock_http_calls': count,
                        'pilot_records_checked': 8, 'saved_models_checked': 12, 'split_checks': len(parity),
                        'rows_checked': sum(r['n'] for r in parity),
                        'max_probability_delta': max(r['max_probability_delta'] for r in parity),
                        'environment': environment(), 'protocol_sha256': file_hash(path),
                        'note': 'Mock responses deleted; no synthetic study scores exported. Original model files unchanged.'}
    write_json(out / 'verification.json', verification)
    preflight(path)
    return verification


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare', 'preflight', 'run', 'combine', 'verify-windows'])
    p.add_argument('--protocol', type=Path, default=DEFAULT_PLAN)
    p.add_argument('--phase', choices=['pilot', 'test', 'generic'], default='pilot')
    p.add_argument('--out', type=Path)
    p.add_argument('--cache-dir', type=Path, default=ROOT / 'runs/astra_filter_comparison_shared_cache')
    p.add_argument('--live', action='store_true')
    p.add_argument('--domain', type=Path)
    p.add_argument('--generic', type=Path)
    a = p.parse_args()
    if a.command == 'prepare':
        result = prepare(a.protocol)
    elif a.command == 'preflight':
        plan = preflight(a.protocol)
        result = {'status': 'preflight_passed_no_api_calls', 'checked_files': len(plan['files']),
                  'phases': {k: {'items': v['n'], 'systems': len(v['systems'])} for k, v in plan['phases'].items()}}
    else:
        if not a.out: p.error('--out is required')
        if a.command == 'run':
            result = run(a.protocol, a.phase, a.out, a.cache_dir, a.live)
        elif a.command == 'verify-windows':
            result = verify_windows(a.protocol, a.out)
        else:
            if not a.domain or not a.generic: p.error('--domain and --generic required')
            result = combine(a.protocol, a.domain, a.generic, a.out)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
