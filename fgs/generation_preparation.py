"""Prepare an existing random split for generation; no resplitting or API calls."""
import csv
from pathlib import Path

from .api import GENERATION_PROMPT, REFINEMENT_PROMPT
from .io import file_hash, new_output, read_json, read_jsonl, write_csv, write_json, write_jsonl
from .pipeline import VARIANTS


def prepare_generation(split_dir, out):
    split_dir = Path(split_dir)
    manifest = read_json(split_dir / 'split_manifest.json')
    if manifest.get('method') != 'stratified_random_by_qa_row':
        raise ValueError('An explicit existing random QA split is required')
    source_checksums = read_json(split_dir / 'checksums.json')
    splits, hashes, all_ids = {}, {}, set()
    for name in ('train', 'validation', 'test'):
        path = split_dir / name / 'qa.json'
        rows = read_json(path)
        if [r['id'] for r in rows] != manifest['splits'][name]['qa_ids']:
            raise ValueError('QA order/ids differ from frozen split manifest')
        if file_hash(path) != source_checksums[f'{name}/qa.json']:
            raise ValueError('QA input hash differs from frozen split')
        ids = [r['id'] for r in rows]
        if len(ids) != len(set(ids)) or all_ids.intersection(ids):
            raise ValueError('QA ids must be unique across splits')
        all_ids.update(ids)
        for r in rows:
            if r.get('review_status') != 'approved' or not r.get('reviewer_id'):
                raise ValueError('Only approved QA with review provenance is permitted')
            if type(r.get('label')) is not int or r['label'] not in (0, 1):
                raise ValueError('Invalid relevance label')
            if not all(isinstance(r.get(k), str) and r[k].strip() for k in ('question', 'answer')):
                raise ValueError('Empty QA content')
        related = [r for r in rows if r['label'] == 1]
        if len(related) != manifest['splits'][name]['related']:
            raise ValueError('Related count differs from frozen manifest')
        splits[name] = rows
        hashes[f'{name}/qa.json'] = file_hash(path)
    out = new_output(out)
    counts, mapping = {}, []
    for name, rows in splits.items():
        related = [r for r in rows if r['label'] == 1]
        chats = [{'messages': [{'role': 'system', 'content': GENERATION_PROMPT},
                               {'role': 'user', 'content': r['question']},
                               {'role': 'assistant', 'content': r['answer']}]} for r in related]
        write_jsonl(out / 'generation' / f'{name}.jsonl', chats)
        for index, r in enumerate(related, 1):
            mapping.append({'split': name, 'row': index, 'qa_id': r['id'],
                            'group_id': r['group_id'], 'origin': r['origin'],
                            'reviewer_id': r['reviewer_id']})
        counts[name] = len(chats)
        if name != 'test':
            write_csv(out / 'refinement' / f'{name}_draft_collection.csv',
                      [{'qa_id': r['id'], 'split': name, 'group_id': r['group_id'],
                        'question': r['question'], 'reference_answer': r['answer'],
                        'draft': '', 'draft_model': '', 'draft_generated_at': '',
                        'draft_request_id': '', 'draft_input_sha256': '',
                        'corrected_answer': '', 'review_status': 'pending',
                        'reviewer_id': '', 'revision_reason': ''} for r in related])
    write_csv(out / 'generation' / 'PRIVATE_row_mapping.csv', mapping)
    # Reference is excluded from the model input artifact, including unrelated answers.
    evaluation, requests = [], []
    for r in splits['test']:
        synthetic = r['origin'] != 'legacy_paired'
        evaluation.append({'id': r['id'], 'question': r['question'],
                           'reference': r['answer'] if r['label'] == 1 else '',
                           'expected_action': 'answer' if r['label'] == 1 else 'abstain',
                           'category': 'related' if r['label'] == 1 else 'unrelated',
                           'synthetic': synthetic, 'synthetic_basis': 'non-legacy origin; legacy authorship unknown',
                           'group_id': r['group_id'], 'origin': r['origin']})
        requests.append({'id': r['id'], 'messages': [{'role': 'system', 'content': GENERATION_PROMPT},
                                                  {'role': 'user', 'content': r['question']}]})
    write_jsonl(out / 'evaluation' / 'internal_test_items.jsonl', evaluation)
    write_jsonl(out / 'evaluation' / 'internal_test_requests_no_reference.jsonl', requests)
    write_json(out / 'prompts.json', {'generation': GENERATION_PROMPT, 'refinement': REFINEMENT_PROMPT})
    write_json(out / 'generation_training_config.template.json',
               {'base_url': 'https://api.openai.com/v1', 'api_key_env': 'OPENAI_API_KEY',
                'base_model': 'YOUR_SUPPORTED_BASE_MODEL', 'task': 'generation', 'seed': 42,
                'hyperparameters': {'n_epochs': 'auto', 'batch_size': 'auto', 'learning_rate_multiplier': 'auto'}})
    plan = {'status': 'draft_not_preregistered_not_executed', 'split_seed': manifest['seed'],
            'split_policy': manifest['method'], 'source_independent': False,
            'evaluation_items': len(evaluation), 'related': sum(r['category'] == 'related' for r in evaluation),
            'unrelated': sum(r['category'] == 'unrelated' for r in evaluation),
            'variants': {name: vars(v) for name, v in VARIANTS.items()},
            'replicates': 1, 'max_generation_attempts': 3, 'max_refinements': 3,
            'classifier_seed': 42, 'thresholds': {'question': 0.5, 'answer': 0.5},
            'classifier_choice': 'word2vec seed42 fixed before generation evaluation; tfidf comparison separately',
            'generator_model': None, 'refiner_model': None, 'independent_guard_model': None,
            'guard_note': 'No actual independent guard configured or validated',
            'conservative_generation_refinement_call_bound': len(evaluation) * len(VARIANTS) * 6,
            'call_bound_note': 'Loose bound before query rejection/cache; excludes guard and LLM-filter calls; not a monetary estimate',
            'metrics': ['coverage', 'action_accuracy', 'character_f1', 'rouge_l_character_f1'],
            'reference_scoring': 'Related references only; refusals score zero, no accepted-only cherry picking',
            'limitations': ['lexical scores are not religious fidelity or safety',
                           'domain-unrelated abstention is this protocol policy, not a harmfulness label',
                           'random split permits shared texts/source groups',
                           'synthetic flag is conservative origin-based, not verified legacy authorship'],
            'missing': ['training platform/model/budget', 'real generation checkpoint',
                        'reviewed draft-to-correction training pairs', 'real refinement checkpoint',
                        'modern base-model baseline configuration', 'independent expert evaluation']}
    write_json(out / 'experiment_plan.json', plan)
    audit = {'status': 'offline_prepared_no_upload_no_training', 'qa_counts': {k: len(v) for k,v in splits.items()},
             'generation_counts': counts, 'evaluation_n': len(evaluation),
             'qa_ids_preserved': True, 'question_answer_content_preserved': True,
             'system_prompt_aligned_with_inference': True, 'test_reference_excluded_from_request_messages': True,
             'source_manifest_sha256': file_hash(split_dir / 'split_manifest.json'),
             'source_hashes': hashes, 'refinement_approved_pairs': 0,
             'review': 'user-confirmed QA approval; independent refinement review not completed',
             'cross_split_overlap': 'retained as requested; not suitable for strict group-disjoint prepare_job without an explicit policy extension'}
    write_json(out / 'preparation_audit.json', audit)
    write_json(out / 'checksums.json', {str(p.relative_to(out)): file_hash(p) for p in sorted(out.rglob('*')) if p.is_file()})
    return audit


def export_refinement(collection, out):
    """Only actual reviewed draft/correction pairs. References never stand in for drafts."""
    with Path(collection).open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    package = Path(collection).resolve().parent.parent
    mapping_path = package / 'generation/PRIVATE_row_mapping.csv'
    if not mapping_path.exists():
        raise ValueError('Refinement collection requires its original generation package mapping')
    hashes = read_json(package / 'checksums.json')
    inputs = [mapping_path] + [package / 'generation' / (name + '.jsonl') for name in ('train','validation')]
    for path in inputs:
        if file_hash(path) != hashes[str(path.relative_to(package))]:
            raise ValueError('Generation mapping or development messages changed')
    with mapping_path.open(encoding='utf-8-sig', newline='') as f:
        mapping = list(csv.DictReader(f))
    allowed = {}
    for name in ('train','validation'):
        chats = read_jsonl(package / 'generation' / (name + '.jsonl'))
        for m in mapping:
            if m['split'] == name:
                allowed[m['qa_id']] = (name, chats[int(m['row'])-1]['messages'][1]['content'])
    result, seen = [], set()
    for r in rows:
        if r['review_status'] != 'approved':
            continue
        if r['split'] not in ('train', 'validation') or r['qa_id'] in seen:
            raise ValueError('Only unique development QA rows may train refinement')
        if allowed.get(r['qa_id']) != (r['split'], r['question']):
            raise ValueError('Refinement QA identity/question differs from original development mapping')
        required = ('question', 'draft', 'draft_model', 'draft_generated_at', 'draft_request_id',
                    'draft_input_sha256', 'corrected_answer', 'reviewer_id', 'revision_reason')
        if not all(r.get(k, '').strip() for k in required):
            raise ValueError('Approved draft/correction requires text, provenance, reviewer, and reason')
        if r['draft'].strip() == r['corrected_answer'].strip():
            raise ValueError('An unchanged copy is not a correction pair')
        seen.add(r['qa_id'])
        import json
        result.append({'messages': [{'role':'system','content':REFINEMENT_PROMPT},
                                   {'role':'user','content':json.dumps({'question':r['question'],'draft':r['draft']},ensure_ascii=False)},
                                   {'role':'assistant','content':r['corrected_answer']}]})
    if not result:
        raise ValueError('No approved real draft/correction pairs; not writing a fake dataset')
    path=Path(out)
    if path.exists():
        raise ValueError('Refinement output already exists')
    write_jsonl(path,result)
    write_json(path.with_suffix('.audit.json'), {'source_sha256':file_hash(collection),'approved_pairs':len(result),
                                               'qa_ids':sorted(seen),'output_sha256':file_hash(path)})
    return len(result)
