"""Import the round-100 review sheet, preserving immutable provenance and nulls."""
import argparse
import csv
import io
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fgs.conversion import csv_text
from fgs.data import normalize
from fgs.dataset import validate_records
from fgs.io import file_hash, read_json, read_jsonl


def export_reviewed(collection, out):
    collection, out = Path(collection), Path(out)
    if out.exists():
        raise FileExistsError('Output exists; choose a new version')
    canonical_path = collection / 'candidate_records.jsonl'
    checksums = read_json(collection / 'file_checksums.json')
    if file_hash(canonical_path) != checksums['candidate_records.jsonl']:
        raise ValueError('Canonical records changed; rebuild the collection before importing')
    canonical = read_jsonl(canonical_path)
    by_id = {r['id']: r for r in canonical}
    if len(by_id) != len(canonical):
        raise ValueError('Duplicate canonical IDs')
    # Compare immutable metadata to the actual CSV representation, including Excel escaping.
    original_csv, fields = csv_text(canonical)
    baseline = {r['id']: r for r in csv.DictReader(io.StringIO(original_csv))}
    review_path = collection / 'records_review.csv'
    with review_path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)) or set(reader.fieldnames) != set(fields):
            raise ValueError('Review columns differ from the exported sheet')
        reviewed = list(reader)
    if any(None in r or any(v is None for v in r.values()) for r in reviewed):
        raise ValueError('Malformed CSV row')
    if len({r['id'] for r in reviewed}) != len(reviewed) or {r['id'] for r in reviewed} != set(by_id):
        raise ValueError('Review IDs must match the full canonical sheet; reject rows instead of deleting them')
    approved = {}
    review_hash = file_hash(review_path)
    for row in reviewed:
        source = by_id[row['id']]
        task = source['task']
        text_fields = ['text'] if task.endswith('filter') else ['question', 'answer']
        editable = set(text_fields + ['review_status', 'reviewer_id', 'review_notes'])
        if task.endswith('filter'):
            editable.add('label')
        if any(row[k] != baseline[row['id']][k] for k in fields if k not in editable):
            raise ValueError(f'Immutable provenance or task field changed: {row["id"]}')
        if row['review_status'] not in {'pending', 'approved', 'rejected'}:
            raise ValueError(f'Unknown review status: {row["id"]}')
        if row['review_status'] != 'approved':
            continue
        if not row['reviewer_id'].strip():
            raise ValueError(f'Approved row requires reviewer_id: {row["id"]}')
        result = dict(source)
        for key in editable:
            if key == 'label':
                if row[key] not in {'0', '1'}:
                    raise ValueError('Reviewed label must be 0 or 1')
                result[key] = int(row[key])
            elif key in text_fields and row[key] == baseline[row['id']][key]:
                # Preserve the original unescaped text when a formula-safe CSV cell is unchanged.
                result[key] = source[key]
            else:
                result[key] = row[key]
        result['review_file_sha256'] = review_hash
        approved[result['id']] = result
    if not approved:
        raise ValueError('No approved records; complete human review first')
    for r in approved.values():
        if r['task'] != 'generation':
            continue
        for field, ref in [('question', 'question_record_id'), ('answer', 'answer_record_id')]:
            linked = approved.get(r[ref])
            if linked is None or linked['label'] != 1 or normalize(linked['text']) != normalize(r[field]):
                raise ValueError(f'Generation requires matching approved related classification rows: {r["id"]} / {field}')
    rows = list(approved.values())
    validate_records(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    return {'approved_task_rows': len(rows), 'skipped_task_rows': len(canonical)-len(rows),
            'output': str(out.resolve()), 'split_hint': 'development'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--collection', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = export_reviewed(args.collection, args.out)
    except (ValueError, FileExistsError) as exc:
        parser.exit(2, str(exc) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
