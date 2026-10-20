"""Import reviewed candidate QAs without mixing negative answers into SFT targets."""
import csv
from pathlib import Path
from .conversion import parse_records
from .io import file_hash, write_jsonl


def import_reviewed_qa(source, out):
    source, out = Path(source), Path(out)
    if out.exists():
        raise ValueError("Output exists; use a new version")
    if source.suffix.lower() == ".csv":
        with source.open(encoding="utf-8-sig", newline="") as h:
            rows = list(csv.DictReader(h))
    else:
        rows = parse_records(source.read_text(encoding="utf-8-sig"))
    ids, records, approved = set(), [], 0
    for r in rows:
        if not r.get("id") or r["id"] in ids:
            raise ValueError("QA ids must be nonempty and unique")
        ids.add(r["id"])
        if r.get("review_status") != "approved":
            continue
        for key in ["question", "answer", "source_id", "group_id", "reviewer_id"]:
            if not isinstance(r.get(key), str) or not r[key].strip():
                raise ValueError(f"Approved QA requires {key}: {r['id']}")
        for key in ["question_label", "answer_label"]:
            if str(r.get(key)) not in {"0", "1"}:
                raise ValueError("Reviewed labels must be 0 or 1")
        approved += 1
        common = {"source_id": r["source_id"], "group_id": r["group_id"], "source_url": r.get("source_url", ""),
                  "review_status": "approved", "reviewer_id": r["reviewer_id"], "review_notes": r.get("review_notes", ""),
                  "review_file_sha256": file_hash(source), "content_origin": r.get("content_origin", "unspecified"),
                  "synthetic": str(r.get("synthetic", True)).lower() == "true", "split_hint": "development"}
        for task, field in [("question_filter", "question"), ("answer_filter", "answer")]:
            records.append(dict(common, id=r["id"] + "_" + task, task=task, text=r[field], label=int(r[field + "_label"])))
        if int(r["question_label"]) == int(r["answer_label"]) == 1:
            records.append(dict(common, id=r["id"] + "_generation", task="generation", question=r["question"], answer=r["answer"]))
    if not records:
        raise ValueError("No approved QAs; complete source/label review first")
    write_jsonl(out, records)
    return {"approved_pairs": approved, "skipped_pairs": len(rows) - approved, "task_rows": len(records),
            "output": str(out.resolve()), "split_hint": "development"}
