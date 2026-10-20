"""Corpus counts and lexical overlap checks, without loading legacy code or secrets."""
from collections import Counter
from pathlib import Path

from .data import normalize, grams
from .io import file_hash, new_output, read_jsonl, write_csv, write_json


def corpus_inventory(root, out):
    root = Path(root)
    selected = ["filter/question/正面提問.txt", "filter/question/負面提問.txt",
                "filter/anwser/正向回答.txt", "filter/anwser/無關回答.txt",
                "question/正面提問.txt", "question/負面提問.txt",
                "conversation/正面提問.txt", "conversation/負面提問.txt",
                "conversation/all_con.jsonl", "conversation/forgpt35.jsonl",
                "filter/anwser fine tune/202410修正/1216forgpt35.jsonl"]
    report = []
    for relative in selected:
        path = root / relative
        if not path.is_file():
            report.append({"file": relative, "status": "missing"})
            continue
        row = {"file": relative, "status": "present", "sha256": file_hash(path)}
        if path.suffix == ".txt":
            lines = [s.strip() for s in path.read_text(encoding="utf-8-sig").splitlines() if s.strip()]
            row.update(records=len(lines), normalized_unique=len({normalize(s) for s in lines}))
        else:
            records = read_jsonl(path)
            prompts, answers = [], []
            for r in records:
                if "prompt" in r:
                    prompts.append(normalize(r["prompt"]))
                    answers.append(normalize(r["completion"]))
                else:
                    prompts.append("|".join(normalize(m["content"]) for m in r["messages"] if m["role"] == "user"))
                    answers.append("|".join(normalize(m["content"]) for m in r["messages"] if m["role"] == "assistant"))
            row.update(records=len(records), unique_user_inputs=len(set(prompts)), unique_assistant_outputs=len(set(answers)),
                       unique_pairs=len(set(zip(prompts, answers))))
        report.append(row)
    out = new_output(out)
    write_json(out / "inventory.json", report)
    return report


def overlap_audit(evaluation, training, out, threshold=.9):
    """Find exact/near evaluation questions or references in training messages.

    Lexical screening does not establish absence of semantic or pretraining overlap.
    """
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0,1]")
    train, eval_rows = read_jsonl(training), read_jsonl(evaluation)
    if not train or not eval_rows or any(not r.get("id") or not r.get("question") for r in eval_rows):
        raise ValueError("Nonempty training and evaluation records with id/question required")
    texts = []
    for i, row in enumerate(train, 1):
        if "prompt" in row:
            texts.extend([(i, "user", normalize(row["prompt"])), (i, "assistant", normalize(row["completion"]))])
        elif "messages" in row:
            texts.extend((i, m["role"], normalize(m["content"])) for m in row["messages"] if m["role"] in {"user", "assistant"})
        elif "text" in row:
            texts.append((i, "text", normalize(row["text"])))
        else:
            raise ValueError(f"Unknown training schema at row {i}")
    prepared = [(i, role, t, grams(t)) for i, role, t in texts if t]
    hits = []
    for row in eval_rows:
        for field in ["question", "reference"]:
            text = normalize(row.get(field, ""))
            if not text:
                continue
            tokens = grams(text)
            for i, role, other, other_tokens in prepared:
                score = len(tokens & other_tokens) / len(tokens | other_tokens)
                if text == other or score >= threshold:
                    hits.append({"evaluation_id": row["id"], "field": field, "training_row": i, "training_role": role,
                                 "match": "exact" if text == other else "near", "jaccard": score})
    out = new_output(out)
    write_csv(out / "overlaps.csv", hits, ["evaluation_id", "field", "training_row", "training_role", "match", "jaccard"])
    report = {"evaluation_sha256": file_hash(evaluation), "training_sha256": file_hash(training), "threshold": threshold,
              "evaluation_items": len(eval_rows), "training_records": len(train), "matching_items": len({h["evaluation_id"] for h in hits}),
              "matching_pairs": len(hits), "warning": "Zero lexical matches cannot rule out paraphrases or earlier exposure by pretrained models."}
    write_json(out / "report.json", report)
    return report
