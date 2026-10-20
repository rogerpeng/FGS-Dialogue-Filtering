"""One globally grouped dataset for every module, not four independent splits."""
from collections import Counter, defaultdict
from pathlib import Path
import json
import random

from .api import GENERATION_PROMPT, REFINEMENT_PROMPT
from .data import grams, normalize
from .io import digest, file_hash, new_output, read_json, read_jsonl, write_csv, write_json, write_jsonl

TASKS = {"question_filter", "answer_filter", "generation", "refinement", "evaluation"}
HINTS = {"auto", "development", "test", "external"}


def validate_records(rows):
    ids, labels = set(), {}
    for r in rows:
        if r.get("review_status") is not None and r["review_status"] != "approved":
            raise ValueError(f"Candidate data requires human review before study freeze: {r.get('id')}")
        for field in ["id", "task", "source_id", "group_id"]:
            if not isinstance(r.get(field), str) or not r[field].strip():
                raise ValueError(f"Every record needs a nonempty {field}")
        if r["id"] in ids or r["task"] not in TASKS:
            raise ValueError(f"Duplicate id or unknown task: {r['id']}")
        ids.add(r["id"])
        if r.get("split_hint", "auto") not in HINTS:
            raise ValueError("split_hint must be auto/development/test/external")
        task = r["task"]
        fields = ["text"] if task.endswith("filter") else ["question"]
        if task in {"generation", "refinement"}:
            fields += ["answer"]
        if task == "refinement":
            fields += ["draft"]
        if any(not isinstance(r.get(f), str) or not normalize(r[f]) for f in fields):
            raise ValueError(f"Missing task text in {r['id']}: {fields}")
        if task.endswith("filter"):
            if type(r.get("label")) is not int or r["label"] not in (0, 1):
                raise ValueError("Classification label must be integer 0 or 1")
            key = task, normalize(r["text"])
            if key in labels and labels[key] != r["label"]:
                raise ValueError(f"Contradictory labels require adjudication: {r['id']}")
            labels[key] = r["label"]
        if task == "evaluation" and r.get("expected_action") not in {None, "answer", "abstain"}:
            raise ValueError("expected_action must be answer/abstain or omitted")
    if not rows:
        raise ValueError("Dataset is empty")


def record_texts(row):
    return {normalize(row[f]) for f in ["text", "question", "answer", "draft", "reference"] if row.get(f)}


def joint_components(rows, near_threshold=.9):
    if not 0 < near_threshold <= 1:
        raise ValueError("near_threshold must be in (0,1]")
    parents = list(range(len(rows)))
    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    def union(a, b):
        a, b = root(a), root(b)
        parents[max(a, b)] = min(a, b)
    group_owner, text_owner = {}, {}
    for i, row in enumerate(rows):
        group = row["group_id"]
        if group in group_owner:
            union(i, group_owner[group])
        group_owner[group] = i
        for text in record_texts(row):
            if text in text_owner:
                union(i, text_owner[text])
            else:
                text_owner[text] = i
    # Near-duplicate checks are corpus-wide and label-independent.
    indexed, token_sets, near_edges = defaultdict(set), {}, 0
    for text in sorted(text_owner):
        token_set = grams(text)
        candidates = set()
        for token in token_set:
            candidates.update(indexed[token])
        for other in sorted(candidates):
            other_set = token_sets[other]
            if min(len(token_set), len(other_set)) / max(len(token_set), len(other_set)) < near_threshold:
                continue
            if len(token_set & other_set) / len(token_set | other_set) >= near_threshold:
                union(text_owner[text], text_owner[other])
                near_edges += 1
        token_sets[text] = token_set
        for token in token_set:
            indexed[token].add(text)
    components = defaultdict(list)
    for i, row in enumerate(rows):
        components[root(i)].append(row)
    return list(components.values()), near_edges


def allocate_components(components, seed=42, fractions=(.6, .2, .2)):
    if len(fractions) != 3 or any(v <= 0 for v in fractions) or abs(sum(fractions) - 1) > 1e-9:
        raise ValueError("Three positive fractions summing to 1 required")
    bins = lambda rs: Counter(f"{r['task']}:{r.get('label', '')}" for r in rs)
    total = bins([r for c in components for r in c if r.get("split_hint") != "external"])
    candidates = []
    for c in components:
        hints = {r.get("split_hint", "auto") for r in c} - {"auto"}
        if len(hints) > 1:
            raise ValueError(f"Linked records have conflicting split hints {sorted(hints)}; inspect {c[0]['id']}")
        candidates.append((c, next(iter(hints), "auto"), bins(c)))
    # Choose only by label/task balance, never by model performance.
    best = None
    for attempt in range(150):
        rng = random.Random(seed + attempt)
        order = list(candidates)
        rng.shuffle(order)
        order.sort(key=lambda x: len(x[0]), reverse=True)
        counts = {s: Counter() for s in ["train", "validation", "test", "external"]}
        assigned = {s: [] for s in counts}
        for c, hint, counts_c in order:
            choices = ["train", "validation"] if hint == "development" else [hint] if hint in {"test", "external"} else ["train", "validation", "test"]
            rng.shuffle(choices)
            def cost(split):
                if split == "external":
                    return 0
                f = fractions[["train", "validation", "test"].index(split)]
                return sum(((counts[split][k] + n - total[k] * f) ** 2 - (counts[split][k] - total[k] * f) ** 2) / max(total[k], 1) for k, n in counts_c.items())
            split = min(choices, key=cost)
            assigned[split].append(c)
            counts[split].update(counts_c)
        required = set(total)
        missing = sum(counts[s][k] == 0 for s in ["train", "validation", "test"] for k in required)
        imbalance = sum((counts[s][k] - total[k] * f) ** 2 / max(total[k], 1)
                        for s, f in zip(["train", "validation", "test"], fractions) for k in required)
        score = missing * 1e9 + imbalance
        if best is None or score < best[0]:
            best = score, assigned, counts
    # Classification and both fine-tuning tasks must exist in train/validation.
    for s in ["train", "validation"]:
        for key in total:
            if not key.startswith("evaluation:") and not best[2][s][key]:
                raise ValueError(f"Too few independent source groups for {s}/{key}; add groups or resolve giant duplicate components")
    return best[1]


def freeze_dataset(sources, out, seed=42, fractions=(.6, .2, .2), near_threshold=.9):
    rows, source_manifest = [], []
    for path in sources:
        rows.extend(read_jsonl(path))
        source_manifest.append({"path": str(path), "sha256": file_hash(path)})
    validate_records(rows)
    components, edges = joint_components(rows, near_threshold)
    assigned = allocate_components(components, seed, fractions)
    out = new_output(out)
    all_records, split_manifest, duplicates_removed = [], {}, 0
    for split, groups in assigned.items():
        split_manifest[split] = {}
        records = []
        for group in groups:
            component = digest("|".join(sorted(r["id"] for r in group)))[:20]
            records.extend(dict(r, component_id=component, split=split) for r in group)
        all_records.extend(records)
        for task in sorted(TASKS):
            subset, seen = [], set()
            for r in records:
                if r["task"] != task:
                    continue
                # Dedup only identical task content, retain alternate answers.
                key = json.dumps({k: normalize(r[k]) if isinstance(r[k], str) else r[k]
                                  for k in ["text", "question", "answer", "draft", "reference", "label"] if k in r}, sort_keys=True)
                if key in seen:
                    duplicates_removed += 1
                    continue
                seen.add(key)
                if task.endswith("filter"):
                    value = {"id": r["id"], "text": r["text"], "label": r["label"], "group_id": r["component_id"],
                             "source_id": r["source_id"]}
                elif task in {"generation", "refinement"}:
                    user = r["question"] if task == "generation" else json.dumps({"question": r["question"], "draft": r["draft"]}, ensure_ascii=False)
                    value = {"messages": [{"role": "system", "content": GENERATION_PROMPT if task == "generation" else REFINEMENT_PROMPT},
                                          {"role": "user", "content": user}, {"role": "assistant", "content": r["answer"]}]}
                else:
                    value = {k: r[k] for k in ["id", "question", "reference", "category", "expected_action", "synthetic"] if k in r}
                    value["group_id"] = r["component_id"]
                subset.append(value)
            path = out / task / f"{split}.jsonl"
            write_jsonl(path, subset)
            split_manifest[split][task] = {"n": len(subset), "sha256": file_hash(path)}
    write_jsonl(out / "lineage.jsonl", all_records)
    write_csv(out / "allocation.csv", [{"id": r["id"], "task": r["task"], "source_id": r["source_id"], "group_id": r["group_id"],
                                       "component_id": r["component_id"], "split": r["split"]} for r in all_records])
    manifest = {"schema_version": 1, "seed": seed, "fractions": fractions, "near_threshold": near_threshold,
                "sources": source_manifest, "records": len(rows), "global_components": len(components),
                "largest_component": max(map(len, components)), "near_edges": edges,
                "duplicate_task_records_removed": duplicates_removed, "splits": split_manifest,
                "caveat": "Group IDs and annotation quality need human verification. Exact/lexical grouping is not semantic proof."}
    write_json(out / "manifest.json", manifest)
    return manifest


def verify_dataset(folder):
    folder = Path(folder)
    manifest = read_json(folder / "manifest.json")
    for split, tasks in manifest["splits"].items():
        for task, item in tasks.items():
            if file_hash(folder / task / f"{split}.jsonl") != item["sha256"]:
                raise ValueError(f"Frozen data changed: {task}/{split}")
    return manifest
