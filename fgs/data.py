"""Legacy import, duplicate quarantine, near-duplicate groups, and frozen splits."""
from collections import Counter, defaultdict
from pathlib import Path
import csv
import random
import unicodedata

from .io import digest, file_hash, new_output, read_jsonl, write_csv, write_json, write_jsonl


def normalize(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    return "".join(c for c in text if not c.isspace() and not unicodedata.category(c).startswith("P"))


def grams(text):
    return {text[i:i + 3] for i in range(max(1, len(text) - 2))}


def clean_group(rows, near_threshold=0.9):
    """Same normalized text with contradictory labels is excluded, never relabeled.

    Same-label exact duplicates retain all provenance. Near duplicates AND explicit
    group_id links form connected components before splitting. Labels do not drive
    near-duplicate grouping. This is lexical grouping, not semantic adjudication.
    """
    if not 0 < near_threshold <= 1:
        raise ValueError("near_threshold must be in (0, 1]")
    buckets = defaultdict(list)
    for row in rows:
        if row["label"] not in (0, 1) or not normalize(row["text"]):
            raise ValueError("Nonempty text and binary 0/1 label required")
        buckets[normalize(row["text"])].append(row)
    clean, excluded = [], []
    for text, items in sorted(buckets.items()):
        if len({r["label"] for r in items}) != 1:
            excluded.extend(dict(r, reason="conflicting_exact_labels") for r in items)
            continue
        clean.append({"id": digest(text)[:20], "text": items[0]["text"], "label": items[0]["label"],
                      "sources": [r.get("source", r.get("id", "unknown")) for r in items],
                      "original_group_ids": sorted({r["group_id"] for r in items if r.get("group_id")})})
    parents = list(range(len(clean)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def union(a, b):
        a, b = find(a), find(b)
        parents[max(a, b)] = min(a, b)

    seen_groups, inverted, near = {}, defaultdict(set), []
    sets = [grams(normalize(r["text"])) for r in clean]
    for i, row in enumerate(clean):
        for group in row["original_group_ids"]:
            if group in seen_groups:
                union(i, seen_groups[group])
            seen_groups[group] = i
        candidates = set()
        for gram in sets[i]:
            candidates.update(inverted[gram])
        for j in sorted(candidates):
            if min(len(sets[i]), len(sets[j])) / max(len(sets[i]), len(sets[j])) < near_threshold:
                continue
            score = len(sets[i] & sets[j]) / len(sets[i] | sets[j])
            if score >= near_threshold:
                union(i, j)
                near.append({"id_a": clean[j]["id"], "id_b": row["id"], "jaccard": score,
                             "label_conflict": clean[j]["label"] != row["label"]})
        for gram in sets[i]:
            inverted[gram].add(i)
    for i, row in enumerate(clean):
        row["group_id"] = clean[find(i)]["id"]
    return clean, excluded, near


def split_rows(rows, seed=42, test_fraction=0.2, validation_fraction=0.2):
    from sklearn.model_selection import GroupShuffleSplit
    import numpy as np
    if not 0 < test_fraction < 1 or not 0 < validation_fraction < 1 - test_fraction:
        raise ValueError("Invalid split fractions")

    def partition(items, fraction, split_seed):
        y = np.array([r["label"] for r in items])
        groups = [r["group_id"] for r in items]
        best = None
        for train, held in GroupShuffleSplit(n_splits=200, test_size=fraction, random_state=split_seed).split(y, y, groups):
            if len(set(y[train])) < 2 or len(set(y[held])) < 2:
                continue
            score = abs(len(held) / len(y) - fraction) + abs(float(y[train].mean() - y.mean())) + abs(float(y[held].mean() - y.mean()))
            if best is None or score < best[0]:
                best = score, train, held
        if best is None:
            raise ValueError("Too few independent groups to form two-class splits; inspect grouping/data")
        return [items[i] for i in best[1]], [items[i] for i in best[2]]

    train_val, test = partition(rows, test_fraction, seed)
    train, val = partition(train_val, validation_fraction / (1 - test_fraction), seed + 1)
    splits = {"train": train, "validation": val, "test": test}
    assert_disjoint(splits)
    return splits


def assert_disjoint(splits):
    seen_ids, seen_texts, seen_groups = set(), set(), set()
    for name, rows in splits.items():
        ids = {r["id"] for r in rows}
        texts = {normalize(r["text"]) for r in rows}
        groups = {r["group_id"] for r in rows}
        if len(ids) != len(rows) or seen_ids & ids or seen_texts & texts or seen_groups & groups:
            raise ValueError(f"Duplicate or cross-split leakage in {name}")
        seen_ids |= ids
        seen_texts |= texts
        seen_groups |= groups


def prepare(positive, negative, out, seed=42, near_threshold=0.9):
    rows, sources = [], []
    for label, path in [(1, Path(positive)), (0, Path(negative))]:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        sources.append({"file": str(path), "sha256": file_hash(path), "physical_lines": len(lines),
                        "nonblank_lines": sum(bool(x.strip()) for x in lines), "label": label})
        rows.extend({"text": line.strip(), "label": label, "source": f"{path.name}:{i}"}
                    for i, line in enumerate(lines, 1) if line.strip())
    return prepare_rows(rows, out, seed, near_threshold, sources)


def prepare_rows(rows, out, seed=42, near_threshold=0.9, sources=None):
    clean, excluded, near = clean_group(rows, near_threshold)
    splits = split_rows(clean, seed)
    out = new_output(out)
    for name, items in splits.items():
        write_jsonl(out / f"{name}.jsonl", items)
    write_jsonl(out / "excluded.jsonl", excluded)
    write_csv(out / "near_duplicates.csv", near, ["id_a", "id_b", "jaccard", "label_conflict"])
    report = {"seed": seed, "near_threshold": near_threshold, "sources": sources or [],
              "raw_nonblank": len(rows), "unique_retained": len(clean), "conflicting_rows_excluded": len(excluded),
              "exact_duplicates_removed": len(rows) - len(excluded) - len(clean),
              "near_duplicate_edges": len(near), "independent_groups": len({r["group_id"] for r in clean}),
              "requested_fractions": {"train": 0.6, "validation": 0.2, "test": 0.2},
              "splits": {k: {"n": len(v), "labels": dict(Counter(r["label"] for r in v)),
                             "groups": len({r["group_id"] for r in v}), "sha256": file_hash(out / f"{k}.jsonl")}
                         for k, v in splits.items()},
              "caveat": "Lexical grouping cannot exclude semantic/source leakage; add original_group_ids via group_id where known."}
    write_json(out / "manifest.json", report)
    return report


def import_legacy_ratings(path, out):
    """The 12-column legacy dia.csv has NO header. Zero and \\N are missing.

    Export stored B1/B2/full mappings without asserting they are historically valid:
    the old app used shared global random order and did not save all three texts.
    """
    rows, counts = [], Counter()
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for line, r in enumerate(csv.reader(handle), 1):
            if len(r) != 12:
                raise ValueError(f"Expected 12 columns at row {line}; got {len(r)}")
            for system, raw in zip(["b1", "b2", "full"], r[9:12]):
                if raw in ("", "0", "\\N"):
                    counts["missing_ratings"] += 1
                    continue
                try:
                    score = int(raw)
                except ValueError as exc:
                    raise ValueError(f"Invalid score at row {line}") from exc
                if not 1 <= score <= 5:
                    raise ValueError(f"Score out of range at row {line}")
                rows.append({"participant_id": "p_" + digest(r[1])[:12], "item_id": r[0],
                             "system": system, "score": score, "mapping_verified": "false"})
            counts["dialogue_rows"] += 1
    out = new_output(out)
    write_csv(out / "ratings.csv", rows, ["participant_id", "item_id", "system", "score", "mapping_verified"])
    report = dict(counts, valid_rating_cells=len(rows), participants=len({r["participant_id"] for r in rows}),
                  items=len({r["item_id"] for r in rows}), source_sha256=file_hash(path),
                  warning="UNVERIFIED legacy mapping: global display order could be overwritten by concurrent requests; verify before inference.")
    write_json(out / "audit.json", report)
    return report


def export_finetune(source, out, seed=42):
    """Split chat fine-tuning data by connected repeated user OR assistant content.

    Export only; no upload, no paid fine-tuning. Keep historic system messages as
    data, but document their persona limitations before any deployment.
    """
    rows = read_jsonl(source)
    parents = list(range(len(rows)))
    seen = {}

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for i, row in enumerate(rows):
        if "messages" not in row and "prompt" in row and "completion" in row:
            row = rows[i] = {"messages": [{"role": "user", "content": row["prompt"]}, {"role": "assistant", "content": row["completion"]}]}
        messages = row.get("messages", [])
        if not messages or not any(m.get("role") == "user" for m in messages) or messages[-1].get("role") != "assistant":
            raise ValueError(f"Invalid chat training record {i + 1}")
        for m in messages:
            if m.get("role") not in {"system", "user", "assistant"} or not isinstance(m.get("content"), str) or not m["content"].strip():
                raise ValueError(f"Invalid message in record {i + 1}")
            if m["role"] == "system":
                continue
            key = (m["role"], normalize(m["content"]))
            if key in seen:
                parents[root(i)] = root(seen[key])
            seen[key] = i
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[root(i)].append(row)
    keys = sorted(groups)
    if len(keys) < 3:
        raise ValueError("Fewer than 3 independent conversation groups")
    random.Random(seed).shuffle(keys)
    n_test = max(1, round(len(keys) * 0.15))
    n_val = max(1, round(len(keys) * 0.15))
    split_keys = {"test": keys[:n_test], "validation": keys[n_test:n_test + n_val], "train": keys[n_test + n_val:]}
    out = new_output(out)
    for name, group_keys in split_keys.items():
        write_jsonl(out / f"{name}.jsonl", [r for k in group_keys for r in groups[k]])
    report = {"source_sha256": file_hash(source), "seed": seed, "raw_records": len(rows), "groups": len(groups),
              "splits": {k: {"n": sum(len(groups[g]) for g in v), "sha256": file_hash(out / f"{k}.jsonl")} for k, v in split_keys.items()},
              "warning": "New split only; old fine-tuned models may have seen every record. Not an independent evaluation of those models. Review personas/consent before upload."}
    write_json(out / "manifest.json", report)
    return report
