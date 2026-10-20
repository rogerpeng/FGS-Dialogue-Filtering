"""Generation experiments, lexical metrics, blinded rating sheets and annotations."""
from collections import Counter, defaultdict
import csv
import random
from pathlib import Path
import time
import numpy as np
import joblib

from .api import APIBackend, ChatClient, LLMFilter, LLMGuard
from .data import normalize
from .io import digest, environment, file_hash, new_output, read_json, read_jsonl, write_csv, write_json, write_jsonl
from .metrics import classification
from .pipeline import DialoguePipeline, VARIANTS


def lexical_scores(answer, reference):
    a, b = normalize(answer), normalize(reference)
    if not a or not b:
        return {"character_f1": 0.0, "rouge_l_character_f1": 0.0}
    if len(a) > 5000 or len(b) > 5000:
        raise ValueError("Reference metric limited to 5000 normalized characters; review unusually long records")
    overlap = sum((Counter(a) & Counter(b)).values())
    f1 = 2 * overlap / (len(a) + len(b))
    previous = [0] * (len(b) + 1)
    for char in a:
        current = [0]
        for j, other in enumerate(b, 1):
            current.append(previous[j - 1] + 1 if char == other else max(previous[j], current[-1]))
        previous = current
    return {"character_f1": f1, "rouge_l_character_f1": 2 * previous[-1] / (len(a) + len(b))}


def generation_experiment(items_path, config_path, out, question_model=None, answer_model=None,
                          variants=("b1", "b2", "full", "no_query", "no_answer"), replicates=1, live=False,
                          cache_dir=None):
    items = read_jsonl(items_path)
    if not items or len({r["id"] for r in items}) != len(items) or replicates < 1:
        raise ValueError("Unique nonempty evaluation items and positive replicates required")
    for item in items:
        if not isinstance(item["id"], str) or not item["id"] or not item.get("question", "").strip():
            raise ValueError("Every item needs a nonempty string id and question")
    config = read_json(config_path)
    selected = [VARIANTS[name] for name in variants]
    if any(v.query_filter for v in selected) and not question_model and not config.get("filter_model"):
        raise ValueError("Provide question-model or filter_model")
    if any(v.answer_filter for v in selected) and not answer_model and not config.get("filter_model"):
        raise ValueError("Provide answer-model or filter_model")
    out = new_output(out)
    client = ChatClient(config, cache_dir or out / "cache", live)
    backend = APIBackend(client, config["generator_model"], config["refiner_model"])
    qf = joblib.load(question_model) if question_model else (LLMFilter(client, config["filter_model"], "question") if config.get("filter_model") else None)
    af = joblib.load(answer_model) if answer_model else (LLMFilter(client, config["filter_model"], "answer") if config.get("filter_model") else None)
    guard = LLMGuard(client, config["guard_model"]) if config.get("guard_model") else None
    pipeline = DialoguePipeline(backend, qf, af, guard,
                                config.get("max_generation_attempts", 3), config.get("max_refinements", 3))
    write_json(out / "config.json", {**config, "variants": list(variants), "replicates": replicates,
                                     "items_sha256": file_hash(items_path),
                                     "question_model_sha256": file_hash(question_model) if question_model else None,
                                     "answer_model_sha256": file_hash(answer_model) if answer_model else None})
    write_json(out / "environment.json", environment())
    results = []
    for replicate in range(replicates):
        for item in items:
            for variant in selected:
                start_event = len(client.events)
                row = pipeline.run(item["question"], variant, item["id"], replicate)
                row["api_events"] = client.events[start_event:]
                row["reference"] = item.get("reference", "")
                row["category"] = item.get("category", "unspecified")
                row["expected_action"] = item.get("expected_action", "unlabeled")
                row["synthetic"] = bool(item.get("synthetic", False))
                results.append(row)
                # Save incremental progress, including provider failures.
                write_jsonl(out / "responses.jsonl", results)
                if client.fatal_error is not None:
                    write_json(out / "failure.json", {"status": "stopped_before_remaining_items", "error": client.fatal_error,
                                                      "network_calls": client.calls, "records_saved": len(results),
                                                      "reported_usage": client.billed_usage,
                                                      "note": "Partial run is not a completed evaluation; no automatic network retry."})
                    raise RuntimeError("Experiment stopped on API/budget failure; inspect failure.json")
    report = summarize_responses(results)
    report["api_accounting"] = {"network_calls": client.calls, "logical_calls": client.logical_calls,
                                 "cache_hits": client.cache_hits, "reported_new_call_tokens": client.billed_usage,
                                 "usage_complete": sum(not e["cache_hit"] and "total_tokens" in e.get("usage", {}) for e in client.events) == client.calls}
    report["latency_note"] = "Wall times with cache hits are not model speed comparisons. api_events separates network time and cache reuse."
    write_json(out / "summary.json", report)
    return report


def summarize_responses(rows):
    if not rows:
        raise ValueError("No response records")
    keys = [(r["id"], r["replicate"], r["system"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate response keys")
    summary = []
    for system in sorted({r["system"] for r in rows}):
        records = [r for r in rows if r["system"] == system]
        scored = [lexical_scores(r["answer"], r["reference"]) if r["status"] == "accepted"
                  else {"character_f1": 0.0, "rouge_l_character_f1": 0.0}
                  for r in records if r.get("reference")]
        labeled = [r for r in records if r.get("expected_action") in {"answer", "abstain"}]
        summary.append({"system": system, "n": len(records), "statuses": dict(Counter(r["status"] for r in records)),
                        "coverage": sum(r["status"] == "accepted" for r in records) / len(records),
                        "reference_n": len(scored),
                        "mean_character_f1": float(np.mean([s["character_f1"] for s in scored])) if scored else None,
                        "mean_rouge_l_character_f1": float(np.mean([s["rouge_l_character_f1"] for s in scored])) if scored else None,
                        "action_labeled_n": len(labeled),
                        "action_accuracy": sum((r["status"] == "accepted") == (r["expected_action"] == "answer")
                                               and r["status"] != "provider_or_filter_error" for r in labeled) / len(labeled) if labeled else None})
    return {"systems": summary, "synthetic_records": sum(bool(r.get("synthetic")) for r in rows),
            "interpretation": "Lexical overlap and action labels do not establish doctrinal truth. Abstentions get zero reference score; no accepted-only cherry picking. Synthetic fixtures are not study evidence."}


def blind_sheets(responses_path, out, participants, seed=42, max_items=None, one_replicate_per_item=True):
    rows = read_jsonl(responses_path)
    if not participants or len(set(participants)) != len(participants):
        raise ValueError("Unique pseudonymous participant ids required")
    summarize_responses(rows)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["id"], r["replicate"])].append(r)
    system_sets = [{r["system"] for r in group} for group in groups.values()]
    if any(s != system_sets[0] for s in system_sets):
        raise ValueError("Every item/replicate must have the same systems")
    if any(r["status"] == "provider_or_filter_error" for r in rows):
        raise ValueError("Resolve provider errors before producing rating sheets")
    out = new_output(out)
    rng, mapping = random.Random(seed), []
    base_items = sorted({key[0] for key in groups})
    rng.shuffle(base_items)
    system_order = sorted(system_sets[0])
    rng.shuffle(system_order)
    if max_items is not None and (not isinstance(max_items, int) or not 1 <= max_items <= len(base_items)):
        raise ValueError("max_items must be within available distinct question count")
    per_person = max_items or len(base_items)
    for person_index, participant in enumerate(participants):
        sheet = []
        # Cyclic question assignment balances exposure; each person sees one draw per question.
        selected_ids = [base_items[(person_index * per_person + j) % len(base_items)] for j in range(per_person)]
        order = []
        for j, item_id in enumerate(selected_ids):
            choices = sorted(key for key in groups if key[0] == item_id)
            order.extend([choices[(person_index + j) % len(choices)]] if one_replicate_per_item else choices)
        rng.shuffle(order)
        for question_index, (item_id, replicate) in enumerate(order):
            by_system = {r["system"]: r for r in groups[(item_id, replicate)]}
            shift = (person_index + question_index) % len(system_order)
            rotated = system_order[shift:] + system_order[:shift]
            candidates = [by_system[name] for name in rotated]
            for slot, row in enumerate(candidates, 1):
                token = digest(f"{seed}|{participant}|{item_id}|{replicate}|{slot}")[:24]
                sheet.append({"rating_id": token, "item_id": f"{item_id}::r{replicate}", "display_position": slot,
                              "question": row["question"], "answer": row["answer"], "score": "", "notes": ""})
                mapping.append({"rating_id": token, "participant_id": participant, "item_id": f"{item_id}::r{replicate}",
                                "system": row["system"], "display_position": slot})
        # Hash filenames to avoid using untrusted participant IDs as paths.
        write_csv(out / f"participant_{digest(participant)[:12]}.csv", sheet)
    write_csv(out / "PRIVATE_mapping.csv", mapping)
    write_json(out / "design.json", {"seed": seed, "participants": participants, "responses_sha256": file_hash(responses_path),
                                      "items_per_participant": per_person, "one_replicate_per_item": one_replicate_per_item,
                                      "assignments": len(mapping), "position_design": "randomized cyclic Latin rotations; not full carryover balancing",
                                      "note": "Keep PRIVATE_mapping.csv from raters. Fixed assignments; question blocks reduce fatigue. Not all partial block sizes achieve perfect balance."})


def unblind(mapping_path, sheets, out):
    with Path(mapping_path).open(encoding="utf-8-sig", newline="") as h:
        mapping_rows = list(csv.DictReader(h))
    mapping = {r["rating_id"]: r for r in mapping_rows}
    if len(mapping) != len(mapping_rows):
        raise ValueError("Duplicate rating ids in private mapping")
    output, seen = [], set()
    for path in sheets:
        with Path(path).open(encoding="utf-8-sig", newline="") as h:
            for row in csv.DictReader(h):
                token = row["rating_id"]
                if token not in mapping or token in seen:
                    raise ValueError("Unknown or repeated rating_id")
                seen.add(token)
                if not row["score"].strip():
                    continue
                if row["score"] not in {"1", "2", "3", "4", "5"}:
                    raise ValueError("Rating must be an integer 1–5")
                original = mapping[token]
                output.append({"participant_id": original["participant_id"], "item_id": original["item_id"],
                               "system": original["system"], "score": row["score"], "mapping_verified": "true",
                               "mapping_sha256": file_hash(mapping_path)})
    if not output:
        raise ValueError("No completed ratings")
    if Path(out).exists():
        raise ValueError("Output file exists")
    write_csv(out, output)


def llm_classification(data_dir, config_path, out, role="question", live=False):
    from .experiments import load_splits
    rows = load_splits(data_dir)["test"]
    config = read_json(config_path)
    out = new_output(out)
    client = ChatClient(config, out / "cache", live)
    model = LLMFilter(client, config["filter_model"], role)
    predictions = []
    for row in rows:
        result = model.classify(row["text"])
        predictions.append({"id": row["id"], "group_id": row["group_id"], "label": row["label"], "prediction": result["label"], "reason": result.get("reason", "")})
        write_csv(out / "predictions.csv", predictions)
    report = classification([r["label"] for r in predictions], [r["prediction"] for r in predictions])
    # Hard decisions are not calibrated probabilities; do not report misleading probabilistic metrics.
    for field in ["roc_auc", "average_precision", "log_loss"]:
        report.pop(field)
    write_json(out / "metrics.json", {"metrics": report, "test_sha256": file_hash(Path(data_dir) / "test.jsonl"),
                                     "config": config, "network_calls": client.calls, "prompt": "fgs.api.FILTER_PROMPT"})
    return report


def annotation_summary(path, out):
    """Expert binary judgments and pairwise Cohen kappa on shared items."""
    from sklearn.metrics import cohen_kappa_score
    with Path(path).open(encoding="utf-8-sig", newline="") as h:
        rows = list(csv.DictReader(h))
    axes = ["doctrinal_error", "harmful", "biased", "fabricated_quote"]
    seen = set()
    for r in rows:
        key = r["rater_id"], r["item_id"], r["system"]
        if not all(key) or key in seen:
            raise ValueError("Duplicate or empty annotation key")
        seen.add(key)
        if any(r[a] not in {"", "0", "1"} for a in axes):
            raise ValueError("Annotation labels must be 0, 1, or blank for unassessed")
    rates, agreement = [], []
    for system in sorted({r["system"] for r in rows}):
        for axis in axes:
            selected = [r for r in rows if r["system"] == system and r[axis] != ""]
            rates.append({"system": system, "axis": axis, "judgments": len(selected),
                          "positive_judgment_rate": sum(int(r[axis]) for r in selected) / len(selected) if selected else None})
            raters = sorted({r["rater_id"] for r in selected})
            for i, a in enumerate(raters):
                for b in raters[i + 1:]:
                    x = {r["item_id"]: int(r[axis]) for r in selected if r["rater_id"] == a}
                    y = {r["item_id"]: int(r[axis]) for r in selected if r["rater_id"] == b}
                    keys = sorted(x.keys() & y.keys())
                    values = [x[k] for k in keys] + [y[k] for k in keys]
                    kappa = float(cohen_kappa_score([x[k] for k in keys], [y[k] for k in keys])) if keys and len(set(values)) > 1 else None
                    agreement.append({"system": system, "axis": axis, "rater_a": a, "rater_b": b, "overlap": len(keys),
                                      "kappa": kappa if kappa is not None and np.isfinite(kappa) else None})
    out = new_output(out)
    write_csv(out / "judgment_rates.csv", rates)
    write_csv(out / "agreement.csv", agreement, ["system", "axis", "rater_a", "rater_b", "overlap", "kappa"])
    write_json(out / "provenance.json", {"source_sha256": file_hash(path),
                                        "note": "Rates are per judgment, not adjudicated per answer. Blanks are not negatives. Resolve disagreements before paper-level prevalence claims."})
    return rates
