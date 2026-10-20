"""Validation-selected baselines, fixed test sets, learning curves and controls."""
import time
from pathlib import Path
import joblib
import numpy as np

from .data import assert_disjoint
from .io import environment, file_hash, new_output, read_jsonl, write_csv, write_json
from .metrics import classification, group_bootstrap_accuracy, threshold_from_validation
from .models import TextClassifier


def assert_random_splits(path, splits):
    import json
    from .data import normalize
    manifest = json.loads((Path(path) / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("split_policy") != "stratified_random_by_qa_row":
        raise ValueError("Random-overlap mode requires an explicit random QA split manifest")
    seen_ids, labels = set(), {}
    for name, rows in splits.items():
        ids = {r["id"] for r in rows}
        if len(ids) != len(rows) or seen_ids & ids:
            raise ValueError(f"Repeated record ids in {name}")
        seen_ids.update(ids)
        for r in rows:
            text = normalize(r["text"])
            if not text or not r.get("group_id") or type(r["label"]) is not int or r["label"] not in (0, 1):
                raise ValueError(f"Invalid classifier record in {name}")
            if text in labels and labels[text] != r["label"]:
                raise ValueError("Contradictory labels across random splits")
            labels[text] = r["label"]


def load_splits(path, include_test=True, allow_random_overlap=False):
    path = Path(path)
    names = ["train", "validation", "test"] if include_test else ["train", "validation"]
    splits = {name: read_jsonl(path / f"{name}.jsonl") for name in names}
    if allow_random_overlap:
        assert_random_splits(path, splits)
    else:
        assert_disjoint(splits)
    for name, rows in splits.items():
        if {r["label"] for r in rows} != {0, 1}:
            raise ValueError(f"Both classes required in {name}")
    return splits


def training_subset(rows, fraction, seed):
    if not 0 < fraction <= 1:
        raise ValueError("train_fraction must be in (0,1]")
    if fraction == 1:
        return rows
    groups = sorted({r["group_id"] for r in rows})
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    chosen = set(groups[:max(2, round(len(groups) * fraction))])
    subset = [r for r in rows if r["group_id"] in chosen]
    if {r["label"] for r in subset} != {0, 1}:
        raise ValueError("Learning-curve subset lacks one class; increase fraction")
    return subset


def run_experiments(data_dir, out, kinds=("tfidf", "word2vec"), seeds=(42, 43, 44),
                    epochs=300, patience=25, learning_rates=(0.001, 0.01), Cs=(0.1, 1.0, 10.0),
                    embedding_model=None, embedding_revision=None, local_files_only=False,
                    min_precision=None, train_fraction=1.0, permute_labels=False, evaluate_test=False,
                    allow_random_overlap=False):
    if not kinds or not seeds or not learning_rates or not Cs:
        raise ValueError("Nonempty model, seed, and hyperparameter lists required")
    if len(set(kinds)) != len(kinds) or len(set(seeds)) != len(seeds):
        raise ValueError("Models and seeds must be unique")
    if any(x <= 0 for x in list(learning_rates) + list(Cs)):
        raise ValueError("Learning rates and C values must be positive")
    if any(k.startswith("transformer") for k in kinds) and not embedding_model:
        raise ValueError("Provide --embedding-model for the transformer baseline")
    splits = load_splits(data_dir, include_test=evaluate_test, allow_random_overlap=allow_random_overlap)
    out = new_output(out)
    config = dict(kinds=list(kinds), seeds=list(seeds), epochs=epochs, patience=patience,
                  learning_rates=list(learning_rates), Cs=list(Cs), embedding_model=embedding_model,
                  embedding_revision=embedding_revision, local_files_only=local_files_only,
                  min_precision=min_precision, train_fraction=train_fraction, permute_labels=permute_labels,
                  evaluate_test=evaluate_test, allow_random_overlap=allow_random_overlap)
    write_json(out / "config.json", config)
    write_json(out / "environment.json", environment())
    write_json(out / "data_hashes.json", {k: file_hash(Path(data_dir) / f"{k}.jsonl") for k in splits})
    results = []
    for seed in seeds:
        train = training_subset(splits["train"], train_fraction, seed)
        val = splits["validation"]
        tx, ty = [r["text"] for r in train], np.array([r["label"] for r in train])
        vx, vy = [r["text"] for r in val], np.array([r["label"] for r in val])
        if permute_labels:
            ty = np.random.default_rng(seed).permutation(ty)
        for kind in kinds:
            run_dir = out / f"{kind}_seed{seed}"
            run_dir.mkdir()
            best_model, best_score, tuning = None, None, []
            start = time.perf_counter()
            grid = [{"learning_rate": x} for x in learning_rates] if kind in {"word2vec", "transformer_mlp"} else [{"C": x} for x in Cs]
            for number, hp in enumerate(grid):
                model = TextClassifier(kind=kind, seed=seed, epochs=epochs, patience=patience,
                                       embedding_model=embedding_model, embedding_revision=embedding_revision,
                                       local_files_only=local_files_only, cache_dir=str(out / ".cache"), **hp)
                model.fit(tx, ty, vx, vy)
                vp = model.probabilities(vx)
                model.threshold = threshold_from_validation(vy, vp, min_precision)
                vm = classification(vy, vp, model.threshold)
                score = (vm["macro_f1"], -vm["log_loss"])
                tuning.append({"trial": number, **hp, "threshold": model.threshold, "best_epoch": model.best_epoch,
                               "validation_macro_f1": vm["macro_f1"], "validation_log_loss": vm["log_loss"]})
                if model.history:
                    write_csv(run_dir / f"training_curve_trial{number}.csv", model.history)
                if best_score is None or score > best_score:
                    best_score, best_model = score, model
            fit_seconds = time.perf_counter() - start
            model = best_model
            write_csv(run_dir / "validation_search.csv", tuning)
            model_path = run_dir / "model.joblib"
            joblib.dump(model, model_path)
            reports = {}
            evaluation_splits = [("train", train), ("validation", val)]
            if evaluate_test:
                evaluation_splits.append(("test", splits["test"]))
            for split_name, rows in evaluation_splits:
                texts, labels = [r["text"] for r in rows], [r["label"] for r in rows]
                if split_name == "train" and permute_labels:
                    labels = ty.tolist()
                started = time.perf_counter()
                p = model.probabilities(texts)
                seconds = time.perf_counter() - started
                metric = classification(labels, p, model.threshold)
                metric["inference_ms_per_item_batch"] = seconds * 1000 / len(rows)
                metric["oov_token_rate"] = model.oov_rate(texts)
                if split_name == "test":
                    metric["accuracy_ci95"] = group_bootstrap_accuracy(labels, p, [r["group_id"] for r in rows], model.threshold, seed)
                reports[split_name] = metric
                predictions = [{"id": r["id"], "group_id": r["group_id"], "text": r["text"], "label": label,
                                "source_label": r["label"],
                                "probability": float(prob), "prediction": int(prob >= model.threshold),
                                "error_type": ("false_accept" if prob >= model.threshold else "false_reject")
                                if int(prob >= model.threshold) != label else ""}
                               for r, prob, label in zip(rows, p, labels)]
                write_csv(run_dir / f"{split_name}_predictions.csv", predictions)
                if split_name == "test":
                    write_csv(run_dir / "failure_cases.csv", [dict(r, reviewed_category="", reviewer_notes="")
                                                             for r in predictions if r["error_type"]],
                              list(predictions[0]) + ["reviewed_category", "reviewer_notes"])
            record = {"kind": kind, "seed": seed, "train_n": len(train), "threshold": model.threshold,
                      "best_epoch": model.best_epoch, "search_fit_seconds": fit_seconds,
                      "artifact_bytes": model_path.stat().st_size, "metrics": reports,
                      "train_test_accuracy_gap": reports["train"]["accuracy"] - reports["test"]["accuracy"] if evaluate_test else None,
                      "permuted_training_labels": permute_labels,
                      "train_label_basis": "permuted fit labels" if permute_labels else "source labels",
                      "size_note": "Transformer artifact excludes external encoder weights; do not compare as total model size."}
            write_json(run_dir / "metrics.json", record)
            results.append(record)
            reporting_split = "test" if evaluate_test else "validation"
            print(f"{kind} seed={seed}: {reporting_split} macro-F1={reports[reporting_split]['macro_f1']:.4f}", flush=True)
    summary = []
    for kind in kinds:
        runs = [r for r in results if r["kind"] == kind]
        for metric in ["accuracy", "macro_f1", "precision", "recall", "false_accept_rate"]:
            reporting_split = "test" if evaluate_test else "validation"
            values = [r["metrics"][reporting_split][metric] for r in runs]
            summary.append({"model": kind, "split": reporting_split, "metric": metric, "mean": float(np.mean(values)),
                            "seed_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None, "runs": len(values)})
    write_csv(out / "summary.csv", summary)
    write_json(out / "results.json", results)
    return summary


def evaluate_saved(data_dir, trained_dir, out, split="test", allow_random_overlap=False):
    """Evaluate frozen checkpoints without fitting or selecting anything on holdout."""
    from .io import read_json
    if split not in {"test", "external"}:
        raise ValueError("Only test/external are final evaluation splits")
    data_dir, trained_dir = Path(data_dir), Path(trained_dir)
    saved_hashes = read_json(trained_dir / "data_hashes.json")
    for name in ["train", "validation"]:
        if file_hash(data_dir / f"{name}.jsonl") != saved_hashes[name]:
            raise ValueError("Training/validation dataset changed after fitting")
    rows = read_jsonl(data_dir / f"{split}.jsonl")
    if not rows or {r["label"] for r in rows} != {0, 1}:
        raise ValueError("Final classification split requires both classes")
    all_rows = {name: read_jsonl(data_dir / f"{name}.jsonl") for name in ["train", "validation"]}
    if allow_random_overlap:
        from .io import read_json
        if not read_json(trained_dir / "config.json").get("allow_random_overlap"):
            raise ValueError("Saved training run did not declare random-overlap mode")
        assert_random_splits(data_dir, dict(all_rows, **{split: rows}))
    else:
        assert_disjoint(dict(all_rows, **{split: rows}))
    artifacts = sorted(trained_dir.glob("*/model.joblib"))
    if not artifacts:
        raise ValueError("No trained checkpoints")
    out = new_output(out)
    summary = []
    for artifact in artifacts:
        model = joblib.load(artifact)
        labels = [r["label"] for r in rows]
        started = time.perf_counter()
        p = model.probabilities([r["text"] for r in rows])
        elapsed = time.perf_counter() - started
        result = classification(labels, p, model.threshold)
        result["accuracy_ci95"] = group_bootstrap_accuracy(labels, p, [r["group_id"] for r in rows], model.threshold, model.seed)
        folder = out / artifact.parent.name
        predictions = [{"id": r["id"], "group_id": r["group_id"], "text": r["text"], "label": r["label"],
                        "prediction": int(prob >= model.threshold), "probability": float(prob),
                        "error_type": "" if int(prob >= model.threshold) == r["label"] else
                        "false_accept" if prob >= model.threshold else "false_reject"} for r, prob in zip(rows, p)]
        write_csv(folder / f"{split}_predictions.csv", predictions)
        write_csv(folder / "failure_cases.csv", [r for r in predictions if r["error_type"]], list(predictions[0]))
        record = {"kind": model.kind, "seed": model.seed, "split": split, "metrics": {split: result},
                  "threshold": model.threshold, "model_sha256": file_hash(artifact),
                  "holdout_sha256": file_hash(data_dir / f"{split}.jsonl"), "inference_seconds": elapsed,
                  "allow_random_overlap": allow_random_overlap}
        write_json(folder / "metrics.json", record)
        summary.append(record)
    write_json(out / "results.json", summary)
    write_csv(out / "summary.csv", [{"model": r["kind"], "seed": r["seed"], "split": split,
                                    **{k: v for k, v in r["metrics"][split].items() if isinstance(v, (int, float))}} for r in summary])
    return summary
