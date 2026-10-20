"""Frozen study configuration, resumable phases, explicit holdout evaluation."""
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import json
import re

from .dataset import freeze_dataset, verify_dataset
from .io import digest, environment, file_hash, new_output, read_json, read_jsonl, write_json, write_jsonl

PHASES = ("develop", "finetune-prepare", "finetune-submit", "finetune-sync", "generate-dev", "test", "external-test",
          "llm-test", "generate-test", "generate-external", "blind", "analyze", "report")


def safe_name(value):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", value):
        raise ValueError(f"Invalid artifact name: {value}")
    return value


def validate_plan(plan):
    if plan.get("schema_version") != 1 or not isinstance(plan.get("sources"), list) or not plan["sources"]:
        raise ValueError("Study plan needs schema_version=1 and nonempty sources")
    cfg = plan.get("classifiers", {})
    allowed = {"kinds", "seeds", "epochs", "patience", "learning_rates", "Cs", "embedding_model", "embedding_revision",
               "local_files_only", "min_precision"}
    if set(cfg) - allowed or not cfg.get("kinds") or not cfg.get("seeds"):
        raise ValueError("Invalid classifier configuration")
    if len(set(cfg["kinds"])) != len(cfg["kinds"]) or len(set(cfg["seeds"])) != len(cfg["seeds"]):
        raise ValueError("Repeated classifier kinds/seeds")
    if plan.get("primary_seed", cfg["seeds"][0]) not in cfg["seeds"]:
        raise ValueError("primary_seed must be one of the trained seeds")
    fractions = plan.get("learning_fractions", [])
    if any(not 0 < f <= 1 for f in fractions) or len({int(f * 1000) for f in fractions}) != len(fractions):
        raise ValueError("Learning fractions must be distinct, in (0,1], at thousandth precision")
    for pair in plan.get("classifier_contrasts", []):
        if len(pair) != 2 or pair[0] == pair[1] or not set(pair) <= set(cfg["kinds"]):
            raise ValueError("Classifier contrasts must name configured models")
    if any(k.startswith("transformer") for k in cfg["kinds"]) and (not cfg.get("embedding_model") or not cfg.get("embedding_revision")):
        raise ValueError("Pin embedding_model and embedding_revision for formal transformer experiments")
    jobs = plan.get("fine_tuning", [])
    names = [safe_name(j["name"]) for j in jobs]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate fine-tuning job names")
    systems = plan.get("generation_systems", [])
    names = [safe_name(s["name"]) for s in systems]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate generation system names")
    from .pipeline import VARIANTS
    system_names = set()
    for s in systems:
        variants = s.get("variants", list(VARIANTS))
        if not variants or len(set(variants)) != len(variants) or set(variants) - set(VARIANTS):
            raise ValueError("Invalid or repeated ablation variants")
        if not isinstance(s.get("replicates", 1), int) or s.get("replicates", 1) < 1:
            raise ValueError("replicates must be positive integer")
        system_names.update(s["name"] + "__" + v for v in variants)
    comparisons = plan.get("human_evaluation", {}).get("contrasts", [])
    if any(len(pair) != 2 or pair[0] == pair[1] or not set(pair) <= system_names for pair in comparisons):
        raise ValueError("Human contrasts must name distinct configured system__variant pairs")
    if len({tuple(p) for p in comparisons}) != len(comparisons):
        raise ValueError("Duplicate human contrasts")
    return plan


def freeze_study(plan_path, out):
    plan_path = Path(plan_path).resolve()
    plan = validate_plan(read_json(plan_path))
    sources = [(plan_path.parent / p).resolve() for p in plan["sources"]]
    for p in sources:
        if not p.is_file():
            raise ValueError(f"Dataset not supplied yet: {p}")
    out = new_output(out)
    manifest = freeze_dataset(sources, out / "data", **plan.get("split", {}))
    for task in ["question_filter", "answer_filter"]:
        for split in ["train", "validation", "test"]:
            if {r["label"] for r in read_jsonl(out / "data" / task / f"{split}.jsonl")} != {0, 1}:
                raise ValueError(f"Study requires both classes in {task}/{split}; add independent groups")
    if plan.get("generation_systems") and any(not manifest["splits"][s]["evaluation"]["n"] for s in ["validation", "test"]):
        raise ValueError("Generation study requires validation and test evaluation items")
    write_json(out / "protocol.json", plan)
    lock = {"protocol_sha256": file_hash(out / "protocol.json"), "dataset_manifest_sha256": file_hash(out / "data/manifest.json"),
            "source_sha256": environment()["source_sha256"], "created_utc": datetime.now(timezone.utc).isoformat(),
            "note": "This is an internal frozen protocol, not a public preregistration."}
    write_json(out / "lock.json", lock)
    write_json(out / "state.json", {"phases": {}, "attempts": []})
    return {"study": str(out), "records": manifest["records"], "components": manifest["global_components"], "next": "study-run --phase develop"}


def verify_study(folder):
    folder = Path(folder)
    lock = read_json(folder / "lock.json")
    if file_hash(folder / "protocol.json") != lock["protocol_sha256"] or file_hash(folder / "data/manifest.json") != lock["dataset_manifest_sha256"]:
        raise ValueError("Frozen protocol/manifest changed; create a new version rather than overwriting")
    if environment()["source_sha256"] != lock["source_sha256"]:
        raise ValueError("Research code changed after study freeze; create a new version to avoid mixed-code results")
    verify_dataset(folder / "data")
    return read_json(folder / "protocol.json")


def phase_path(folder, phase):
    state = read_json(Path(folder) / "state.json")
    record = state["phases"].get(phase)
    if not record or record["status"] != "completed":
        raise ValueError(f"Required phase not complete: {phase}")
    path = Path(folder) / record["path"]
    for relative, expected in record["files"].items():
        if file_hash(path / relative) != expected:
            raise ValueError(f"Completed artifact changed: {phase}/{relative}")
    return path


def artifact_hashes(folder):
    return {p.relative_to(folder).as_posix(): file_hash(p) for p in sorted(Path(folder).rglob("*"))
            if p.is_file() and ".cache" not in p.parts and "cache" not in p.parts}


@contextmanager
def phase_lock(folder):
    lock = Path(folder) / ".running"
    try:
        with lock.open("x", encoding="utf-8") as h:
            h.write(datetime.now(timezone.utc).isoformat())
    except FileExistsError as exc:
        raise ValueError("Study phase already running; after a crash verify no process remains before removing .running") from exc
    try:
        yield
    finally:
        lock.unlink()


def _model_reference(folder, value):
    if isinstance(value, str):
        if not value or value.startswith("YOUR_"):
            raise ValueError("Model identifier has not been configured")
        return value
    if not isinstance(value, dict) or set(value) != {"fine_tune_job"}:
        raise ValueError("Use model id or {fine_tune_job: name}")
    name = safe_name(value["fine_tune_job"])
    job = read_json(Path(folder) / "cloud" / name / "job.json")
    if job.get("status") != "succeeded" or not job.get("fine_tuned_model"):
        raise ValueError(f"Fine-tuning not ready: {name}; run finetune-sync")
    prepared = read_json(phase_path(folder, "finetune-prepare") / "jobs.json")
    if prepared[name] != job["fingerprint"]:
        raise ValueError("Fine-tuning provenance does not match this study")
    return job["fine_tuned_model"]


def _generate(folder, plan, out, split, live):
    from .evaluation import generation_experiment
    from .pipeline import VARIANTS
    systems = plan.get("generation_systems", [])
    if not systems:
        raise ValueError("Configure generation_systems first")
    items_path = folder / "data/evaluation" / f"{split}.jsonl"
    if not read_jsonl(items_path):
        raise ValueError(f"No evaluation items in {split}")
    merged = []
    for sys in systems:
        name = sys["name"]
        config = dict(sys["api"])
        config["generator_model"] = _model_reference(folder, sys["generator"])
        config["refiner_model"] = _model_reference(folder, sys["refiner"])
        config_path = out / f"{name}.json"
        write_json(config_path, config)
        local_paths = {}
        for task, key in [("question_filter", "question_model"), ("answer_filter", "answer_model")]:
            ref = sys.get(key)
            if ref:
                develop = phase_path(folder, "develop")
                local_paths[key] = develop / f"{task}/main/{safe_name(ref)}/model.joblib"
        result_dir = out / name
        result = generation_experiment(items_path, config_path, result_dir,
                                       local_paths.get("question_model"), local_paths.get("answer_model"),
                                       sys.get("variants", list(VARIANTS)), sys.get("replicates", 1), live,
                                       cache_dir=folder / "api_cache" / name)
        if any(r["statuses"].get("provider_or_filter_error") for r in result["systems"]):
            raise ValueError(f"Provider/filter errors in {name}; phase remains incomplete, cache is retained")
        for r in read_jsonl(result_dir / "responses.jsonl"):
            r["system"] = name + "__" + r["system"]
            r["split"] = split
            merged.append(r)
    write_jsonl(out / "responses.jsonl", merged)
    return {"records": len(merged), "split": split}


def run_phase(folder, phase, live=False, ratings=None, annotations=None):
    folder = Path(folder).resolve()
    if phase not in PHASES:
        raise ValueError("Unknown phase")
    plan = verify_study(folder)
    state = read_json(folder / "state.json")
    # Cloud sync intentionally queries again until succeeded, rather than caching a running state.
    if phase in state["phases"] and state["phases"][phase]["status"] == "completed" and phase not in {"finetune-sync", "report"}:
        return {"status": "already_completed", "path": str(phase_path(folder, phase))}
    with phase_lock(folder):
        attempt = len(state["attempts"]) + 1
        out = new_output(folder / "artifacts" / f"{phase}-{attempt:03}")
        record = {"phase": phase, "status": "running", "path": out.relative_to(folder).as_posix()}
        state["attempts"].append(record)
        write_json(folder / "state.json", state)
        try:
            result = _execute(folder, plan, phase, out, live, ratings, annotations)
            write_json(out / "phase_result.json", result)
            record.update(status="completed", files=artifact_hashes(out))
            state["phases"][phase] = dict(record)
            write_json(folder / "state.json", state)
            return {"status": "completed", "phase": phase, "path": str(out), "result": result}
        except Exception as exc:
            record.update(status="failed", error_type=type(exc).__name__)
            write_json(folder / "state.json", state)
            raise


def _execute(folder, plan, phase, out, live, ratings, annotations):
    if phase == "develop":
        from .experiments import run_experiments
        cfg = dict(plan["classifiers"])
        paths = []
        for task in ["question_filter", "answer_filter"]:
            jobs = [("main", 1., False)]
            jobs += [(f"fraction_{int(f*1000)}", f, False) for f in plan.get("learning_fractions", []) if f != 1]
            if plan.get("permutation_control", True):
                jobs.append(("permuted", 1., True))
            for name, fraction, permute in jobs:
                path = out / task / name
                run_experiments(folder / "data" / task, path, **cfg, train_fraction=fraction,
                                permute_labels=permute, evaluate_test=False)
                paths.append(str(path.relative_to(out)))
        return {"training_jobs": paths, "test_opened": False}
    if phase in {"test", "external-test"}:
        from .experiments import evaluate_saved
        develop = phase_path(folder, "develop")
        paths = []
        for task in ["question_filter", "answer_filter"]:
            for train_dir in sorted((develop / task).iterdir()):
                evaluate_saved(folder / "data" / task, train_dir, out / task / train_dir.name,
                               split="external" if phase == "external-test" else "test")
                paths.append(f"{task}/{train_dir.name}")
        return {"evaluated_jobs": paths, "no_training_or_selection": True}
    if phase == "finetune-prepare":
        from .finetune import prepare_job
        result = {}
        for spec in plan.get("fine_tuning", []):
            cfg = {k: v for k, v in spec.items() if k != "name"}
            path = folder / "cloud" / spec["name"]
            if path.exists():
                job = read_json(path / "job.json")
                inputs = {"config": cfg, **{s + "_sha256": file_hash(folder / "data" / cfg["task"] / f"{s}.jsonl")
                                            for s in ["train", "validation"]}}
                if job["fingerprint"] != digest(json.dumps(inputs, sort_keys=True)) or any(
                    file_hash(path / f"{s}.jsonl") != inputs[s + "_sha256"] for s in ["train", "validation"]):
                    raise ValueError("Existing cloud preparation differs from frozen study")
            else:
                job = prepare_job(folder / "data" / cfg["task"] / "train.jsonl", folder / "data" / cfg["task"] / "validation.jsonl", cfg, path)
            result[spec["name"]] = job["fingerprint"]
        if not result:
            raise ValueError("No fine_tuning jobs configured")
        write_json(out / "jobs.json", result)
        return result
    if phase in {"finetune-submit", "finetune-sync"}:
        from .finetune import submit_job, sync_job
        prepared = read_json(phase_path(folder, "finetune-prepare") / "jobs.json")
        result = {}
        for name, fingerprint in prepared.items():
            path = folder / "cloud" / name
            if read_json(path / "job.json")["fingerprint"] != fingerprint:
                raise ValueError("Cloud preparation fingerprint mismatch")
            job = (submit_job if phase == "finetune-submit" else sync_job)(path, live=live)
            result[name] = {"job_id": job.get("job_id"), "status": job["status"], "model": job.get("fine_tuned_model")}
        return result
    if phase == "llm-test":
        from .evaluation import llm_classification
        results = {}
        for i, spec in enumerate(plan.get("llm_classifiers", [])):
            name = safe_name(spec["name"])
            config_path = out / f"{name}.json"
            write_json(config_path, spec["api"])
            for task, role in [("question_filter", "question"), ("answer_filter", "answer")]:
                results[f"{name}/{task}"] = llm_classification(folder / "data" / task, config_path, out / name / task, role, live)
        if not results:
            raise ValueError("No llm_classifiers configured")
        return results
    if phase in {"generate-dev", "generate-test", "generate-external"}:
        if phase != "generate-dev":
            phase_path(folder, "generate-dev")
        return _generate(folder, plan, out, {"generate-dev": "validation", "generate-test": "test", "generate-external": "external"}[phase], live)
    if phase == "blind":
        from .evaluation import blind_sheets
        cfg = plan["human_evaluation"]
        responses = phase_path(folder, "generate-test") / "responses.jsonl"
        blind_sheets(responses, out / "sheets", cfg["participants"], cfg.get("seed", 42),
                     max_items=cfg.get("items_per_participant"), one_replicate_per_item=True)
        return {"sheets": "sheets", "n_participants_planned": len(cfg["participants"])}
    if phase == "analyze":
        from .statistics import analyze_ratings
        from .evaluation import annotation_summary
        if not ratings:
            raise ValueError("Provide --ratings after collecting and unblinding real ratings")
        import csv
        mapping = phase_path(folder, "blind") / "sheets/PRIVATE_mapping.csv"
        with mapping.open(encoding="utf-8-sig", newline="") as h:
            allowed = {(r["participant_id"], r["item_id"], r["system"]) for r in csv.DictReader(h)}
        source = Path(ratings)
        with source.open(encoding="utf-8-sig", newline="") as h:
            rows = list(csv.DictReader(h))
        if not rows or any(r.get("mapping_sha256") != file_hash(mapping) or
                           (r["participant_id"], r["item_id"], r["system"]) not in allowed for r in rows):
            raise ValueError("Ratings must be unblinded from this study's registered mapping")
        (out / "ratings.csv").write_bytes(source.read_bytes())
        cfg = plan["human_evaluation"]
        result = analyze_ratings(out / "ratings.csv", out / "human", contrasts=cfg.get("contrasts"),
                                 reference=cfg.get("reference", "full"), seed=cfg.get("seed", 42))
        if annotations:
            (out / "annotations.csv").write_bytes(Path(annotations).read_bytes())
            annotation_summary(out / "annotations.csv", out / "expert")
        return {"comparisons": len(result["comparisons"]), "ratings_sha256": file_hash(source)}
    if phase == "report":
        from .reporting import study_report
        return study_report(folder, out)
    raise ValueError("Unhandled phase")
