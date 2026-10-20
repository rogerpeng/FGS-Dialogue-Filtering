"""Build paper tables from verified phase receipts; never invent missing results."""
from .io import read_json, read_jsonl, write_csv, write_json


def study_report(folder, out):
    from .study import phase_path
    from .evaluation import summarize_responses
    state = read_json(folder / "state.json")
    plan = read_json(folder / "protocol.json")
    tables, completed = {}, []
    for phase, record in state["phases"].items():
        if record["status"] != "completed" or phase == "report":
            continue
        root = phase_path(folder, phase)
        completed.append(phase)
        if phase in {"develop", "test", "external-test"}:
            rows = []
            for path in sorted(root.glob("*/*/*/metrics.json")):
                result = read_json(path)
                for split, metrics in result["metrics"].items():
                    rows.append({"task": path.relative_to(root).parts[0],
                                 "experiment": path.relative_to(root).parts[1],
                                 "model": result["kind"], "seed": result["seed"], "split": split,
                                 **{k: v for k, v in metrics.items() if isinstance(v, (int, float))}})
            if rows:
                fields = sorted(set().union(*(r.keys() for r in rows)))
                write_csv(out / f"{phase}_classifiers.csv", rows, fields)
                tables[phase] = len(rows)
            if phase in {"test", "external-test"} and plan.get("classifier_contrasts"):
                from .statistics import compare_classifiers, holm
                comparisons = []
                split = "external" if phase == "external-test" else "test"
                for task in ["question_filter", "answer_filter"]:
                    # A predeclared seed is primary; other seeds describe training variability.
                    seed = plan.get("primary_seed", plan["classifiers"]["seeds"][0])
                    for a, b in plan["classifier_contrasts"]:
                        result = compare_classifiers(root / task / "main" / f"{a}_seed{seed}" / f"{split}_predictions.csv",
                            root / task / "main" / f"{b}_seed{seed}" / f"{split}_predictions.csv",
                            out / f"{phase}_comparisons" / task / f"{a}_vs_{b}", seed)
                        comparisons.append({"task": task, "a": a, "b": b, "seed": seed, **result})
                valid = [c for c in comparisons if c["group_analysis"]["wilcoxon_p"] is not None]
                for c, p in zip(valid, holm([c["group_analysis"]["wilcoxon_p"] for c in valid])):
                    c["group_analysis"]["wilcoxon_p_holm"] = p
                write_json(out / f"{phase}_planned_comparisons.json", comparisons)
        if phase.startswith("generate-"):
            responses = read_jsonl(root / "responses.jsonl")
            summary = summarize_responses(responses)
            write_json(out / f"{phase}.json", summary)
            write_csv(out / f"{phase}.csv", summary["systems"])
            categories = {}
            for category in sorted({r.get("category", "unspecified") for r in responses}):
                categories[category] = summarize_responses([r for r in responses if r.get("category", "unspecified") == category])
            write_json(out / f"{phase}_categories.json", categories)
            tables[phase] = summary
        if phase == "llm-test":
            for path in root.rglob("*.json"):
                if path.name in {"metrics.json", "results.json"}:
                    write_json(out / ("llm_" + "_".join(path.relative_to(root).parts)), read_json(path))
        if phase == "analyze":
            stats = read_json(root / "human/statistics.json")
            write_csv(out / "human_descriptive.csv", stats["descriptive"])
            rows = []
            for c in stats["comparisons"]:
                rows.append({"reference": c["reference"], "comparator": c["comparator"],
                             "matched_pairs": c["matched_rating_pairs"],
                             "mean_reference": c["matched_mean_reference"],
                             "mean_comparator": c["matched_mean_comparator"],
                             "supplementary_percent": c["relative_change_percent"],
                             **(c["participant_analysis"] or {}),
                             "crossed_ci95": (c["crossed_bootstrap"] or {}).get("ci95")})
            if rows:
                write_csv(out / "human_comparisons.csv", rows, sorted(set().union(*(r.keys() for r in rows))))
            write_json(out / "human_statistics.json", stats)
            if (root / "expert").exists():
                for path in (root / "expert").glob("*.json"):
                    write_json(out / ("expert_" + path.name), read_json(path))
                for path in (root / "expert").glob("*.csv"):
                    (out / ("expert_" + path.name)).write_bytes(path.read_bytes())
    required = ["develop", "test", "generate-dev", "generate-test", "blind", "analyze"]
    if plan.get("fine_tuning"):
        required += ["finetune-prepare", "finetune-submit", "finetune-sync"]
    if plan.get("llm_classifiers"):
        required += ["llm-test"]
    missing = sorted(set(required) - set(completed))
    report = {"completed": completed, "missing_phases": missing, "tables": tables,
              "note": "Phase completion does not establish scientific adequacy. Check cloud success, expert labels, sample size and external validity."}
    write_json(out / "completeness.json", report)
    (out / "REPORT.md").write_text("# Experiment report\n\n" +
        "Completed: " + ", ".join(completed) + "\n\nMissing phases: " + (", ".join(missing) or "none") +
        "\n\nTables are generated from verified artifacts. Seeds describe training variability, not additional independent participants. "
        "Human primary effects are paired rating-point differences; percentages on Likert scores are supplementary arithmetic only. "
        "Missing phases/results must not be reported as zero or as completed experiments.\n", encoding="utf-8")
    return {"completed": completed, "missing_phases": missing, "output": str(out)}
