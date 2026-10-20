"""Use `python -m fgs --help` for all research workflows."""
import argparse
import json
from pathlib import Path
import sys

from .io import read_jsonl, write_json


def parser():
    p = argparse.ArgumentParser(description="FGS dialogue revision 3 research toolkit")
    sub = p.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Import binary text files or labeled JSONL and freeze grouped splits")
    prepare.add_argument("--positive")
    prepare.add_argument("--negative")
    prepare.add_argument("--input", help="Alternative JSONL: text,label,optional group_id/source")
    prepare.add_argument("--out", required=True)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--near-threshold", type=float, default=.9)

    train = sub.add_parser("train", help="Train and select on validation only")
    train.add_argument("--data", required=True)
    train.add_argument("--out", required=True)
    train.add_argument("--models", nargs="+", choices=["tfidf", "word2vec", "word2vec_linear", "transformer", "transformer_mlp"], default=["tfidf", "word2vec"])
    train.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    train.add_argument("--epochs", type=int, default=300)
    train.add_argument("--patience", type=int, default=25)
    train.add_argument("--learning-rates", nargs="+", type=float, default=[.001, .01])
    train.add_argument("--cs", nargs="+", type=float, default=[.1, 1., 10.])
    train.add_argument("--embedding-model")
    train.add_argument("--embedding-revision")
    train.add_argument("--local-files-only", action="store_true")
    train.add_argument("--min-precision", type=float)
    train.add_argument("--train-fraction", type=float, default=1.)
    train.add_argument("--permute-labels", action="store_true", help="Negative control only; never mix with real-label results")
    train.add_argument("--allow-random-overlap", action="store_true", help="Use an explicitly manifested random QA split; shared text/source groups are permitted and must be disclosed")

    ft = sub.add_parser("prepare-finetune", help="Export group-disjoint fine-tuning train/validation/test; no upload")
    ft.add_argument("--input", required=True)
    ft.add_argument("--out", required=True)
    ft.add_argument("--seed", type=int, default=42)

    legacy = sub.add_parser("import-ratings", help="Audit headerless 12-column legacy dia.csv")
    legacy.add_argument("--input", required=True)
    legacy.add_argument("--out", required=True)

    stats = sub.add_parser("statistics", help="Participant-level paired analysis, effect sizes and Holm correction")
    stats.add_argument("--input", required=True)
    stats.add_argument("--out", required=True)
    stats.add_argument("--reference", default="full")
    stats.add_argument("--seed", type=int, default=42)
    stats.add_argument("--allow-unverified", action="store_true")

    llm = sub.add_parser("llm-classify", help="LLM hard-label baseline on the same frozen test split")
    llm.add_argument("--data", required=True)
    llm.add_argument("--config", required=True)
    llm.add_argument("--out", required=True)
    llm.add_argument("--role", choices=["question", "answer"], default="question")
    llm.add_argument("--live", action="store_true")

    gen = sub.add_parser("generate", help="Run bounded dialogue pipeline variants; live API opt-in")
    gen.add_argument("--items", required=True)
    gen.add_argument("--config", required=True)
    gen.add_argument("--out", required=True)
    gen.add_argument("--question-model")
    gen.add_argument("--answer-model")
    from .pipeline import VARIANTS
    gen.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=["b1", "b2", "full", "no_query", "no_answer"])
    gen.add_argument("--replicates", type=int, default=1)
    gen.add_argument("--cache-dir")
    gen.add_argument("--live", action="store_true")

    blind = sub.add_parser("blind", help="Create separate blinded sheets and PRIVATE immutable mapping")
    blind.add_argument("--responses", required=True)
    blind.add_argument("--participants", nargs="+", required=True)
    blind.add_argument("--out", required=True)
    blind.add_argument("--seed", type=int, default=42)
    unblind = sub.add_parser("unblind", help="Validate and map completed score sheets")
    unblind.add_argument("--mapping", required=True)
    unblind.add_argument("--sheets", nargs="+", required=True)
    unblind.add_argument("--out", required=True)
    ann = sub.add_parser("annotations", help="Summarize independent expert labels and inter-rater kappa")
    ann.add_argument("--input", required=True)
    ann.add_argument("--out", required=True)

    summary = sub.add_parser("summarize", help="Summarize saved generation outcomes and reference metrics")
    summary.add_argument("--responses", required=True)
    summary.add_argument("--out", required=True)
    demo = sub.add_parser("demo", help="Offline fixture demonstration, NOT real LLM inference")
    demo.add_argument("--question", default="如何解決人際衝突？")
    demo.add_argument("--question-model")
    demo.add_argument("--answer-model")
    bundle = sub.add_parser("bundle", help="Create allowlisted source-only ZIP; does not publish")
    bundle.add_argument("--out", required=True)
    plots = sub.add_parser("plot", help="Generate training curves and confusion matrices as PNG/PDF; requires [plots]")
    plots.add_argument("--run", required=True, help="One model/seed run directory containing metrics.json")
    plots.add_argument("--out", required=True)
    audit = sub.add_parser("inventory", help="Count and fingerprint known corpus versions without loading old code")
    audit.add_argument("--legacy-root", default="..")
    audit.add_argument("--out", required=True)
    overlap = sub.add_parser("audit-overlap", help="Screen evaluation questions/references against generation training data")
    overlap.add_argument("--evaluation", required=True)
    overlap.add_argument("--training", required=True)
    overlap.add_argument("--out", required=True)
    overlap.add_argument("--threshold", type=float, default=.9)
    compare = sub.add_parser("compare-classifiers", help="Pair test predictions; group effects and McNemar sensitivity")
    compare.add_argument("--a", required=True)
    compare.add_argument("--b", required=True)
    compare.add_argument("--out", required=True)
    compare.add_argument("--seed", type=int, default=42)
    web = sub.add_parser("serve", help="Loopback-only browser demo; offline by default")
    web.add_argument("--port", type=int, default=8765)
    web.add_argument("--config")
    web.add_argument("--question-model")
    web.add_argument("--answer-model")
    web.add_argument("--cache-dir", default="runs/web_cache")
    web.add_argument("--live", action="store_true")
    from .study import PHASES
    freeze = sub.add_parser("study-freeze", help="Freeze joint data, protocol and code hashes")
    freeze.add_argument("--plan", required=True)
    freeze.add_argument("--out", required=True)
    phase = sub.add_parser("study-run", help="Run a resumable frozen study phase")
    phase.add_argument("--study", required=True)
    phase.add_argument("--phase", choices=PHASES, required=True)
    phase.add_argument("--live", action="store_true")
    phase.add_argument("--ratings")
    phase.add_argument("--annotations")
    power = sub.add_parser("power", help="Prospective sample-size sensitivity simulation")
    power.add_argument("--config", required=True)
    power.add_argument("--out", required=True)
    evaluate = sub.add_parser("evaluate", help="Evaluate saved classifiers without fitting")
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument("--trained", required=True)
    evaluate.add_argument("--out", required=True)
    evaluate.add_argument("--split", choices=["test", "external"], default="test")
    evaluate.add_argument("--allow-random-overlap", action="store_true", help="Evaluate an explicitly manifested random split with the same declared training policy")
    blind.add_argument("--max-items", type=int)
    train.add_argument("--evaluate-test", action="store_true", help="Exploratory legacy mode; prefer separate evaluate")
    convert = sub.add_parser("json-to-csv", help="Convert JSON/JSONL to Excel-friendly UTF-8 CSV")
    convert.add_argument("--input", required=True)
    convert.add_argument("--out", required=True)
    convert.add_argument("--records-key")
    convert.add_argument("--columns", nargs="+")
    convert.add_argument("--chat-pairs", action="store_true")
    convert.add_argument("--raw-cells", action="store_true", help="Preserve formula-like strings without Excel protection")
    reviewed = sub.add_parser("import-reviewed-qa", help="Import approved QA CSV/JSON/JSONL into development task records")
    reviewed.add_argument("--input", required=True)
    reviewed.add_argument("--out", required=True)
    return p


def dispatch(a):
    if a.command == "import-reviewed-qa":
        from .qa_import import import_reviewed_qa
        return import_reviewed_qa(a.input, a.out)
    if a.command == "json-to-csv":
        from .conversion import convert_json
        return convert_json(a.input, a.out, a.records_key, a.columns, a.chat_pairs, not a.raw_cells)
    if a.command == "study-freeze":
        from .study import freeze_study
        return freeze_study(a.plan, a.out)
    if a.command == "study-run":
        from .study import run_phase
        return run_phase(a.study, a.phase, a.live, a.ratings, a.annotations)
    if a.command == "power":
        from .planning import simulate_power
        from .io import read_json
        return simulate_power(read_json(a.config), a.out)
    if a.command == "evaluate":
        from .experiments import evaluate_saved
        return evaluate_saved(a.data, a.trained, a.out, a.split, allow_random_overlap=a.allow_random_overlap)
    if a.command == "prepare":
        from .data import prepare, prepare_rows
        if a.input:
            if a.positive or a.negative:
                raise ValueError("Use --input OR --positive/--negative")
            return prepare_rows(read_jsonl(a.input), a.out, a.seed, a.near_threshold)
        if not a.positive or not a.negative:
            raise ValueError("Provide both --positive and --negative")
        return prepare(a.positive, a.negative, a.out, a.seed, a.near_threshold)
    if a.command == "train":
        from .experiments import run_experiments
        return run_experiments(a.data, a.out, a.models, a.seeds, a.epochs, a.patience, a.learning_rates,
                               a.cs, a.embedding_model, a.embedding_revision, a.local_files_only,
                               a.min_precision, a.train_fraction, a.permute_labels, a.evaluate_test,
                               allow_random_overlap=a.allow_random_overlap)
    if a.command == "prepare-finetune":
        from .data import export_finetune
        return export_finetune(a.input, a.out, a.seed)
    if a.command == "import-ratings":
        from .data import import_legacy_ratings
        return import_legacy_ratings(a.input, a.out)
    if a.command == "statistics":
        from .statistics import analyze_ratings
        return analyze_ratings(a.input, a.out, a.reference, a.seed, a.allow_unverified)
    if a.command == "llm-classify":
        from .evaluation import llm_classification
        return llm_classification(a.data, a.config, a.out, a.role, a.live)
    if a.command == "generate":
        from .evaluation import generation_experiment
        return generation_experiment(a.items, a.config, a.out, a.question_model, a.answer_model,
                                     a.variants, a.replicates, a.live, a.cache_dir)
    if a.command == "blind":
        from .evaluation import blind_sheets
        blind_sheets(a.responses, a.out, a.participants, a.seed, max_items=a.max_items)
        return {"output": a.out}
    if a.command == "unblind":
        from .evaluation import unblind
        unblind(a.mapping, a.sheets, a.out)
        return {"output": a.out}
    if a.command == "annotations":
        from .evaluation import annotation_summary
        return annotation_summary(a.input, a.out)
    if a.command == "summarize":
        from .evaluation import summarize_responses
        result = summarize_responses(read_jsonl(a.responses))
        if Path(a.out).exists():
            raise ValueError("Output file exists")
        write_json(a.out, result)
        return result
    if a.command == "demo":
        import joblib
        from .pipeline import DialoguePipeline, FixtureBackend, VARIANTS
        qf = joblib.load(a.question_model) if a.question_model else None
        af = joblib.load(a.answer_model) if a.answer_model else None
        if bool(qf) != bool(af):
            raise ValueError("Supply both model paths or neither")
        row = DialoguePipeline(FixtureBackend(), qf, af).run(a.question, VARIANTS["full"] if qf else VARIANTS["refinement_only"])
        return {"DEMO_ONLY": True, "note": "Fixed synthetic text; not a live LLM or research result", **row}
    if a.command == "bundle":
        from .release import bundle
        return bundle(a.out)
    if a.command == "plot":
        from .plots import plot_run
        return plot_run(a.run, a.out)
    if a.command == "inventory":
        from .audit import corpus_inventory
        return corpus_inventory(a.legacy_root, a.out)
    if a.command == "audit-overlap":
        from .audit import overlap_audit
        return overlap_audit(a.evaluation, a.training, a.out, a.threshold)
    if a.command == "compare-classifiers":
        from .statistics import compare_classifiers
        return compare_classifiers(a.a, a.b, a.out, a.seed)
    if a.command == "serve":
        from .web import serve
        serve(a.port, a.config, a.question_model, a.answer_model, a.live, a.cache_dir)


def main():
    try:
        result = dispatch(parser().parse_args())
        if result is not None:
            print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
            if isinstance(result, dict) and "systems" in result and any(s.get("statuses", {}).get("provider_or_filter_error", 0) for s in result["systems"]):
                print("Run contains provider/filter errors; inspect saved traces before using results.", file=sys.stderr)
                raise SystemExit(3)
    except (ValueError, FileNotFoundError, FileExistsError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
