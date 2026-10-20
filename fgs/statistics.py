"""Paired ratings with participant-level inference and item-level sensitivity.

Repeated ratings are NOT treated as independent people. Missing pairs are
reported, never filled with zero. No inference can repair an invalid legacy map.
"""
from collections import defaultdict
import csv
import itertools
from pathlib import Path
import numpy as np
from scipy.stats import rankdata, wilcoxon

from .io import file_hash, new_output, write_csv, write_json


def holm(pvalues):
    order = np.argsort(pvalues)
    adjusted, previous = [0.0] * len(pvalues), 0.0
    for rank, idx in enumerate(order):
        previous = max(previous, min(1.0, float(pvalues[idx]) * (len(pvalues) - rank)))
        adjusted[int(idx)] = previous
    return adjusted


def paired_summary(differences, seed=42, repeats=5000):
    d = np.asarray(differences, dtype=float)
    if not len(d) or not np.isfinite(d).all():
        raise ValueError("Nonempty finite paired differences required")
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if len(d) > 1 else None
    nz = d[d != 0]
    ranks = rankdata(np.abs(nz)) if len(nz) else np.array([])
    rbc = float(np.sum(ranks * np.sign(nz)) / ranks.sum()) if len(nz) else 0.0
    result = {"n_units": len(d), "mean_difference": mean, "median_difference": float(np.median(d)),
              "cohen_dz": mean / sd if sd and sd > 0 else None, "rank_biserial": rbc,
              "nonzero_units": len(nz)}
    if len(d) < 2:
        return dict(result, ci95=None, wilcoxon_p=None, wilcoxon_statistic=None, sign_flip_p=None, warning="Insufficient independent units for inference")
    rng = np.random.default_rng(seed)
    boot = [float(rng.choice(d, size=len(d), replace=True).mean()) for _ in range(repeats)]
    result["ci95"] = [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]
    test = wilcoxon(d, zero_method="wilcox", alternative="two-sided", method="auto") if len(nz) else None
    result["wilcoxon_p"] = float(test.pvalue) if test else 1.0
    result["wilcoxon_statistic"] = float(test.statistic) if test else 0.0
    result["wilcoxon_method"] = "scipy auto, two-sided, zero_method=wilcox"
    if len(d) <= 15:
        null = [abs(float(np.mean(d * np.array(signs)))) for signs in itertools.product([-1, 1], repeat=len(d))]
        result["sign_flip_p"] = sum(v >= abs(mean) - 1e-12 for v in null) / len(null)
        result["sign_flip_method"] = "exact two-sided"
    else:
        count = sum(abs(float(np.mean(d * rng.choice([-1, 1], size=len(d))))) >= abs(mean) - 1e-12 for _ in range(repeats))
        result["sign_flip_p"] = (count + 1) / (repeats + 1)
        result["sign_flip_method"] = "Monte Carlo two-sided, plus-one correction"
    return result


def analyze_ratings(path, out, reference="full", seed=42, allow_unverified=False, contrasts=None):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"participant_id", "item_id", "system", "score"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"Ratings require {sorted(required)}")
        source = list(reader)
    ratings, missing, unverified = {}, 0, 0
    for row in source:
        if row["score"] in ("", "0", "\\N"):
            missing += 1
            continue
        if row.get("mapping_verified", "true").lower() != "true":
            unverified += 1
        score = float(row["score"])
        if not np.isfinite(score) or score not in (1, 2, 3, 4, 5):
            raise ValueError("Scores must be integers 1–5; blank/0/\\N denotes missing")
        key = row["participant_id"], row["item_id"], row["system"]
        if not all(key) or key in ratings:
            raise ValueError(f"Empty identifier or duplicate participant/item/system key: {key}")
        ratings[key] = score
    if unverified and not allow_unverified:
        raise ValueError("Unverified legacy mapping. Resolve provenance first, or --allow-unverified for explicitly exploratory output")
    systems = sorted({k[2] for k in ratings})
    if (contrasts is None and reference not in systems) or len(systems) < 2:
        raise ValueError("Reference and at least one comparator required")
    contrasts = contrasts if contrasts is not None else [[reference, c] for c in systems if c != reference]
    if not contrasts or len({tuple(c) for c in contrasts}) != len(contrasts) or any(len(c) != 2 or c[0] == c[1] or not set(c) <= set(systems) for c in contrasts):
        raise ValueError("Invalid planned contrasts or missing system ratings")
    descriptive = []
    for system in systems:
        values = [s for k, s in ratings.items() if k[2] == system]
        person_items = defaultdict(lambda: defaultdict(list))
        for (person, item, sys), value in ratings.items():
            if sys == system:
                base = item.rsplit("::r", 1)[0] if "::r" in item and item.rsplit("::r", 1)[1].isdigit() else item
                person_items[person][base].append(value)
        person_means = [np.mean([np.mean(scores) for scores in items.values()]) for items in person_items.values()]
        mean_ci = paired_summary(person_means, seed)["ci95"] if len(person_means) > 1 else None
        descriptive.append({"system": system, "rating_cells": len(values), "mean": float(np.mean(values)),
                            "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                            "participant_weighted_mean": float(np.mean(person_means)), "participant_mean_ci95": mean_ci,
                            "participants": len({k[0] for k in ratings if k[2] == system}),
                            "items": len({k[1] for k in ratings if k[2] == system})})
    comparisons = []
    for reference, comparator in contrasts:
        a = {(p, i): s for (p, i, sys), s in ratings.items() if sys == reference}
        b = {(p, i): s for (p, i, sys), s in ratings.items() if sys == comparator}
        pairs = sorted(a.keys() & b.keys())
        by_person, by_item, base_pairs = defaultdict(list), defaultdict(list), defaultdict(list)
        for p, i in pairs:
            delta = a[p, i] - b[p, i]
            # Blinding appends replicate ids; they are not independent questions.
            base_item = i.rsplit("::r", 1)[0] if "::r" in i and i.rsplit("::r", 1)[1].isdigit() else i
            base_pairs[p, base_item].append(delta)
        for (p, item), values in base_pairs.items():
            by_person[p].append(float(np.mean(values)))
            by_item[item].append(float(np.mean(values)))
        base_a, base_b = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
        for p, i in pairs:
            base = i.rsplit("::r", 1)[0] if "::r" in i and i.rsplit("::r", 1)[1].isdigit() else i
            base_a[p][base].append(a[p, i])
            base_b[p][base].append(b[p, i])
        matched_a = float(np.mean([np.mean([np.mean(x) for x in v.values()]) for v in base_a.values()])) if pairs else None
        matched_b = float(np.mean([np.mean([np.mean(x) for x in v.values()]) for v in base_b.values()])) if pairs else None
        item = {"reference": reference, "comparator": comparator, "matched_rating_pairs": len(pairs),
                "matched_mean_reference": matched_a, "matched_mean_comparator": matched_b,
                "relative_change_percent": (matched_a - matched_b) / matched_b * 100 if matched_b else None,
                "relative_change_note": "Supplementary arithmetic only; Likert scores are not a ratio scale. Primary effect is rating-point difference.",
                "unmatched_reference": len(a.keys() - b.keys()), "unmatched_comparator": len(b.keys() - a.keys()),
                "primary_unit": "participant mean of matched item differences; equal participant weights",
                "participant_analysis": paired_summary([np.mean(v) for v in by_person.values()], seed) if pairs else None,
                "item_sensitivity": paired_summary([np.mean(v) for v in by_item.values()], seed) if pairs else None,
                "crossed_bootstrap": crossed_bootstrap(base_pairs, seed) if pairs else None}
        comparisons.append(item)
    for analysis in ["participant_analysis", "item_sensitivity"]:
        valid = [c for c in comparisons if c[analysis] and c[analysis]["wilcoxon_p"] is not None]
        for c, adjusted in zip(valid, holm([c[analysis]["wilcoxon_p"] for c in valid])):
            c[analysis]["wilcoxon_p_holm"] = adjusted
        for c, adjusted in zip(valid, holm([c[analysis]["sign_flip_p"] for c in valid])):
            c[analysis]["sign_flip_p_holm"] = adjusted
    out = new_output(out)
    report = {"source_sha256": file_hash(path), "seed": seed, "missing_cells": missing,
              "unverified_cells": unverified, "exploratory_only": bool(unverified),
              "descriptive": descriptive, "comparisons": comparisons, "planned_contrasts": contrasts,
              "limitations": ["Likert differences use a working interval-scale assumption.",
                              "Participant inference is conditional on sampled questions; crossed person/item effects require further modeling.",
                              "Item-level analysis is sensitivity analysis, not an independent replication.",
                              "Wilcoxon and sign-flip assumptions must be assessed; p-values do not validate randomization or data provenance."]}
    write_json(out / "statistics.json", report)
    write_csv(out / "descriptive.csv", descriptive)
    return report


def crossed_bootstrap(base_pairs, seed=42, repeats=2000):
    """Two-way (pigeonhole) resampling of people AND items; sensitivity analysis."""
    people = sorted({p for p, _ in base_pairs})
    items = sorted({i for _, i in base_pairs})
    if min(len(people), len(items)) < 2:
        return {"ci95": None, "reason": "Need at least two people and two independent items"}
    pi, ii = {v: k for k, v in enumerate(people)}, {v: k for k, v in enumerate(items)}
    values = [(pi[p], ii[i], float(np.mean(v))) for (p, i), v in base_pairs.items()]
    rng, estimates = np.random.default_rng(seed), []
    for _ in range(repeats):
        wp = np.bincount(rng.integers(len(people), size=len(people)), minlength=len(people))
        wi = np.bincount(rng.integers(len(items), size=len(items)), minlength=len(items))
        person_sum, person_weight = np.zeros(len(people)), np.zeros(len(people))
        for p, i, value in values:
            person_sum[p] += wi[i] * value
            person_weight[p] += wi[i]
        valid = (person_weight > 0) & (wp > 0)
        if valid.any():
            estimates.append(float(np.average(person_sum[valid] / person_weight[valid], weights=wp[valid])))
    return {"ci95": [float(np.quantile(estimates, .025)), float(np.quantile(estimates, .975))] if estimates else None,
            "requested_repeats": repeats, "valid_repeats": len(estimates), "people": len(people), "items": len(items),
            "method": "two-way bootstrap of participant-weighted item means; sensitivity, not a mixed-effects model"}


def compare_classifiers(a_path, b_path, out, seed=42):
    """Pair fixed-test predictions by ID. Primary effect uses independent groups."""
    from scipy.stats import binomtest
    def load(path):
        with Path(path).open(encoding="utf-8-sig", newline="") as h:
            rows = list(csv.DictReader(h))
        result = {r["id"]: r for r in rows}
        if len(rows) != len(result) or not rows:
            raise ValueError("Unique nonempty prediction IDs required")
        return result
    a, b = load(a_path), load(b_path)
    if a.keys() != b.keys():
        raise ValueError("Predictions must cover exactly the same test IDs")
    groups = defaultdict(list)
    a_only = b_only = 0
    for key in sorted(a):
        x, y = a[key], b[key]
        if x["label"] != y["label"] or x["group_id"] != y["group_id"]:
            raise ValueError("Gold label/group mismatch")
        if any(row[field] not in {"0", "1"} for row in [x, y] for field in ["label", "prediction"]):
            raise ValueError("Binary gold labels and predictions required")
        ac, bc = x["prediction"] == x["label"], y["prediction"] == y["label"]
        a_only += ac and not bc
        b_only += bc and not ac
        groups[x["group_id"]].append(int(ac) - int(bc))
    report = {"a_sha256": file_hash(a_path), "b_sha256": file_hash(b_path), "n_paired_items": len(a),
              "a_correct_b_wrong": a_only, "a_wrong_b_correct": b_only,
              "accuracy_difference_a_minus_b": (a_only - b_only) / len(a),
              "group_analysis": paired_summary([np.mean(v) for v in groups.values()], seed),
              "mcnemar_exact_p": float(binomtest(a_only, a_only + b_only, .5).pvalue) if a_only + b_only else 1.,
              "note": "McNemar assumes independent items; grouped difference is primary when items share a source. No correction across separately invoked comparisons; predefine and correct the comparison family."}
    out = new_output(out)
    write_json(out / "comparison.json", report)
    return report
