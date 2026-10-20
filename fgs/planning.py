"""Prospective power sensitivity, not a post-hoc observed-power claim."""
import numpy as np
from scipy.stats import wilcoxon
from .io import new_output, write_csv, write_json


def simulate_power(config, out):
    ns = config.get("participants", [20, 30, 40, 60])
    items = int(config.get("items_per_participant", 20))
    repeats = int(config.get("simulations", 1000))
    alpha = float(config.get("comparison_alpha", .025))
    effect = float(config.get("latent_difference", .3))
    subject_sd = float(config.get("participant_difference_sd", .4))
    item_sd = float(config.get("item_difference_sd", .3))
    noise_sd = float(config.get("rating_noise_sd", .8))
    seed = int(config.get("seed", 42))
    if not ns or min(ns) < 3 or items < 2 or repeats < 20 or not 0 < alpha < 1 or min(subject_sd, item_sd, noise_sd) < 0:
        raise ValueError("Invalid planning assumptions")
    rng, rows = np.random.default_rng(seed), []
    for n in ns:
        reject, observed = 0, []
        for _ in range(repeats):
            baseline = 3.0 + rng.normal(0, .5, (n, 1)) + rng.normal(0, .4, (1, items))
            difference = effect + rng.normal(0, subject_sd, (n, 1)) + rng.normal(0, item_sd, (1, items))
            a = np.clip(np.rint(baseline + rng.normal(0, noise_sd, (n, items))), 1, 5)
            b = np.clip(np.rint(baseline + difference + rng.normal(0, noise_sd, (n, items))), 1, 5)
            d = (b - a).mean(axis=1)
            p = wilcoxon(d, alternative="two-sided", zero_method="wilcox", method="auto").pvalue if np.any(d) else 1.
            reject += p < alpha
            observed.append(float(d.mean()))
        power = reject / repeats
        rows.append({"participants": n, "items_per_participant": items, "simulations": repeats,
                     "estimated_power": power, "monte_carlo_se": float(np.sqrt(power * (1-power) / repeats)),
                     "mean_observed_rating_difference": float(np.mean(observed))})
    out = new_output(out)
    write_csv(out / "power_sensitivity.csv", rows)
    write_json(out / "assumptions.json", {**config, "seed": seed,
               "purpose": "Prospective sensitivity under stated latent-score assumptions; not real study data or a recruitment guarantee",
               "design": "All participants rate the same items in both conditions; independent random person/item differences and rating noise; round and clip to 1–5"})
    return rows
