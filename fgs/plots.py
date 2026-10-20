"""Standard Matplotlib research plots from saved, measured outputs."""
import csv
from pathlib import Path
import numpy as np
from .io import new_output, read_json


def plot_run(run, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    run, out = Path(run), new_output(out)
    report = read_json(run / "metrics.json")
    split = next(k for k in ["test", "external", "validation"] if k in report["metrics"])
    matrix = report["metrics"][split]["confusion_matrix"]
    values = np.array([[matrix["tn"], matrix["fp"]], [matrix["fn"], matrix["tp"]]])
    fig, ax = plt.subplots(figsize=(4.8, 4.3), layout="constrained")
    ax.imshow(values, cmap="Blues")
    for (i, j), value in np.ndenumerate(values):
        ax.text(j, i, str(value), ha="center", va="center", color="white" if value > values.max() / 2 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Off-topic", "Relevant"], yticklabels=["Off-topic", "Relevant"],
           xlabel="Predicted class", ylabel="Gold class", title=f"{report['kind']} | {split} | seed {report['seed']}")
    fig.savefig(out / "confusion_matrix.png", dpi=300)
    fig.savefig(out / "confusion_matrix.pdf")
    plt.close(fig)
    for path in sorted(run.glob("training_curve_trial*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as h:
            rows = list(csv.DictReader(h))
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
        epochs = [int(r["epoch"]) for r in rows]
        for ax, metric in zip(axes, ["loss", "accuracy"]):
            for split in ["train", "validation"]:
                ax.plot(epochs, [float(r[f"{split}_{metric}"]) for r in rows], label=split)
            ax.set(xlabel="Epoch", ylabel=metric.capitalize())
            ax.legend()
        fig.savefig(out / f"{path.stem}.png", dpi=300)
        fig.savefig(out / f"{path.stem}.pdf")
        plt.close(fig)
    return {"output": str(out), "files": [p.name for p in sorted(out.iterdir())]}
