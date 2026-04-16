import json
import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = "binary_suffix_experiment/sweep_v2"
FIG_DIR = "figures/sweep_v2"

BIT_COLORS = ["#d62728", "#1f77b4", "#9467bd", "#e8c838", "#e377c2",
              "#2ca02c", "#ff7f0e", "#17becf"]


def nu_2(n):
    if n == 0:
        return float("inf")
    v = 0
    while n % 2 == 0:
        v += 1
        n //= 2
    return v


def load_result(base, p):
    path = os.path.join(DATA_DIR, f"B{base}_p{p}_results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def compute_per_bit_acc(confusion_matrices, p):
    """Compute true per-bit accuracy from confusion matrices.

    The stored bit_accs is cumulative (bits 0..k all correct), not per-bit.
    This computes the correct per-bit metric: for each bit independently,
    what fraction of predictions got that bit right.
    """
    K = 2 ** p
    n_epochs = len(confusion_matrices)
    accs = np.zeros((p, n_epochs))
    for t, cm_list in enumerate(confusion_matrices):
        cm = np.array(cm_list)
        total = cm.sum()
        if total == 0:
            continue
        for bit_idx in range(p):
            correct = 0
            for true_cls in range(K):
                true_bit = (true_cls >> bit_idx) & 1
                for pred_cls in range(K):
                    pred_bit = (pred_cls >> bit_idx) & 1
                    if pred_bit == true_bit:
                        correct += cm[true_cls, pred_cls]
            accs[bit_idx, t] = correct / total
    return accs


def plot_bits(result, out_path):
    """Per-bit accuracy plot with corrected labels."""
    B = result["base"]
    p = result["p"]
    hist = result["history"]
    epochs = np.array(hist["epoch"]) + 1
    test_acc = np.array(hist["test_acc"])
    confusion_matrices = hist.get("confusion", [])

    if not confusion_matrices:
        return

    n_bits = p
    bit_accs = compute_per_bit_acc(confusion_matrices, p)

    fig, ax = plt.subplots(figsize=(14, 4.5))

    for b_idx in range(n_bits):
        color = BIT_COLORS[b_idx % len(BIT_COLORS)]
        # Shift label: bit 0 -> "bit 1", bit 1 -> "bit 2", etc.
        ax.plot(epochs, bit_accs[b_idx], color=color, lw=2.5, label=f"bit {b_idx + 1}")

    ax.plot(epochs, test_acc, color="gray", ls="--", lw=1.5, alpha=0.7,
            label="overall test acc")

    ax.axhline(0.5, color="gray", ls=":", lw=1, alpha=0.4)
    ax.set_xlabel("epoch", fontsize=12)
    ax.set_ylabel("accuracy", fontsize=12)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(epochs[0], epochs[-1])
    ax.set_title(f"B={B}, p={p} — Per-bit accuracy from confusion matrices",
                 fontsize=13)
    ax.legend(loc="right", fontsize=10, framealpha=0.9)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_heatmap(out_path):
    """Accuracy heatmap with cleaned-up title."""
    bases = [8, 24, 4, 12, 28, 2, 10, 3, 5]
    p_values = [1, 2, 3, 4, 5]

    grid_acc = np.full((len(bases), len(p_values)), np.nan)
    grid_ep = np.full((len(bases), len(p_values)), np.nan)

    for i, B in enumerate(bases):
        for j, p in enumerate(p_values):
            r = load_result(B, p)
            if r is None:
                continue
            grid_acc[i, j] = r.get("best_acc", r.get("final_acc", 0))
            grid_ep[i, j] = r.get("epochs_completed", 0)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(grid_acc, cmap="RdYlGn", vmin=0, vmax=1,
                   aspect="auto", interpolation="nearest")

    ax.set_xticks(range(len(p_values)))
    ax.set_xticklabels([str(p) for p in p_values], fontsize=11)
    ax.set_yticks(range(len(bases)))
    ylabels = [f"B={B} (\u03bd\u2082={nu_2(B)})" for B in bases]
    ax.set_yticklabels(ylabels, fontsize=11)

    ax.set_xlabel("p", fontsize=13)
    ax.set_ylabel("Base B", fontsize=13)

    for i in range(len(bases)):
        for j in range(len(p_values)):
            if np.isnan(grid_acc[i, j]):
                continue
            acc = grid_acc[i, j]
            ep = int(grid_ep[i, j])
            color = "white" if acc < 0.5 else "black"
            ax.text(j, i, f"{acc:.2f}\n({ep}ep)",
                    ha="center", va="center", fontsize=9, color=color,
                    fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Best accuracy", fontsize=11)
    ax.set_title("Accuracy heatmap", fontsize=14)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    # Regenerate all bit accuracy plots
    for f in sorted(glob(os.path.join(DATA_DIR, "B*_results.json"))):
        r = json.load(open(f))
        B, p = r["base"], r["p"]
        if "confusion" not in r.get("history", {}):
            continue
        out = os.path.join(FIG_DIR, f"B{B}_p{p}_bits.png")
        plot_bits(r, out)

    # Regenerate heatmap
    plot_heatmap(os.path.join(FIG_DIR, "heatmap.png"))


if __name__ == "__main__":
    main()
