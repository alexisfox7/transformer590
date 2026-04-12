"""Plot stepwise / progressive grokking from saved results JSONs.

Usage:
    python analyze.py                       # auto-pick interesting configs
    python analyze.py --config 5 2          # plot just (B=5, p=2)
    python analyze.py --all                 # plot every config in results dir
"""
import argparse
import json
import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = "binary_suffix_experiment"
FIG_DIR = "figures"


def load_result(base, p, results_dir=RESULTS_DIR):
    path = os.path.join(results_dir, f"B{base}_p{p}_results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_all(results_dir=RESULTS_DIR):
    out = {}
    for f in sorted(glob(os.path.join(results_dir, "B*_results.json"))):
        name = os.path.basename(f)
        if any(v in name for v in ["mod", "curriculum", "sinusoidal", "rope", "alibi"]):
            continue
        r = json.load(open(f))
        if "p" not in r:
            continue
        out[(r["base"], r["p"])] = r
    return out


def class_acc_matrix(result):
    """Return (num_classes, n_epochs) array of per-class accuracies over training."""
    K = result["num_classes"]
    hist = result["history"]["class_acc"]
    n_epochs = len(hist)
    M = np.full((K, n_epochs), np.nan)
    for t, entry in enumerate(hist):
        for k_str, acc in entry.items():
            M[int(k_str), t] = acc
    return M


def plot_grokking(result, out_path, show_bits=True, log_x=True):
    """Three-panel figure: test acc curve, per-class heatmap, per-class lines."""
    B = result["base"]
    p = result["p"]
    K = result["num_classes"]
    hist = result["history"]
    epochs = np.array(hist["epoch"]) + 1
    train_acc = np.array(hist["train_acc"])
    test_acc = np.array(hist["test_acc"])
    train_loss = np.array(hist["train_loss"])

    M = class_acc_matrix(result)  # K x T

    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1.3, 1.3])

    x_label = "epoch (log scale)" if log_x else "epoch"

    # ── Row 1: overall curves ────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(epochs, train_acc, color="#4a90e2", lw=2, label="train acc")
    ax1.plot(epochs, test_acc, color="#d9534f", lw=2, label="test acc")
    ax1.axhline(1.0 / K, color="gray", ls="--", lw=1, alpha=0.6, label=f"chance (1/{K})")
    ax1.axhline(0.90, color="green", ls=":", lw=1, alpha=0.5, label="success (0.90)")
    ax1.set_ylabel("accuracy")
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_xlim(epochs[0], epochs[-1])
    if log_x:
        ax1.set_xscale("log")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="lower right", fontsize=9)
    ax1.set_title(
        f"B={B}, p={p}  (predicting n mod {2**p},  ν₂(B)={nu_2(B)},  best={result.get('best_acc', 0):.4f},  epochs={result.get('epochs_completed', len(epochs))})",
        fontsize=12,
    )

    # ── Row 2: per-class heatmap ─────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    im = ax2.imshow(
        M,
        aspect="auto",
        origin="lower",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        interpolation="nearest",
        extent=[epochs[0] - 0.5, epochs[-1] + 0.5, -0.5, K - 0.5],
    )
    if log_x:
        ax2.set_xscale("log")
    ax2.set_ylabel("class index (n mod 2^p)")
    ax2.set_xlabel("")
    cbar = fig.colorbar(im, ax=ax2, shrink=0.85, pad=0.01)
    cbar.set_label("per-class accuracy")
    ax2.set_yticks(range(K))
    if K <= 16:
        ax2.set_yticklabels([str(k) for k in range(K)])
    else:
        step = max(1, K // 16)
        ax2.set_yticks(range(0, K, step))
        ax2.set_yticklabels([str(k) for k in range(0, K, step)])

    # ── Row 3: per-class lines, colored by low-bit identity if small p ──
    ax3 = fig.add_subplot(gs[2])
    if show_bits and p <= 3:
        cmap_even = plt.cm.Blues
        cmap_odd = plt.cm.Oranges
        even_classes = [k for k in range(K) if k % 2 == 0]
        odd_classes = [k for k in range(K) if k % 2 == 1]
        for i, k in enumerate(even_classes):
            color = cmap_even(0.4 + 0.5 * i / max(1, len(even_classes) - 1))
            ax3.plot(epochs, M[k], color=color, lw=1.5, label=f"class {k} (even)")
        for i, k in enumerate(odd_classes):
            color = cmap_odd(0.4 + 0.5 * i / max(1, len(odd_classes) - 1))
            ax3.plot(epochs, M[k], color=color, lw=1.5, label=f"class {k} (odd)")
    else:
        cmap = plt.cm.viridis
        for k in range(K):
            ax3.plot(epochs, M[k], color=cmap(k / max(1, K - 1)), lw=1, alpha=0.7)
    ax3.axhline(1.0 / K, color="gray", ls="--", lw=1, alpha=0.5)
    if log_x:
        ax3.set_xscale("log")
    ax3.set_ylim(-0.02, 1.02)
    ax3.set_xlim(epochs[0], epochs[-1])
    ax3.set_xlabel(x_label)
    ax3.set_ylabel("per-class accuracy")
    ax3.grid(alpha=0.3)
    if K <= 8:
        ax3.legend(loc="lower right", fontsize=8, ncol=2)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def nu_2(n):
    if n == 0:
        return float("inf")
    v = 0
    while n % 2 == 0:
        v += 1
        n //= 2
    return v


def pick_grokky(results, top_k=6):
    """Return configs ranked by how stepwise their dynamics look.
    Heuristic: long training runs that aren't instant wins."""
    scored = []
    for (B, p), r in results.items():
        ep = r.get("epochs_completed", 0)
        if ep < 8:  # trivial, skip
            continue
        scored.append((ep, B, p))
    scored.sort(reverse=True)
    return [(B, p) for _, B, p in scored[:top_k]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs=2, type=int, metavar=("BASE", "P"),
                    help="Plot just this single (B, p) config")
    ap.add_argument("--all", action="store_true", help="Plot every config in results dir")
    ap.add_argument("--linear", action="store_true", help="Use linear x-axis instead of log")
    ap.add_argument("--results_dir", default=RESULTS_DIR)
    ap.add_argument("--fig_dir", default=FIG_DIR)
    args = ap.parse_args()

    os.makedirs(args.fig_dir, exist_ok=True)
    suffix = "_linear" if args.linear else ""
    use_log = not args.linear

    if args.config:
        B, p = args.config
        r = load_result(B, p, args.results_dir)
        if r is None:
            raise SystemExit(f"no result for B={B}, p={p}")
        plot_grokking(r, os.path.join(args.fig_dir, f"grokking_B{B}_p{p}{suffix}.png"), log_x=use_log)
        return

    all_results = load_all(args.results_dir)
    print(f"loaded {len(all_results)} results")

    if args.all:
        targets = list(all_results.keys())
    else:
        targets = pick_grokky(all_results, top_k=8)
        print(f"plotting top {len(targets)} grokkiest configs: {targets}")

    for B, p in targets:
        r = all_results.get((B, p))
        if r is None:
            continue
        plot_grokking(r, os.path.join(args.fig_dir, f"grokking_B{B}_p{p}{suffix}.png"), log_x=use_log)


if __name__ == "__main__":
    main()
