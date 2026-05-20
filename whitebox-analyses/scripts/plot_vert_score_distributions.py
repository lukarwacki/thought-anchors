import os
import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attention_analysis.receiver_head_funcs import (
    get_all_problems_vert_scores,
    get_3d_ar_skewness,
    get_3d_ar_variance,
)


def plot_global_histogram(all_scores_flat, output_dir, model_name, proximity_ignore, dpi, bins):
    fig = plt.figure(figsize=(5, 3.5))
    plt.hist(all_scores_flat, bins=bins, color="dodgerblue")
    plt.title("Empirical distribution of vertical attention scores", fontsize=11, pad=8)
    plt.xlabel("Vertical score", labelpad=7)
    plt.ylabel("Count", labelpad=7)
    plt.gca().spines[["top", "right"]].set_visible(False)
    plt.subplots_adjust(bottom=0.15, top=0.88, left=0.13, right=0.97)
    suffix = f"pi{proximity_ignore}"
    fp_out = output_dir / f"global_hist_{model_name}_{suffix}.png"
    plt.savefig(fp_out, dpi=dpi)
    plt.close()
    print(f"Saved {fp_out}")


def plot_stat_heatmap(mean_matrix, stat_name, cmap, output_dir, model_name, proximity_ignore, dpi):
    n_layers, n_heads = mean_matrix.shape
    fig, ax = plt.subplots(figsize=(max(6, n_heads * 0.22), max(4, n_layers * 0.18)))
    sns.heatmap(
        mean_matrix,
        ax=ax,
        cmap=cmap,
        center=0 if cmap == "RdBu_r" else None,
        xticklabels=5,
        yticklabels=5,
    )
    ax.set_xlabel("Head", labelpad=7)
    ax.set_ylabel("Layer", labelpad=7)
    ax.set_title(f"Mean {stat_name} per head (layer × head)", fontsize=11, pad=8)
    plt.tight_layout()
    suffix = f"pi{proximity_ignore}"
    fp_out = output_dir / f"{stat_name}_heatmap_{model_name}_{suffix}.png"
    plt.savefig(fp_out, dpi=dpi)
    plt.close()
    print(f"Saved {fp_out}")


def plot_kde_overlay(resp_layer_head_verts, response_idxs, output_dir, model_name, proximity_ignore, dpi):
    fig = plt.figure(figsize=(6, 4))
    ax = fig.gca()

    colors = {1: "dodgerblue", 0: "tomato"}
    labels_added = set()

    for i, (problem_id, is_correct) in enumerate(response_idxs):
        flat = resp_layer_head_verts[i].flatten()
        flat = flat[~np.isnan(flat)]
        label = ("Correct" if is_correct else "Incorrect") if is_correct not in labels_added else None
        sns.kdeplot(flat, ax=ax, color=colors[is_correct], alpha=0.4, linewidth=1, label=label)
        labels_added.add(is_correct)

    ax.set_xlabel("Vertical score", labelpad=7)
    ax.set_ylabel("Density", labelpad=7)
    ax.set_title("Per-problem vertical score distributions", fontsize=11, pad=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    plt.subplots_adjust(bottom=0.13, top=0.88, left=0.12, right=0.97)
    suffix = f"pi{proximity_ignore}"
    fp_out = output_dir / f"kde_overlay_{model_name}_{suffix}.png"
    plt.savefig(fp_out, dpi=dpi)
    plt.close()
    print(f"Saved {fp_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnostic plots for vertical attention score distributions")
    parser.add_argument("--model-name", type=str, default="qwen-14b")
    parser.add_argument("--proximity-ignore", type=int, default=4)
    parser.add_argument("--control-depth", action="store_true")
    parser.add_argument("--output-dir", type=str, default="plots/vert_score_distributions")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--bins", type=int, default=100)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resp_layer_head_verts, response_idxs = get_all_problems_vert_scores(
        model_name=args.model_name,
        proximity_ignore=args.proximity_ignore,
        control_depth=args.control_depth,
    )

    # Figure 1: global histogram
    all_scores_flat = np.concatenate([v.flatten() for v in resp_layer_head_verts])
    all_scores_flat = all_scores_flat[~np.isnan(all_scores_flat)]
    plot_global_histogram(
        all_scores_flat, output_dir, args.model_name, args.proximity_ignore, args.dpi, args.bins
    )

    # Figures 2a & 2b: heatmaps of mean skewness and mean variance
    skew_matrices = np.array([get_3d_ar_skewness(v) for v in resp_layer_head_verts])
    var_matrices = np.array([get_3d_ar_variance(v) for v in resp_layer_head_verts])

    mean_skew = np.nanmean(skew_matrices, axis=0)
    mean_var = np.nanmean(var_matrices, axis=0)

    # Zero out layer 0 (no interesting attention)
    mean_skew[0, :] = np.nan
    mean_var[0, :] = np.nan

    plot_stat_heatmap(
        mean_skew, "skewness", "RdBu_r", output_dir, args.model_name, args.proximity_ignore, args.dpi
    )
    plot_stat_heatmap(
        mean_var, "variance", "viridis", output_dir, args.model_name, args.proximity_ignore, args.dpi
    )

    # Figure 3: per-problem KDE overlay
    plot_kde_overlay(
        resp_layer_head_verts, response_idxs, output_dir, args.model_name, args.proximity_ignore, args.dpi
    )
