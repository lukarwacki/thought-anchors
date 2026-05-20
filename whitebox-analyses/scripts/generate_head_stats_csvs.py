import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attention_analysis.receiver_head_funcs import (
    get_all_problems_vert_scores,
    get_3d_ar_kurtosis,
    get_3d_ar_skewness,
    get_3d_ar_variance,
)


def build_head_stats_df(model_name, proximity_ignore, control_depth):
    resp_layer_head_verts, response_idxs = get_all_problems_vert_scores(
        model_name=model_name,
        proximity_ignore=proximity_ignore,
        control_depth=control_depth,
    )

    n_layers, n_heads = resp_layer_head_verts[0].shape[:2]
    layers = np.arange(n_layers)
    heads = np.arange(n_heads)

    rows = []
    for i, (problem_id, is_correct) in enumerate(response_idxs):
        verts = resp_layer_head_verts[i]
        kurt = get_3d_ar_kurtosis(verts)
        skew = get_3d_ar_skewness(verts)
        var = get_3d_ar_variance(verts)

        for layer in layers:
            for head in heads:
                rows.append({
                    "problem_id": problem_id,
                    "is_correct": is_correct,
                    "layer": layer,
                    "head": head,
                    "kurtosis": kurt[layer, head],
                    "skewness": skew[layer, head],
                    "variance": var[layer, head],
                })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate per-head stat CSVs for GAMLSS")
    parser.add_argument("--model-name", type=str, default="qwen-14b")
    parser.add_argument("--proximity-ignore", type=int, default=4)
    parser.add_argument("--control-depth", action="store_true")
    parser.add_argument("--output-dir", type=str, default="csvs")
    args = parser.parse_args()

    df = build_head_stats_df(args.model_name, args.proximity_ignore, args.control_depth)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"pi{args.proximity_ignore}" + ("_cd" if args.control_depth else "")
    fp_out = output_dir / f"head_stats_{args.model_name}_{suffix}.csv"
    df.to_csv(fp_out, index=False)
    print(f"Saved {len(df)} rows to {fp_out}")
