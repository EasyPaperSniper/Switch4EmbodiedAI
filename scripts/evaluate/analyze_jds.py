import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# === DATASET (from your sheet) ===
from data.jds_scores import data as jds_scores


# === Loop through data folder and load all metrics similar to batch_compare_all ===
from scripts.evaluate.batch_compare_all import discover_all_reference_songs

reference_songs = discover_all_reference_songs()
available_ref_songs = discover_all_reference_songs(reference_base)
print(f"Available reference songs: {available_ref_songs}")


# === Helper function to get DataFrame for a song & person ===
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# --- UPDATED helper: select rows by run type and/or exact condition names ---
def get_song_df(
    data, song, metric="DTW", name="Jeonghwan",
    include_run_types=None,   # e.g., {"normal"} or {"normal", "exaggeration", "upperbody"}
    include_conditions=None   # e.g., {"normal_1", "normal_2"}
):
    """
    include_run_types filters by the prefix before '_' (e.g., 'normal' in 'normal_1').
    include_conditions filters by exact condition names.
    If either is None, that filter is not applied.
    Both can be used together (intersection).
    """
    rows = []
    for cond, metrics in data[song].items():
        run_type = cond.split("_", 1)[0]  # 'normal' from 'normal_1'
        if include_run_types is not None and run_type not in include_run_types:
            continue
        if include_conditions is not None and cond not in include_conditions:
            continue

        jds = metrics.get("JDS", {}).get(name)
        val = metrics.get(metric, {}).get(name)
        if jds is not None and val is not None:
            rows.append((cond, jds, val))

    return pd.DataFrame(rows, columns=["Condition", "JDS", metric])


# --- UPDATED plotting: save to file and accept filters ---
import os, re
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

import os, re
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def plot_correlation_merged(
    data,
    song,
    metric="DTW",
    names_to_include=("Jeonghwan", "Wontaek"),   # ← choose who to merge
    include_run_types=None,                      # e.g., {"normal"} or {"normal","upperbody"}
    include_conditions=None,                     # e.g., {"normal_1","normal_2"}
    save_dir="plots",
    annotate=True
):
    """
    Merge data across the specified names and compute/plot correlation for JDS vs metric.
    """
    os.makedirs(save_dir, exist_ok=True)

    all_jds, all_vals, all_labels = [], [], []

    for cond, metrics in data[song].items():
        run_type = cond.split("_", 1)[0]
        if include_run_types and run_type not in include_run_types:
            continue
        if include_conditions and cond not in include_conditions:
            continue

        for name in names_to_include:
            jds = metrics.get("JDS", {}).get(name)
            val = metrics.get(metric, {}).get(name)
            if jds is not None and val is not None:
                all_jds.append(jds)
                all_vals.append(val)
                all_labels.append(f"{cond} ({name})")

    if len(all_jds) < 2:
        print(f"[skip] Not enough data points for {song} / {metric} with names={names_to_include}")
        return

    # correlation
    # r: Pearson correlation coefficient (range -1 to 1)
    #    - Measures linear relationship strength and direction between JDS and the metric.
    #      r > 0 → as JDS increases, metric tends to increase.
    #      r < 0 → as JDS increases, metric tends to decrease.
    # p: two-tailed p-value testing the null hypothesis (r = 0).
    #    - Small p (< 0.05) → statistically significant correlation.
    #    - Large p → no significant evidence of linear relationship.
    r, p = pearsonr(all_jds, all_vals)

    # plot
    plt.figure(figsize=(6, 5))
    plt.scatter(all_jds, all_vals, label=f"merged r={r:.2f}, p={p:.3f}")

    # regression line
    m, b = np.polyfit(all_jds, all_vals, 1)
    xs = np.linspace(min(all_jds), max(all_jds), 100)
    plt.plot(xs, m * xs + b, linestyle="--", alpha=0.8)

    # annotations
    if annotate:
        for j, v, lbl in zip(all_jds, all_vals, all_labels):
            plt.annotate(lbl, (j, v), xytext=(5, 5), textcoords="offset points", fontsize=8)

    # title / labels
    title_parts = [song, f"JDS vs {metric}", f"names={tuple(names_to_include)}"]
    if include_run_types:
        title_parts.append(f"experiment types={sorted(include_run_types)}")
    if include_conditions:
        title_parts.append("subset conditions")
    plt.title(" | ".join(title_parts))
    plt.xlabel("JDS")
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    # === CAPTION: explanation for r and p values ===
    caption_text = (
        "r: Pearson correlation coefficient (−1 ≤ r ≤ 1)\n"
        "  • r > 0 → metric increases with JDS (positive correlation)\n"
        "  • r < 0 → metric decreases with JDS (negative correlation)\n"
        "p: two-tailed p-value testing H₀: r = 0\n"
        "  • p < 0.05 → statistically significant correlation"
    )
    plt.figtext(
        0.5, -0.05, caption_text,
        ha="center", va="top", fontsize=8, wrap=True
    )

    # filename
    def safe(s): return re.sub(r"[^A-Za-z0-9_]+", "_", s)
    runpart = ""
    if include_run_types:
        runpart += "_types_" + "_".join(sorted(include_run_types))
    if include_conditions:
        runpart += "_conds_" + "_".join(sorted(safe(c) for c in include_conditions))
    namepart = "_names_" + "_".join(safe(n) for n in names_to_include)

    filename = f"{safe(song)}_{metric.replace('-', '')}_merged{namepart}{runpart}.png"
    filepath = os.path.join(save_dir, filename)

    # adjust layout for caption space
    plt.subplots_adjust(bottom=0.25)

    plt.savefig(filepath, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved merged plot with caption: {filepath}")


# === Plot examples ===
plot_correlation_merged(data, "Old_Town_Road", "DTW", names_to_include=("Jeonghwan", "Wontaek"), include_run_types={"normal", "exaggeration", "controllerarm"})
plot_correlation_merged(data, "Old_Town_Road", "DTW", names_to_include=("Jeonghwan", "Wontaek"), include_run_types={"normal", "exaggeration", "controllerarm", "upperbody"})
plot_correlation_merged(data, "Old_Town_Road", "PA-MPJPE", names_to_include=("Jeonghwan", "Wontaek"), include_run_types={"normal", "exaggeration", "controllerarm"})
plot_correlation_merged(data, "Old_Town_Road", "PA-MPJPE", names_to_include=("Jeonghwan", "Wontaek"), include_run_types={"normal", "exaggeration", "controllerarm", "upperbody"})
plot_correlation_merged(data, "Old_Town_Road", "MPJPE", names_to_include=("Jeonghwan", "Wontaek"), include_run_types={"normal", "exaggeration", "controllerarm"})
plot_correlation_merged(data, "Old_Town_Road", "MPJPE", names_to_include=("Jeonghwan", "Wontaek"), include_run_types={"normal", "exaggeration", "controllerarm", "upperbody"})
plot_correlation_merged(data, "Unstoppable", "DTW", names_to_include=("Jeonghwan", "Wontaek"), include_run_types={"normal", "controllerarm"})
plot_correlation_merged(data, "Unstoppable", "DTW", names_to_include=("Jeonghwan", "Wontaek"), include_run_types={"normal", "controllerarm", "upperbody"})
plot_correlation_merged(data, "Unstoppable", "PA-MPJPE", names_to_include=("Jeonghwan", "Wontaek"), include_run_types={"normal", "controllerarm"})
plot_correlation_merged(data, "Unstoppable", "PA-MPJPE", names_to_include=("Jeonghwan", "Wontaek"), include_run_types={"normal", "controllerarm", "upperbody"})
plot_correlation_merged(data, "Unstoppable", "MPJPE", names_to_include=("Jeonghwan", "Wontaek"), include_run_types={"normal", "controllerarm"})
plot_correlation_merged(data, "Unstoppable", "MPJPE", names_to_include=("Jeonghwan", "Wontaek"), include_run_types={"normal", "controllerarm", "upperbody"})
