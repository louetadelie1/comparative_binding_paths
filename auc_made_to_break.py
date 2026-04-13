import seaborn as sns
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from scipy.stats import spearmanr,pearsonr
import math
import pickle
import itertools
import pickle
import itertools
import math
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import smplotlib
from matplotlib.colors import LinearSegmentedColormap, Normalize

def cleanup(d):
    return {tuple(int(k) for k in key): float(value[0]) for key, value in d.items()}


def enrichment_factor(source_rank, target_rank, k):
    actives = source_rank[:k]
    topk_target = target_rank[:k]
    H = k
    N = len(target_rank)
    hits = len(set(actives) & set(topk_target))
    return (hits / k) / (H / N)

paths = [[
    "/Users/adelielouet/Documents/science/dd_proj/mass_producing_idp_simulations/protein_simualtions/output_post_processed/msm_output/ZINC000000030986/p_eq_keys_trip.pckl",
    "/Users/adelielouet/Documents/science/dd_proj/mass_producing_idp_simulations/protein_simualtions/output_post_processed/msm_output/ZINC000000038519/p_eq_keys_trip.pckl",
    "/Users/adelielouet/Documents/science/dd_proj/mass_producing_idp_simulations/protein_simualtions/output_post_processed/msm_output/ZINC000000052225/p_eq_keys_trip.pckl",
    "/Users/adelielouet/Documents/science/dd_proj/mass_producing_idp_simulations/protein_simualtions/output_post_processed/msm_output/ZINC000000057966/p_eq_keys_trip.pckl",
    "/Users/adelielouet/Documents/science/dd_proj/mass_producing_idp_simulations/protein_simualtions/output_post_processed/msm_output/abeta_gabis/p_eq_keys_trip.pckl",
]]

label_map = {
    "ZINC000000030986": "ZINC000000030986",
    "ZINC000000038519": "ZINC000000038519",
    "ZINC000000052225":  "ZINC000000052225",
    "ZINC000000057966": "ZINC000000057966",
    "abeta_gabis": "G5",
}

title='ABETA no restarts - 10us'

def get_cmap(inp):
    if "_ph" in inp.split('/')[-3]: 
        return LinearSegmentedColormap.from_list(
            "abeta_blue", ["#edf0fa", "#2f3f7f"]
        )
    elif "medin" in inp.split('/')[-3]:
        return LinearSegmentedColormap.from_list(
            "medin_green", ["#edf8f2", "#5a8f7b"]
        )
    elif "alpha_syn" in inp.split('/')[-3]:
        return LinearSegmentedColormap.from_list(
            "alpha_purple", ["#e0d4f7", "#5e3c99"]
        )
    else:
        return LinearSegmentedColormap.from_list(
            "abeta_blue_2", ["#eef4fa", "#355f8c"])
    

for system in paths:
# palette = sns.color_palette(["#edf0fa", "#2f3f7f"])
    matrices = []
    for path in system:
        with open(path, "rb") as f:
            data = pickle.load(f)
        cleaned = {
            tuple(int(k) for k in key): float(val[0])
            for key, val in data.items()
        }

        cleaned_sorted = dict(
            sorted(cleaned.items(), key=lambda x: x[1], reverse=True)
        )

        matrices.append(cleaned_sorted)

    ranked_lists = [list(m.keys()) for m in matrices]
    #pairs = list(itertools.permutations(range(len(ranked_lists)), 2))
    pairs = list(itertools.combinations(range(len(ranked_lists)), 2))

    # ==================================================
    # PART 1 — ENRICHMENT FACTOR BAR PLOTS (ONE FIGURE)
    # ==================================================
    # cutoffs = [20, 50, 100, 200, 400]

    # cols = 2
    # rows = math.ceil(len(pairs) / cols)
    # fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows))
    # axes = axes.flatten()

    n_panels = len(pairs)
    ncols = 3
    nrows = math.ceil(n_panels / ncols)

    # fig, axes = plt.subplots(
    #     nrows, ncols,
    #     #figsize=(4.2 * ncols, 3.2 * nrows),
    #     figsize=(8 * ncols, 5 * nrows),

    #     sharey=True
    # )
    # axes = np.array(axes).reshape(-1)
    # width = 0.35

    # # for ax, lig in zip(axes, pairs):

    # for idx, (i, j) in enumerate(pairs):

    #     ranked_A = ranked_lists[i]
    #     ranked_B = ranked_lists[j]

    #     ef_AtoB = []
    #     ef_BtoA = []

    #     for k in cutoffs:
    #         actives_A = ranked_A[:k]
    #         actives_B = ranked_B[:k]
    #         N = len(ranked_B)

    #         hits_AB = len(set(actives_A) & set(actives_B))
    #         ef_AtoB.append((hits_AB / k) / (k / N))

    #         hits_BA = len(set(actives_B) & set(actives_A))
    #         ef_BtoA.append((hits_BA / k) / (k / N))

    #     ax = axes[idx]
    #     x = np.arange(len(cutoffs))
    #     system_names = [p.split('/')[-2] for p in system]

    #     ax.bar(
    #         x - width / 2,
    #         ef_AtoB,
    #         width,
    #         color=get_cmap(path)(0.5),
    #         label=f"{label_map.get(system_names[i], system_names[i])} → "
    #             f"{label_map.get(system_names[j], system_names[j])}"
    #     )

    #     ax.bar(
    #         x + width / 2,
    #         ef_BtoA,
    #         width,
    #         color=get_cmap(path)(0.75),
    #         label=f"{label_map.get(system_names[j], system_names[j])} → "
    #             f"{label_map.get(system_names[i], system_names[i])}"
    #     )

    #     ax.set_xticks(x)
    #     ax.tick_params(axis="x", labelsize=15)
    #     ax.tick_params(axis="y", labelsize=15)
    #     ax.set_xticklabels(cutoffs)
    #     ax.set_ylabel("Enrichment Factor")
    #     ax.set_xlabel("Top-k cutoff")
    #     ax.set_title(
    #         f"{label_map.get(system_names[i], system_names[i])} vs "
    #         f"{label_map.get(system_names[j], system_names[j])}",fontsize=20
    #     )
    #     ax.legend(fontsize=15)
    #     ax.grid(alpha=0.3)

    #     if idx % ncols != 0:   # not first column
    #         ax.set_ylabel("")
    #         ax.tick_params(labelleft=False)

    # for ax in axes[len(pairs):]:
    #     ax.axis("off")

    # plt.suptitle("Pairwise Enrichment Factors (All Systems)", fontsize=16)
    # plt.tight_layout()
    # plt.show()

    # ==================================================
    # PART 2 — AUC ENRICHMENT CURVES (ONE FIGURE, FIXED)
    # ==================================================
    TOP_ACTIVES = 10

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(8 * ncols, 5 * nrows),
        sharey=True
    )
    axes = np.array(axes).reshape(-1)

    for idx, (i, j) in enumerate(pairs):

        ranked_A = ranked_lists[i]
        ranked_B = ranked_lists[j]

        system_names = [p.split('/')[-2] for p in system]

        # Sizes
        N_A = len(ranked_A)
        N_B = len(ranked_B)

        # Actives
        actives_A = set(ranked_A[:TOP_ACTIVES])
        actives_B = set(ranked_B[:TOP_ACTIVES])

        # ======================
        # A → B
        # ======================
        hits_AB = 0
        frac_scan_AB = []
        frac_find_AB = []

        for n, item in enumerate(ranked_B):
            if item in actives_A:
                hits_AB += 1
            frac_scan_AB.append((n + 1) / N_B)
            frac_find_AB.append(min(hits_AB / TOP_ACTIVES, 1.0))

        auc_AB = np.trapz(frac_find_AB, frac_scan_AB)

        # ======================
        # B → A
        # ======================
        hits_BA = 0
        frac_scan_BA = []
        frac_find_BA = []

        for n, item in enumerate(ranked_A):
            if item in actives_B:
                hits_BA += 1
            frac_scan_BA.append((n + 1) / N_A)
            frac_find_BA.append(min(hits_BA / TOP_ACTIVES, 1.0))

        auc_BA = np.trapz(frac_find_BA, frac_scan_BA)

        # ======================
        # Plot
        # ======================
        ax = axes[idx]

        ax.plot(
            frac_scan_AB,
            frac_find_AB,
            color=get_cmap(path)(0.5),
            linewidth=2,
            label=f"{label_map.get(system_names[i], system_names[i])} → "
                f"{label_map.get(system_names[j], system_names[j])}"
        )

        ax.plot(
            frac_scan_BA,
            frac_find_BA,
            color=get_cmap(path)(0.75),
            linestyle="--",
            linewidth=2,
            label=f"{label_map.get(system_names[j], system_names[j])} → "
                f"{label_map.get(system_names[i], system_names[i])}"
        )

        # Random baseline
        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)

        ax.set_xlabel("Fraction of library scanned", fontsize=18)
        ax.set_ylabel(f"Fraction of top {TOP_ACTIVES} actives found", fontsize=18)

        ax.set_title(
            f"{label_map.get(system_names[i], system_names[i])} vs "
            f"{label_map.get(system_names[j], system_names[j])}",
            fontsize=18
        )

        ax.legend(fontsize=14)
        ax.grid(alpha=0.3)

        if idx % ncols != 0:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)

        ax.text(
            0.05, 0.95,
            f"AUC {label_map.get(system_names[i], system_names[i])} → "
            f"{label_map.get(system_names[j], system_names[j])}: {auc_AB:.3f}\n"
            f"AUC {label_map.get(system_names[j], system_names[j])} → "
            f"{label_map.get(system_names[i], system_names[i])}: {auc_BA:.3f}",
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )

    # Disable unused axes
    for ax in axes[len(pairs):]:
        ax.axis("off")

    plt.suptitle(
        f"AUC Enrichment Curves (TOP_ACTIVES={TOP_ACTIVES}) – {title}",
        fontsize=20
    )
    plt.tight_layout()
    plt.show()



# ==================================================
# PART 3 — TOP-N RANK LOCATION COMPARISONS (TRUE BOTH WAYS)
# ==================================================
TOP_ACTIVES = 30

n_panels = 2 * len(pairs)
ncols = 3
nrows = math.ceil(n_panels / ncols)

fig, axes = plt.subplots(
    nrows, ncols,
    figsize=(8 * ncols, 5 * nrows),
    sharex=False,
    sharey=False
)
axes = np.array(axes).reshape(-1)

system_names = [p.split('/')[-2] for p in system]

panel_idx = 0

for (i, j) in pairs:

    ranked_A = ranked_lists[i]
    ranked_B = ranked_lists[j]

    N_A = len(ranked_A)
    N_B = len(ranked_B)

    index_A = {item: k for k, item in enumerate(ranked_A)}
    index_B = {item: k for k, item in enumerate(ranked_B)}

    # ======================
    # A → B
    # ======================
    topA = ranked_A[:TOP_ACTIVES]
    shared_AtoB = [item for item in topA if item in index_B]

    x_A = [index_A[item] + 1 for item in shared_AtoB]
    y_B = [index_B[item] + 1 for item in shared_AtoB]

    ax = axes[panel_idx]
    panel_idx += 1

    ax.scatter(x_A, y_B, s=60, alpha=0.8)
    ax.plot([1, TOP_ACTIVES], [1, TOP_ACTIVES], "--", color="gray")

    ax.set_xlim(0, TOP_ACTIVES)
    ax.set_ylim(-5, N_B)

    ax.set_xlabel(
        f"{label_map.get(system_names[i], system_names[i])} rank (top-{TOP_ACTIVES})",
        fontsize=14
    )
    ax.set_ylabel(
        f"{label_map.get(system_names[j], system_names[j])} rank",
        fontsize=14
    )

    ax.set_title(
        f"{label_map.get(system_names[i], system_names[i])} → "
        f"{label_map.get(system_names[j], system_names[j])}",
        fontsize=18
    )

    ax.grid(alpha=0.3)
    ax.text(
        0.05, 0.95,
        f"Shared: {len(shared_AtoB)} / {TOP_ACTIVES}",
        transform=ax.transAxes,
        fontsize=13,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    # ======================
    # B → A
    # ======================
    topB = ranked_B[:TOP_ACTIVES]
    shared_BtoA = [item for item in topB if item in index_A]

    x_B = [index_B[item] + 1 for item in shared_BtoA]
    y_A = [index_A[item] + 1 for item in shared_BtoA]

    ax = axes[panel_idx]
    panel_idx += 1

    ax.scatter(x_B, y_A, s=60, alpha=0.8)
    ax.plot([1, TOP_ACTIVES], [1, TOP_ACTIVES], "--", color="gray")

    ax.set_xlim(0, TOP_ACTIVES)
    ax.set_ylim(-5, N_A)

    ax.set_xlabel(
        f"{label_map.get(system_names[j], system_names[j])} rank (top-{TOP_ACTIVES})",
        fontsize=14
    )
    ax.set_ylabel(
        f"{label_map.get(system_names[i], system_names[i])} rank",
        fontsize=14
    )

    ax.set_title(
        f"{label_map.get(system_names[j], system_names[j])} → "
        f"{label_map.get(system_names[i], system_names[i])}",
        fontsize=18
    )

    ax.grid(alpha=0.3)
    ax.text(
        0.05, 0.95,
        f"Shared: {len(shared_BtoA)} / {TOP_ACTIVES}",
        transform=ax.transAxes,
        fontsize=13,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

# Turn off unused axes
for ax in axes[panel_idx:]:
    ax.axis("off")

plt.suptitle(
    f"Top-{TOP_ACTIVES} Rank Location Comparisons (Both Directions)",
    fontsize=22
)
plt.tight_layout()
plt.show()
