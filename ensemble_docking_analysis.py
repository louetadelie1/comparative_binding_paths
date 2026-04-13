import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import math

df = pd.read_pickle(
    "/Users/adelielouet/Documents/science/dd_proj/comparative_binding_paths_final_draft/ads_youre_an_idiot.pkl"
)

label_map = {
    "alpha_syn_lig_40": "Lig 40",
    "alpha_syn_lig_20": "Lig 20",
    "alpha_syn_lig_4":  "Lig 4",
    "alpha_syn_lig_26": "Lig 26",
    "alpha_syn_lig_30": "Lig 30",
    "alpha_syn_lig_12": "Lig 12",
    "alpha_syn_lig_50": "Lig 50",
    "cm8": "CM8",
    "cm10": "CM10",
    "urea": "Urea",
    "D4": "D4",
    "D8": "D8",
    "G5_new_proto": "G5",
    "ph5_278k": "pH5 278k",
    "ph5_298k": "pH5 298k",
    "ph7_278k": "pH7 278k",
}


def get_cmap(system):
    if system.startswith("fausadil"):
        return LinearSegmentedColormap.from_list(
            "alpha_purple", ["#e0d4f7", "#5e3c99"]
        )
    elif system.startswith("medin"):
        return LinearSegmentedColormap.from_list(
            "medin_green", ["#edf8f2", "#5a8f7b"]
        )
    elif system.startswith("abeta_g5_deriv"):
        return LinearSegmentedColormap.from_list(
            "abeta_blue", ["#eef4fa", "#355f8c"]
        )
    elif system.startswith("diff_ph_abeta"):
        return LinearSegmentedColormap.from_list(
            "abeta_blue", ["#edf0fa", "#2f3f7f"]
        )
    else:
        return plt.cm.viridis


for system, df_sys in df.groupby("protein_system"):

    cmap = get_cmap(system)
    ligands = list(df_sys["ligand"].drop_duplicates())

    n_panels = len(ligands)
    ncols = 3
    nrows = math.ceil(n_panels / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        #figsize=(4.2 * ncols, 3.2 * nrows),
        figsize=(8 * ncols, 5 * nrows),

        sharey=True
    )
    axes = np.array(axes).reshape(-1)

    for ax, lig in zip(axes, ligands):

        above_35 = df_sys[df_sys["ligand"] == lig].copy()
        title_ = label_map.get(lig, lig)

        # --- select pockets PER LIGAND (THIS IS THE KEY FIX) ---
        pockets = np.sort(above_35["pocket_number"].unique())[:30]
        above_35 = above_35[above_35["pocket_number"].isin(pockets)]

        if above_35.empty:
            ax.set_title(title_)
            ax.axis("off")
            continue

        above_35["all_stp_score"] = above_35["all_stp_score"].apply(list)

        # --- explode ---
        long_df = (
            above_35[["pocket_number", "all_stp_score"]]
            .explode("all_stp_score")
            .reset_index(drop=True)
            .rename(columns={"all_stp_score": "score"})
        )

        n_ligands = len(above_35["all_stp_score"].iloc[0])
        norm = Normalize(vmin=0, vmax=n_ligands - 1)

        long_df["ligand_id"] = np.tile(
            np.arange(n_ligands),
            len(above_35)
        )

        pocket_map = {p: i for i, p in enumerate(pockets)}
        long_df["pocket_idx"] = long_df["pocket_number"].map(pocket_map)

        # --- plot each docked ligand ---
        for ligand_id, g in long_df.groupby("ligand_id"):
            g = g.sort_values("pocket_idx")

            ax.plot(
                g["pocket_idx"],
                g["score"],
                color=cmap(norm(ligand_id)),
                linewidth=1,
                alpha=0.85
            )

        ax.set_title(title_, fontsize=20)
        ax.invert_yaxis()
        # ax.grid(True, linestyle="--", alpha=0.25)
        ax.set_xticks(range(len(pockets)))
        ax.tick_params(axis="x", labelsize=15,rotation=90)
        ax.tick_params(axis="y", labelsize=15)

    for j in range(len(ligands), len(axes)):
        axes[j].axis("off")

    # fig.suptitle(system, fontsize=15, y=1.02)
    fig.supxlabel("Pocket index")
    fig.supylabel("Vina Docking Score")

    plt.tight_layout()
    plt.show()
