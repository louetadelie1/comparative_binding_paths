import os
import glob
import itertools
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker
import seaborn as sns
import MDAnalysis as mda
from MDAnalysis.analysis import contacts
import mdtraj as md
import smplotlib


def contacts_within_cutoff(u, group_a, group_b, radius=3.0):
    timeseries = []
    for ts in u.trajectory:
        dist = contacts.distance_array(group_a.positions, group_b.positions)
        n_contacts = contacts.contact_matrix(dist, radius).sum()
        timeseries.append([n_contacts])
    return np.array(timeseries)

path = os.getcwd()
cutoff = 0.45


systems = {
    "medin": {
        "proteins": ["medin_cm10", "medin_cm8", "medin_urea"],
        "label":'Medin',
        "cmap": LinearSegmentedColormap.from_list("medin_green",["#edf8f2", "#5a8f7b"])},

    "abeta": {
        "proteins": ["abeta_d8", "abeta_d4", "abeta_g5_new_protocol"],
        "label":'A\u03B242',
        "cmap": LinearSegmentedColormap.from_list("abeta_steel",["#eef4fa", "#355f8c"])},
        
    "abeta_ph": {
        "proteins": ["ph_5_278k", "ph_7_278k", "ph_5_298k"],
        "label":'A\u03B242 Ph',
        "cmap": LinearSegmentedColormap.from_list("ph_indigo",["#edf0fa", "#2f3f7f"])}
        }

for system_name, system in systems.items():

    cmap = system["cmap"]
    plot_data = []

    for protein_name in system["proteins"]:
        if protein_name == 'medin_cm8':
            pdb = '/Users/adelielouet/Documents/science/medin/cm8/cm8/protein_ligand.gro'
            xtc = '/Users/adelielouet/Documents/science/medin/cm8/cm8/concatenate_cm8.xtc'
            label='Medin'
            ligand='CM8'

        elif protein_name == 'medin_cm10':
            pdb = '/Users/adelielouet/Documents/science/medin/cm10/cm10/protein_ligand.gro'
            xtc = '/Users/adelielouet/Documents/science/medin/cm10/cm10/concatenate_cm10.xtc'
            label='Medin'
            ligand='CM10'

        elif protein_name == 'medin_urea':
            pdb = '/Users/adelielouet/Documents/science/medin/urea/urea_post_run/protein_urea.gro'
            xtc = '/Users/adelielouet/Documents/science/medin/urea/urea_post_run/concatenate_urea.xtc'
            label='Medin'
            ligand='Urea'

        elif protein_name == 'abeta_d4':
            pdb = '/Users/adelielouet/Documents/science/shared_files/maria/trajectories/D4/complex_noW_3.gro'
            label='A\u03B242' 
            xtc='/Users/adelielouet/Documents/science/shared_files/maria/trajectories/D4/traj_all-skip-10-noW-PBC_2.xtc'
            ligand='D4'

        elif protein_name == 'abeta_d8':
            pdb = '/Users/adelielouet/Documents/science/shared_files/maria/trajectories/D8/complex_noW_2.gro'
            label='A\u03B242' 
            xtc='/Users/adelielouet/Documents/science/shared_files/maria/trajectories/D8/traj_all-skip-10-noW-PBC_3.xtc'
            ligand='D8'

        elif protein_name == 'abeta_g5_new_protocol':
            pdb = '/Users/adelielouet/Documents/science/shared_files/maria/trajectories/G5_new_proto/complex_noW_4.gro'
            xtc = '/Users/adelielouet/Documents/science/shared_files/maria/trajectories/G5_new_proto/traj_all-skip-10-noW-PBC_1.xtc'
            label='A\u03B242' 
            ligand='G5'

        elif protein_name == 'ph_5_278k':
            pdb = '/Users/adelielouet/Documents/science/dd_proj/abeta_ph_simulations/ph5_278k/template.pdb'
            xtc = '/Users/adelielouet/Documents/science/dd_proj/abeta_ph_simulations/ph5_278k/all_concatenated.xtc'
            label='A\u03B242 pH' 
            ligand='pH5 278k'

        elif protein_name == 'ph_7_278k':
            pdb = '/Users/adelielouet/Documents/science/dd_proj/abeta_ph_simulations/ph7_278k/template.pdb'
            xtc = '/Users/adelielouet/Documents/science/dd_proj/abeta_ph_simulations/ph7_278k/all_concatenated.xtc'
            label='A\u03B242 pH' 
            ligand='pH7 278k'

        elif protein_name == 'ph_5_298k':
            pdb = '/Users/adelielouet/Documents/science/dd_proj/abeta_ph_simulations/ph5_298k/template.pdb'
            xtc = '/Users/adelielouet/Documents/science/dd_proj/abeta_ph_simulations/ph5_298k/all_concatenated.xtc'
            label='A\u03B242 pH' 
            ligand='pH5 298k'

        traj = md.load(xtc, top=pdb)
        protein_atoms = traj.topology.select("protein")
        ligand_atoms = traj.topology.select("not protein")
        print(traj,pdb)
        neighbors = md.compute_neighbors(
            traj, cutoff, query_indices=ligand_atoms, haystack_indices=protein_atoms
        )

        residue_contact_counts = {}
        for frame_neighbors in neighbors:
            res_ids = {traj.topology.atom(i).residue.index for i in frame_neighbors}
            for r in res_ids:
                residue_contact_counts[r] = residue_contact_counts.get(r, 0) + 1

        n_frames = traj.n_frames
        for r, count in residue_contact_counts.items():
            plot_data.append({
                "Residue": f"{traj.topology.residue(r).name[0]}{traj.topology.residue(r).resSeq}",
                "Frequency": count / n_frames,
                "Ligand": ligand
            })

    df = pd.DataFrame(plot_data)
    combined_df = df.groupby(["Residue", "Ligand"], as_index=False).sum()

    ligands = combined_df["Ligand"].unique()
    sample_points = np.linspace(0.25, 0.95, len(ligands))
    colors = [cmap(x) for x in sample_points]


    fig, ax = plt.subplots(figsize=(10, 5))

    for i, ligand in enumerate(ligands):
        subset = combined_df[combined_df["Ligand"] == ligand]
        ax.plot(
            subset["Residue"],
            subset["Frequency"],
            marker="o",
            label=ligand,
            color=colors[i],
            markersize=5
        )

    ax.legend(fontsize=13,frameon=False)
    ax.set_title(label, fontsize=14)
    ax.set_ylabel("Contact Frequency", fontsize=12)
    ax.set_xlabel("Residues", fontsize=12)
    ax.tick_params(axis="x", rotation=90, labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.legend(fontsize=9, frameon=False)

    plt.tight_layout()
    plt.show()


########## On the chem cluster for the alpha-syn lgiands (/chem2/scratch/adlouet/alpha_syn/cterm)
path = os.getcwd()
ligands = ['lig_12', 'lig_20', 'lig_26', 'lig_30', 'lig_4']
labels = ['L12', 'L20', 'L26', 'L30', 'L4']

save_dir = os.path.join(path, "heatmaps")
os.makedirs(save_dir, exist_ok=True)
cutoff = 0.45

plot_data = []

for idx,protein_name in enumerate(ligands):
    pdb = sorted(glob.glob(f'{path}/{protein_name}/*/*/*pdb'))[0]
    DCD_files = sorted(glob.glob(f'{path}/{protein_name}/*/*/*dcd'))


    global_residue_labels = None

    traj = md.load(DCD_files,top=pdb)

    protein_atoms = traj.topology.select("protein")
    ligand_atoms = traj.topology.select("not protein")

    neighbors = md.compute_neighbors(
        traj, cutoff, query_indices=ligand_atoms, haystack_indices=protein_atoms
    )

    residue_contact_counts = {}
    for frame_neighbors in neighbors:
        res_ids = {traj.topology.atom(i).residue.index for i in frame_neighbors}
        for r in res_ids:
            residue_contact_counts[r] = residue_contact_counts.get(r, 0) + 1

    n_frames = traj.n_frames
    residues = sorted(residue_contact_counts.keys())
    freqs = np.array([residue_contact_counts[r] / n_frames for r in residues])

    residue_labels = [
        f"{traj.topology.residue(r).name[0]}{traj.topology.residue(r).resSeq}"
        for r in residues
    ]

    if global_residue_labels is None:
        global_residue_labels = residue_labels  # for consistent x-axis

    for res_label, freq in zip(residue_labels, freqs):
        plot_data.append({
            "Residue": res_label,
            "Frequency": freq,
            "Ligand": labels[idx]
        })

df = pd.DataFrame(plot_data)

combined_df = df.groupby(['Residue', 'Ligand'], as_index=False).agg({
    'Frequency': 'sum'  # or 'mean' if averaging makes more sense
})
ligands = combined_df['Ligand'].unique()

fig, axes = plt.subplots(len(ligands), 1, figsize=(8, 10), sharex=True)

# colors = [sns.color_palette("Blues", 4)[3], 
#         sns.color_palette("Purples", 4s)[3]]     

for i, ligand in enumerate(ligands):
    subset = combined_df[combined_df['Ligand'] == ligand]
    sns.pointplot(
        data=subset,
        x="Residue",
        y="Frequency",
        color="#ab80fa",
        marker='o',
        scale=0.7,
        ax=axes[i]
    )
    axes[i].set_title(f'{ligand}', fontsize=14)
    axes[i].set_ylabel("Frequency", fontsize=12)
    axes[i].tick_params(axis='x', rotation=90, labelsize=11)
    axes[i].tick_params(axis='y', labelsize=12, width=2, length=8)
    #axes[i].grid(True, which='major', linestyle='--', alpha=0.5)
    axes[i].grid(False)

axes[1].set_xlabel("Residue", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, f'contact_freq_{protein_name}.pdf'), format='pdf')
plt.show()


fig, ax = plt.subplots(figsize=(10, 5))

greens = sns.color_palette("Purples", n_colors=len(ligands)+2)[2:]  # skip lightest tones

for i, ligand in enumerate(ligands):
    subset = combined_df[combined_df['Ligand'] == ligand]
    ax.plot(
        subset['Residue'],
        subset['Frequency'],
        marker='o',
        label=ligand,
        color=greens[i]
    )

ax.set_title(r"$\alpha$-Synuclein C Terminal", fontsize=14)
ax.set_ylabel("Frequency", fontsize=12)
ax.set_xlabel("Residue", fontsize=12)
ax.tick_params(axis='x', rotation=90, labelsize=10)
ax.tick_params(axis='y', labelsize=10)
ax.grid(False)
ax.legend(fontsize=13,frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, f'contact_freq_combined_2.pdf'), format='pdf')
plt.show()