import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis import contacts
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import math
import matplotlib.ticker as ticker
from sklearn.preprocessing import normalize
import smplotlib

# Protein trajectories
# proteins = ['medin_cm8','medin_urea','abeta_d8', 'abeta_d4','abeta_g5_new_protocol','ph_5_278k','ph_5_298k','ph_7_278k']
# labels = ['CM8','Urea','D8','D4','G5','pH5 278k','pH5 298k','pH7 278k']
proteins = ['ph_5_278k','ph_5_298k','ph_7_278k']
labels = ['pH5 278k','pH5 298k','pH7 278k']

MAX_FRAMES = 10

def contacts_within_cutoff(u, group_a, group_b, radius=3.0):
    timeseries = []
    for ts in u.trajectory:
        dist = contacts.distance_array(group_a.positions, group_b.positions)
        n_contacts = contacts.contact_matrix(dist, radius).sum()
        timeseries.append(n_contacts)
    return np.array(timeseries)

for idx,protein_name in enumerate(proteins):
    try:
        print(f"Processing {protein_name}...")

        if protein_name == 'abeta':
            pdb = '/Users/adelielouet/Documents/science/AB_G5_original_simu_analysis/trajectories/Gabis_paper/template_G5.pdb'
            xtc = '/Users/adelielouet/Documents/science/AB_G5_original_simu_analysis/trajectories/Gabis_paper/traj_all-skip-0-noW_G5.xtc'

        elif protein_name == 'medin_cm8':
            pdb = '/Users/adelielouet/Documents/science/medin/cm8/cm8_post_run/protein_ligand.gro'
            xtc = '/Users/adelielouet/Documents/science/medin/cm8/cm8/concatenate_cm8.xtc'
            
        elif protein_name == 'medin_cm10':
            pdb = '/Users/adelielouet/Documents/science/medin/cm10/cm10_post_run/protein_ligand.gro'
            xtc = '/Users/adelielouet/Documents/science/medin/cm10/cm10/concatenate_cm10.xtc'

        elif protein_name == 'medin_urea':
            pdb = '/Users/adelielouet/Documents/science/medin/urea/urea_post_run/protein_urea.gro'
            xtc = '/Users/adelielouet/Documents/science/medin/urea/urea_post_run/concatenate_urea.xtc'
            resname_ligand = "1UNL"

        elif protein_name == 'abeta_d4':
            pdb = '/Users/adelielouet/Documents/science/shared_files/maria/trajectories/D4/complex_noW_3.gro'
            xtc='/Users/adelielouet/Documents/science/shared_files/maria/trajectories/D4/traj_all-skip-10-noW-PBC_2.xtc'

        elif protein_name == 'abeta_d8':
            pdb = '/Users/adelielouet/Documents/science/shared_files/maria/trajectories/D8/complex_noW_2.gro'
            xtc='/Users/adelielouet/Documents/science/shared_files/maria/trajectories/D8/traj_all-skip-10-noW-PBC_3.xtc'

        elif protein_name == 'abeta_g5_new_protocol':
            pdb = '/Users/adelielouet/Documents/science/shared_files/maria/trajectories/G5_new_proto/complex_noW_4.gro'
            xtc = '/Users/adelielouet/Documents/science/shared_files/maria/trajectories/G5_new_proto/traj_all-skip-10-noW-PBC_1.xtc'

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

        else:
            print(f"Skipping {protein_name}: configuration not defined.")
            continue

        u = mda.Universe(pdb, xtc)
        resid_list = [f"resid {res.resid}" for res in u.select_atoms("protein").residues]
        ca_df = pd.DataFrame(index=range(len(u.trajectory)))
        # ca_df = pd.DataFrame(index=range(MAX_FRAMES))

        for y in resid_list:
            ligand = u.select_atoms('not protein')
            pocket = u.select_atoms(y)
            ca = contacts_within_cutoff(u, ligand, pocket, radius=3.0)
            ca_df[y] = ca.flatten()

        # Convert >0 contacts to 1, 0 → NaN
        ca_df = ca_df.applymap(lambda x: 1 if x > 0 else np.nan)

        # ---- Build Transition Matrix ----
        sequence = ca_df.fillna(0).to_numpy()
        protein = u.select_atoms('protein')
        residues = [res.resname for res in protein.residues]
        num_residues = len(residues)

        transition_matrix = np.zeros((num_residues, num_residues))

        for step in range(len(sequence) - 1):
            current_contact = sequence[step]
            next_contact = sequence[step + 1]

            for i in range(num_residues):
                for j in range(num_residues):
                    if current_contact[i] == 1 and next_contact[j] == 1:
                        transition_matrix[i][j] += 1

        transition_matrix_norm = normalize(transition_matrix, axis=1, norm='l1')
   
        fig, ax = plt.subplots(figsize=(8,6))

        if protein_name.startswith('alpha'):
            cmap = LinearSegmentedColormap.from_list("alpha_purple", ["#e0d4f7", "#5e3c99"])
        elif protein_name.startswith('medin'):
            cmap = LinearSegmentedColormap.from_list("medin_green", ["#edf8f2", "#5a8f7b"])
        elif protein_name.startswith('abeta'):
            cmap = LinearSegmentedColormap.from_list("abeta_blue", ["#eef4fa", "#355f8c"])
        elif protein_name.startswith('ph'):
            cmap = LinearSegmentedColormap.from_list("abeta_blue", ["#edf0fa", "#2f3f7f"])
                        
        else:
            cmap = "viridis"

        sns.heatmap(transition_matrix_norm, cmap=cmap, cbar_kws={"shrink": 0.8}, ax=ax)

        ax.set_title(labels[idx], fontsize=25, pad=15)
        ax.set_ylabel("From Residues [i]", fontsize=14)
        ax.set_xlabel("To Residues [j]", fontsize=14)
        ax.tick_params(axis='y', labelrotation=0)

        plt.tight_layout()
        plt.savefig(f"/Users/adelielouet/Documents/science/dd_proj/msm_full_model_paper/pickled_files/{protein_name}/figures/heatmap_rial.pdf", format='pdf')

        plt.show()

        print(f"Finished {protein_name} successfully!\n")

    except Exception as e:
        print(f"Error processing {protein_name}: {e}")
        continue

    