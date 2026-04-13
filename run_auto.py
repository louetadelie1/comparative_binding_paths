import sys, os, pickle, math, itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import networkx as nx
import itertools
import math
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from comparative_binding_paths_final_draft.population_equilibrium import *
from comparative_binding_paths_final_draft.clustering_uplets import *

# List of proteins to run
proteins = ['ZINC000000030986', 'ZINC000000038519', 'ZINC000000052225', 'ZINC000000057966'#,'abeta_gabis'
# 'alpha_syn_lig_12'
#, 'alpha_syn_lig_20', 
#    'alpha_syn_lig_26', 'alpha_syn_lig_30', 'alpha_syn_lig_4','alpha_syn_lig_40','alpha_syn_lig_50','abeta_gabis'
#     'medin_cm10','medin_cm8','medin_urea',
# 'abeta_d8','abeta_d4','abeta_g5_new_protocol'
# ,'abeta_ph5_298k','abeta_ph7_278k','abeta_ph5_298k'
]

# we will be including the following for the publication:
# proteins = ['alpha_syn_lig_40','alpha_syn_lig_50','alpha_syn_lig_12','abeta_gabis']

# proteins = ['abeta_ph5_278_v2']#,'abeta_ph5_278k','abeta_ph7_278k']
output_dir='/Users/adelielouet/Documents/science/dd_proj/mass_producing_idp_simulations/protein_simualtions/output_post_processed'

def sigmoid_magnification(val, A, k, x0):
    return A / (1 + math.exp(-k * (val - x0)))

plot_data = []

for protein_name in proteins:
    try:
        print(f"Processing {protein_name}...")

        # === Protein-specific settings == #
        if protein_name == 'abeta_gabis':
            pdb = '/Users/adelielouet/Documents/science/AB_G5_original_simu_analysis/trajectories/Gabis_paper/template_G5.pdb'
            label="abeta_gabis"
            distances_com = pickle.load(open(f'/Users/adelielouet/Documents/science/dd_proj/msm_full_model_paper/pickled_files/abeta_gabis/d_24_t_com_avg.pkl', 'rb'))
            distances_closest = pickle.load(open(f'/Users/adelielouet/Documents/science/dd_proj/msm_full_model_paper/pickled_files/abeta_gabis/d_24_t_closest.pkl', 'rb'))
            w_com =  0.3
            w_closest =  0.7
            w_eq=[1.0] *len(distances_com)   # this is for systems that don't need to be reweighted, assign each frame a weight of 1.0
            combined_threshold=True
            uplet_type=3
            
        elif protein_name == 'medin_cm8':
            pdb = '/Users/adelielouet/Documents/science/medin/cm8/cm8_post_run/protein_ligand.gro'
            xtc = '/Users/adelielouet/Documents/science/medin/cm8/cm8/concatenate_cm8.xtc'
            w_file='/Users/adelielouet/Documents/science/dd_proj/msm_full_model_paper/pickled_files/medin_cm8/weights/COLVAR_REWEIGHT'
            distances_com = pickle.load(open(f'/Users/adelielouet/Documents/science/dd_proj/msm_full_model_paper/pickled_files/medin_cm8/distances/d_24_t_com_avg.pkl', 'rb'))
            distances_closest = pickle.load(open(f'/Users/adelielouet/Documents/science/dd_proj/msm_full_model_paper/pickled_files/medin_cm8/distances/d_24_t_closest.pkl', 'rb'))
            resname_ligand = "1UNL"
            special_char = 'Medin_cm8'
            w_eq = process_weights(w_file)
            w_com = 0.45
            w_closest = 0.55
            combined_threshold=True
            uplet_type=3

        elif protein_name == 'medin_cm10':
            pdb = '/Users/adelielouet/Documents/science/medin/cm10/cm10_post_run/protein_ligand.gro'
            xtc = '/Users/adelielouet/Documents/science/medin/cm10/cm10/concatenate_cm10.xtc'
            w_file='/Users/adelielouet/Documents/science/dd_proj/msm_full_model_paper/pickled_files/medin_cm10/weights/COLVAR_REWEIGHT'
            distances_com = pickle.load(open(f'/Users/adelielouet/Documents/science/dd_proj/msm_full_model_paper/pickled_files/medin_cm10/distances/d_24_t_com_avg.pkl', 'rb'))
            distances_closest = pickle.load(open(f'/Users/adelielouet/Documents/science/dd_proj/msm_full_model_paper/pickled_files/medin_cm10/distances/d_24_t_closest.pkl', 'rb'))
            resname_ligand = "1UNL"
            special_char = 'Medin_cm10'
            w_file = None
            w_com = 0.45
            w_closest = 0.55
            # w_eq=[1.0] *len(distances_com)   # this is for systems that don't need to be reweighted, assign each frame a weight of 1.0
            combined_threshold=True
            uplet_type=3

        elif protein_name == 'medin_urea':
            pdb = '/Users/adelielouet/Documents/science/medin/urea/urea_post_run/protein_urea.gro'
            xtc = '/Users/adelielouet/Documents/science/medin/urea/urea_post_run/concatenate_urea.xtc'
            resname_ligand = "1UNL"
            distances_com = pickle.load(open(f'/Users/adelielouet/Documents/science/dd_proj/msm_full_model_paper/pickled_files/medin_urea/distances/d_24_t_com_avg.pkl', 'rb'))
            distances_closest = pickle.load(open(f'/Users/adelielouet/Documents/science/dd_proj/msm_full_model_paper/pickled_files/medin_urea/distances/d_24_t_closest.pkl', 'rb'))
            w_file = None
            w_com = 0.45
            w_closest = 0.55
            # w_eq=[1.0] *len(distances_com)   # this is for systems that don't need to be reweighted, assign each frame a weight of 1.0
            combined_threshold=True
            uplet_type=3

        elif protein_name == 'alpha_syn_lig_40':
            pdb = '/Users/adelielouet/Documents/science/alpha_syn/DE_SHAW/lig_40/ligand_40_alpha_syn_c_term.pdb'
            # xtc = '/Users/adelielouet/Documents/science/medin/urea/urea_post_run/concatenate_urea.xtc'
            # resname_ligand = "1UNL"
            distances_com = pickle.load(open(f'/Users/adelielouet/Documents/science/dd_proj/msm_full_model_paper/pickled_files/alpha_syn_lig_40/distances/t_resnum_com.pkl', 'rb'))
            distances_closest = pickle.load(open(f'/Users/adelielouet/Documents/science/dd_proj/msm_full_model_paper/pickled_files/alpha_syn_lig_40/distances/t_resnum_closest.pkl', 'rb'))
            w_com = 0.3
            w_closest = 0.7
            w_eq=[1.0] *len(distances_com)   # this is for systems that don't need to be reweighted, assign each frame a weight of 1.0
            combined_threshold=True
            uplet_type=3

        elif protein_name == 'alpha_syn_lig_50':
            pdb = '/Users/adelielouet/Documents/science/alpha_syn/DE_SHAW/lig_50/ligand_50_alpha_syn_c_term.pdb'
            # xtc = '/Users/adelielouet/Documents/science/medin/urea/urea_post_run/concatenate_urea.xtc'
            # resname_ligand = "1UNL"
            distances_com = pickle.load(open(f'/Users/adelielouet/Documents/science/dd_proj/msm_full_model_paper/pickled_files/alpha_syn_lig_50/distances/t_resnum_com.pkl', 'rb'))
            distances_closest = pickle.load(open(f'/Users/adelielouet/Documents/science/dd_proj/msm_full_model_paper/pickled_files/alpha_syn_lig_50/distances/t_resnum_closest.pkl', 'rb'))
            w_com = 0.3
            w_closest = 0.7
            w_eq=[1.0] *len(distances_com)   # this is for systems that don't need to be reweighted, assign each frame a weight of 1.0
            combined_threshold=True
            uplet_type=3

        elif protein_name == 'alpha_syn_lig_12':
            pdb = '/Users/adelielouet/Documents/science/alpha_syn/DE_SHAW/lig_12/DESRES-Trajectory_jacs2022-5447858-no-water-glue/jacs2022-5447858-no-water-glue/lig_12_cterm.pdb'
            distances_com = pickle.load(open(f'/Users/adelielouet/Documents/science/alpha_syn/DE_SHAW/lig_12/DESRES-Trajectory_jacs2022-5447858-no-water-glue/jacs2022-5447858-no-water-glue/t_resnum_com.pkl', 'rb'))
            distances_closest = pickle.load(open(f'/Users/adelielouet/Documents/science/alpha_syn/DE_SHAW/lig_12/DESRES-Trajectory_jacs2022-5447858-no-water-glue/jacs2022-5447858-no-water-glue/t_resnum_closest.pkl', 'rb'))
            w_com = 0.3
            w_closest = 0.7
            w_eq=[1.0] *len(distances_com)   # this is for systems that don't need to be reweighted, assign each frame a weight of 1.0
            combined_threshold=True
            uplet_type=3

        elif protein_name == 'alpha_syn_lig_20':
            pdb = '/Users/adelielouet/Documents/science/alpha_syn/DE_SHAW/lig_20/DESRES-Trajectory_jacs2022-5447842-no-water-glue/jacs2022-5447842-no-water-glue/lig_20_cterm.pdb'
            distances_com = pickle.load(open(f'/Users/adelielouet/Documents/science/alpha_syn/DE_SHAW/lig_20/DESRES-Trajectory_jacs2022-5447842-no-water-glue/jacs2022-5447842-no-water-glue/t_resnum_com.pkl', 'rb'))
            distances_closest = pickle.load(open(f'/Users/adelielouet/Documents/science/alpha_syn/DE_SHAW/lig_20/DESRES-Trajectory_jacs2022-5447842-no-water-glue/jacs2022-5447842-no-water-glue/t_resnum_closest.pkl', 'rb'))
            w_com = 0.3
            w_closest = 0.7
            w_eq=[1.0] *len(distances_com)   # this is for systems that don't need to be reweighted, assign each frame a weight of 1.0
            combined_threshold=True
            uplet_type=3

        elif protein_name == 'alpha_syn_lig_26':
            pdb = '/Users/adelielouet/Documents/science/alpha_syn/DE_SHAW/lig_26/DESRES-Trajectory_jacs2022-5447843-no-water-glue/jacs2022-5447843-no-water-glue/lig_26_cterm.pdb'
            distances_com = pickle.load(open(f'/Users/adelielouet/Documents/science/alpha_syn/DE_SHAW/lig_26/DESRES-Trajectory_jacs2022-5447843-no-water-glue/jacs2022-5447843-no-water-glue/t_resnum_com.pkl', 'rb'))
            distances_closest = pickle.load(open(f'/Users/adelielouet/Documents/science/alpha_syn/DE_SHAW/lig_26/DESRES-Trajectory_jacs2022-5447843-no-water-glue/jacs2022-5447843-no-water-glue/t_resnum_closest.pkl', 'rb'))
            w_com = 0.3
            w_closest = 0.7
            w_eq=[1.0] *len(distances_com)   # this is for systems that don't need to be reweighted, assign each frame a weight of 1.0
            combined_threshold=True
            uplet_type=3

        elif protein_name == 'alpha_syn_lig_30':
            pdb = '/Users/adelielouet/Documents/science/alpha_syn/DE_SHAW/lig_30/DESRES-Trajectory_jacs2022-5447857-no-water-glue/jacs2022-5447857-no-water-glue/lig_30_cterm.pdb'
            distances_com = pickle.load(open(f'/Users/adelielouet/Documents/science/alpha_syn/DE_SHAW/lig_30/DESRES-Trajectory_jacs2022-5447857-no-water-glue/jacs2022-5447857-no-water-glue/t_resnum_com.pkl', 'rb'))
            distances_closest = pickle.load(open(f'/Users/adelielouet/Documents/science/alpha_syn/DE_SHAW/lig_30/DESRES-Trajectory_jacs2022-5447857-no-water-glue/jacs2022-5447857-no-water-glue/t_resnum_closest.pkl', 'rb'))
            w_com =  0.3
            w_closest = 0.7
            w_eq=[1.0] *len(distances_com)   # this is for systems that don't need to be reweighted, assign each frame a weight of 1.0
            combined_threshold=True
            uplet_type=3

        elif protein_name == 'alpha_syn_lig_4':
            pdb = '/Users/adelielouet/Documents/science/alpha_syn/DE_SHAW/lig_4/DESRES-Trajectory_jacs2022-12293914-no-water-glue/jacs2022-12293914-no-water-glue/lig_4_cterm.pdb'
            distances_com = pickle.load(open(f'/Users/adelielouet/Documents/science/alpha_syn/DE_SHAW/lig_4/DESRES-Trajectory_jacs2022-12293914-no-water-glue/jacs2022-12293914-no-water-glue/t_resnum_com.pkl', 'rb'))
            distances_closest = pickle.load(open(f'/Users/adelielouet/Documents/science/alpha_syn/DE_SHAW/lig_4/DESRES-Trajectory_jacs2022-12293914-no-water-glue/jacs2022-12293914-no-water-glue/t_resnum_closest.pkl', 'rb'))
            w_com =  0.3
            w_closest =  0.7
            w_eq=[1.0] *len(distances_com)   # this is for systems that don't need to be reweighted, assign each frame a weight of 1.0
            combined_threshold=True
            uplet_type=3

        elif protein_name == 'abeta_d4':
            pdb = '/Users/adelielouet/Documents/science/shared_files/maria/trajectories/D4/complex_noW_3.gro'
            distances_com = pickle.load(open(f'/Users/adelielouet/Documents/science/dd_proj/msm_full_model_paper/pickled_files/abeta_D4/distances/d_24_t_com_avg.pkl', 'rb'))
            distances_closest = pickle.load(open(f'/Users/adelielouet/Documents/science/dd_proj/msm_full_model_paper/pickled_files/abeta_D4/distances/d_24_t_closest.pkl', 'rb'))
            w_com = 0.3
            w_closest = 0.7
            w_eq=[1.0] *len(distances_com)   # this is for systems that don't need to be reweighted, assign each frame a weight of 1.0
            combined_threshold=True
            uplet_type=3

        elif protein_name == 'abeta_d8':
            pdb = '/Users/adelielouet/Documents/science/shared_files/maria/trajectories/D8/complex_noW_2.gro'
            distances_com = pickle.load(open(f'/Users/adelielouet/Documents/science/dd_proj/msm_full_model_paper/pickled_files/abeta_D8/distances/d_24_t_com_avg.pkl', 'rb'))
            distances_closest = pickle.load(open(f'/Users/adelielouet/Documents/science/dd_proj/msm_full_model_paper/pickled_files/abeta_D8/distances/d_24_t_closest.pkl', 'rb'))
            w_com = 0.3
            w_closest =0.7
            w_eq=[1.0] *len(distances_com)   # this is for systems that don't need to be reweighted, assign each frame a weight of 1.0
            combined_threshold=True
            uplet_type=3

        elif protein_name == 'abeta_g5_new_protocol':
            pdb = '/Users/adelielouet/Documents/science/shared_files/maria/trajectories/G5_new_proto/complex_noW_4.gro'
            distances_com = pickle.load(open(f'/Users/adelielouet/Documents/science/dd_proj/msm_full_model_paper/pickled_files/abeta_G5_new_protocol/distances/d_24_t_com_avg.pkl', 'rb'))
            distances_closest = pickle.load(open(f'/Users/adelielouet/Documents/science/dd_proj/msm_full_model_paper/pickled_files/abeta_G5_new_protocol/distances/d_24_t_closest.pkl', 'rb'))
            w_com = 0.3
            w_closest = 0.7
            w_eq=[1.0] *len(distances_com)   # this is for systems that don't need to be reweighted, assign each frame a weight of 1.0
            combined_threshold=True
            uplet_type=3

        elif protein_name == 'ph_5_278k':
            pdb = '/Users/adelielouet/Documents/science/dd_proj/abeta_ph_simulations/ph5_278k/Complex_AB.pdb'
            label=protein_name.split('abeta_')[1]
            distances_com = pickle.load(open(f'//Users/adelielouet/Documents/science/dd_proj/abeta_ph_simulations/pickled_files/{label}/distances_2/d_24_t_com_avg.pkl', 'rb'))
            distances_closest = pickle.load(open(f'//Users/adelielouet/Documents/science/dd_proj/abeta_ph_simulations/pickled_files/{label}/distances_2/d_24_t_closest.pkl', 'rb'))
            w_com =  0.3
            w_closest =  0.7
            w_eq=[1.0] *len(distances_com)   # this is for systems that don't need to be reweighted, assign each frame a weight of 1.0
            combined_threshold=True
            uplet_type=3

        elif protein_name == 'ph_7_278k':
            pdb = '/Users/adelielouet/Documents/science/dd_proj/abeta_ph_simulations/ph5_278k/Complex_AB.pdb'
            label=protein_name.split('abeta_')[1]
            distances_com = pickle.load(open(f'//Users/adelielouet/Documents/science/dd_proj/abeta_ph_simulations/pickled_files/{label}/distances_2/d_24_t_com_avg.pkl', 'rb'))
            distances_closest = pickle.load(open(f'//Users/adelielouet/Documents/science/dd_proj/abeta_ph_simulations/pickled_files/{label}/distances_2/d_24_t_closest.pkl', 'rb'))
            w_com =  0.3
            w_closest =  0.7
            w_eq=[1.0] *len(distances_com)   # this is for systems that don't need to be reweighted, assign each frame a weight of 1.0
            combined_threshold=True
            uplet_type=3

        elif protein_name == 'ph_5_298k':
            pdb = '/Users/adelielouet/Documents/science/dd_proj/abeta_ph_simulations/ph5_278k/Complex_AB.pdb'
            label=protein_name.split('abeta_')[1]
            distances_com = pickle.load(open(f'//Users/adelielouet/Documents/science/dd_proj/abeta_ph_simulations/pickled_files/{label}/distances_2/d_24_t_com_avg.pkl', 'rb'))
            distances_closest = pickle.load(open(f'//Users/adelielouet/Documents/science/dd_proj/abeta_ph_simulations/pickled_files/{label}/distances_2/d_24_t_closest.pkl', 'rb'))
            w_com =  0.3
            w_closest =  0.7
            w_eq=[1.0] *len(distances_com)   # this is for systems that don't need to be reweighted, assign each frame a weight of 1.0
            combined_threshold=True
            uplet_type=3

        elif protein_name == 'abeta_ph5_278_v2':
            pdb = '/Users/adelielouet/Documents/science/dd_proj/abeta_ph_simulations/ph5_278k/Complex_AB.pdb'
            label=protein_name.split('abeta_')[1]
            distances_com = pickle.load(open(f'//Users/adelielouet/Documents/science/dd_proj/abeta_ph_simulations/pickled_files/{label}/distances_2/d_24_t_com_avg.pkl', 'rb'))
            distances_closest = pickle.load(open(f'//Users/adelielouet/Documents/science/dd_proj/abeta_ph_simulations/pickled_files/{label}/distances_2/d_24_t_closest.pkl', 'rb'))
            w_com =  0.3
            w_closest =  0.7
            w_eq=[1.0] *len(distances_com)   # this is for systems that don't need to be reweighted, assign each frame a weight of 1.0
            combined_threshold=True
            uplet_type=3

        elif protein_name == 'ZINC000000030986':            
            pdb = f'/Users/adelielouet/Documents/science/dd_proj/mass_producing_idp_simulations/protein_simualtions/output_post_processed/abeta/{protein_name}/template.pdb'
            label=protein_name
            distances_com = pickle.load(open(f'/Users/adelielouet/Documents/science/dd_proj/mass_producing_idp_simulations/protein_simualtions/output_post_processed/abeta/{protein_name}/distances/all_concatenated_1.xtc/d_24_t_com_avg.pkl', 'rb'))
            distances_closest = pickle.load(open(f'/Users/adelielouet/Documents/science/dd_proj/mass_producing_idp_simulations/protein_simualtions/output_post_processed/abeta/{protein_name}/distances/all_concatenated_1.xtc/d_24_t_closest.pkl', 'rb'))
            w_com =  0.3
            w_closest =  0.7
            w_eq=[1.0] *len(distances_com)   # this is for systems that don't need to be reweighted, assign each frame a weight of 1.0
            combined_threshold=True
            uplet_type=3
            restart='1'

        elif protein_name == 'ZINC000000038519':            
            pdb = f'/Users/adelielouet/Documents/science/dd_proj/mass_producing_idp_simulations/protein_simualtions/output_post_processed/abeta/{protein_name}/template.pdb'
            label=protein_name
            distances_com = pickle.load(open(f'/Users/adelielouet/Documents/science/dd_proj/mass_producing_idp_simulations/protein_simualtions/output_post_processed/abeta/{protein_name}/distances/all_concatenated_1.xtc/d_24_t_com_avg.pkl', 'rb'))
            distances_closest = pickle.load(open(f'/Users/adelielouet/Documents/science/dd_proj/mass_producing_idp_simulations/protein_simualtions/output_post_processed/abeta/{protein_name}/distances/all_concatenated_1.xtc/d_24_t_closest.pkl', 'rb'))
            w_com =  0.3
            w_closest =  0.7
            w_eq=[1.0] *len(distances_com)   # this is for systems that don't need to be reweighted, assign each frame a weight of 1.0
            combined_threshold=True
            uplet_type=3
            restart='1'

        elif protein_name == 'ZINC000000052225':            
            pdb = f'/Users/adelielouet/Documents/science/dd_proj/mass_producing_idp_simulations/protein_simualtions/output_post_processed/abeta/{protein_name}/template.pdb'
            label=protein_name
            distances_com = pickle.load(open(f'/Users/adelielouet/Documents/science/dd_proj/mass_producing_idp_simulations/protein_simualtions/output_post_processed/abeta/{protein_name}/distances/all_concatenated_1.xtc/d_24_t_com_avg.pkl', 'rb'))
            distances_closest = pickle.load(open(f'/Users/adelielouet/Documents/science/dd_proj/mass_producing_idp_simulations/protein_simualtions/output_post_processed/abeta/{protein_name}/distances/all_concatenated_1.xtc/d_24_t_closest.pkl', 'rb'))
            w_com =  0.3
            w_closest =  0.7
            w_eq=[1.0] *len(distances_com)   # this is for systems that don't need to be reweighted, assign each frame a weight of 1.0
            combined_threshold=True
            uplet_type=3
            restart='1'

        elif protein_name == 'ZINC000000057966':            
            pdb = f'/Users/adelielouet/Documents/science/dd_proj/mass_producing_idp_simulations/protein_simualtions/output_post_processed/abeta/{protein_name}/template.pdb'
            label=protein_name
            distances_com = pickle.load(open(f'/Users/adelielouet/Documents/science/dd_proj/mass_producing_idp_simulations/protein_simualtions/output_post_processed/abeta/{protein_name}/distances/all_concatenated_1.xtc/d_24_t_com_avg.pkl', 'rb'))
            distances_closest = pickle.load(open(f'/Users/adelielouet/Documents/science/dd_proj/mass_producing_idp_simulations/protein_simualtions/output_post_processed/abeta/{protein_name}/distances/all_concatenated_1.xtc/d_24_t_closest.pkl', 'rb'))
            w_com =  0.3
            w_closest =  0.7
            w_eq=[1.0] *len(distances_com)   # this is for systems that don't need to be reweighted, assign each frame a weight of 1.0
            combined_threshold=True
            uplet_type=3
            restart='1'

        # elif protein_name == 'abeta_fpocket':
        #     pdb = '/Users/adelielouet/Documents/science/dd_proj/abeta_ph_simulations/ph5_278k/Complex_AB.pdb'
        #     label='abeta_fpocket'
        #     distances_com = pickle.load(open(f'/Users/adelielouet/Documents/science/dd_proj/fpocket_idrome/testing_w_abeta/distances/d_24_t.pkl', 'rb'))
        #     distances_closest = pickle.load(open(f'/Users/adelielouet/Documents/science/dd_proj/fpocket_idrome/testing_w_abeta/distances/d_24_t_closest.pkl', 'rb'))
        #     w_com =  0.3
        #     w_closest =  0.7
        #     w_eq=[1.0] *len(distances_com)   # this is for systems that don't need to be reweighted, assign each frame a weight of 1.0
        #     combined_threshold=True
        #     uplet_type=3
        else:
            print(f"Skipping {protein_name}: configuration not defined.")
            continue
        
        
        # def trim(distances,third=1):
        #     w_split=np.array_split(distances_com,16)
        #     second_third_all_reps=[]
        #     third_third_all_reps=[]

        #     for chunk in w_split:
        #         trimmed=np.array_split(chunk,3)
        #         second_third=trimmed[third]
        #         # third_third=trimmed[2]
        #         second_third_all_reps.append(second_third)
        #         # third_third_all_reps.append(third_third)

        #     second_third_all_reps_w_trimmed = np.concatenate(second_third_all_reps)
        #     return(second_third_all_reps_w_trimmed)

        def trim(distances, third=2):
            w_split = np.array_split(distances, 48)
            second_third_all_reps = []

            for chunk in w_split:
                trimmed = np.array_split(chunk, 3)
                second_third = trimmed[third]
                second_third_all_reps.append(second_third)

            return np.concatenate(second_third_all_reps)
            # third_third_all_reps_w_trimmed = np.concatenate(third_third_all_reps)

        # === Create directories ===
        os.makedirs(f"{output_dir}/msm_output/{protein_name}", exist_ok=True)

        print(len(distances_closest))
        # === Population at equilibrium ===
        x_normed, filtered_keys,Kd_kinetic_weighted, Kd_pop_weighted,transition_matrix = transition_matrix_custom(
            pdb, distances_com, distances_closest, w_file=None, n_reps=None, 
            trim_fraction=None, combined_threshold=combined_threshold, w_com=w_com, w_closest=w_closest,uplet_type=uplet_type
        )

        equilibrium_matrix, P_eq, P_eq_keys = solving_states_at_equilirum(x_normed, filtered_keys)
        dictionary_transitions_sorted, filtered_merged_output = kd_dictionary(x_normed, filtered_keys,transition_matrix) # rather than x_normed,filtered_keys

        # === Clustered Uplets ===
        if protein_name.split('_')[0] == 'alpha':
            resolution=2

        elif protein_name.split('_')[0] == 'medin':
            resolution=3

        elif protein_name.split('_')[0] == 'abeta':
            resolution=3

        elif protein_name.split('_')[0] == 'ph':
            resolution=3
        else:
            resolution=2
        parts, G, pos, values, communities,betCent = network_graph_microstates(filtered_merged_output,resolution=resolution)

        # === Microstates plot ===
        # if protein_name.split('_')[0] == 'alpha':
        #     node_size = [(v * 500) for v in betCent.values()]
        #     cmap = LinearSegmentedColormap.from_list("alpha_purple", ["#cab4f2", "#5e3c99"])  # light → dark purple

        # elif protein_name.split('_')[0] == 'medin':
        #     node_size = [(v * 500)**2 for v in betCent.values()]
        #     cmap = LinearSegmentedColormap.from_list("medin_green", ["#d9f0d3", "#1b7837"])  # light → dark green

        # elif protein_name.split('_')[0] == 'abeta':
        #     node_size = [(v * 500)**2 for v in betCent.values()]
        #     cmap = LinearSegmentedColormap.from_list("abeta_blue", ["#d1e5f0", "#2166ac"])  # light → dark blue

        # elif protein_name.split('_')[0] == 'ph':
        #     node_size = [(v * 500)**2 for v in betCent.values()]
        #     cmap = LinearSegmentedColormap.from_list("abeta_blue", ["#d1e5f0", "#ac6721"])  # light → dark blue

        # === Microstates plot ===
        if protein_name.split('_')[0] == 'alpha':
            node_size = [(v * 500) for v in betCent.values()]
            cmap = LinearSegmentedColormap.from_list(
                "alpha_violet_gray",
                ["#efedf5", "#756bb1"]  # very light violet → muted violet-gray
            )

        elif protein_name.split('_')[0] == 'medin':
            node_size = [(v * 500)**2 for v in betCent.values()]
            cmap = LinearSegmentedColormap.from_list(
                "medin_green_gray",
                ["#edf8f2", "#5a8f7b"]  # pale green-gray → desaturated teal-green
            )

        elif protein_name.split('_')[0] == 'abeta':
            node_size = [(v * 500)**2 for v in betCent.values()]
            cmap = LinearSegmentedColormap.from_list(
                "abeta_slate_blue",
                ["#e6eef6", "#4f6d8a"]  # light slate → restrained blue-gray
            )

        elif protein_name.split('_')[0] == 'ph':
            node_size = [(v * 500)**2 for v in betCent.values()]
            cmap = LinearSegmentedColormap.from_list(
                "ph_warm_gray",
                ["#f0f0f0", "#4a4a4a"]  # light neutral → charcoal
            )
        else:
            node_size = [(v * 500)**2 for v in betCent.values()]
            cmap = LinearSegmentedColormap.from_list(
                "abeta_slate_blue",
                ["#e6eef6", "#4f6d8a"]  # light slate → restrained blue-gray
            )

        norm = plt.Normalize(vmin=min(values), vmax=max(values))
        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        for spine in ax.spines.values(): spine.set_visible(False)
        nx.draw_networkx_nodes(G, pos=pos, node_color=[cmap(norm(v)) for v in values], node_size=node_size, alpha=0.9, linewidths=0.3)
        nx.draw_networkx_edges(G, pos=pos, alpha=0.05, width=0.5)
        plt.axis("off")
        # plt.figtext(0.05, 0.93, f"Kd from populations: {Kd_pop_weighted:.3e} M")
        # plt.figtext(0.05, 0.87, f"Kd from kinetics: {Kd_kinetic_weighted:.3e} M")
        #plt.figtext(0.05, 0.93, fr"$K_d^{{\mathrm{{pop}}}}$: {Kd_pop_weighted:.3e} M",fontsize=17)
        plt.figtext(0.05, 0.87, fr"$K_d^{{\mathrm{{kinetic}}}}$: {Kd_kinetic_weighted:.3e} M",fontsize=17)

        plt.tight_layout()
        plt.title(f"{protein_name} Microstates Network Graph", fontsize=14, weight="bold")

        # plt.savefig(f"/Users/adelielouet/Documents/science/dd_proj/msm_full_model_paper/pickled_files/{protein_name}/figures/microstates_triplet.pdf", format='pdf')
        plt.show()
        plt.close()

        # === Macrostates plot ===
        kd_centrality_ordered, inv_map, inv_map_vals, pos = network_graph_macrostates(parts, P_eq_keys, communities, G)

    #    color_cycle = itertools.cycle([plt.cm.Purples(i) for i in np.linspace(0.3, 1, len(inv_map_vals))])
        color_cycle = itertools.cycle([cmap(i) for i in np.linspace(0.3, 1, len(inv_map_vals))])

        plt.figure(figsize=(8, 8))
        for nodes,color in zip(inv_map_vals,color_cycle):
            kd = [P_eq_keys[node] for node in nodes]
            sizes_shared = [sum(kd)] * len(nodes)
            sigmoid_values = [sigmoid_magnification(val, 1000, 200, 0.025) for val in sizes_shared]
            nx.draw_networkx_nodes(
                G,
                pos=pos,
                nodelist=nodes,
                node_color=[color],
                node_size=sigmoid_values,
                alpha=0.85,
                edgecolors="none"
            )

        G.remove_edges_from(list(nx.selfloop_edges(G)))
        nx.draw_networkx_edges(
            G,
            pos=pos,
            alpha=0.15,
            edge_color="gray",
            width=0.6,
            connectionstyle="arc3,rad=0.1"
        )

        ax = plt.gca()
        ax.set_axis_off()
        for spine in ax.spines.values():
            spine.set_visible(False)
        plt.title(f"{protein_name} Macrostates Network Graph", fontsize=14, weight="bold")
        # plt.savefig(f"/Users/adelielouet/Documents/science/dd_proj/msm_full_model_paper/pickled_files/{protein_name}/figures/macrostatess_triplet.pdf", format='pdf')
        plt.show()

        plot_data.append({"System":protein_name,"Uplet": uplet_type,"Threshold": combined_threshold,"Number": len(filtered_keys)})

        # # === Save outputs ===
        with open(f"{output_dir}/msm_output/{protein_name}/p_eq_keys_trip_{restart}.pckl", "wb") as f:
            pickle.dump(P_eq_keys, f)
        # with open(f"{output_dir}/msm_output/{protein_name}/filtered_merged_trip.pckl", "wb") as f:
        #     pickle.dump(filtered_merged_output, f)
        # with open(f"{output_dir}/msm_output/{protein_name}/dictionary_transitions_trip.pckl", "wb") as f:
        #     pickle.dump(dictionary_transitions_sorted, f)

        print(f"Finished {protein_name} successfully!\n")

    except Exception as e:
        print(f"Error processing {protein_name}: {e}")
        continue


