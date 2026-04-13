import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis import contacts
import mdtraj as md
import itertools
import glob
import os

### The following is fitted for alpha_synuclein from DE Shaw. Made to handle dcd files
# xtc_files=glob.glob("/Users/adelielouet/Documents/science/shared_files/maria/trajectories/G5_new_proto/*xtc")
# gro_files=glob.glob("/Users/adelielouet/Documents/science/shared_files/maria/trajectories/G5_new_proto/*gro")

# xtc_files=("/Users/adelielouet/Documents/science/medin/urea/urea_post_run/concatenate_urea.xtc","/Users/adelielouet/Documents/science/shared_files/maria/trajectories/G5_new_proto/traj_all-skip-10-noW-PBC_1.xtc")
# gro_files=("/Users/adelielouet/Documents/science/medin/urea/urea_post_run/Complex_box.gro","/Users/adelielouet/Documents/science/shared_files/maria/trajectories/G5_new_proto/complex_noW_4.gro ")
xtc_files=('/chem2/scratch/adlouet/abeta_ph_simulations/ph5_278k/prod/pbc_mol.xtc','/chem2/scratch/adlouet/abeta_ph_simulations/ph5_298k/prod/pbc_mol.xtc','/chem2/scratch/adlouet/abeta_ph_simulations/ph7_278k/prod/pbc_mol.xtc')
gro_files=('/chem2/scratch/adlouet/abeta_ph_simulations/ph5_278k/prep/Complex_box.gro','/chem2/scratch/adlouet/abeta_ph_simulations/ph5_278k/prep/Complex_box.gro','/chem2/scratch/adlouet/abeta_ph_simulations/ph5_278k/prep/Complex_box.gro')

labels=("ph5_278k","ph5_298k","ph7_278k")

for xtc, gro,label in zip(xtc_files[1],gro_files[1],labels[1]):
    traj = md.load(xtc, top=gro)
    topology = traj.topology
    protein = traj.top.select('protein')
    ligand_indices = topology.select('not protein')
   # label=(gro.split('/'))[-2]
    # ### CALCULATING THE AVERAGE BETWEEN COM OF LIGAND AND COM OF EACH RESIUDUE && TAKING SMALLEST DISTANCE BETWEEN LIGAND AND PROTEIN AA

    com_calcualtion=[]
    clostest_calcualtion=[]
    for resid in range(0,traj.topology.n_residues-1): #-1 beacus -1 is ligand
        res_x=topology.select(f'resid {resid}')
        pairs = list(itertools.product(res_x, ligand_indices))
        distances=md.compute_distances(traj, pairs, periodic=True, opt=True)
        com_calcualtion.append((np.average(distances,axis=1)).tolist())
        clostest_calcualtion.append(np.min(distances,axis=1).tolist())

    d_24_t=np.reshape(com_calcualtion,((traj.topology.n_residues-1), len(traj))).T
    d_24_t_closest=np.reshape(clostest_calcualtion,((traj.topology.n_residues-1), len(traj))).T
    
    os.makedirs(f"/chem2/scratch/adlouet/abeta_ph_simulations/pickled_files/{label}/distances", exist_ok=True) 

    with open(f'/chem2/scratch/adlouet/abeta_ph_simulations/pickled_files/{label}/distances/d_24_t_closest.pkl', 'wb') as file:
            pickle.dump(d_24_t_closest, file)

    with open(f'/chem2/scratch/adlouet/abeta_ph_simulations/pickled_files/{label}/distances/d_24_t_com_avg.pkl', 'wb') as file:
            pickle.dump(d_24_t, file)

