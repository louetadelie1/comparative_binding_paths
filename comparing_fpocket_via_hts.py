import glob
import MDAnalysis as mda
from matplotlib import pyplot as plt
import pandas as pd
import re
from collections import Counter
from scipy.spatial import distance_matrix
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
from collections import OrderedDict
import shutil
import smplotlib


def distance_finder(one,two):
    [x1,y1,z1] = one  # first coordinates
    [x2,y2,z2] = two[:3]  # second coordinates
    v=(((x2-x1)**2)+((y2-y1)**2)+((z2-z1)**2))**(1/2)
    answer=(v,two[3])
    return answer

class Centroid_fpocket:
    def __init__(self, apo_pockets,type_mol):
        self.apo_pockets = apo_pockets
        self.type_mol = type_mol
        
    def mda(self):
        apo_pock = mda.Universe(self.apo_pockets)
        select_atom = apo_pock.select_atoms(self.type_mol)
        select_atom_pos = select_atom.atoms.positions
        select_atom_resids = select_atom.resids
        select_atom_resids=select_atom_resids
        select_atom_pos=select_atom_pos
        return (select_atom_resids,select_atom_pos)
        
    def mda_to_df(self):
        select_atom_resids, select_atom_pos = self.mda()
        # print(select_atom_resids)
        result_STP = list(OrderedDict.fromkeys(select_atom_resids))
        data_STP = {'x': select_atom_pos[:, 0], 'y': select_atom_pos[:, 1], 'z': select_atom_pos[:, 2], 'name': select_atom_resids}
        df_STP = pd.DataFrame(data_STP)
        return (df_STP,result_STP)
    
    def center(self, nested_array_list):
        a = np.array(nested_array_list)
        mean = np.mean(a, axis=0)
        return mean[0], mean[1], mean[2]

    def calculate_centroid(self):
        df_STP,result_STP = self.mda_to_df()
        centroid = []
        for l in result_STP:
            STP1 = df_STP[df_STP['name'] == l]
            stp_pos1 = STP1[['x', 'y', 'z']].values
            centroid_point = self.center(stp_pos1)
            a = list(centroid_point)
            a.append(l)
            centroid.append(a)
        return centroid
    
def clean_numpy_list(lst):
    return [item.item() if hasattr(item, "item") else item for item in lst]
    
class Retrive_pocket_residues:
    def __init__(self, single_pocket_path):
        self.single_pocket_path = single_pocket_path
        
    def md_analysis_fix(self,cutoff=55.5):
        liga_centroid_instance = Centroid_fpocket(self.single_pocket_path, 'not protein')
        liga_resid,liga_pos = liga_centroid_instance.mda()      
        
        protein_centroid_instance = Centroid_fpocket(self.single_pocket_path, 'protein')
        protein_resid,protein_pos = protein_centroid_instance.mda()      
        
        #This one does each atom of the ligand * each atom of protein - size of matrix is 2017 (woth 38 nested distance for 38 atoms in ligand) and 2017 atoms in protein
        dist_matrix = distance.cdist(protein_pos, liga_pos, 'euclidean')    
            
        list_prot_resid_distance=[]
        prot_resids=[]
        for i,j in zip((enumerate(dist_matrix)),protein_resid):
            if (min(i[1])) <= cutoff:
                evalue=(j,i[1])
                list_prot_resid_distance.append(evalue)
                prot_resids.append(j)
                
        return list_prot_resid_distance,prot_resids
        
    def return_path_resids(self):
        _, prot_resids = self.md_analysis_fix()
        orga_prot_resids = list(dict.fromkeys(prot_resids))
        complete_list=[(self.single_pocket_path,orga_prot_resids)]
        return complete_list


apo_path=('/Users/adelielouet/Documents/science/dd_proj/fpocket_idrome/fpocket_hts/testing_abeta_subset/apo/')
holo_path=('/Users/adelielouet/Documents/science/dd_proj/fpocket_idrome/fpocket_hts/testing_abeta_subset/holo/')
dock_path=('/Users/adelielouet/Documents/science/dd_proj/fpocket_idrome/fpocket_hts/testing_abeta_subset/dock/frame_pockets/')

rows = []
for pdb_file in range(0,100):
    try:
        apo_pockets=apo_path+(f'pair{pdb_file}_out/pair{pdb_file}_out.pdb')
        get_apo_centroids = Centroid_fpocket(apo_pockets,'resname STP')
        apo_centroids = get_apo_centroids.calculate_centroid()
        stp_cooridnates=[clean_numpy_list(x) for x in apo_centroids]
        # comapring to og
        holo_pdb=holo_path+(f'pair{pdb_file}.pdb')
        get_holo_centroids = Centroid_fpocket(holo_pdb,'not protein')
        ligand_centroids = get_holo_centroids.calculate_centroid()
        ligand_coordinates=[clean_numpy_list(x) for x in ligand_centroids]
        x1_coords, y1_coords, z1_coords,liga_resid=ligand_coordinates[0]
        for stp in stp_cooridnates:
            x2_coords,y2_coords,z2_coords,stp_number=stp
            p1 = np.array([x1_coords, y1_coords, z1_coords])
            p2 = np.array([x2_coords, y2_coords, z2_coords])

            [x1,y1,z1] = p1  # first coordinates
            [x2,y2,z2] = p2
            v=(((x2-x1)**2)+((y2-y1)**2)+((z2-z1)**2))**(1/2)


            squared_dist = np.sum((p1-p2)**2, axis=0)
            dist = np.sqrt(squared_dist)

            stp_pocket=[]
            results=glob.glob(dock_path+f"pair{pdb_file}/stp_{stp[3]}/*_docked_results")
            results_10_random=np.random.choice(results, 60)
            for res in results_10_random:
                f=float(list(filter(None, linecache.getline(res, 2).split(' ')))[-3])
                stp_pocket.append(f)
            avg=np.mean(stp_pocket)
            # print(pdb_file,stp[3],avg,dist)

            rows.append({
                "pdb": pdb_file,
                "stp": stp_number,
                "avg": avg,
                "dist": dist
            })
    except:
        pass
df = pd.DataFrame(rows)

df_wait = df.dropna(axis=0)

df_wait["true_rank"] = df_wait.groupby("pdb")["dist"].rank(ascending=True)
df_wait["my_pred_rank"] = df_wait.groupby("pdb")["avg"].rank(ascending=True)
df_wait["fpock_pred_rank"] = df_wait.groupby("pdb")["stp"].rank(ascending=False)

df_wait["true_best"] = df_wait.groupby("pdb")["true_rank"].transform(lambda x: x == 1)
df_wait["my_best"]   = df_wait.groupby("pdb")["my_pred_rank"].transform(lambda x: x == 1)
df_wait["fp_best"]   = df_wait.groupby("pdb")["fpock_pred_rank"].transform(lambda x: x == 1)

# Per-PDB correctness
my_correct  = (df_wait["true_best"] & df_wait["my_best"]).groupby(df_wait["pdb"]).max()
fp_correct  = (df_wait["true_best"] & df_wait["fp_best"]).groupby(df_wait["pdb"]).max()

my_acc = my_correct.mean()
fp_acc = fp_correct.mean()

print("My Method Top-1 Accuracy:", my_acc)
print("FPocket Top-1 Accuracy:", fp_acc)

plt.figure(figsize=(4.2, 4.2))

methods = ["Ensemble Docking", "FPocket"]
values = [my_acc, fp_acc]

bars = plt.bar(
    methods,
    values,
    width=0.55,
    color=["#4E79A7", "#8F77B5"],  # muted blue → soft purple
    edgecolor="black",
    linewidth=1.0
)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 0.025,
        f"{height:.2f}",
        ha="center",
        va="bottom",
        fontsize=11
    )

plt.ylabel("Accuracy", fontsize=13)
plt.ylim(0, 1.05)

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.tick_params(axis="x", labelsize=12)
ax.tick_params(axis="y", labelsize=12)

plt.title(
    "Recovery of the True Pocket",
    fontsize=14,
    pad=10
)

plt.tight_layout()
plt.show()
