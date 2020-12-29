from cxsParser import HirshfeldSurface as HS
import os
import trimesh
import numpy as np
import pandas as pd
from tqdm import tqdm

cxs_dir = os.path.join('data/SA_cxs_12_21')
label_file = os.path.join('data', 'metavalent_labels_12_21.csv')
folderlist = os.listdir(cxs_dir)

labels = pd.read_csv(label_file)
labels = labels[['Name', 'Category']]
output_folder = os.path.join('outputs', 'descriptors_12_21')
# if not os.path.isdir(output_folder):     # If the result folder doesn't exist then create one
#     os.mkdir(output_folder)

descs = ['shape_idx', 'curvedness', 'd_norm']
stats = ['mean', 'std']
attr = ['area', 'volume']

si_bins = 4
curvedness_bins = 5

# cxs_names = []
# area = []
# volume = []
# dnorm_mean = []
# si_mean = []
# curv_mean = []
# dnorm_std = []
# si_std = []
# curv_std = []
# globularity = []
surfaces = []

trash = ['CePd3_mp-2092', 'Ru_mp-8639', 'NaF_mp-682']

for f in tqdm(folderlist):

    if f in trash:
        continue

    cxs_files = os.listdir(os.path.join(cxs_dir, f))

    if not len(cxs_files):
        trash.append(f)
        continue

    filename = os.path.join(cxs_dir, f, cxs_files[0])
    hs = HS(file_path=filename)
    atoms = hs.unit_cell.Atom.unique().tolist()   # Extracting the Atoms in a compound
    n_atom = len(atoms)

    # Initialize the properties
    surf_properties = {}
    surf_properties['Name'] = f
    for i in range(3):
        for desc in descs:
            for stat in stats:
                key = desc + '_' + stat + '_' + str(i + 1)
                surf_properties[key] = 0

        for att in attr:
            key = att + '_' + str(i + 1)
            surf_properties[key] = 0

        surf_properties['G_' + str(i + 1)] = 0

        # Ranges
        for j in range(si_bins):
            key = 'shape_idx' + '_' + str(i + 1) + '_' + str(j + 1)
            surf_properties[key] = 0
        for j in range(curvedness_bins):
            key = 'curvedness' + '_' + str(i + 1) + '_' + str(j + 1)
            surf_properties[key] = 0

    for cxs in cxs_files:

        # cxs_names.append(cxs.split('.')[0])
        filename = os.path.join(cxs_dir, f, cxs)
        hs = HS(file_path=filename)
        atom_ind = atoms.index(hs.unit_cell.Atom[hs.atoms_inside_surface.Atom_Id[0] - 1])  # Index of atom
        atom_count = hs.unit_cell.Atom.value_counts()[hs.
            unit_cell.Atom[hs.atoms_inside_surface.Atom_Id[0] - 1]]  # How many times it is repeated

        mesh = trimesh.Trimesh(vertices=hs.vtx, faces=hs.idx)
        # mesh_prop = {}
        for att in attr:
            key = att + '_' + str(atom_ind + 1)
            surf_properties[key] = getattr(mesh, att)

        surf_properties['G_' + str(atom_ind + 1)] = ((36 * np.pi * (mesh.volume ** 2)) ** (1/3)) / (mesh.area)

        for desc in descs:
            for stat in stats:
                func = getattr(np, stat)
                key = desc + '_' + stat + '_' + str(atom_ind + 1)
                surf_properties[key] = surf_properties[key] + func(getattr(hs, desc)) / atom_count

        # Ranging values
        a, _ = np.histogram(hs.shape_idx, range=(-1, 1), bins=si_bins)
        a = a/sum(a)
        for j in range(si_bins):
            key = 'shape_idx' + '_' + str(atom_ind + 1) + '_' + str(j + 1)
            surf_properties[key] = surf_properties[key] + a[j] / atom_count
        a, _ = np.histogram(hs.curvedness, range=(-4, 1), bins=curvedness_bins)
        a = a / sum(a)
        for j in range(curvedness_bins):
            key = 'curvedness' + '_' + str(atom_ind + 1) + '_' + str(j + 1)
            surf_properties[key] = surf_properties[key] + a[j] / atom_count

    surfaces.append(surf_properties)
        # area.append(mesh.area)
        # volume.append(mesh.volume)
        # globularity.append(((36 * np.pi * (mesh.volume ** 2)) ** (1/3)) / (mesh.area))
        #
        # dnorm_mean.append(np.mean(hs.d_norm))
        # dnorm_std.append(np.std(hs.d_norm))
        # si_mean.append(np.mean(hs.shape_idx))
        # si_std.append(np.std(hs.shape_idx))
        # curv_mean.append(np.mean(hs.curvedness))
        # curv_std.append(np.std(hs.curvedness))

df = pd.DataFrame(surfaces)
df = df.merge(labels, on='Name')

import seaborn as sns
import matplotlib.pyplot as plt

i = 0

for desc in descs:
    for stat in stats:
        key = desc + '_' + stat + '_' + str(i + 1)
        sns.displot(data=df, x=key, hue='Category', kind='kde')
        plt.savefig('figs/figs_12_28/' + key)
        plt.close()
for att in attr:
    key = att + '_' + str(i + 1)
    sns.displot(data=df, x=key, hue='Category', kind='kde')
    plt.savefig('figs/figs_12_28/' + key)
    plt.close()

for j in range(si_bins):
    key = 'shape_idx' + '_' + str(i + 1) + '_' + str(j + 1)
    sns.histplot(data=df, x=key, hue='Category')
    plt.savefig('figs/figs_12_28/' + key)
    plt.close()

# df = pd.DataFrame({'Area': area, 'Volume': volume,
#                    'dnorm_mean': dnorm_mean, 'dnorm_std': dnorm_std,
#                    'si_mean': si_mean, 'si_std': si_std, 'globularity': globularity,
#                    'curv_mean': curv_mean, 'curv_std': curv_std}, index=cxs_names)

df.to_csv(os.path.join(output_folder, 'cxs_descriptors_2.csv'))