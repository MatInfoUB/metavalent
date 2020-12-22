from cxsParser import HirshfeldSurface as HS
import os
import trimesh
import numpy as np
import pandas as pd

cxs_dir = os.path.join('data/SA_cxs_old')
folderlist = os.listdir(cxs_dir)

output_folder = os.path.join('outputs', 'descriptors_12_21')
if not os.path.isdir(output_folder):
    os.mkdir(output_folder)

cxs_names = []
area = []
volume = []
dnorm_mean = []
si_mean = []
curv_mean = []
dnorm_std = []
si_std = []
curv_std = []

for f in folderlist:
    cxs_files = os.listdir(os.path.join(cxs_dir, f))
    for cxs in cxs_files:
        cxs_names.append(cxs.split('.')[0])
        filename = os.path.join(cxs_dir, f, cxs)
        hs = HS(file_path=filename)
        mesh = trimesh.Trimesh(vertices=hs.vtx, faces=hs.idx)
        area.append(mesh.area)
        volume.append(mesh.volume)

        dnorm_mean.append(np.mean(hs.d_norm))
        dnorm_std.append(np.std(hs.d_norm))
        si_mean.append(np.mean(hs.shape_idx))
        si_std.append(np.std(hs.shape_idx))
        curv_mean.append(np.mean(hs.curvedness))
        curv_std.append(np.std(hs.curvedness))

df = pd.DataFrame({'Area': area, 'Volume': volume,
                   'dnorm_mean': dnorm_mean, 'dnorm_std': dnorm_std,
                   'si_mean': si_mean, 'si_std': si_std,
                   'curv_mean': curv_mean, 'curv_std': curv_std}, index=cxs_names)

df.to_csv(os.path.join(output_folder, 'cxs_descriptors.csv'))