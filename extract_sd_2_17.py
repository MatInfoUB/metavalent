from cxsParser import HirshfeldSurface as HS
import os
import numpy as np
from tqdm import tqdm

datadir = 'data/org_inorg_cxs_2D'
cxslist = os.listdir(datadir)
cxslist = [os.path.join(datadir, cxs) for cxs in cxslist]
cxslist.sort()

descs = ['shape_idx', 'curvedness', 'd_norm']
stats = ['mean', 'std']
attr = ['area', 'volume']

max_inorganic = max([len([files for files in os.listdir(cxslist_i) if
                          'inorganic' in files]) for cxslist_i in cxslist])
max_organic = max([len([files for files in os.listdir(cxslist_i) if
                          'inorganic' not in files]) for cxslist_i in cxslist])
# inorganic_names = [files.split('_')[1] + files.split('_')[2]
#            for files in os.listdir(cxslist[0]) if 'inorganic' in files]
# inorganic_names = [c.split('.')[0] for c in inorganic_names]
# inorganic_names.sort()
inorganic_names = ['inorganic_' + str(i) for i in range(max_inorganic)]
organic_names = ['organic_' + str(i) for i in range(max_organic)]

columns = []
for col in inorganic_names:
    for desc in descs:
        for stat in stats:
            columns.append(col + '_' + desc + '_' + stat)

for col in organic_names:
    for desc in descs:
        for stat in stats:
            columns.append(col + '_' + desc + '_' + stat)

inds = []
surfaces = []
for cxsfolder in tqdm(cxslist[:5]):
    ind = cxsfolder.split('\\')[-1]
    inds.append(ind)
    cxsfiles = os.listdir(cxsfolder)
    cxsfiles.sort()
    inorganics = len([files for files in cxsfiles if
                          'inorganic' in files])
    inorganic_names = ['inorganic_' + str(i) for i in range(inorganics)]
    inorganics = [os.path.join(cxsfolder, ind + '_inorganic_' + str(i) + '.cxs')
                  for i in range(inorganics)]

    surface_prop = {}
    for c_file, inorg in zip(inorganics, inorganic_names):
        hs = HS(c_file)
        for desc in descs:
            for stat in stats:
                func = getattr(np, stat)
                surface_prop[inorg+'_' + desc + '_' + stat] = func(getattr(hs, desc))

    organics = len([files for files in cxsfiles if
                      'inorganic' not in files])
    organic_names = ['organic_' + str(i) for i in range(organics)]
    organics = [os.path.join(cxsfolder, ind + '_organic_' + str(i) + '.cxs')
                for i in range(organics)]
    for c_file, inorg in zip(organics, organic_names):
        hs = HS(c_file)
        for desc in descs:
            for stat in stats:
                func = getattr(np, stat)
                surface_prop[inorg + '_' + desc + '_' + stat] = func(getattr(hs, desc))
    surfaces.append(surface_prop)

import pandas as pd
df = pd.DataFrame(surfaces)