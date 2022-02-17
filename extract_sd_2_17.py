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

inorganic_names = [files.split('_')[1] + files.split('_')[2]
           for files in os.listdir(cxslist[0]) if 'inorganic' in files]
inorganic_names = [c.split('.')[0] for c in inorganic_names]
inorganic_names.sort()

columns = []
for col in inorganic_names:
    for desc in descs:
        for stat in stats:
            columns.append(col+'_'+desc+'_'+stat)

inds = []
surfaces = []
for cxsfolder in tqdm(cxslist):
    ind = cxsfolder.split('/')[-1]
    inds.append(ind)
    cxsfiles = os.listdir(cxsfolder)
    cxsfiles.sort()
    inorganics = [os.path.join(cxsfolder, cxs) for cxs in cxsfiles if 'inorganic' in cxs]
    inorganics.sort()
    surface_prop = {}
    for c_file, inorg in zip(inorganics, inorganic_names):
        hs = HS(c_file)
        for desc in descs:
            for stat in stats:
                func = getattr(np, stat)
                surface_prop[inorg+'_' + desc + '_' + stat] = func(getattr(hs, desc))

    organics = [os.path.join(cxsfolder, cxs) for cxs in cxsfiles if 'inorganic' not in cxs][0]
    hs = HS(organics)
    for desc in descs:
        for stat in stats:
            func = getattr(np, stat)
            surface_prop['organic_' + desc + '_' + stat] = func(getattr(hs, desc))
    surfaces.append(surface_prop)

import pandas as pd
df = pd.DataFrame(surfaces)