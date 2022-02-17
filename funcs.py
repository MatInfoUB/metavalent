from os.path import basename, join
from time import time

import numpy as np
from sklearn.metrics import pairwise
import matplotlib as mpl
import colorsys

import trimesh

#mpl.use('agg')
import matplotlib.pyplot as plt
from matplotlib import colors, cm
from matplotlib.colors import LogNorm


plt.style.use('seaborn-poster')


# mpl.rcParams.update({'axes.titlesize': 28})
# mpl.rcParams.update({'axes.labelsize': 24})
# mpl.rcParams.update({'xtick.labelsize': 22})
# mpl.rcParams.update({'ytick.labelsize': 22})


RWB = np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [1.0, 0.0, 0.0]])
RGB = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

COLORS = {
    'd_norm': RWB,
    'd_i': RGB,
    'd_e': RGB,
    'curvedness': RGB,
    'shape_index': RGB,
    'deformation_density': RWB,
    'electron_density': RWB,
    'promolecule_density': RWB,
    'electric_potential': RWB,
    'orbital': RWB,
}

def find_tag(abs_path):
    return basename(abs_path).strip().split(".")[0]


def str_to_num_list(str_list):
    str_list[::] = list(map(str.strip, str_list))
    # sum([item.is_integer() for item in str_list[0:3]]) == len(str_list[0:3]):
    # sum([item.is_integer() for item in str_list[0]]) == len(str_list[0]):

    if '.' in str_list[0]:
        try:
            str_list[::] = list(map(float, str_list))
        except ValueError:
            str_list[::] = [list(map(float, (str_.split()))) for str_ in str_list]
    else:
        try:
            str_list[::] = list(map(int, str_list))
        except ValueError:
            str_list[::] = [list(map(int, (str_.split()))) for str_ in str_list]


def keys_vals(str_list, f, str_list_size, output_type=None):
    for _ in range(str_list_size):
        str_list.append(next(f))
    if output_type == 'str':
        return str_list
    if next(f).split()[0] != "end":
        raise ValueError("The number of vertice in the file did not\
        match the atcual number vertice in the file: di/de values.")
    else:
        pass
    str_to_num_list(str_list)


def fingerprint(d_i, d_e, bin_num):
    x = np.asarray(d_i, dtype='float64')
    y = np.asarray(d_e, dtype='float64')
    edg_min, edg_max = 0, 4
    # edg_min, edg_max = 0, 3
    bin_arr = np.linspace(edg_min, edg_max, bin_num)
    hist, xedges, yedges = np.histogram2d(x=x, y=y, bins=(bin_arr, bin_arr), normed=True)
    return hist.T


def plot_signature1(d_i, d_e, plot_title, bin_num, file_path):
    hist = fingerprint(d_i, d_e, bin_num)
    edg_min, edg_max = 0, 4
    # edg_min, edg_max = 0, 3
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_title(plot_title)
    ax.set_xlabel('$d_i$')
    ax.set_ylabel('$d_e$')
    plt.xlim(edg_min, edg_max)
    plt.ylim(edg_min, edg_max)
    plt.xticks([0, 1, 2, 3, 4])
    plt.yticks([0, 1, 2, 3, 4])
    # plt.xticks([0, 1, 2, 3])
    # plt.yticks([0, 1, 2, 3])
    plt.imshow(hist, plt.get_cmap('jet'), norm=LogNorm(), aspect='equal', interpolation='nearest',
               origin='lower', extent=[edg_min, edg_max, edg_min, edg_max])
    plt.gcf()
    plt.savefig(file_path, bbox_inches='tight', dpi=600)


def plot_signature2(d_i, d_e, plot_title, bin_num, file_path):
    x = np.asarray(d_i, dtype='float64')
    y = np.asarray(d_e, dtype='float64')
    edg_min, edg_max = 0, 4
    # edg_min, edg_max = 0, 3
    # prepare canvas
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_title(plot_title)
    ax.set_xlabel('$d_i$')
    ax.set_ylabel('$d_e$')
    ax.hist2d(x, y, bins=np.linspace(edg_min, edg_max, bin_num),
              cmap=plt.get_cmap('jet'), norm=LogNorm())
    plt.xlim(edg_min, edg_max)
    plt.ylim(edg_min, edg_max)
    plt.xticks([0, 1, 2, 3, 4])
    plt.yticks([0, 1, 2, 3, 4])
    # plt.xticks([0, 1, 2, 3])
    # plt.yticks([0, 1, 2, 3])
    plt.gcf()
    plt.savefig(file_path, bbox_inches='tight', dpi=600)


# Function: prepare colorScales for Hirshfeld Surface
def surface_color_scale(surface_property, surfaceIdx, color_scheme):  ## 'bwr_r'
    color_scale = []
    color_range = [np.average([surface_property[i] for i in idx]) for idx in surfaceIdx]
    color_map = mpl.cm.get_cmap(color_scheme)
    color_norm = mpl.colors.Normalize(min(color_range), max(color_range))
    color_scale += [color_map(color_norm(value)) for value in color_range]
    return color_scale
    

def clamp(v, vmin, vmax):
    return max(min(v, vmax), vmin)

def hmap(value, vmin, vmax, reverse, hmin, hmax):
    newval = clamp(value, vmin, vmax)

    range_ratio = 0.0
    r = vmax - vmin
    if r > 1e-6:
        range_ratio = (hmax - hmin) / r

    if reverse:
        h = 1.0 - range_ratio * (newval - vmin)
    else:
        h = range_ratio * (newval - vmin)

    return colorsys.hsv_to_rgb(clamp(h, hmin, hmax), 1.0, 1.0)

def pairwise_distances_(x):
    return pairwise.pairwise_distances(X=x, Y=None, metric='euclidean')

def colormap(prop, scheme='d_norm', minval=None, maxval=None):
    vmin = minval if minval else prop.min()
    vmax = maxval if maxval else prop.max()

    colors = np.zeros((prop.shape[0], 3), dtype=np.float32)

    if scheme in {'d_norm', 'electric_potential', 'orbital',
            'deformation_density', 'electron_density'}:
        start = COLORS[scheme][0,:]
        mid = COLORS[scheme][1,:]
        end = COLORS[scheme][2,:]
        for i, value in enumerate(prop):
            colors[i,:] = cmap(value, vmin, vmax, start, mid, end)
    else:
        hmax = 240.0/359.0
        hmin = 0.0
        for i, value in enumerate(prop):
            colors[i,:] = hmap(value, vmin, vmax, False, hmin, hmax)
    return colors

def cmap(value, vmin, vmax, start, mid, end):
    if value < 0:
        factor = 1.0 - value / vmin
        color = start
    else:
        factor = 1.0 - value / vmax
        color = end
    res = color + (mid - color) * factor
    return res


