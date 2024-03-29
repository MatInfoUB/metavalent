from funcs import find_tag, keys_vals, fingerprint, \
    plot_signature1, plot_signature2
from os.path import dirname, join
import pandas as pd

class HirshfeldSurface:
    """
    A class to read the output file of Hirshfeld calculation from CrystalXplore.
    """

    def __init__(self, file_path):
        self.abs_path = file_path
        self.tag = find_tag(file_path)  # fetch sys_name from file path
        # value:                        # key:
        self.vtx = []  # begin vertices ##
        self.idx = []  # begin indices ##
        # self.vtx_norms = []             # begin vertex_normals ##
        self.d_i = []  # begin d_i ##
        self.d_i_face_atoms = []
        self.d_e = []  # begin d_e ##
        self.d_e_face_atoms = []
        # self.d_norm_i = []              # begin d_norm_i ##
        # self.d_norm_e = []              # begin d_norm_e ##
        self.d_norm = []  # begin d_norm ##
        self.unit_cell = []
        self.atoms_inside_surface = []
        self.atoms_outside_surface = []
        self.shape_idx = []             # begin shape_index ##
        self.curvedness = []            # begin curvedness ##
        # self.atoms_in = []              # begin atoms_inside_surface ##
        # self.atoms_out = []             # begin atoms_outside_surface ##

        with open(file_path, "r") as fi:
            for line in fi:
                words = [word for word in line.split(' ')]
                try:
                    key = words[0] + words[1]
                except IndexError:
                    key = "null"
                if key == "beginunit_cell":
                    keys_vals(self.unit_cell, fi, int(words[-1]), output_type='str')
                    df = pd.DataFrame(columns=['Serial', 'Atom', 'x', 'y', 'z',
                                               'no', 'd'])
                    for i, unit in enumerate(self.unit_cell):
                        df.loc[i] = unit.strip().split(' ')
                    self.unit_cell = df
                elif key == "beginvertices":
                    keys_vals(self.vtx, fi, int(words[-1]))
                elif key == 'beginatoms_outside_surface':
                    keys_vals(self.atoms_outside_surface, fi, int(words[-1]))
                    df = pd.DataFrame(columns=['Atom_Id', 'x', 'y', 'z'])
                    for i, unit in enumerate(self.atoms_outside_surface):
                        df.loc[i] = unit
                    self.atoms_outside_surface = df
                elif key == 'beginatoms_inside_surface':
                    keys_vals(self.atoms_inside_surface, fi, int(words[-1]))
                    df = pd.DataFrame(columns=['Atom_Id', 'x', 'y', 'z'])
                    for i, unit in enumerate(self.atoms_inside_surface):
                        df.loc[i] = unit
                    self.atoms_inside_surface = df
                elif key == "beginindices":
                    keys_vals(self.idx, fi, int(words[-1]))
                # elif key == "beginvertex_normals":
                #     keys_vals(self.vtx_norms, fi, int(words[-1]))
                elif key == "begind_i":
                    keys_vals(self.d_i, fi, int(words[-1]))
                elif key == "begind_e":
                    keys_vals(self.d_e, fi, int(words[-1]))
                # elif key == "begind_norm_i":
                #     keys_vals(self.d_norm_i, fi, int(words[-1]))
                # elif key == "begind_norm_e":
                #     keys_vals(self.d_norm_e, fi, int(words[-1]))
                elif key == "begind_norm":
                    keys_vals(self.d_norm, fi, int(words[-1]))
                elif key == 'begind_i_face_atoms':
                    keys_vals(self.d_i_face_atoms, fi, int(words[-1]))
                elif key == 'begind_e_face_atoms':
                    keys_vals(self.d_e_face_atoms, fi, int(words[-1]))
                elif key == "beginshape_index":
                    keys_vals(self.shape_idx, fi, int(words[-1]))
                elif key == "begincurvedness":
                    keys_vals(self.curvedness, fi, int(words[-1]))
                # elif key == "beginatoms_inside_surface":
                #     keys_vals(self.atoms_in, fi, int(words[-1]))
                # elif key == "beginatoms_outside_surface":
                #     keys_vals(self.atoms_out, fi, int(words[-1]))
                else:
                    continue

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del self

    def atoms_list(self):

        return list(self.unit_cell['Atom'].unique())

    def fingerprint_(self):
        return fingerprint(self.d_i, self.d_e, 101)

    def signature_(self):
        plot_signature1(self.d_i, self.d_e, self.tag, 251,
                        file_path=join(dirname(self.abs_path), self.tag + "_hi.png"))

    def signature(self):
        plot_signature2(self.d_i, self.d_e, self.tag, 101,
                        file_path=join(dirname(self.abs_path), self.tag + "_lo.png"))


