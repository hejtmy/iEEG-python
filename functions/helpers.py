import numpy as np
import random
from math import sqrt


def remove_unnamed(pd_dataframe):
    cols = [c for c in pd_dataframe.columns if 'Unnamed' not in c]
    return pd_dataframe[cols]


#returns indices of duplicates in array
def find_duplicates(arr):
    vals, ids, count = np.unique(arr, return_index=True, return_counts=True)
    return ids[count > 1]


def random_matplotlib_color():
    r = lambda: random.randint(0,255)
    rgb = (r()/255, r()/255,r()/255)
    return tuple(rgb)


# helper for matplotlib lines and iterating through generation of graphs
def int_to_linestyle(num):
    mapp = {
        0 : '-',
        1 : '--',
        2 : '-.',
        3 : ':',
        4 : 'None'
    }
    if num > 4: return '-'
    return mapp[num]


# Returns "logical" nrow, ncol layout of N elements.
# Gravitates towards squares if possible
def nrow_ncol(n_elements, nrow = 0, ncol = 0):
    if nrow > 0 & ncol > 0:
        return nrow, ncol
    if ncol > 0:
        return int(np.ceil(n_elements / ncol)), ncol
    if nrow > 0:
        return nrow, int(np.ceil(n_elements / nrow))
    # we try to fit in a smallest box. Not perfect, but will do
    square_n = round(sqrt(n_elements))
    return square_n, int(np.ceil(n_elements / square_n))


def layout_position(idx, nrow, ncol):
    i_row = idx % nrow
    i_col = (idx - i_row) / ncol
    return int(i_row), int(i_col)
