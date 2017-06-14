import numpy as np
import random

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
