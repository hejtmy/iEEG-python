import numpy as np

def remove_unnamed(pd_dataframe):
    cols = [c for c in pd_dataframe.columns if 'Unnamed' not in c]
    return pd_dataframe[cols]

#returns indices of duplicates in array
def find_duplicates(arr):
    vals, ids, count = np.unique(arr, return_index=True, return_counts=True)
    return ids[count > 1]
