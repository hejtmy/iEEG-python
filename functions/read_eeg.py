import numpy as np
import h5py

def read_mat(path):
    mat = h5py.File(path)
    data = mat['eeg'][()] #stores is as a numpy array - [()]
    return(data)