import numpy as np
import h5py
import mne

def read_mat(path):
    mat = h5py.File(path)
    data = mat['eeg'][()] #stores is as a numpy array - [()]
    data = data * 1e-09 #because of stupid scaling
    return(data)

def numpy_mne(data, frequency):
    n_channels = data.shape[0]
    ch_names = list(range(1, n_channels + 1))
    ch_names = [str(ch) for ch in ch_names]
    ch_types = ["seeg"] * n_channels
    info = mne.create_info(ch_names = ch_names, sfreq = frequency, ch_types = ch_types)
    raw = mne.io.RawArray(data, info)
    return(raw)