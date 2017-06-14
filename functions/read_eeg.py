import h5py
import mne
import pandas as pd
from functions import helpers

def read_mat(path):
    mat = h5py.File(path)
    data = mat['eeg'][()] #stores is as a numpy array - [()]
    return(data)


def numpy_mne(data, frequency):
    n_channels = data.shape[0]
    ch_names = list(range(1, n_channels + 1))
    ch_names = ["SEEG_" + str(ch) for ch in ch_names]
    ch_types = ["seeg"] * n_channels
    info = mne.create_info(ch_names = ch_names, sfreq = frequency, ch_types = ch_types)
    raw = mne.io.RawArray(data, info)
    return(raw)


def read_montage(path):
    pd_montage = pd.read_csv(path)
    pd_montage = helpers.remove_unnamed(pd_montage)
    pd_montage.neurologyLabel = pd_montage.headboxNumber
    pd_montage = pd_montage.drop('headboxNumber', axis = 1)
    return pd_montage
