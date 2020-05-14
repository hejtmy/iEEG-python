import h5py
import mne
import pandas as pd
from functions import helpers


def read_mat(path):
    mat = h5py.File(path)
    data = mat['eeg'][()]  # stores is as a numpy array - [()]
    return(data)


def numpy_mne(data, frequency, montage=None):
    n_channels = data.shape[0]
    ch_nums = list(range(1, n_channels + 1))
    if montage is None:
        ch_names = ["SEEG_" + str(ch) for ch in ch_nums]
        ch_types = ["seeg"] * n_channels
    else:
        if montage.shape[0] != data.shape[0]:
            print("Montage has differnet number of channels {} that are in the data {}".format(str(montage.shape[0]), str(data.shape[0])))
            return numpy_mne(data, frequency)
        ch_types = montage.signalType
        ch_names = [ch_type + "_" + str(montage.numberOnAmplifier[i]) for i, ch_type in enumerate(ch_types)]
        ch_types = [mne_type(type.lower()) for type in ch_types]
    info = mne.create_info(ch_names=ch_names, sfreq=frequency, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)
    return(raw)


def mne_type(str):
    if str in ['ecog', 'bio', 'seeg', 'hbr', 'eog', 'grad', 'ecg', 'mag', 'eeg', 'stim', 'ref_meg', 'hbo', 'misc']:
        return str
    else:
        return "misc"


def read_montage(path):
    pd_montage = pd.read_csv(path)
    pd_montage = helpers.remove_unnamed(pd_montage)
    pd_montage.neurologyLabel = pd_montage.headboxNumber
    pd_montage = pd_montage.drop('headboxNumber', axis=1)
    return pd_montage
