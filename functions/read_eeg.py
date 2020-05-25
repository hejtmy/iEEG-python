import h5py
import mne
import pandas as pd
import re
import os
from functions import helpers


def read_mat(path):
    """Reads mat file into numpy array

    Parameters
    ----------
    path : [type]
        [description]
    """
    mat = h5py.File(path)
    data = mat['eeg'][()]  # stores is as a numpy array - [()]
    return(data)


def eeg_mat_to_mne(data, frequency, montage=None):
    """Converts passed matrix to a mne raw object

    Parameters
    ------------
    data : matrix
        read with read_mat and h5py
    frequency : int
        recording frequency
    montage : pandas.DataFrame
        read with read_montage
    Returns
    -----------
    """
    n_channels = data.shape[0]
    ch_nums = list(range(1, n_channels + 1))
    if montage is None:
        ch_names = ["SEEG_" + str(ch) for ch in ch_nums]
        ch_types = ["seeg"] * n_channels
    else:
        if montage.shape[0] != data.shape[0]:
            print("""Montage has differnet number of channels {}
                  than are in the data {}""".format(str(montage.shape[0]),
                                                    str(data.shape[0])))
            return eeg_mat_to_mne(data, frequency)
        ch_types = [mne_type(ch_type) for ch_type in montage.signalType]
        ch_names = [ch_type + "_" + str(montage.numberOnAmplifier[i]) for i,
                    ch_type in enumerate(ch_types)]
    info = mne.create_info(ch_names=ch_names, sfreq=frequency,
                           ch_types=ch_types)
    raw = mne.io.RawArray(data, info)
    return(raw)


def mne_type(ch_type):
    """Recodes passed string to a valid MNE type

    Parameters
    ----------
    ch_type : str
        channel type as exported from the headers

    Returns
    ----------
    valid vonversion of the channel to a MNE type
    """
    ch_type = ch_type.lower()
    if ch_type in ['ecog', 'bio', 'seeg', 'hbr', 'eog', 'grad', 'ecg', 'mag',
                   'eeg', 'stim', 'ref_meg', 'hbo', 'misc']:
        return ch_type
    else:
        return 'misc'


def read_montage(path):
    pd_montage = pd.read_csv(path)
    pd_montage = helpers.remove_unnamed(pd_montage)
    pd_montage.neurologyLabel = pd_montage.headboxNumber
    pd_montage = pd_montage.drop('headboxNumber', axis=1)
    return pd_montage


def get_frequency(eeg_path):
    """Estimates frequency from file names in given folder

    Parameters
    ----------
    eeg_path : str
        folder with eeg mat filesp

    Returns
    -------
    int or None
        estimated frequency or None if none was found
    """
    ptr = 'prep_.*_(\\d+)\\.mat'
    for root, dirs, files in os.walk(eeg_path):
        for file in files:
            res = re.match(ptr, file)
            if res:
                return int(res.group(1))
    return None
