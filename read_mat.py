import numpy
import h5py
import pyedflib

def read_mat(path):
    mat = h5py.File(path)

    list(mat['EEG']['referenced']['bipolar'].keys())

    bip_data = mat['EEG']['referenced']['bipolar']['d'][()] #stores is as a numpy array - [()]
    orig_data = mat['EEG']['original']['d'][()] #stores is as a numpy array - [()]

    numpy.save("D:\\IntracranialElectrodes\\Data\\p83\\BVAAlloEgo\\EEG\\Preprocessed\\250_bipolar_eeg", bip_data)
    numpy.save("D:\\IntracranialElectrodes\\Data\\p83\\BVAAlloEgo\\EEG\\Preprocessed\\250_original_eeg", orig_data)

path = "D:\\IntracranialElectrodes\\Data\\p83\\UTAlloEgo\\EEG\\Preprocessed\\prep_UTAlloEgo_250_bipolar.mat"
read_mat(path)