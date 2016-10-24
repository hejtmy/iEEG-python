import mne
import numpy as np
import pandas as pd

from functions import read_eeg as eegrd
from functions import read_experiment_data as exprd
path = "D:\\IntracranialElectrodes\\Data\\p83\\UTAlloEgo\\EEG\\Preprocessed\\prep_UTAlloEgo_250_bipolar.mat"
path = "U:\\OneDrive\\FGU\\iEEG\\p83\\UTAlloEgo\\EEG\\Preprocessed\\prep_bipolar_250.mat"

path_events = "U:\\OneDrive\\FGU\\iEEG\\p83\\UTAlloEgo\\EEG\\Preprocessed\\p83_UT.csv"
data = eegrd.read_mat(path)

ch_names = list(range(1, 51))
ch_names = [str(ch) for ch in ch_names]
ch_types = ["seeg"] * 50
info = mne.create_info(ch_names = ch_names, sfreq = 250, ch_types = ch_types)
raw = mne.io.RawArray(data, info)
events = exprd.read_events(path_events)