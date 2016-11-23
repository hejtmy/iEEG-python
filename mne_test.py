import mne

from functions import read_eeg as eegrd
from functions import read_experiment_data as exprd
path = "D:\\IntracranialElectrodes\\Data\\p126\\UnityAlloEgo\\EEG\\Preprocessed\\prep_perHeadbox_250.mat"
path = "U:\\OneDrive\\FGU\\iEEG\\p83\\UTAlloEgo\\EEG\\Preprocessed\\prep_bipolar_250.mat"

path_events = "U:\\OneDrive\\FGU\\iEEG\\p83\\UTAlloEgo\\EEG\\Preprocessed\\p83_UT.csv"
path_events = "D:\\IntracranialElectrodes\\Data\\p126\\UnityAlloEgo\\EEG\\Preprocessed\\p126_unity.csv"
data = eegrd.read_mat(path)

raw = eegrd.numpy_mne(data, 250)
events = exprd.read_events(path_events)

