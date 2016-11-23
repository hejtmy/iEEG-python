import mne

from functions import read_eeg as eegrd
from functions import read_experiment_data as exprd
path = "D:\\IntracranialElectrodes\\Data\\p126\\UnityAlloEgo\\EEG\\Preprocessed\\prep_bipolar_250.mat"
path = "U:\\OneDrive\\FGU\\iEEG\\p83\\UTAlloEgo\\EEG\\Preprocessed\\prep_bipolar_250.mat"

FREQUENCY = 250

path_events = "U:\\OneDrive\\FGU\\iEEG\\p83\\UTAlloEgo\\EEG\\Preprocessed\\p83_UT.csv"
path_events = "D:\\IntracranialElectrodes\\Data\\p126\\UnityAlloEgo\\EEG\\Preprocessed\\p126_unity.csv"
data = eegrd.read_mat(path)

raw = eegrd.numpy_mne(data, FREQUENCY)
pd_events = exprd.read_events(path_events)
mne_events, mapp = exprd.mne_epochs_from_pd(pd_events, FREQUENCY)

epochs = mne.Epochs(raw, mne_events, [-1, 2], event_id = mapp, add_eeg_ref = False)

from mne.datasets import sample  # noqa
data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
print(raw_fname)
raw = mne.io.read_raw_fif(raw_fname, add_eeg_ref=False)
print(raw)
print(raw.info)
start, stop = raw.time_as_index([100, 115])  # 100 s to 115 s data segment
data, times = raw[:, start:stop]
print(data.shape)
print(times.shape)
data, times = raw[2:20:3, start:stop]  # access underlying data
raw.plot()