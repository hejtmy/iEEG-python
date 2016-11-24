import mne

from functions import read_eeg as eegrd
from functions import read_experiment_data as exprd
path = "U:\\OneDrive\\FGU\\iEEG\\p83\\UTAlloEgo\\EEG\\Preprocessed\\prep_bipolar_250.mat"

FREQUENCY = 250

path_events = "U:\\OneDrive\\FGU\\iEEG\\p83\\UTAlloEgo\\EEG\\Preprocessed\\p83_UT.csv"
data = eegrd.read_mat(path)

raw = eegrd.numpy_mne(data, FREQUENCY)
pd_events = exprd.read_events(path_events)
mne_events, mapp = exprd.mne_epochs_from_pd(pd_events, FREQUENCY)

raw.plot(events = mne_events, scalings = 'auto')


epochs = mne.Epochs(raw, mne_events, event_id = mapp, tmin=-0.5, tmax=2, add_eeg_ref = False)
epochs.plot(block=True, scalings='auto')

onsets = epochs['onsets_500_1500']
onsets.plot(scalings = 'auto', n_epochs = 5)

stops = epochs['stops_500_1500']
stops.plot(scalings = 'auto', n_epochs = 5)
