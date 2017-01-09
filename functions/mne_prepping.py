from functions import read_eeg as eegrd
from functions import read_experiment_data as exprd

def load_raw(data_path, frequency):
    data = eegrd.read_mat(data_path)
    data = data * 1e-09  # because of stupid scaling
    raw = eegrd.numpy_mne(data, frequency)
    return raw

def load_events(events_path, frequency):
    pd_events = exprd.read_events(events_path)
    mne_events, mapp = exprd.mne_epochs_from_pd(pd_events, frequency)
    return mne_events, mapp