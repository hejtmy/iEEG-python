import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from functions import mne_prepping as mneprep
from functions import mne_helpers as mnehelp
from functions import read_eeg as readeegr

from mne.stats import permutation_cluster_test
from mne.time_frequency import psd_multitaper

#base_path = "D:\\IntracranialElectrodes\\Data\\p126\\UnityAlloEgo\\EEG\\Preprocessed\\"
base_path = "U:\\OneDrive\\FGU\\iEEG\\p126\\"

path_original_vr = base_path + "UnityAlloEgo\\EEG\\Preprocessed\\prep_250.mat"
path_bip_vr = base_path + "UnityAlloEgo\\EEG\\Preprocessed\\prep_bipolar_250.mat"
path_unity_events = base_path + "UnityAlloEgo\\EEG\\Preprocessed\\p126_unity.csv"
path_onset_events = base_path + "UnityAlloEgo\\EEG\\Preprocessed\\p126_onsets.csv"
path_montage = base_path + "UnityAlloEgo\\EEG\\Preprocessed\\p126_montage.csv"

path_perhead_bva = "prep_perHeadbox_250.mat"
path_bip_bva =  base_path + "prep_bipolar_250.mat"
path_events_bva = "base_path + p126_BVA500_1500.csv"

FREQUENCY = 250

# Loading Unnity data
raw_original_vr = mneprep.load_raw(path_original_vr, FREQUENCY)
raw_bip_vr = mneprep.load_raw(path_bip_vr, FREQUENCY)

pd_unity_events = mneprep.load_unity_events(path_unity_events)
pd_matlab_events = mneprep.load_matlab_events(path_onset_events)
pd_events = pd.concat([pd_unity_events, pd_matlab_events])
pd_events = mneprep.clear_pd(pd_events)
mne_events_vr, mapp_vr = mneprep.pd_to_mne_events(pd_events, FREQUENCY)
pd_montage = readeegr.read_montage(path_montage)

raw_original_vr.plot(events = mne_events_vr, scalings='auto')

# loading BVA data
raw_perhead_bva = mneprep.load_raw(path_perhead_bva, FREQUENCY)
raw_bip_bva = mneprep.load_raw(path_bip_bva, FREQUENCY)
mne_events_bva, mapp_bva = mneprep.load_events(path_events_bva, FREQUENCY)

# {'ArduinoPulseStop': blue, 'onsets_500_1500': green, 'stops_500_1500': red}
raw_original_vr.plot(events = mne_events_vr, scalings='auto')

raw_original_vr.info["bads"] = ['SEEG_55', 'SEEG_56', 'SEEG_57', 'SEEG_58', 'SEEG_59']

# {'c': 1, 'f': 2, 'g': 3, 'onsets_500_1500': 4, 'stops_500_1500': 5}
raw_perhead_bva.plot(events=mne_events_bva, scalings='auto',
                     event_color={1: 'blue', 2: 'blue', 3: 'blue', 4: 'green', 5: 'red'})
raw_bip_bva.plot(events=mne_events_bva, scalings='auto',
                 event_color={1: 'blue', 2: 'blue', 3: 'blue', 4: 'green', 5: 'red'})

raw_perhead_bva.info["bads"] = ['SEEG_55', 'SEEG_56', 'SEEG_57', 'SEEG_58', 'SEEG_59']
raw_bip_bva.info["bads"] = ['SEEG_45', 'SEEG_46', 'SEEG_47', 'SEEG_48']

## Epoching
epochs_original_vr = mne.Epochs(raw_original_vr, mne_events_vr, event_id=mapp_vr, tmin=-3, tmax=3, add_eeg_ref=False)
epochs_bip_vr = mne.Epochs(raw_bip_vr, mne_events_vr, event_id=mapp_vr, tmin=-3, tmax=3, add_eeg_ref=False)

epochs_bip_bva = mne.Epochs(raw_bip_bva, mne_events_vr, event_id=mapp_vr, tmin=-3, tmax=3, add_eeg_ref=False)

epochs_original_vr['onsets_500_1500', 'stops_500_1500'].plot(block=True, scalings='auto')
epochs_bip_vr['onsets_500_1500', 'stops_500_1500'].plot(block=True, scalings='auto')

epochs_original_vr['onsets_500_1500', 'stops_500_1500'].plot(block=True, scalings='auto')
epochs_bip_bva['onsets_500_1500', 'stops_500_1500'].plot(block=True, scalings='auto')

freqs = np.arange(2, 30, 1)
n_cycles = freqs / 2

picks_original = mnehelp.def_picks(epochs_original_vr)
box = mnehelp.custom_box_layout(picks_original, 8)
plot_picks_perhead = range(0, len(picks_original))

power_onset_perhead_vr = tfr_morlet(epochs_perhead_vr['onsets_500_1500'], freqs=freqs, n_cycles=n_cycles,
                                    picks=picks_perhead, return_itc=False)
# NEED to pass picks because default IGNORES SEEG channels
power_onset_perhead_vr.plot_topo(picks=plot_picks_perhead, baseline=(-2., -1.5), mode='logratio', layout=box)

power_onset_perhead_bva = tfr_morlet(epochs_perhead_bva['onsets_500_1500'], freqs=freqs, n_cycles=n_cycles,
                                     picks=picks_perhead, return_itc=False)
# NEED to pass picks because default IGNORES SEEG channels
power_onset_perhead_bva.plot_topo(picks=plot_picks_perhead, baseline=(-2., -1.5), mode='logratio', layout=box)

## BIPORAL
picks_bip = mnehelp.def_picks(epochs_bip_vr['onsets_500_1500'])
box_bip = mnehelp.custom_box_layout(picks_bip)
plot_picks_bip = range(0, len(picks_bip))
power_onset_bip_vr = tfr_morlet(epochs_bip_vr['onsets_500_1500'], freqs=freqs, n_cycles=n_cycles, picks=picks_bip,
                                return_itc=False)

power_onset_bip_vr.plot_topo(picks=plot_picks_bip, baseline=(-2., -1.5), mode='logratio', layout=box_bip)
