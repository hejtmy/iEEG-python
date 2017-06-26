import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from functions import mne_prepping as mneprep
from functions import mne_helpers as mnehelp
from functions import read_eeg as readeegr

from mne.stats import permutation_cluster_test
from mne.time_frequency import psd_multitaper

from mne.time_frequency import tfr_multitaper, tfr_stockwell, tfr_morlet
from mne.minimum_norm import read_inverse_operator, source_band_induced_power

base_path = "D:\\IntracranialElectrodes\\Data\\p136\\"
base_path = "U:\\OneDrive\\FGU\\iEEG\\p129\\"

path_original = base_path + "\\UnityAlloEgo\\EEG\\Preprocessed\\prep_256.mat"
path_perhead_vr = base_path + "\\UnityAlloEgo\\EEG\\Preprocessed\\prep_perHeadbox_256.mat"
path_bip_vr = base_path + "\\UnityAlloEgo\\EEG\\Preprocessed\\prep_bipolar_256.mat"
path_perelectrode_vr = base_path + "\\UnityAlloEgo\\EEG\\Preprocessed\\prep_perElectrode_256.mat"
path_unity_events = base_path + "UnityAlloEgo\\EEG\\Preprocessed\\p129_unity.csv"
path_onset_events = base_path + "UnityAlloEgo\\EEG\\Preprocessed\\p129_onsets.csv"
path_montage = base_path + "UnityAlloEgo\\EEG\\Preprocessed\\p129_montage.csv"

FREQUENCY = 256

# Loading Unity data
raw_original_vr  = mneprep.load_raw(path_original, FREQUENCY)
raw_perhead_vr = mneprep.load_raw(path_perhead_vr, FREQUENCY)
raw_perelectrode_vr = mneprep.load_raw(path_perelectrode_vr, FREQUENCY)
raw_bip_vr = mneprep.load_raw(path_bip_vr, FREQUENCY)

pd_unity_events = mneprep.load_unity_events(path_unity_events)
pd_matlab_events = mneprep.load_matlab_events(path_onset_events)
pd_events = pd.concat([pd_unity_events, pd_matlab_events])
pd_events = mneprep.clear_pd(pd_events)
mne_events_vr, mapp_vr = mneprep.pd_to_mne_events(pd_events, FREQUENCY)

# loading montage
pd_montage = readeegr.read_montage(path_montage)

#raw_perhead_vr.plot(events = mne_events_vr, scalings = 'auto')
#raw_bip_vr.plot(events = mne_events_vr, scalings = 'auto')

raw_original_vr.info["bads"] = ['SEEG_34', 'SEEG_35', 'SEEG_36', 'SEEG_37', 'SEEG_38', 'SEEG_39', 'SEEG_40', 'SEEG_47']
raw_perelectrode_vr.info["bads"] = ['SEEG_40', 'SEEG_34', 'SEEG_35', 'SEEG_36', 'SEEG_37', 'SEEG_38', 'SEEG_39', 'SEEG_47']
raw_perhead_vr.info["bads"] = ['SEEG_23', 'SEEG_34', 'SEEG_35', 'SEEG_47','SEEG_67']
raw_bip_vr.info["bads"] = ['SEEG_21', 'SEEG_24','SEEG_42','SEEG_43','SEEG_60']

## Epoching
reject = {'mag': 4e-12, 'eog': 200e-6}
epochs_perhead_vr = mne.Epochs(raw_perhead_vr, mne_events_vr, event_id = mapp_vr, tmin = -3, tmax = 3, add_eeg_ref = False)
epochs_perelectrode_vr = mne.Epochs(raw_perelectrode_vr, mne_events_vr, event_id = mapp_vr, tmin = -3, tmax = 3, add_eeg_ref = False)
epochs_bip_vr = mne.Epochs(raw_bip_vr, mne_events_vr, event_id = mapp_vr, tmin=-3, tmax = 3, add_eeg_ref = False)

#epochs_perelectrode_vr['onsets_500_1500', 'stops_500_1500'].plot(block = True, scalings = 'auto')
#epochs_perhead_vr['onsets_500_1500', 'stops_500_1500'].plot(block = True, scalings = 'auto')
#epochs_bip_vr['onsets_500_1500', 'stops_500_1500'].plot(block = True, scalings = 'auto')

# PICKS
pick_orig_hip = mnehelp.picks_all_localised(raw_original_vr, pd_montage, 'Hi')
pick_orig_hip_names = mne.pick_info(raw_original_vr.info, pick_orig_hip)['ch_names']
pick_orig_ins = mnehelp.picks_all_localised(raw_original_vr, pd_montage, 'Ins')
pick_orig_all = mnehelp.picks_all(raw_original_vr)

plot_pick_orig_hip = range(len(pick_orig_hip))

# TIME FREQ
freqs = np.arange(2, 10, .5)
n_cycles = 6

box = mnehelp.custom_box_layout(pick_orig_hip_names, 3)

picks_perhead = mnehelp.def_picks(epochs_perhead_vr)
box =  mnehelp.custom_box_layout(picks_perhead)
plot_picks_perhead = range(0, len(picks_perhead))

power_all_perhead_vr = tfr_morlet(epochs_perhead_vr, freqs = freqs, n_cycles = n_cycles, picks = picks_perhead, return_itc = False)

power_onset_perhead_vr = tfr_morlet(epochs_perhead_vr['onsets_500_1500'], freqs = freqs, n_cycles = n_cycles, picks = picks_perhead, return_itc = False)

# NEED to pass picks because default IGNORES SEEG channels
power_onset_perhead_vr.plot_topo(picks = plot_picks_perhead,  baseline=(-3, -2), mode = 'logratio', layout = box)

power_onset_perhead_vr_divided = power_onset_perhead_vr
power_onset_perhead_vr_divided.data = power_onset_perhead_vr.data/power_all_perhead_vr.data
power_onset_perhead_vr_divided.plot_topo(picks = plot_picks_perhead,  baseline=(-3, -2), mode = 'logratio', layout = box)

power_stops_perhead_vr = tfr_morlet(epochs_perhead_vr['stops_500_1500'], freqs = freqs, n_cycles = n_cycles, picks = picks_perhead, return_itc = False)
# NEED to pass picks because default IGNORES SEEG channels
power_stops_perhead_vr.plot_topo(picks = plot_picks_perhead,  baseline=(-3, -2), mode = 'logratio', layout = box)


## BIPOLAR
picks_bip = mnehelp.def_picks(epochs_bip_vr['onsets_500_1500'])
box_bip =  mnehelp.custom_box_layout(picks_bip)
plot_picks_bip = range(0, len(picks_bip))
power_onset_bip_vr = tfr_morlet(epochs_bip_vr['onsets_500_1500'], freqs = freqs, n_cycles=n_cycles, picks = picks_bip, return_itc = False)

power_onset_bip_vr.plot_topo(picks = plot_picks_bip, baseline=(-3, -2), mode = 'logratio', layout = box_bip)