import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from functions import mne_prepping as mneprep
from functions import mne_helpers as mnehelp
from functions import read_eeg as readeegr
from functions import mne_stats as mnestats

from mne.stats import permutation_cluster_test
from mne.time_frequency import psd_multitaper, tfr_multitaper, tfr_stockwell, tfr_morlet

base_path = "D:\\OneDrive\\FGU\\iEEG\\p142\\"
base_path = "U:\\OneDrive\\FGU\\iEEG\\p142\\"

path_original_vr = base_path + "UnityAlloEgo\\EEG\\Preprocessed\\prep_250.mat"
path_perhead_vr = base_path + "UnityAlloEgo\\EEG\\Preprocessed\\prep_perHeadbox_250.mat"
path_perelectrode_vr = base_path + "UnityAlloEgo\\EEG\\Preprocessed\\prep_perElectrode_250.mat"
path_bipolar_vr = base_path + "UnityAlloEgo\\EEG\\Preprocessed\\prep_bipolar_250.mat"
path_unity_events = base_path + "UnityAlloEgo\\EEG\\Preprocessed\\p142_unity.csv"
path_onset_events = base_path + "UnityAlloEgo\\EEG\\Preprocessed\\p142_onsets.csv"
path_montage = base_path + "UnityAlloEgo\\EEG\\Preprocessed\\p142_montage.csv"
path_montage_referenced = base_path + "UnityAlloEgo\\EEG\\Preprocessed\\p142_montage_referenced.csv"

FREQUENCY = 250

pd_montage = readeegr.read_montage(path_montage)
pd_montage_referenced = readeegr.read_montage(path_montage_referenced)

# Loading Unnity data
raw_original_vr = mneprep.load_raw(path_original_vr, FREQUENCY)
raw_perhead_vr = mneprep.load_raw(path_perhead_vr, FREQUENCY, pd_montage_referenced)
raw_perelectrode_vr = mneprep.load_raw(path_perelectrode_vr, FREQUENCY, pd_montage_referenced)
raw_bipolar_vr = mneprep.load_raw(path_bipolar_vr, FREQUENCY)

pd_unity_events = mneprep.load_unity_events(path_unity_events)
pd_matlab_events = mneprep.load_matlab_events(path_onset_events)
pd_events = pd.concat([pd_unity_events, pd_matlab_events])
pd_events = mneprep.clear_pd(pd_events)
mne_events_vr, mapp_vr = mneprep.pd_to_mne_events(pd_events, FREQUENCY)


#raw_original_vr.plot(events = mne_events_vr, scalings='auto')
#raw_perhead_vr.plot(events = mne_events_vr, scalings='auto')

## Epoching
epochs_perhead_vr = mne.Epochs(raw_perhead_vr, mne_events_vr, event_id=mapp_vr, tmin=-3, tmax=3, add_eeg_ref=False, baseline=None)

# PICKS
pick_perhead_hip = mnehelp.picks_all_localised(raw_perhead_vr, pd_montage_referenced, 'Hi')
pick_perhead_hip_names = mne.pick_info(raw_perhead_vr.info, pick_perhead_hip)['ch_names']
pick_perhead_ins = mnehelp.picks_all_localised(raw_perhead_vr, pd_montage_referenced, 'Ins')
pick_perhead_all = mnehelp.picks_all(epochs_perhead_vr)


# BAD EPOCHS
#epochs_perhead_vr.plot(scalings = 'auto')
#

#epochs_perhead_vr.plot(block = True, scalings = 'auto', picks=pick_perhead_hip)
#epochs_perhead_vr.plot(block = True, scalings = 'auto')
#mnehelp.get_dropped_epoch_indices(epochs_perhead_vr.drop_log)


# TFR ANALYSIS ---------------
freqs = np.arange(2, 11, 1)
n_cycles = 6

box = mnehelp.custom_box_layout(pick_perhead_hip_names, 3)
plot_pick_perhead_hip = range(len(pick_perhead_hip))


# POINTING
power_point_perhead_vr_ego = tfr_morlet(epochs_perhead_vr['pointingEnded_Ego'], freqs=freqs, n_cycles=n_cycles, picks = pick_perhead_all, return_itc=False)
power_point_perhead_vr_allo = tfr_morlet(epochs_perhead_vr['pointingEnded_Allo'], freqs=freqs, n_cycles=n_cycles, picks = pick_perhead_all, return_itc=False)

power_trials_point_perhead_ego = tfr_morlet(epochs_perhead_vr['pointingEnded_Ego'], freqs=freqs, n_cycles = n_cycles,picks = pick_perhead_all,  return_itc = False, average = False)
power_trials_point_perhead_allo = tfr_morlet(epochs_perhead_vr['pointingEnded_Allo'], freqs = freqs, n_cycles = n_cycles, picks = pick_perhead_all, return_itc = False, average = False)

# ONSETS
power_onset_perhead_vr = tfr_morlet(epochs_perhead_vr['onsets_500_1500'], freqs = freqs, n_cycles = n_cycles, picks = pick_perhead_all, return_itc = False)
power_stop_perhead_vr = tfr_morlet(epochs_perhead_vr['stops_500_1500'], freqs = freqs, n_cycles = n_cycles, picks = pick_perhead_all, return_itc = False)

power_trials_onset_perhead_vr = tfr_morlet(epochs_perhead_vr['onsets_500_1500'], freqs = freqs, n_cycles = n_cycles, picks = pick_perhead_all, return_itc = False, average = False)
power_trials_stop_perhead_vr = tfr_morlet(epochs_perhead_vr['stops_500_1500'], freqs = freqs, n_cycles = n_cycles, picks = pick_perhead_all,  return_itc = False, average = False)


## BASELINES ----------------
power_onset_perhead_vr.apply_baseline(mode = 'ratio', baseline = (-2, -1))
power_stop_perhead_vr.apply_baseline(mode = 'ratio', baseline = (-2, -1))

power_trials_onset_perhead_vr.apply_baseline(mode = 'ratio', baseline = (-2, -1))
power_trials_stop_perhead_vr.apply_baseline(mode = 'ratio', baseline = (-2, -1))

power_point_perhead_vr_ego.apply_baseline(mode = 'ratio', baseline = (-2, -1))
power_point_perhead_vr_allo.apply_baseline(mode = 'ratio', baseline = (-2, -1))

power_trials_point_perhead_ego.apply_baseline(mode = 'ratio', baseline = (-2, -1))
power_trials_point_perhead_allo.apply_baseline(mode = 'ratio', baseline = (-2, -1))


### LFO BANDS
lfo_bands = [[2, 4], [4, 9]]

power_onset_perhead_vr_lfo = mnehelp.band_power(power_onset_perhead_vr, lfo_bands)
power_stop_perhead_vr_lfo = mnehelp.band_power(power_stop_perhead_vr, lfo_bands)

power_trials_onset_perhead_vr_lfo = mnehelp.band_power(power_trials_onset_perhead_vr, lfo_bands)
power_trials_stop_orig_vr_lfo = mnehelp.band_power(power_trials_stop_perhead_vr, lfo_bands)

power_point_perhead_vr_ego_lfo = mnehelp.band_power(power_point_perhead_vr_ego, lfo_bands)
power_point_perhead_vr_allo_lfo = mnehelp.band_power(power_point_perhead_vr_allo, lfo_bands)

power_trial_point_perhead_vr_ego_lfo = mnehelp.band_power(power_trials_point_perhead_ego, lfo_bands)
power_trial_point_perhead_vr_allo_lfo = mnehelp.band_power(power_trials_point_perhead_allo, lfo_bands)


## POWER OVER TIME --------
mnehelp.plot_power_time([power_stop_perhead_vr_lfo, power_onset_perhead_vr_lfo], pick_perhead_hip, 0, event_names = ['stop', 'onset'], pick_names = pick_perhead_hip_names)

mnehelp.plot_power_time([power_point_perhead_vr_ego_lfo, power_point_perhead_vr_allo_lfo], pick_perhead_hip, 1, event_names = ['ego', 'allo'], pick_names = pick_perhead_hip_names)

mnehelp.plot_power_time([power_point_perhead_vr_ego, power_point_perhead_vr_allo], pick_perhead_hip, 1, event_names = ['ego', 'allo'], pick_names = pick_perhead_hip_names)

## STATISTICS
wilcox_ego_allo_hip_lfo, wilcox_freqs = mnestats.wilcox_tfr_power(power_trial_point_perhead_vr_ego_lfo, power_trial_point_perhead_vr_allo_lfo, pick_channels = pick_perhead_hip_names)
wilcox_ego_allo_hip, wilcox_freqs = mnestats.wilcox_tfr_power(power_trials_point_perhead_ego, power_trials_point_perhead_allo)

#mnestats.plot_wilcox_box(wilcox_ego_allo, 250)
#mnestats.plot_wilcox_box(wilcox_ego_allo_lfo, 256)


#TEST
mnehelp.plot_power_time([power_stop_perhead_vr_lfo, power_onset_perhead_vr_lfo], pick_perhead_all, 0, event_names = ['stop', 'onset'])
