import mne
import numpy as np
import pandas as pd

from functions import mne_prepping as mneprep
from functions import mne_helpers as mnehelp
from functions import read_eeg as readeegr
from functions import mne_stats as mnestats

from mne.time_frequency import tfr_multitaper, tfr_stockwell, tfr_morlet
from mne.minimum_norm import read_inverse_operator, source_band_induced_power

################### PREPARATION ----------------------------------

base_path = "D:\\IntracranialElectrodes\\Data\\p136\\"
base_path = "U:\\OneDrive\\FGU\\iEEG\\p136\\"

path_original_vr = base_path + "UnityAlloEgo\\EEG\\Preprocessed\\prep_256.mat"
path_perhead_vr = base_path + "UnityAlloEgo\\EEG\\Preprocessed\\prep_perHeadbox_256.mat"
path_perelectrode_vr = base_path + "UnityAlloEgo\\EEG\\Preprocessed\\prep_perElectrode_256.mat"
path_bip_vr = base_path + "UnityAlloEgo\\EEG\\Preprocessed\\prep_bipolar_256.mat"

path_unity_events = base_path + "UnityAlloEgo\\EEG\\Preprocessed\\p136_unity.csv"
path_onset_events = base_path + "UnityAlloEgo\\EEG\\Preprocessed\\p136_onsets.csv"
path_montage = base_path + "UnityAlloEgo\\EEG\\Preprocessed\\p136_montage.csv"
path_montage_referenced = base_path + "UnityAlloEgo\\EEG\\Preprocessed\\p136_montage_referenced.csv"


FREQUENCY = 256
runfile('M:/Vyzkum/AV/FGU/IntracranialElectrodes/iEEG-python/base_setup.py', wdir='M:/Vyzkum/AV/FGU/IntracranialElectrodes/iEEG-python')

# PICKS
pick_perhead_hip = mnehelp.picks_all_localised(raw_perhead_vr, pd_montage_referenced, 'Hi')
pick_perhead_hip_names = mne.pick_info(raw_perhead_vr.info, pick_perhead_hip)['ch_names']
pick_perhead_ins = mnehelp.picks_all_localised(raw_perhead_vr, pd_montage_referenced, 'Ins')
pick_perhead_all = mnehelp.picks_all(raw_perhead_vr)

# BAD EPOCHS
bad_epochs_original = [10, 42, 51, 52, 74, 82, 92, 93, 114, 129, 130, 131, 161, 165, 181, 185, 222, 255, 268, 269, 275, 291, 299, 300, 301, 302, 311, 323, 324, 325, 326, 327, 328, 346, 347, 348, 349, 353, 366, 367, 368, 369, 370, 376,397, 398, 427, 435, 438, 450, 461, 489, 490, 491, 492, 493, 494, 499, 518]

bad_epochs_perhead = [74, 92, 93, 114, 129, 130, 131, 161, 185, 255, 268, 269, 275, 302, 324, 325, 326, 366, 367, 368, 369, 370, 397, 398, 427, 435, 438, 461]
epochs_perhead_vr.drop(bad_epochs_perhead)
#epochs_perhead_vr.plot(block = True, scalings = 'auto', picks=pick_perhead_hip)
#epochs_perhead_vr.drop(bad_epochs)
#epochs_original_vr.plot(block = True, scalings = 'auto', picks=pick_orig_hip)

########### ANALYSES ----------------------------------

# TIME FREQ

freqs = np.arange(2, 11, 1)
n_cycles = 6
box = mnehelp.custom_box_layout(pick_perhead_hip_names, 3)
plot_pick_perhead_hip = range(len(pick_perhead_hip))

runfile('M:/Vyzkum/AV/FGU/IntracranialElectrodes/iEEG-python/tfr_perhead_unity.py', wdir='M:/Vyzkum/AV/FGU/IntracranialElectrodes/iEEG-python')

## BASELINES ----------------
baseline = (-2, -1)
runfile('M:/Vyzkum/AV/FGU/IntracranialElectrodes/iEEG-python/baselines.py', wdir='M:/Vyzkum/AV/FGU/IntracranialElectrodes/iEEG-python')

### LFO BANDS
lfo_bands = [[2, 4], [4, 9]]
runfile('M:/Vyzkum/AV/FGU/IntracranialElectrodes/iEEG-python/lfo_collapse.py', wdir='M:/Vyzkum/AV/FGU/IntracranialElectrodes/iEEG-python')


## PLOTS
power_point_perhead_vr_ego.plot_topo(picks = pick_perhead_hip, baseline=baseline, mode='logratio', layout=box)
power_point_perhead_vr_allo.plot_topo(picks = pick_perhead_hip, baseline=baseline, mode='logratio', layout=box)

power_point_perhead_vr_ego_lfo.plot_topo(picks = pick_perhead_hip, baseline=baseline, mode='logratio', layout=box)
power_point_perhead_vr_allo_lfo.plot_topo(picks = pick_perhead_hip, baseline=baseline, mode='logratio', layout=box)

power_onset_perhead_vr.plot_topo(picks = plot_pick_perhead_hip, baseline=(-3, -2), mode = 'logratio', layout = box)
power_stop_perhead_vr.plot_topo(picks = plot_pick_perhead_hip, baseline=(-3, -2), mode = 'logratio', layout = box)


## POWER OVER TIME --------
mnehelp.plot_power_time([power_stop_perhead_vr_lfo, power_onset_perhead_vr_lfo], pick_perhead_hip, 0, event_names = ['stop', 'onset'], pick_names = pick_perhead_hip_names)

mnehelp.plot_power_time([power_point_perhead_vr_ego_lfo, power_point_perhead_vr_allo_lfo], pick_perhead_hip, 0, event_names = ['ego', 'allo'], pick_names = pick_perhead_hip_names)

mnehelp.plot_power_time([power_point_perhead_vr_ego, power_point_perhead_vr_allo], pick_perhead_hip, 0, event_names = ['ego', 'allo'], pick_names = pick_perhead_hip_names)


## STATISTICS -------------------
# POinting 
wilcox_ego_allo, wilcox_freqs = mnestats.wilcox_tfr_power(power_trials_point_perhead_ego, power_trials_point_perhead_allo, picks = pick_perhead_hip_names)
mnestats.plot_wilcox_box(wilcox_ego_allo, FREQUENCY, freqs = wilcox_freqs)

wilcox_allo_ego_lfo, wilcox_freqs_lfo = mnestats.wilcox_tfr_power(power_trial_point_perhead_vr_ego_lfo, power_trial_point_perhead_vr_allo_lfo,  picks = pick_perhead_hip_names)
mnestats.plot_wilcox_box(wilcox_allo_ego_lfo, FREQUENCY, pick_names = pick_perhead_hip_names)

# Onsets
wilcox_stops_onsets_lfo, wilcox_freqs_lfo = mnestats.wilcox_tfr_power(power_trials_stop_perhead_vr_lfo, power_trials_onset_perhead_vr_lfo, picks = pick_perhead_hip_names)
mnestats.plot_wilcox_box(wilcox_stops_onsets_lfo, FREQUENCY, pick_names = pick_perhead_hip_names)