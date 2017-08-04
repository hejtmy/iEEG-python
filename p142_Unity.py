import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from functions import mne_prepping as mneprep
from functions import mne_helpers as mnehelp
from functions import mne_plot_helpers as mneplothelp
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
runfile('M:/Vyzkum/AV/FGU/IntracranialElectrodes/iEEG-python/base_setup.py', wdir='M:/Vyzkum/AV/FGU/IntracranialElectrodes/iEEG-python')

# PICKS
pick_perhead_hip = mnehelp.picks_all_localised(raw_perhead_vr, pd_montage_referenced, 'Hi')
pick_perhead_hip_names = mne.pick_info(raw_perhead_vr.info, pick_perhead_hip)['ch_names']
pick_perhead_ent = mnehelp.picks_all_localised(raw_perhead_vr, pd_montage_referenced, 'Ent')
pick_perhead_ent_names = mne.pick_info(raw_perhead_vr.info, pick_perhead_ent)['ch_names']
pick_perhead_all = mnehelp.picks_all(raw_perhead_vr)


# BAD EPOCHS
#epochs_perhead_vr.plot(scalings = 'auto')

#epochs_perhead_vr.plot(block = True, scalings = 'auto', picks=pick_perhead_hip)
#epochs_perhead_vr.plot(block = True, scalings = 'auto')
#mnehelp.get_dropped_epoch_indices(epochs_perhead_vr.drop_log)
bad_epochs = [0, 1, 6, 8, 9, 10, 11, 12, 13, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 49, 50, 51, 52, 54, 55, 56, 57, 58, 59, 70, 71, 72, 73, 74, 76, 77, 78, 82, 83, 88, 89, 90, 91, 92, 93, 104, 105, 106, 107, 108, 109, 110, 117, 118, 119, 120, 121, 122, 123, 124, 127, 128, 129, 130, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 154, 155, 156, 157, 159, 160, 161, 162, 163, 164, 169, 170, 171, 172, 173, 174, 177, 178, 179, 180, 181, 182, 185, 186, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 213, 214, 215, 217, 218, 219, 220, 221, 222, 223, 233, 234, 236, 237, 238, 239, 240, 244, 245, 246, 247, 248, 249, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 287, 288, 289, 290, 291, 295, 296, 297, 298, 299, 300, 301, 302, 307, 308, 309, 310, 311, 315, 327, 328, 329, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346]
epochs_perhead_vr.drop(bad_epochs)

# TFR ANALYSIS ---------------
freqs = np.arange(2, 11, 1)
n_cycles = 6

runfile('M:/Vyzkum/AV/FGU/IntracranialElectrodes/iEEG-python/tfr_perhead_unity.py', wdir='M:/Vyzkum/AV/FGU/IntracranialElectrodes/iEEG-python')

## BASELINES ----------------
onset_baseline = (-0.5, 0)
baseline = (-3, -2)
mode = 'ratio'
runfile('M:/Vyzkum/AV/FGU/IntracranialElectrodes/iEEG-python/baselines.py', wdir='M:/Vyzkum/AV/FGU/IntracranialElectrodes/iEEG-python')

### LFO BANDS
lfo_bands = [[2, 4], [4, 9]]
runfile('M:/Vyzkum/AV/FGU/IntracranialElectrodes/iEEG-python/lfo_collapse.py', wdir='M:/Vyzkum/AV/FGU/IntracranialElectrodes/iEEG-python')

###
box = mnehelp.custom_box_layout(pick_perhead_hip_names, 3)
plot_pick_perhead_hip = range(len(pick_perhead_hip))
power_stop_perhead_vr.plot_topo(picks = pick_perhead_hip, layout = box, baseline = (-3, -2))

## POWER OVER TIME --------
#Onsets
mnehelp.plot_power_time([power_stop_perhead_vr_lfo, power_onset_perhead_vr_lfo], pick_perhead_hip, 0, event_names = ['stop', 'onset'], pick_names = pick_perhead_hip_names)
mnehelp.plot_power_time_average ([power_stop_perhead_vr_lfo, power_onset_perhead_vr_lfo], pick_perhead_hip, 0, event_names = ['stop', 'onset'])
#point end
mnehelp.plot_power_time([power_point_perhead_vr_ego_lfo, power_point_perhead_vr_allo_lfo], pick_perhead_hip, 0, event_names = ['ego', 'allo'], pick_names = pick_perhead_hip_names)
mnehelp.plot_power_time_average ([power_point_perhead_vr_ego_lfo, power_point_perhead_vr_allo_lfo], pick_perhead_hip, 0, event_names = ['ego', 'allo'])
mneplothelp.plot_epochs_power(power_trial_point_perhead_vr_allo_lfo, 0, pick_perhead_hip_names)
#point start
mnehelp.plot_power_time([power_point_start_perhead_vr_ego_lfo, power_point_start_perhead_vr_allo_lfo], pick_perhead_hip, 0, event_names = ['ego', 'allo'], pick_names = pick_perhead_hip_names)
mnehelp.plot_power_time_average ([power_point_start_perhead_vr_ego_lfo, power_point_start_perhead_vr_allo_lfo], pick_perhead_hip, 0, event_names = ['ego', 'allo'],)

## STATISTICS -------------------
# POinting 
wilcox_allo_ego_lfo, wilcox_freqs_lfo = mnestats.wilcox_tfr_power(power_trial_point_perhead_vr_ego_lfo, power_trial_point_perhead_vr_allo_lfo,  picks = pick_perhead_hip_names)
mnestats.plot_wilcox_box(wilcox_allo_ego_lfo, FREQUENCY, pick_names = pick_perhead_hip_names)
#pointing start
wilcox_allo_ego_start_lfo, wilcox_freqs_lfo = mnestats.wilcox_tfr_power(power_trial_point_start_perhead_vr_ego_lfo, power_trial_point_start_perhead_vr_allo_lfo,  picks = pick_perhead_hip_names)
mnestats.plot_wilcox_box(wilcox_allo_ego_start_lfo, FREQUENCY, pick_names = pick_perhead_hip_names)
# Onsets
wilcox_stops_onsets_lfo, wilcox_freqs_lfo = mnestats.wilcox_tfr_power(power_trials_stop_perhead_vr_lfo, power_trials_onset_perhead_vr_lfo, picks = pick_perhead_hip_names)
mnestats.plot_wilcox_box(wilcox_stops_onsets_lfo, FREQUENCY, pick_names = pick_perhead_hip_names)
