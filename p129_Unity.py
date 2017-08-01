import mne
import numpy as np
import pandas as pd

from functions import mne_prepping as mneprep
from functions import mne_helpers as mnehelp
from functions import read_eeg as readeegr
from functions import mne_stats as mnestats

from mne.time_frequency import tfr_multitaper, tfr_stockwell, tfr_morlet
from mne.minimum_norm import read_inverse_operator, source_band_induced_power


base_path = "D:\\IntracranialElectrodes\\Data\\p129\\"
base_path = "U:\\OneDrive\\FGU\\iEEG\\p129\\"

path_original_vr = base_path + "\\UnityAlloEgo\\EEG\\Preprocessed\\prep_256.mat"
path_perhead_vr = base_path + "\\UnityAlloEgo\\EEG\\Preprocessed\\prep_perHeadbox_256.mat"
path_bip_vr = base_path + "\\UnityAlloEgo\\EEG\\Preprocessed\\prep_bipolar_256.mat"
path_perelectrode_vr = base_path + "\\UnityAlloEgo\\EEG\\Preprocessed\\prep_perElectrode_256.mat"
path_unity_events = base_path + "UnityAlloEgo\\EEG\\Preprocessed\\p129_unity.csv"
path_onset_events = base_path + "UnityAlloEgo\\EEG\\Preprocessed\\p129_onsets.csv"
path_montage = base_path + "UnityAlloEgo\\EEG\\Preprocessed\\p129_montage.csv"
path_montage_referenced = base_path + "UnityAlloEgo\\EEG\\Preprocessed\\p129_montage_referenced.csv"

FREQUENCY = 256
runfile('M:/Vyzkum/AV/FGU/IntracranialElectrodes/iEEG-python/base_setup.py', wdir='M:/Vyzkum/AV/FGU/IntracranialElectrodes/iEEG-python')

# PICKS
pick_perhead_hip = mnehelp.picks_all_localised(raw_perhead_vr, pd_montage_referenced, 'Hi')
pick_perhead_hip_names = mne.pick_info(raw_perhead_vr.info, pick_perhead_hip)['ch_names']
pick_perhead_ins = mnehelp.picks_all_localised(raw_perhead_vr, pd_montage_referenced, 'Ins')
pick_perhead_all = mnehelp.picks_all(raw_perhead_vr)

# BAD EPOCHS
bad_epochs_perhead = []
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
baseline = (-3, -2)
runfile('M:/Vyzkum/AV/FGU/IntracranialElectrodes/iEEG-python/baselines.py', wdir='M:/Vyzkum/AV/FGU/IntracranialElectrodes/iEEG-python')

### LFO BANDS
lfo_bands = [[2, 4], [4, 9]]
runfile('M:/Vyzkum/AV/FGU/IntracranialElectrodes/iEEG-python/lfo_collapse.py', wdir='M:/Vyzkum/AV/FGU/IntracranialElectrodes/iEEG-python')
###
power_onset_perhead_vr.plot_topo(picks = pick_perhead_hip, layout = box, baseline = (-2, -1))

## POWER OVER TIME --------
mnehelp.plot_power_time([power_stop_perhead_vr_lfo, power_onset_perhead_vr_lfo], pick_perhead_ins, 0, event_names = ['stop', 'onset'], pick_names = pick_perhead_hip_names)

mnehelp.plot_power_time([power_point_perhead_vr_ego_lfo, power_point_perhead_vr_allo_lfo], pick_perhead_hip, 0, event_names = ['ego', 'allo'], pick_names = pick_perhead_hip_names)

mnehelp.plot_power_time([power_point_perhead_vr_ego, power_point_perhead_vr_allo], pick_perhead_hip, 1, event_names = ['ego', 'allo'], pick_names = pick_perhead_hip_names)

## STATISTICS
wilcox_stop_onset_hip_lfo, wilcox_freqs = mnestats.wilcox_tfr_power(power_trials_stop_orig_vr_lfo, power_trials_onset_perhead_vr_lfo, picks = pick_perhead_hip_names)
mnestats.plot_wilcox_box(wilcox_stop_onset_hip_lfo, 256, pick_names = pick_perhead_hip_names, cutout = 0.01)


wilcox_ego_allo_hip_lfo, wilcox_freqs = mnestats.wilcox_tfr_power(power_trial_point_perhead_vr_ego_lfo, power_trial_point_perhead_vr_allo_lfo, picks = pick_perhead_hip_names)
wilcox_ego_allo_hip, wilcox_freqs = mnestats.wilcox_tfr_power(power_trials_point_perhead_ego, power_trials_point_perhead_allo)

#mnestats.plot_wilcox_box(wilcox_ego_allo, 250)
#mnestats.plot_wilcox_box(wilcox_ego_allo_lfo, 250)

#TEST
mnehelp.plot_power_time([power_stop_perhead_vr_lfo, power_onset_perhead_vr_lfo], pick_perhead_all, 0, event_names = ['stop', 'onset'])
