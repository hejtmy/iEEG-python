import mne
import numpy as np

from functions import mne_helpers as mnehelp
from functions import mne_plot_helpers as mneplothelp
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
pick_perhead_ent = mnehelp.picks_all_localised(raw_perhead_vr, pd_montage_referenced, 'Ent')
pick_perhead_ent_names = mne.pick_info(raw_perhead_vr.info, pick_perhead_hip)['ch_names']
pick_perhead_all = mnehelp.picks_all(raw_perhead_vr)

raw_perhead_vr = None

# BAD EPOCHS
bad_epochs_perhead = [19, 51, 52, 185, 186, 204, 205, 287, 288, 289, 302, 367, 368, 373, 374, 375, 376, 377, 378, 418, 419, 488, 489, 490, 504, 505, 506, 507]
epochs_perhead_vr.drop(bad_epochs_perhead)
#epochs_perhead_vr.plot(block = True, scalings = 'auto', picks=pick_perhead_hip)

########### ANALYSES ----------------------------------
# TIME FREQ
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
power_onset_perhead_vr.plot_topo(picks = pick_perhead_hip, layout = box, baseline = baseline)
power_stop_perhead_vr.plot_topo(picks = pick_perhead_hip, layout = box, baseline = baseline)

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
# Onsets
wilcox_stops_onsets_lfo, wilcox_freqs_lfo = mnestats.wilcox_tfr_power(power_trials_stop_perhead_vr_lfo, power_trials_onset_perhead_vr_lfo, picks = pick_perhead_hip_names)
mnestats.plot_wilcox_box(wilcox_stops_onsets_lfo, FREQUENCY, pick_names = pick_perhead_hip_names)
# POinting 
wilcox_allo_ego_lfo, wilcox_freqs_lfo = mnestats.wilcox_tfr_power(power_trial_point_perhead_vr_ego_lfo, power_trial_point_perhead_vr_allo_lfo,  picks = pick_perhead_hip_names)
mnestats.plot_wilcox_box(wilcox_allo_ego_lfo, FREQUENCY, pick_names = pick_perhead_hip_names)
#pointing start
wilcox_allo_ego_start_lfo, wilcox_freqs_lfo = mnestats.wilcox_tfr_power(power_trial_point_start_perhead_vr_ego_lfo, power_trial_point_start_perhead_vr_allo_lfo,  picks = pick_perhead_hip_names)
mnestats.plot_wilcox_box(wilcox_allo_ego_start_lfo, FREQUENCY, pick_names = pick_perhead_hip_names)

#mnestats.plot_wilcox_box(wilcox_ego_allo, 250)
#mnestats.plot_wilcox_box(wilcox_ego_allo_lfo, 250)

mneplothelp.plot_epochs_power(power_trials_onset_perhead_vr_lfo, 0, pick_perhead_hip_names)
mneplothelp.plot_epochs_power(power_trials_stop_perhead_vr_lfo, 0, pick_perhead_hip_names)

mneplothelp.plot_epochs_power(power_trial_point_perhead_vr_ego_lfo, 0, pick_perhead_hip_names)
mneplothelp.plot_epochs_power(power_trial_point_perhead_vr_allo_lfo, 0, pick_perhead_hip_names)
