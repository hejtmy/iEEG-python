import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from functions import mne_prepping as mneprep
from functions import mne_helpers as mnehelp
from functions import read_eeg as readeegr

from mne.stats import permutation_cluster_test
from mne.time_frequency import psd_multitaper, tfr_multitaper, tfr_stockwell, tfr_morlet

base_path = "D:\\IntracranialElectrodes\\Data\\p126\\"
#base_path = "U:\\OneDrive\\FGU\\iEEG\\p126\\"

path_original_vr = base_path + "UnityAlloEgo\\EEG\\Preprocessed\\prep_250.mat"
path_bip_vr = base_path + "UnityAlloEgo\\EEG\\Preprocessed\\prep_bipolar_250.mat"
path_unity_events = base_path + "UnityAlloEgo\\EEG\\Preprocessed\\p126_unity.csv"
path_onset_events = base_path + "UnityAlloEgo\\EEG\\Preprocessed\\p126_onsets.csv"
path_montage = base_path + "UnityAlloEgo\\EEG\\Preprocessed\\p126_montage.csv"

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

# {'ArduinoPulseStop': blue, 'onsets_500_1500': green, 'stops_500_1500': red}
raw_original_vr.plot(events = mne_events_vr, scalings='auto')

raw_original_vr.info["bads"] = ['SEEG_55', 'SEEG_56', 'SEEG_57', 'SEEG_58', 'SEEG_59']

## Epoching
epochs_original_vr = mne.Epochs(raw_original_vr, mne_events_vr, event_id=mapp_vr, tmin=-3, tmax=3, add_eeg_ref=False, baseline=None)
#epochs_bip_vr = mne.Epochs(raw_bip_vr, mne_events_vr, event_id=mapp_vr, tmin=-3, tmax=3, add_eeg_ref=False, baseline=None)

epochs_original_vr['onsets_500_1500', 'stops_500_1500'].plot(block=True, scalings='auto')
#epochs_bip_vr['onsets_500_1500', 'stops_500_1500'].plot(block=True, scalings='auto')

epochs_original_vr['onsets_500_1500', 'stops_500_1500'].plot(block=True, scalings='auto')

freqs = np.arange(2, 30, 1)
n_cycles = freqs / 2

picks_original = mnehelp.picks_all(epochs_original_vr)
picks_hi = mnehelp.picks_all_localised(epochs_original_vr, pd_montage, 'Hi')
pick_ch_names = mne.pick_info(raw_original_vr.info, picks_original)['ch_names']
box = mnehelp.custom_box_layout(pick_ch_names, 8)
plot_picks_perhead = range(0, len(picks_original))


picks_hi = mnehelp.picks_all_localised(epochs_original_vr, pd_montage, 'Hi')
pick_ch_names = mne.pick_info(raw_original_vr.info, picks_hi)['ch_names']
box = mnehelp.custom_box_layout(picks_hi, 8)
power_original_point_ego_vr = tfr_morlet(epochs_original_vr['pointingEnded_Ego'], freqs=freqs, n_cycles=n_cycles,
                                    picks=picks_hi, return_itc=False)

# NEED to pass picks because default IGNORES SEEG channels
power_original_point_ego_vr.plot_topo(picks = 1, baseline=(-3, -2), mode='logratio', layout=box)

conditions = ["pointingEnded_Ego", "pointingEnded_Allo"]
evoked_dict = dict()
for condition in conditions:
    evoked_dict[condition] = epochs_original_vr[condition].average()

colors = dict(pointingEnded_Ego="Crimson", pointingEnded_Allo="CornFlowerBlue")
mne.viz.plot_compare_evokeds(evoked_dict, picks=picks_hi, colors=colors)