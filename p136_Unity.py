import mne
import numpy as np
import pandas as pd

from functions import mne_prepping as mneprep
from functions import mne_helpers as mnehelp
from mne.time_frequency import tfr_multitaper, tfr_stockwell, tfr_morlet

path_original_vr = "D:\\IntracranialElectrodes\\Data\\p136\\UnityAlloEgo\\EEG\\Preprocessed\\prep_256.mat"
path_perhead_vr = "D:\\IntracranialElectrodes\\Data\\p136\\UnityAlloEgo\\EEG\\Preprocessed\\prep_perHeadbox_256.mat"
path_bip_vr = "D:\\IntracranialElectrodes\\Data\\p136\\UnityAlloEgo\\EEG\\Preprocessed\\prep_bipolar_256.mat"

path_unity_events = "D:\\IntracranialElectrodes\\Data\\p136\\UnityAlloEgo\\EEG\\Preprocessed\\p136_unity.csv"
path_onset_events = "D:\\IntracranialElectrodes\\Data\\p136\\UnityAlloEgo\\EEG\\Preprocessed\\p136_onsets.csv"

FREQUENCY = 250

# Loading Unnity data
raw_original_vr = mneprep.load_raw(path_original_vr, FREQUENCY)
raw_perhead_vr = mneprep.load_raw(path_perhead_vr, FREQUENCY)
raw_bip_vr = mneprep.load_raw(path_bip_vr, FREQUENCY)

pd_unity_events = mneprep.load_unity_events(path_unity_events)
pd_matlab_events = mneprep.load_matlab_events(path_onset_events)
pd_events = pd.concat([pd_unity_events, pd_matlab_events])
pd_events = mneprep.clear_pd(pd_events)
pd_events = mneprep.solve_duplicates(pd_events, FREQUENCY)

mne_events_vr, mapp_vr = mneprep.pd_to_mne_events(pd_events, FREQUENCY)

raw_original_vr.plot(events = mne_events_vr, scalings='auto')
raw_bip_vr.plot(events=mne_events_vr, scalings='auto', event_color = {1: 'blue', 2: 'green', 3: 'red'})

raw_original_vr.info["bads"] = ['47']

## Epoching
epochs_original_vr = mne.Epochs(raw_original_vr, mne_events_vr, event_id=mapp_vr, tmin=-3, tmax=3, add_eeg_ref=False)

onsets_perhead = epochs_original_vr['onsets_500_1500']

epochs_original_vr['onsets_500_1500', 'stops_500_1500'].plot(block=True, scalings='auto')

freqs = np.arange(2, 30, 1)
n_cycles = freqs / 2

picks_perhead = mnehelp.def_picks(epochs_perhead_vr['onsets_500_1500'])
box = mnehelp.custom_box_layout(picks_perhead)
plot_picks_perhead = range(0, len(picks_perhead))

power_onset_perhead_vr = tfr_morlet(epochs_perhead_vr['onsets_500_1500'], freqs=freqs, n_cycles=n_cycles,
                                    picks = picks_perhead, return_itc=False)
# NEED to pass picks because default IGNORES SEEG channels
power_onset_perhead_vr.plot_topo(picks=plot_picks_perhead, baseline=(-2., -1.5), mode='logratio', layout=box)

## BIPORAL
picks_bip = mnehelp.def_picks(epochs_bip_vr['onsets_500_1500'])
box_bip = mnehelp.custom_box_layout(picks_bip)
plot_picks_bip = range(0, len(picks_bip))
power_onset_bip_vr = tfr_morlet(epochs_bip_vr['onsets_500_1500'], freqs=freqs, n_cycles=n_cycles, picks=picks_bip,
                                return_itc=False)

power_onset_bip_vr.plot_topo(picks=plot_picks_bip, baseline=(-2., -1.5), mode='logratio', layout=box_bip)

