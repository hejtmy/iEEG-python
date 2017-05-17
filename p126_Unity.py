import mne
import numpy as np

from functions import mne_prepping as mneprep
from functions import mne_helpers as mnehelp
from mne.time_frequency import tfr_multitaper, tfr_stockwell, tfr_morlet

path_perhead_vr = "D:\\IntracranialElectrodes\\Data\\p126\\UnityAlloEgo\\EEG\\Preprocessed\\prep_perHeadbox_250.mat"
path_bip_vr = "D:\\IntracranialElectrodes\\Data\\p126\\UnityAlloEgo\\EEG\\Preprocessed\\prep_bipolar_250.mat"
path_events_vr = "D:\\IntracranialElectrodes\\Data\\p126\\UnityAlloEgo\\experiment_data\\p126_unity500_1500.csv"

path_perhead_bva = "D:\\IntracranialElectrodes\\Data\\p126\\BVAAlloEgo\\EEG\\Preprocessed\\prep_perHeadbox_250.mat"
path_bip_bva = "D:\\IntracranialElectrodes\\Data\\p126\\BVAAlloEgo\\EEG\\Preprocessed\\prep_bipolar_250.mat"
path_events_bva = "D:\\IntracranialElectrodes\\Data\\p126\\BVAAlloEgo\\experiment_data\\p126_BVA500_1500.csv"

FREQUENCY = 250

# Loading Unnity data
raw_perhead_vr = mneprep.load_raw(path_perhead_vr, FREQUENCY)
raw_bip_vr = mneprep.load_raw(path_bip_vr, FREQUENCY)
mne_events_vr, mapp_vr = mneprep.load_events(path_events_vr, FREQUENCY)

# loading BVA data
raw_perhead_bva = mneprep.load_raw(path_perhead_bva, FREQUENCY)
raw_bip_bva = mneprep.load_raw(path_bip_bva, FREQUENCY)
mne_events_bva, mapp_bva = mneprep.load_events(path_events_bva, FREQUENCY)

# {'ArduinoPulseStop': blue, 'onsets_500_1500': green, 'stops_500_1500': red}
raw_perhead_vr.plot(events = mne_events_vr, scalings = 'auto', event_color = {1 : 'blue', 2 : 'green', 3: 'red'})
raw_bip_vr.plot(events = mne_events_vr, scalings = 'auto', event_color = {1 : 'blue', 2 : 'green', 3: 'red'})

raw_perhead_vr.info["bads"] = ['55', '56', '57', '58', '59']
raw_bip_vr.info["bads"] = ['45', '46', '47', '48']

# {'c': 1, 'f': 2, 'g': 3, 'onsets_500_1500': 4, 'stops_500_1500': 5}
raw_perhead_bva.plot(events = mne_events_bva, scalings = 'auto', event_color = {1 : 'blue', 2 : 'blue', 3 : 'blue', 4 : 'green', 5: 'red'})
raw_bip_bva.plot(events = mne_events_bva, scalings = 'auto', event_color = {1 : 'blue', 2 : 'blue', 3 : 'blue', 4 : 'green', 5: 'red'})

raw_perhead_bva.info["bads"] = ['55', '56', '57', '58', '59']
raw_bip_bva.info["bads"] = ['45', '46', '47', '48']

## Epoching
epochs_perhead_vr = mne.Epochs(raw_perhead_vr, mne_events_vr, event_id = mapp_vr, tmin = -3, tmax = 3, add_eeg_ref = False)
epochs_bip_vr = mne.Epochs(raw_bip_vr, mne_events_vr, event_id = mapp_vr, tmin=-3, tmax = 3, add_eeg_ref = False)

epochs_perhead_bva = mne.Epochs(raw_perhead_bva, mne_events_vr, event_id = mapp_vr, tmin = -3, tmax = 3, add_eeg_ref = False)
epochs_bip_bva = mne.Epochs(raw_bip_bva, mne_events_vr, event_id = mapp_vr, tmin=-3, tmax=3, add_eeg_ref = False)

onsets_perhead = epochs_perhead_vr['onsets_500_1500']
onsets_bip = epochs_bip_vr['onsets_500_1500']
stops_perhead_bva = epochs_perhead_bva['stops_500_1500']
stops_bip_bva = epochs_bip_bva['stops_500_1500']

epochs_perhead_vr['onsets_500_1500', 'stops_500_1500'].plot(block = True, scalings = 'auto')
epochs_bip_vr['onsets_500_1500', 'stops_500_1500'].plot(block = True, scalings = 'auto')

epochs_perhead_bva['onsets_500_1500', 'stops_500_1500'].plot(block = True, scalings = 'auto')
epochs_bip_bva['onsets_500_1500', 'stops_500_1500'].plot(block = True, scalings = 'auto')

freqs = np.arange(2, 30, 1)
n_cycles = freqs / 2

picks_perhead = mnehelp.def_picks(epochs_perhead_vr['onsets_500_1500'])
box =  mnehelp.custom_box_layout(picks_perhead)
plot_picks_perhead = range(0, len(picks_perhead))

power_onset_perhead_vr = tfr_morlet(epochs_perhead_vr['onsets_500_1500'], freqs = freqs, n_cycles = n_cycles, picks = picks_perhead, return_itc = False)
# NEED to pass picks because default IGNORES SEEG channels
power_onset_perhead_vr.plot_topo(picks = plot_picks_perhead, baseline = (-2., -1.5), mode = 'logratio', layout = box)

power_onset_perhead_bva = tfr_morlet(epochs_perhead_bva['onsets_500_1500'], freqs = freqs, n_cycles = n_cycles, picks = picks_perhead, return_itc = False)
# NEED to pass picks because default IGNORES SEEG channels
power_onset_perhead_bva.plot_topo(picks = plot_picks_perhead, baseline = (-2., -1.5), mode = 'logratio', layout = box)

## BIPORAL
picks_bip = mnehelp.def_picks(epochs_bip_vr['onsets_500_1500'])
box_bip =  mnehelp.custom_box_layout(picks_bip)
plot_picks_bip = range(0, len(picks_bip))
power_onset_bip_vr = tfr_morlet(epochs_bip_vr['onsets_500_1500'], freqs = freqs, n_cycles = n_cycles, picks = picks_bip, return_itc = False)

power_onset_bip_vr.plot_topo(picks = plot_picks_bip, baseline=(-2., -1.5), mode = 'logratio', layout = box_bip)