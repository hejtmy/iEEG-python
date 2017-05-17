import mne
import numpy as np
from functions import mne_prepping as mneprep
from functions import mne_helpers as mnehelp
from mne.time_frequency import tfr_multitaper, tfr_stockwell, tfr_morlet

path_perhead_vr = "D:\\IntracranialElectrodes\\Data\\p129\\UnityAlloEgo\\EEG\\Preprocessed\\prep_perHeadbox_256.mat"
path_bip_vr = "D:\\IntracranialElectrodes\\Data\\p129\\UnityAlloEgo\\EEG\\Preprocessed\\prep_bipolar_256.mat"
path_perelectrode_vr = "D:\\IntracranialElectrodes\\Data\\p129\\UnityAlloEgo\\EEG\\Preprocessed\\prep_perElectrode_256.mat"
path_events_vr = "D:\\IntracranialElectrodes\\Data\\p129\\UnityAlloEgo\\experiment_data\\p129_unity500_1500.csv"

FREQUENCY = 256

# Loading Unity data
raw_perhead_vr = mneprep.load_raw(path_perhead_vr, FREQUENCY)
raw_perelectrode_vr = mneprep.load_raw(path_perelectrode_vr, FREQUENCY)
raw_bip_vr = mneprep.load_raw(path_bip_vr, FREQUENCY)
mne_events_vr, mapp_vr = mneprep.load_events(path_events_vr, FREQUENCY)

# {'ArduinoPulseStop': blue, 'onsets_500_1500': green, 'stops_500_1500': red}
#raw_perhead_vr.plot(events = mne_events_vr, scalings = 'auto', event_color = {1 : 'blue', 2 : 'green', 3: 'red'})
#raw_bip_vr.plot(events = mne_events_vr, scalings = 'auto', event_color = {1 : 'blue', 2 : 'green', 3: 'red'})

raw_perelectrode_vr.info["bads"] = ['40', '34', '35', '36', '37', '38', '39', '47']
raw_perhead_vr.info["bads"] = ['23', '34', '35', '47','67']
raw_bip_vr.info["bads"] = ['21', '24','42','43','60']

## Epoching
reject = {'mag': 4e-12, 'eog': 200e-6}
epochs_perhead_vr = mne.Epochs(raw_perhead_vr, mne_events_vr, event_id = mapp_vr, tmin = -3, tmax = 3, add_eeg_ref = False)
epochs_perelectrode_vr = mne.Epochs(raw_perelectrode_vr, mne_events_vr, event_id = mapp_vr, tmin = -3, tmax = 3, add_eeg_ref = False)
epochs_bip_vr = mne.Epochs(raw_bip_vr, mne_events_vr, event_id = mapp_vr, tmin=-3, tmax = 3, add_eeg_ref = False)

onsets_perhead = epochs_perhead_vr['onsets_500_1500']
onsets_bip = epochs_bip_vr['onsets_500_1500']

#epochs_perelectrode_vr['onsets_500_1500', 'stops_500_1500'].plot(block = True, scalings = 'auto')
#epochs_perhead_vr['onsets_500_1500', 'stops_500_1500'].plot(block = True, scalings = 'auto')
#epochs_bip_vr['onsets_500_1500', 'stops_500_1500'].plot(block = True, scalings = 'auto')

freqs = np.arange(2, 30, 1)
n_cycles = freqs / 2

picks_perhead = mnehelp.def_picks(epochs_perhead_vr['onsets_500_1500'])
box =  mnehelp.custom_box_layout(picks_perhead)
plot_picks_perhead = range(0, len(picks_perhead))

power_all_perhead_vr = tfr_morlet(epochs_perhead_vr, freqs = freqs, n_cycles = n_cycles, picks = picks_perhead, return_itc = False)

power_onset_perhead_vr = tfr_morlet(epochs_perhead_vr['onsets_500_1500'], freqs = freqs, n_cycles = n_cycles, picks = picks_perhead, return_itc = False)
# NEED to pass picks because default IGNORES SEEG channels
power_onset_perhead_vr.plot_topo(picks = plot_picks_perhead, baseline=(-2., -1.5), mode = 'logratio', layout = box)

power_onset_perhead_vr_divided = power_onset_perhead_vr
power_onset_perhead_vr_divided.data = power_onset_perhead_vr.data/power_all_perhead_vr.data
power_onset_perhead_vr_divided.plot_topo(picks = plot_picks_perhead, baseline=(-2., -1.5), mode = 'logratio', layout = box)

power_stops_perhead_vr = tfr_morlet(epochs_perhead_vr['stops_500_1500'], freqs = freqs, n_cycles = n_cycles, picks = picks_perhead, return_itc = False)
# NEED to pass picks because default IGNORES SEEG channels
power_stops_perhead_vr.plot_topo(picks = plot_picks_perhead, baseline=(-2., -1.5), mode = 'logratio', layout = box)

## BIPOLAR
picks_bip = mnehelp.def_picks(epochs_bip_vr['onsets_500_1500'])
box_bip =  mnehelp.custom_box_layout(picks_bip)
plot_picks_bip = range(0, len(picks_bip))
power_onset_bip_vr = tfr_morlet(epochs_bip_+vr['onsets_500_1500'], freqs = freqs, n_cycles=n_cycles, picks = picks_bip, return_itc = False)

power_onset_bip_vr.plot_topo(picks = plot_picks_bip, baseline=(-2., -1.5), mode = 'logratio', layout = box_bip)