import mne
import numpy as np

from functions import mne_prepping as mneprep
from functions import mne_helpers as mnehelp
from mne.time_frequency import tfr_multitaper, tfr_stockwell, tfr_morlet

path_bip = "U:\\OneDrive\\FGU\\iEEG\\p83\\UTAlloEgo\\EEG\\Preprocessed\\prep_bipolar_250.mat"
path_perHead = "D:\\IntracranialElectrodes\\Data\\p83\\UTAlloEgo\\EEG\\Preprocessed\\prep_perHeadbox_250.mat"
path_perElec = "D:\\IntracranialElectrodes\\Data\\p83\\v\\EEG\\Preprocessed\\prep_perElectrode_250.mat"
path = "D:\\IntracranialElectrodes\\Data\\p83\\UTAlloEgo\\EEG\\Preprocessed\\prep_250.mat"

FREQUENCY = 250

path_events = "U:\\OneDrive\\FGU\\iEEG\\p83\\UTAlloEgo\\EEG\\Preprocessed\\p83_UT.csv"
path_events = "D:\\IntracranialElectrodes\\Data\\p83\\UTAlloEgo\\experiment_data\\p83_ut.csv"

raw_perhead = mneprep.load_raw(path_perHead, FREQUENCY)
raw_bip = mneprep.load_raw(path_bip, FREQUENCY)
mne_events, mapp = mneprep.load_events(path_events, FREQUENCY)

raw_perhead.plot(events = mne_events, scalings = 'auto')
raw_perhead.plot(events = mne_events, scalings = 'auto', event_color = {4 : 'green', 5: 'red'}) #coloring events 4 and 5

raw_bip.plot(events = mne_events, scalings = 'auto')
raw_bip.plot(events = mne_events, scalings = 'auto', event_color = {4 : 'green', 5: 'red'}) #coloring events 4 and 5

epochs_perHead = mne.Epochs(raw_perhead, mne_events, event_id = mapp, tmin=-0.5, tmax=2, add_eeg_ref = False)
epochs_perHead.plot(block = True, scalings = 'auto')
epochs_perHead['onsets_0_1500', 'stops_500_1500'].plot(block = True, scalings = 'auto')

epochs_bip = mne.Epochs(raw_bip, mne_events, event_id = mapp, tmin=-0.5, tmax=2, add_eeg_ref = False)
epochs_bip.plot(block = True, scalings = 'auto')
epochs_bip['onsets_0_1500', 'stops_500_1500'].plot(block = True, scalings = 'auto')

onsets_perHead = epochs_perHead['onsets_0_1500']
onsets_perHead.plot(scalings = 'auto', n_epochs = 5)
onsets_bip = epochs_bip['onsets_0_1500']
stops_perHead = epochs_perHead['stops_500_1500']
stops_bip = epochs_bip['stops_500_1500']

freqs = np.arange(2, 10, 1)
n_cycles = freqs / 2

picks = mnehelp.def_picks(stops_bip)
power = tfr_morlet(stops_bip, freqs = freqs, n_cycles=n_cycles, picks = picks, return_itc=False)
power.plot([0], baseline=(0., 0.1), mode = 'mean', vmin=-1., vmax=3.,
           title='Sim: Using Morlet wavelet')