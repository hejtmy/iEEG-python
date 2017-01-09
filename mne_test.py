import mne
import numpy as np

from functions import mne_prepping as mneprep
from functions import mne_helpers as mnehelp
from mne.time_frequency import tfr_multitaper, tfr_stockwell, tfr_morlet

path = "U:\\OneDrive\\FGU\\iEEG\\p83\\UTAlloEgo\\EEG\\Preprocessed\\prep_bipolar_250.mat"
path = "D:\\IntracranialElectrodes\\Data\\p83\\UTAlloEgo\\EEG\\Preprocessed\\prep_perHeadbox_250.mat"
path = "D:\\IntracranialElectrodes\\Data\\p83\\v\\EEG\\Preprocessed\\prep_perElectrode_250.mat"
path = "D:\\IntracranialElectrodes\\Data\\p83\\UTAlloEgo\\EEG\\Preprocessed\\prep_250.mat"

FREQUENCY = 250

path_events = "U:\\OneDrive\\FGU\\iEEG\\p83\\UTAlloEgo\\EEG\\Preprocessed\\p83_UT.csv"
path_events = "D:/IntracranialElectrodes/Data/p83/UTAlloEgo/experiment_data/p83_ut.csv"

raw = mneprep.load_raw(path, FREQUENCY)
mne_events, mapp = mneprep.load_events(path_events, FREQUENCY)

raw.plot(events = mne_events, scalings = 'auto')

epochs = mne.Epochs(raw, mne_events, event_id = mapp, tmin=-0.5, tmax=2, add_eeg_ref = False)
epochs.plot(block = True, scalings = 'auto')

onsets = epochs['onsets_500_1500']
onsets.plot(scalings = 'auto', n_epochs = 5)

stops = epochs['stops_500_1500']
stops.plot(scalings = 'auto', n_epochs = 5)

freqs = np.arange(2, 10, 1)
n_cycles = freqs / 2

picks = mnehelp.def_picks(raw)
power = tfr_morlet(onsets, freqs=freqs, n_cycles=n_cycles, picks = picks, return_itc=False)
power.plot()