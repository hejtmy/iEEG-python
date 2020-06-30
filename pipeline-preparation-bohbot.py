import mne
import numpy as np

from functions import mne_prepping as mneprep
from functions import mne_loading as loader
from functions import paths
from mne.time_frequency import tfr_morlet

# %% Setup
base_path = 'E:/OneDrive/FGU/iEEG/Data'
participant = 'p136'
scalings = {'seeg': 5e2, 'ecg': 1e2, 'misc': 1e2}
FULL_EPOCH_TIME = (-1.5, 1.5)
file_paths = paths.prep_unity_alloego_files(base_path, participant)
EEG_TYPE = 'perElectrode'

# %% Raw eeg loading
eeg, montage = loader.load_eeg(file_paths, EEG_TYPE)
eeg.notch_filter(50)

# %% Epoching
pd_events = mneprep.load_preprocessed_events(file_paths)
mne_events, events_mapp = mneprep.pd_to_mne_events(pd_events,
                                                   eeg.info['sfreq'])

epochs = mne.Epochs(eeg, mne_events, event_id=events_mapp, tmin=-5, tmax=5)

# %% epoch visualisation and bad epoch removal
epochs.plot(n_epochs=10, n_channels=25, scalings=scalings)
append = '-' + EEG_TYPE
mneprep.write_bad_epochs(epochs, file_paths, append=append)

# %% Loading already selected
append = '-' + EEG_TYPE
bad_epochs = mneprep.read_bad_epochs(file_paths, append=append)
epochs.drop(bad_epochs)

# %% CONVOLUTION
# Spectral power estimates were computed by convolving the filtered signal
# with six cycle Morlet wavelets at 32 logarithmically spaced frequencies
# ranging from 1 to 45 Hz. Because our hypotheses are concerning LFOs
# we focused on frequencies lower than 32 Hz.
freqs = np.logspace(np.log(1.0), np.log(45.0), num=32)
freqs = freqs[0:10]  # don't need the higher ones
n_cycles = 6

morlet = tfr_morlet(epochs, freqs, n_cycles=n_cycles, decim=5,
                    average=False, return_itc=False, n_jobs=1)

morlet.crop(*FULL_EPOCH_TIME)
append = '-bohbot-' + EEG_TYPE
mneprep.save_tfr_epochs(morlet, file_paths, append=append, overwrite=True)
