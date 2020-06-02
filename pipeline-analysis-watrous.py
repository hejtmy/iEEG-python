# Behavioral correlates of human hippocampal delta and theta
# oscillations during navigation

# Andrew J. Watrous Itzhak Fried and Arne D. Ekstrom

import mne
import numpy as np

from functions import mne_prepping as mneprep
from functions import mne_loading as loader
from functions import paths
from functions import mne_analysis as mneanalysis

# from mne.time_frequency import tfr_multitaper, tfr_stockwell, tfr_morlet

base_path = 'E:/OneDrive/FGU/iEEG/Data'
participant = 'p136'
scalings = {'seeg': 1e2, 'ecg': 1e2, 'misc': 1e2}

file_paths = paths.prep_unity_alloego_files(base_path, participant)
eeg, montage = loader.load_eeg(file_paths, 'bipolar')

pd_events = mneprep.load_preprocessed_events(file_paths)
mne_events, events_mapp = mneprep.pd_to_mne_events(pd_events, eeg.info['sfreq'])

# Preprocessing
eeg.notch_filter(50)

# Raw signals were downsampled to 1,000 Hz, band-passfiltered from 1 to 200 Hz,
# and notch filtered at 60 Hz (59.9–60.9 Hz)to minimize line noise artifacts
# during postprocessing.

# The speed predictor variable represented the subject’s virtual linear speed
# in the environment. The task predictor variable corresponded to whether
# the subject was searching for passengers or stores, and the store viewing
# predictor variable (spatial view) represented whether the subject was
# viewing a goal store, nongoal store, or the background (i.e., no store).

# The goal store was defined as the virtual passenger’s destination store,
# and nongoal stores were all other stores during a given delivery. Periods of
# navigation were subdivided into time points of goal store viewing, nongoal
# store viewing, and background viewing and were segmented into 200-ms epochs.
# These epochs were used to create each type of predictor variable as well as
# the log-power-dependent variable.

# To determine whether an electrode showed a statistically significant effect
# of a behavioral variable, we employed a bootstrap resampling procedure to
# estimate the null hypothesis distribution. This procedure involved shuffling
# the labels of each predictor variable 1,000 times at each electrode and
# recomputing the multilinear regression. This allowed us to produce a pseudo
# t-value distribution separately for each regressor by pooling the resulting
# t-values across electrodes within each frequency band. We then selected
# the pseudo t-value in the 99. 99th percentile (i.e., bootstrap alpha
# threshold 0.001) as the critical t-value for each predictor variable,
# and electrodeswith t-values greater than the critical t-value were
# deemed significant.
