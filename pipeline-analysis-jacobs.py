# Lateralized hippocampal oscillations underliedistinct aspects of
# human spatial memoryand navigation

# Jonathan Miller, Andrew J. Watrous, Melina Tsitsiklis,
# Sang Ah Lee, .... & Joshua Jacobs2

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
# To reduce confounding noise artifacts, we used a bipolar referencing scheme
# where by we identified all pairs of immediately adjacent electrodes and
# calculated the voltage difference between both contacts in the pair.
# The location of these new virtual electrodes was taken to be the midpoint
# between the two physical contacts(inter-electrode spacing: 10 mm).
eeg, montage = loader.load_eeg(file_paths, 'bipolar')

pd_events = mneprep.load_preprocessed_events(file_paths)
mne_events, events_mapp = mneprep.pd_to_mne_events(pd_events, eeg.info['sfreq'])

# Preprocessing

# To further reduce electrical line noise, a band-stop 4th order Butterworth
# filter was applied at 58–62 Hz. To eliminate events contaminated by
# epileptiform activity, we excluded time periods of interest if the
# kurtosis of the voltage trace exceeded a threshold of 563. This in
# the exclusion of an average of 7.1% ± 1.7% of experimental events.
eeg.notch_filter(50)

# Additionally, clinicians at each collaborating hospital identified which
# electrodes were located in seizure onset zones. For our control analysis
# that only included a subset of hippocampal contacts, we excluded a subject’s
# hippocampalelectrodes in a given hemisphere if any MTL electrode in the
# ipsilateral hemispherefell inside of a clinically defined seizure onset zone

# To analyze the spectral properties of the recorded signals, we calculated
# the continuous Morlet wavelet transform (wave number 5) at 50 logarithmically
# spaced frequencies between 1 and 200 Hz. A 3000-ms buffer was added toboth
# ends of all power computations before wavelet decomposition in order to
# minimize edge effects.

# For analyses of item encoding (Figs.3,6), spectral power was computed at each
#  sample in the 0–1500-ms item presentation window and then averaged over time

# For analyses of average power during navigation and non-navigation periods,
# power was computed and then averaged over each variable duration navigation
# and pre-trial baselineperiod (see“Analysis of navigation epochs”below).

# The resulting power values were then log-transformed (with the exception of
# the individual electrode power spectra shown in Fig.4b, we z-transformed the
# log-power values within session). This z-transformation was performed
# separately for every session, electrode, and frequency, by subtracting
# the mean log-power across all event types (encoding, navigation,and baseline)
# and dividing by the standard deviation.

# For time–frequency spectrograms (Fig.4), we averaged log-power into
# overlapping windows of 100 ms each in steps of 50 ms, between−1500
# and 2000 ms relative to item onset. We then z-scored as described
# above, now separately for session, electrode, frequency, and timepoint.

# For the longer timeperiods shown in Fig.5, we averaged log-power into
# overlapping windows of 500 ms each, in steps of 100 ms, between
# 2250 and 3750 ms relative to item onset, and we normalized the z-power
# values based on the mean power of the baselinecondition.

# To determine whether changes in spectral power were due to the presence of
# narrowband oscillations, we performed the oscillation detection procedure of
# Manning. First, we computed the mean power spectra between 1 and 50 Hz for
# each electrode and condition and then used a robust regression to fit
# the 1/f background power spectrum. We labeled any frequency where the
# residual power was greater than one standard deviation above the background
# 1/f as exhibiting narrow band oscillatory activity
