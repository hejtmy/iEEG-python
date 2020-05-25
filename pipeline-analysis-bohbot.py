# Low-frequency theta oscillations in the humanhippocampus during real-world and virtualnavigation

# Ve ́ronique D. Bohbot1, Milagros S. Copara2, Jean Gotman3& Arne D. Ekstrom2,4,5
import mne
import numpy as np

from functions import mne_prepping as mneprep
from functions import mne_loading as loader
from functions import mne_helpers as mnehelp
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

# Epoching
# Needs to give it long tails because of the low frequnecies
epochs = mne.Epochs(eeg, mne_events, event_id=events_mapp, tmin=-5, tmax=5)

# Spectral power estimates were computed by convolving the filtered signal with six cycle Morlet wavelets at 32 logarithmically spaced frequencies ranging from 1 to 45 Hz. Because our hypotheses are concerning LFOs we focused on frequencies lower than 32 Hz.
freqs = np.logspace(np.log(1.0), np.log(45.0), num=32)
freqs = freqs[0:10]  # don't need the higher ones
n_cycles = 6

# Calculate the morlet convolutions and return epochsTFR, not AverageTFR
events = ['onsets_500_1500', 'stops_500_1500']
morlet = mneanalysis.morlet_all_events(
    epochs, freqs, n_cycles, average=False, events=events)

morlet['onsets_500_1500'].data.shape

# These power values were binned into delta (1–4 Hz), theta(4–8 Hz) and alpha (8–12 Hz) frequency bands. Power values were also subsequently log transformed and then z-transformed.
box = mnehelp.custom_box_layout(epochs.info['ch_names'], 6)
pick_all = np.arange(0,112,1)
morlet['onsets_500_1500'].average().plot_topo(picks=pick_all, layout=box, baseline=(-1, -0.5))
morlet['onsets_500_1500'].average().plot(13)

# bin into bins
lfo_bands = [[2, 4], [4, 9]]
morlet_bands = mneanalysis.convolutions_band_power(morlet, lfo_bands)
morlet_bands['onsets_500_1500'].average().plot_topo(picks=pick_all, layout=box, baseline=(-1, -0.5))

# Log transform the data

# Z transform the data - PER CHANNEL

# No baseline???

# Crop the 0 1500
# EpochsTFR.crop

# The number of electrode contacts with significantly different power values across experimental conditions was determined with t-tests across each frequency band at a P value 0.01. Thus,we averaged the log and z-transformed power for delta (1–4 Hz), theta (4–8 Hz),and alpha (8–12 Hz) bands for each condition (for example, Search). 

# The data were epoched with respect to the natural occurrence of the events during the tasks (see average event duration in task description), except for Walk which was one long event that we parsed into 20 s events. 

# We then compared these values using a t-test between conditions (for example, search versus stop period).

# We corrected for multiple comparisons by bootstrapping the power values for each trial between movement conditions and stop conditions, conducting a one tailed t-test at each frequency. We extracted individual t-distributions for each patient as there were different numbers of trials for certain conditions, giving us variable degrees of freedom for each patient. When bootstrapped, each patient’s P value that separated the top 5% of t-values was approximately P 0.01. We applied an alpha value of P 0.01 (two-tailed) to conservatively correct for multiple comparisons across subfrequencies (delta, theta, alpha). 

# For all movement conditions, all counts well exceeded those predicted by chance (that is, 1%, or 1 electrode contact, see Table 3). To compare different conditions (for example, search versus stop), we adopted two different approaches. Consistent with past work, we performed binomial tests, which estimate the confidence intervals for a significant number of electrodecontacts compared with a chance distribution. We then compared the confidence intervals between different conditions, which allowed us to assess for significant differences between conditions. Additionally, we performed Chi Square goodness of fit tests on electrode counts, consistent with past work. We did this by taking the electrode contacts across a frequency band and comparing with an evendistribution of electrode contacts across these three frequency bands. Finally,we performed Fisher’s exact tests on electrode counts across frequency bands. This allowed us to look for crossover interaction effects, in other words, a differentialdistribution of electrodes as a function of both condition and frequency band.