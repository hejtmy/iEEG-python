# Low-frequency theta oscillations in the humanhippocampus during real-world
# and virtual navigation

# Veronique D. Bohbot, Milagros S. Copara, Jean Gotman Arne D. Ekstrom
import mne
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

from functions import mne_prepping as mneprep
from functions import mne_loading as loader
from functions import mne_helpers as mnehelp
from functions import paths
from functions import mne_analysis as mneanalysis
from functions import mne_visualisations as mnevis
from mne.time_frequency import tfr_morlet

# from mne.time_frequency import tfr_multitaper, tfr_stockwell, tfr_morlet

base_path = 'E:/OneDrive/FGU/iEEG/Data'
participant = 'p136'
scalings = {'seeg': 1e2, 'ecg': 1e2, 'misc': 1e2}
FULL_EPOCH_TIME = (-1.0, 1.5)
EPOCH_TIME = (0, 1.0)
BASELINE_TIME = (-1.0, -0.5)

file_paths = paths.prep_unity_alloego_files(base_path, participant)
eeg, montage = loader.load_eeg(file_paths, 'perHeadbox')

pd_events = mneprep.load_preprocessed_events(file_paths)
mne_events, events_mapp = mneprep.pd_to_mne_events(pd_events, eeg.info['sfreq'])

# Preprocessing
eeg.notch_filter(50)

# Epoching
# Needs to give it long tails because of the low frequnecies
epochs = mne.Epochs(eeg, mne_events, event_id=events_mapp, tmin=-5, tmax=5)

# Spectral power estimates were computed by convolving the filtered signal
# with six cycle Morlet wavelets at 32 logarithmically spaced frequencies
# ranging from 1 to 45 Hz. Because our hypotheses are concerning LFOs
# we focused on frequencies lower than 32 Hz.
freqs = np.logspace(np.log(1.0), np.log(45.0), num=32)
freqs = freqs[0:10]  # don't need the higher ones
n_cycles = 6

events = ['onsets_500_1500', 'stops_500_1500']
morlet = tfr_morlet(epochs[events], freqs, n_cycles=n_cycles, decim=5,
                    average=False, return_itc=False, n_jobs=1)

morlet.crop(*FULL_EPOCH_TIME)
# Calculate the morlet convolutions and return epochsTFR, not AverageTFR

# MINE --------
morlet['onsets_500_1500'].data.shape
pick_all = mnehelp.picks_all(eeg, montage)
pick_all_names = mne.pick_info(eeg.info, pick_all)['ch_names']
pick_hip = mnehelp.picks_all_localised(eeg, montage, 'Hi')
pick_hip_names = mne.pick_info(eeg.info, pick_hip)['ch_names']

avg = morlet['onsets_500_1500'].average()
box = mne.channels.layout.make_grid_layout(avg.info, picks=pick_all)
avg.plot_topo(layout=box, picks=pick_all, baseline=(-1, -0.5),
              mode='zscore', title='Average power')

morlet['onsets_500_1500'].average().plot(0)

# These power values were binned into delta (1–4 Hz), theta(4–8 Hz) and
# alpha (8–12 Hz) frequency bands. Power values were also subsequently
# log transformed and then z-transformed.
lfo_bands = [[1, 4], [4, 8], [8, 13]]
morlet = mneanalysis.band_power(morlet, lfo_bands)

# Log transform the data
log_morlet = morlet.copy()
log_morlet.data = np.log(log_morlet.data)
# log_morlet.average().plot(23)

# Z transform the data
# MNE "zlogratio" divides by mean and then log transforms, not what we want.
# we want to do a z-transfrom on log-transformed data
z_morlet = log_morlet.copy().apply_baseline(BASELINE_TIME, mode="zscore")
# z_morlet = log_morlet.copy().apply_baseline(EPOCH_TIME, mode="zscore")
z_morlet.crop(*EPOCH_TIME)

# MNE WAY
# *This technically does each z scoring individually per epoch, not sure
# if this is wanted behavior - maybe this could be done differently
# https://github.com/mne-tools/mne-python/blob/76ee63ff92b0424a304a12532d0cb53c0833a0ec/mne/baseline.py
mnevis.plot_power_heatmap(morlet['onsets_500_1500'].copy().apply_baseline((-1, 2), mode='logratio').average().pick(pick_hip_names))
mnevis.plot_power_heatmap(morlet['stops_500_1500'].copy().apply_baseline((-1, -0.5), mode='logratio').average().pick(pick_hip_names))
mnevis.plot_power_heatmap(log_morlet.copy().average().pick(pick_hip_names))

# The number of electrode contacts with significantly different power values
# across experimental conditions was determined with t-tests across each
# frequency band at a P value 0.01. Thus,we averaged the log and z-transformed
# power for delta (1–4 Hz), theta (4–8 Hz), and alpha (8–12 Hz) bands for each
# condition (for example, Search).
average_onsets = np.average(z_morlet['onsets_500_1500'].data, axis=-1)
average_stops = np.average(z_morlet['stops_500_1500'].data, axis=-1)
# Epochs are epoch x channels x freqs x time
# averaged over time
all_pvalue1 = np.empty(average_onsets.shape[1:3])
all_pvalueInd = np.empty(average_onsets.shape[1:3])
for iChannel in range(0, average_onsets.shape[1]):
    for iFrequency in range(0, average_onsets.shape[2]):
        stat1, pvalue1 = stats.ttest_1samp(
            average_onsets[:, iChannel, iFrequency], 0)
        all_pvalue1[iChannel, iFrequency] = pvalue1
        statInd, pvalueInd = stats.ttest_ind(
            average_onsets[:, iChannel, iFrequency],
            average_stops[:, iChannel, iFrequency])
        all_pvalueInd[iChannel, iFrequency] = pvalueInd

# Channels where low theta is sig stronger in move than still
low_theta_channels = np.where(all_pvalueInd[:, 0] < 0.01)[0]
montage.iloc[low_theta_channels]
# Channels where high theta is sig stronger from still
high_theta_channels = np.where(all_pvalueInd[:, 1] < 0.01)[0]
montage.iloc[high_theta_channels]

# We then compared these values using a t-test between conditions
# (for example, search versus stop period).

# We corrected for multiple comparisons by bootstrapping the power values
# for each trial between movement conditions and stop conditions, conducting
# a one tailed t-test at each frequency.We extracted individual t-distributions
# for each patient as there were different numbers of trials for certain
# conditions, giving us variable degrees of freedom for each patient. When
# bootstrapped, each patient’s P value that separated the top 5% of t-values
# was approximately P 0.01. We applied an alpha value of P 0.01 (two-tailed)
# to conservatively correct for multiple comparisons across subfrequencies
# (delta, theta, alpha).

# For all movement conditions, all counts well exceeded those predicted by
# chance (that is, 1%, or 1 electrode contact). To compare different
# conditions (for example, search versus stop), we adopted two different
# approaches. Consistent with past work, we performed binomial tests, which
# estimate the confidence intervals for a significant number of electrode
# contacts compared with a chance distribution. We then compared the confidence
# intervals between different conditions, which allowed us to assess for
# significant differences between conditions. Additionally, we performed
# Chi Square goodness of fit tests on electrode counts, consistent with
# past work. We did this by taking the electrode contacts across a frequency
# band and comparing with an evendistribution of electrode contacts across
# these three frequency bands. Finally,we performed Fisher’s exact tests
# on electrode counts across frequency bands. This allowed us to look for
# crossover interaction effects, in other words, a differentialdistribution
# of electrodes as a function of both condition and frequency band
