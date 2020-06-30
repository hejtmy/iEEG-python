# Low-frequency theta oscillations in the humanhippocampus during real-world
# and virtual navigation

# Veronique D. Bohbot, Milagros S. Copara, Jean Gotman Arne D. Ekstrom
# %%
import mne
import numpy as np
import scipy.stats as stats

from functions import mne_prepping as mneprep
from functions import mne_loading as loader
from functions import mne_helpers as mnehelp
from functions import paths
from functions import mne_analysis as mneanalysis
from functions import mne_visualisations as mnevis

# from mne.time_frequency import tfr_multitaper, tfr_stockwell, tfr_morlet
# %% Setup
base_path = 'E:/OneDrive/FGU/iEEG/Data'
participant = 'p129'
scalings = {'seeg': 5e2, 'ecg': 1e2, 'misc': 1e2}
FULL_EPOCH_TIME = (-1.5, 1.5)
EPOCH_TIME = (0, 1.5)
BASELINE_TIME = (-0.5, -0.2)
file_paths = paths.prep_unity_alloego_files(base_path, participant)
EEG_TYPE = 'perElectrode'

# %% Raw eeg loading
eeg, montage = loader.load_eeg(file_paths, EEG_TYPE)
eeg.notch_filter(50)

# %% Epoching
pd_events = mneprep.load_preprocessed_events(file_paths)
mne_events, events_mapp = mneprep.pd_to_mne_events(pd_events,
                                                   eeg.info['sfreq'])

# Needs to give it long tails because of the
# low frequnecies later in the convolution
epochs = mne.Epochs(eeg, mne_events, event_id=events_mapp, tmin=-5, tmax=5)

bad_epochs = mneprep.read_bad_epochs(file_paths, append='-perElectrode')
epochs.drop(bad_epochs)

# %% Analysis
morlet = mneprep.load_tfr_epochs(file_paths, append='-bohbot-perElectrode')
montage = loader.load_montage(file_paths, EEG_TYPE)

# %% PICKS
pick_all = mnehelp.picks_all(morlet, montage)
pick_all_names = mne.pick_info(morlet.info, pick_all)['ch_names']
pick_hip = mnehelp.picks_all_localised(morlet, montage, 'Hi')
pick_hip_names = mne.pick_info(morlet.info, pick_hip)['ch_names']

# %% Not important
avg = morlet['onsets_500_1500'].average()
box = mne.channels.layout.make_grid_layout(avg.info, picks=pick_all)
avg.plot_topo(layout=box, picks=pick_all, baseline=BASELINE_TIME,
              mode='zscore', title='Average power')

# %% Processing
# These power values were binned into delta (1–4 Hz), theta(4–8 Hz) and
# alpha (8–12 Hz) frequency bands. Power values were also subsequently
# log transformed and then z-transformed.
lfo_bands = [[1, 4], [4, 8], [8, 13]]
morlet_bands = mneanalysis.band_power(morlet.copy(), lfo_bands)

# Log transform the data
log_morlet = mneanalysis.log_transform(morlet_bands.copy())

# Z transform the data
# z_morlet = mneanalysis.z_transform_baseline(log_morlet.copy(), (-1.0, 1.5))
z_morlet = mneanalysis.z_transform_all(log_morlet.copy())
# z_morlet.crop(*EPOCH_TIME)

# %% Visualisaiotns
# *MNE  technically does each z scoring individually per epoch, not sure
# if this is wanted behavior - maybe this could be done differently
# https://github.com/mne-tools/mne-python/blob/76ee63ff92b0424a304a12532d0cb53c0833a0ec/mne/baseline.py
box = mne.channels.layout.make_grid_layout(z_morlet.info, picks=pick_all)
z_morlet['onsets_500_1500'].average().plot_topo(layout=box, picks=pick_hip, title='Average power')
z_morlet['stops_500_1500'].average().plot_topo(layout=box, picks=pick_all, title='Average power')

# %%
mnevis.plot_power_heatmap(z_morlet['onsets_500_1500'].average().pick(pick_hip_names), ylim=(-0.5, 0.5))
mnevis.plot_power_heatmap(z_morlet['stops_500_1500'].average().pick(pick_hip_names), ylim=(-0.5, 0.5))

# %% Comparisons
# The number of electrode contacts with significantly different power values
# across experimental conditions was determined with t-tests across each
# frequency band at a P value 0.01. Thus,we averaged the log and z-transformed
# power for delta (1–4 Hz), theta (4–8 Hz), and alpha (8–12 Hz) bands for each
# condition (for example, Search).
average_onsets = np.average(z_morlet['onsets_500_1500'].data, axis=-1)
average_stops = np.average(z_morlet['stops_500_1500'].data, axis=-1)
# Epochs are epoch x channels x freqs x timen averaged over time

# We then compared these values using a t-test between conditions
# (for example, search versus stop period).
all_pvalue1 = np.empty(average_onsets.shape[1:3])
all_pvalueInd = np.empty(average_onsets.shape[1:3])
for iChannel in range(0, average_onsets.shape[1]):
    for iFrequency in range(0, average_onsets.shape[2]):
        stat1, pvalue1 = stats.ttest_1samp(
            average_onsets[:, iChannel, iFrequency], 0)
        all_pvalue1[iChannel, iFrequency] = pvalue1*np.sign(stat1)
        # negative means the second is larger
        statInd, pvalueInd = stats.ttest_ind(
            average_onsets[:, iChannel, iFrequency],
            average_stops[:, iChannel, iFrequency],
            equal_var=False)
        all_pvalueInd[iChannel, iFrequency] = pvalueInd*np.sign(statInd)

# %% MOVE VS STILL
# Channels where low theta is sig stronger in move than still
low_theta_channels = np.where((all_pvalueInd[:, 0] < 0.01) & (all_pvalueInd[:, 0] > 0))[0]
montage.iloc[low_theta_channels]
# Channels where high theta is sig stronger from still
high_theta_channels = np.where((all_pvalueInd[:, 1] < 0.01) & (all_pvalueInd[:, 1] > 0))[0]
montage.iloc[high_theta_channels]

# %% STILL VS MOVE
# Channels where low theta is sig stronger in still than move
low_theta_channels = np.where((-all_pvalueInd[:, 0] < 0.001) & (-all_pvalueInd[:, 0] > 0))[0]
montage.iloc[low_theta_channels]
# Channels where high theta is sig stronger in still from move
high_theta_channels = np.where((-all_pvalueInd[:, 1] < 0.001) & (-all_pvalueInd[:, 1] > 0))[0]
montage.iloc[high_theta_channels]


# %%



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


# %%
