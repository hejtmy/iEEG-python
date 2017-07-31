# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 12:57:20 2017

@author: hejtm
"""

# POINTING
power_point_perhead_vr_ego = tfr_morlet(epochs_perhead_vr['pointingEnded_Ego'], freqs=freqs, n_cycles=n_cycles, picks = pick_perhead_all, return_itc=False)
power_point_perhead_vr_allo = tfr_morlet(epochs_perhead_vr['pointingEnded_Allo'], freqs=freqs, n_cycles=n_cycles,picks = pick_perhead_all, return_itc=False)

power_trials_point_perhead_ego = tfr_morlet(epochs_perhead_vr['pointingEnded_Ego'], freqs=freqs, n_cycles = n_cycles, picks = pick_perhead_all, return_itc = False, average = False)
power_trials_point_perhead_allo = tfr_morlet(epochs_perhead_vr['pointingEnded_Allo'], freqs = freqs, n_cycles = n_cycles, picks = pick_perhead_all, return_itc = False, average = False)

power_point_start_perhead_vr_ego = tfr_morlet(epochs_perhead_vr['pointingStarted_Ego'], freqs=freqs, n_cycles=n_cycles, picks = pick_perhead_all, return_itc=False)
power_point_start_perhead_vr_allo = tfr_morlet(epochs_perhead_vr['pointingStarted_Allo'], freqs=freqs, n_cycles=n_cycles,picks = pick_perhead_all, return_itc = False)

power_trials_point_start_perhead_ego = tfr_morlet(epochs_perhead_vr['pointingStarted_Ego'], freqs=freqs, n_cycles = n_cycles, picks = pick_perhead_all, return_itc = False, average = False)
power_trials_point_start_perhead_allo = tfr_morlet(epochs_perhead_vr['pointingStarted_Allo'], freqs = freqs, n_cycles = n_cycles, picks = pick_perhead_all, return_itc = False, average = False)

# ONSETS
power_onset_perhead_vr = tfr_morlet(epochs_perhead_vr['onsets_500_1500'], freqs = freqs, n_cycles = n_cycles, picks = pick_perhead_all, return_itc = False)
power_stop_perhead_vr = tfr_morlet(epochs_perhead_vr['stops_500_1500'], freqs = freqs, n_cycles = n_cycles, picks = pick_perhead_all, return_itc = False)

power_trials_onset_perhead_vr = tfr_morlet(epochs_perhead_vr['onsets_500_1500'], freqs = freqs, n_cycles = n_cycles, picks = pick_perhead_all, return_itc = False, average = False)
power_trials_stop_perhead_vr = tfr_morlet(epochs_perhead_vr['stops_500_1500'], freqs = freqs, n_cycles = n_cycles, picks = pick_perhead_all, return_itc = False, average = False)