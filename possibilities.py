# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 16:29:33 2017

@author: Lukáš
"""

# EVOKED -------------------------------------------------
conditions = ["pointingEnded_Ego", "pointingEnded_Allo"]
evoked_dict = dict()
for condition in conditions:
    evoked_dict[condition] = epochs_perhead_vr[condition].average()

colors = dict(pointingEnded_Ego="Crimson", pointingEnded_Allo="CornFlowerBlue")
mne.viz.plot_compare_evokeds(evoked_dict, picks=pick_perhead_hip, colors=colors)


mnestats.plot_wilcox(wilcox_allo_ego, 0, 256)
for channel in range(len(pick_perhead_hip)):
    mnestats.plot_wilcox(wilcox_allo_ego, channel, 256, freqs = wilcox_freqs)