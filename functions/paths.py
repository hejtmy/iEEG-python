import os
import re


def get_frequency(eeg_path):
    ptr = 'prep_.*_(\\d+)\\.mat'
    for root, dirs, files in os.walk(eeg_path):
        for file in files:
            res = re.match(ptr, file)
            if res:
                return int(res.group(1))
    return None


def participant_path(base, participant):
    return os.path.join(base, participant)


def unity_alloego_path(base, participant):
    return os.path.join(base, participant, 'UnityAlloEgo')


def eeg_path(experiment_path):
    return os.path.join(experiment_path, 'EEG', 'Preprocessed')


def prep_unity_alloego_files(base, participant):
    exp_path = eeg_path(unity_alloego_path(base, participant))
    files = dict()
    files['montage'] = {
        'original': os.path.join(exp_path, participant + '_montage.csv'),
        'referenced': os.path.join(
            exp_path, participant + '_montage_referenced.csv')
    }
    freq = str(get_frequency(exp_path))
    files['EEG'] = {
        'base': os.path.join(exp_path, 'prep_' + freq + '.mat'),
        'bipolar': os.path.join(exp_path, 'prep_bipolar_' + freq + '.mat'),
        'perElectrode': os.path.join(exp_path, 'prep_perElectrode_' + freq + '.mat'),
        'perHeadbox': os.path.join(exp_path, 'prep_perHeadbox_' + freq + '.mat')
    }
    files['experiment'] = {
        'player': os.path.join(exp_path, participant + '_player.csv'),
        'events_eegtime': os.path.join(
            exp_path, participant + '_unity_eegtime_.csv'),
        'events_unitytime': os.path.join(
            exp_path, participant + '_unity_unitytime.csv'),
        'events_timesinceeegstart': os.path.join(
            exp_path, participant + '_unity_timesinceeegstart.csv'),
        'onsets': os.path.join(exp_path, participant + '_onsets.csv')
    }
    return files
