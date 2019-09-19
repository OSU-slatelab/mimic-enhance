from torch.utils.data import Dataset
import os
import json
import torch
import numpy as np
import soundfile as sf
import sys
import copy

class wav_dataset(Dataset):

    def __init__(self, config, phase, ch = None):

        key_list = []
        self.flists = {}
        for ftype, flist in [('clean', config.clean_flist), ('noise', config.noise_flist), ('noisy', config.noisy_flist)]:
            if flist:
                with open(os.path.join(config.base_dir, phase, flist)) as f:
                    self.flists[ftype] = json.load(f)
                    key_list = self.flists[ftype].keys()
                    ch_count = len(self.flists[ftype][list(key_list)[0]])

        if config.senone_file:
            self.flists['senone'] = {}
            with open(os.path.join(config.base_dir, phase, config.senone_file)) as f:
                for line in f:
                    line = line.split()
                    self.flists['senone'][line[0]] = np.array([int(i) for i in line[1:]], np.int64)
            key_list = self.flists['senone'].keys()

        self.flist = []
        for key in key_list:
            list_item = {'id': key}
            index = np.random.randint(ch_count) if ch is None else ch
            for ftype in self.flists:
                if ftype == 'senone':
                    list_item[ftype] = self.flists[ftype][key]
                else:
                    list_item[ftype] = self.flists[ftype][key][index]
            self.flist.append(list_item)

        self.base_dir = config.base_dir

    def __len__(self):
        return len(self.flist)

    def __getitem__(self, idx):

        data = copy.deepcopy(self.flist[idx])

        for ftype in ['clean', 'noisy', 'noise']:
            if ftype in data:
                wav = self._load_wav(data[ftype])
                newlen = len(wav) - len(wav) % 1024
                data[ftype] = torch.from_numpy(wav[np.newaxis, np.newaxis, :newlen])
                if ftype == 'noise':
                    data['noisy'] = data['noise'] + data['clean']
                    del data['noise']

        if 'senone' in data:
            target = torch.from_numpy(data['senone'])
            newlen = len(target) - len(target) % 16
            data['senone'] = target[np.newaxis, :newlen]

        return data

    def _load_wav(self, fname):
        data, sr = sf.read(os.path.join(self.base_dir, fname))
        return np.array(data, dtype=np.float32)

def mag(tensor, truncate = False, log = False):

    spectrogram = torch.stft(
        torch.squeeze(tensor),
        n_fft = 512,
        hop_length = 160,
        win_length = 400,
        window = torch.hann_window(400),
    )

    real = spectrogram[:, :, 0]
    imag = spectrogram[:, :, 1]
    
    magnitude = torch.sqrt(real * real + imag * imag + 1e-8)

    if truncate:
        magnitude = magnitude[None, None, :-1]

    if log:
        return torch.log10(magnitude + 0.1)

    return magnitude
