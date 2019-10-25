import torch

from aecnn import AECNN
from data_io import wav_dataset, mag
from train import parse_args
import numpy as np
import os
import soundfile as sf
import matplotlib.pyplot as plt
from python_speech_features import fbank


def print_spec(array, filename, xAxisRange=None, axes='on'):
    """ Print a spectrogram to a file """

    if xAxisRange:
        array = np.flipud(array.T)[:-3,xAxisRange[0]:xAxisRange[1]]
        extent = [xAxisRange[0] / 100., xAxisRange[1] / 100., 0, 8]
    else:
        array = np.flipud(array.T)
        extent = [0, array.shape[1] / 100., 0, 8]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(array,cmap=plt.cm.jet, interpolation='none', extent=extent, aspect=1./14)

    if axes == 'on':
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (kHz)")
        fig.savefig(filename, format='pdf', bbox_inches='tight')
    else:
        ax.axis('off')
        fig.savefig(filename, format='pdf', bbox_inches=0)

    plt.close(fig)


def energy(data, window = 200):

    e = np.zeros_like(data)
    for i in range(len(data)-2*window):
        i += window
        e[i-window:i+window] += np.sum(data[i-window:i+window] ** 2) / window

    cap = 0.2
    e[e > cap] = cap
    e[e < cap] = 0

    return e

def zero_crossings(data, window = 200):

    z = np.zeros_like(data)
    for i in range(len(data)-2*window-1):
        i += window
        crossed = data[i-window:i+window] * data[i-window+1:i+window+1]
        crossed[crossed > 0] = 0
        crossed[crossed < 0] = 0.3
        z[i-window:i+window] += np.sum(crossed) / window / window

    cap = 0.2
    z[z > cap] = cap
    z[z < cap] = 0

    return -z

def print_wav(data, fname, sr = 16000.):


    e = energy(data)
    z = zero_crossings(data)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    xpoints = np.arange(len(data)) / sr
    ax.plot(xpoints, data, linewidth=0.5)
    ax.plot(xpoints, e, linewidth=0.5) 
    ax.plot(xpoints, z, linewidth=0.5)

    fig.savefig(fname, format='pdf', bbox_inches='tight')

    plt.close(fig)

def run_test(config):
    """ Define our model and test it """

    generator = AECNN(
        channel_counts = config.gchan,
        kernel_size = config.gkernel,
        block_size = config.gblocksize,
        dropout = config.gdrop,
    ).cuda()

    generator.load_state_dict(torch.load(config.gcheckpoints))

    # Initialize datasets
    #ev_dataset = wav_dataset(config, 'et', 4)
    ev_dataset = wav_dataset(config, 'et')


    #count = 0
    #score = {'stoi': 0, 'estoi':0, 'sdr':0}
    example = ev_dataset[361]
    print(example['id'])
    data = np.squeeze(generator(example['noisy'].cuda()).cpu().detach().numpy())
    #clean = np.squeeze(example['clean'].numpy())
    noisy = np.squeeze(example['noisy'].numpy())
    #with sf.SoundFile('clean.wav', 'w', 16000, 1) as w:
    #    w.write(clean)
    with sf.SoundFile('noisy.wav', 'w', 16000, 1) as w:
        w.write(noisy)
    with sf.SoundFile('test.wav', 'w', 16000, 1) as w:
        w.write(data)

    #print_wav(noisy, 'noisy_waveform.pdf')
    #print_wav(clean, 'clean_waveform.pdf')
    #print_wav(data, 'waveform.pdf')
    #data = np.squeeze(generator(example['noisy']).detach().numpy())
    #clean = np.squeeze(example['clean'].numpy())
    #noisy = np.squeeze(example['noisy'].numpy())


    #data, _ = fbank(data,nfilt=80)
    #clean, _ = fbank(clean,nfilt=80)
    #noisy, _ = fbank(noisy,nfilt=80)
    #data, clean, noisy = np.log(data), np.log(clean), np.log(noisy)
    #minimum = min(np.min(data), np.min(clean), np.min(noisy))
    #data, clean, noisy = data - minimum, clean - minimum, noisy - minimum
    #maximum = max(np.max(data), np.max(clean), np.max(noisy))
    #data, clean, noisy = data / maximum, clean / maximum, noisy / maximum
    #print_spec(data, 'spectrogram.pdf', xAxisRange=[110,140])
    #print_spec(clean, 'clean_spec.pdf', xAxisRange=[110,140])
    #print_spec(noisy, 'noisy_spec.pdf', xAxisRange=[110,140])


def main():
    config = parse_args()
    run_test(config)

if __name__=='__main__':
    main()

