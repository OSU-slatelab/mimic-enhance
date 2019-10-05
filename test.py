import torch

from aecnn import AECNN
from data_io import wav_dataset
from train import parse_args
from pystoi.stoi import stoi
import numpy as np
from sdr import si_sdr
import os
import soundfile as sf

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
    ev_dataset = wav_dataset(config, 'et', 4)


    count = 0
    score = {'stoi': 0, 'estoi':0, 'sdr':0}
    for example in ev_dataset:
        data = np.squeeze(generator(example['noisy'].cuda()).cpu().detach().numpy())
        clean = np.squeeze(example['clean'].numpy())
        noisy = np.squeeze(example['noisy'].numpy())
        score['stoi'] += stoi(clean, data, 16000, extended=False)
        score['estoi'] += stoi(clean, data, 16000, extended=True)
        score['sdr'] += si_sdr(data, clean)
        count += 1
        #if count == 1:
        #    with sf.SoundFile('clean.wav', 'w', 16000, 1) as w:
        #        w.write(clean)
        #    with sf.SoundFile('noisy.wav', 'w', 16000, 1) as w:
        #        w.write(noisy)
        #    with sf.SoundFile('test.wav', 'w', 16000, 1) as w:
        #        w.write(data)
        #    break

    print('stoi: %f' % (score['stoi'] / count))
    print('estoi: %f' % (score['estoi'] / count))
    print('sdr: %f' % (score['sdr'] / count))


def main():
    config = parse_args()
    run_test(config)

if __name__=='__main__':
    main()

