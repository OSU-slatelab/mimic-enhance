import torch
import os

from aecnn import AECNN
from data_io import wav_dataset
from train import parse_args
import soundfile as sf
import numpy as np

def run_test(config):
    """ Define our model and test it """

    generator = AECNN(
        channel_counts = config.gchan,
        kernel_size = config.gkernel,
        block_size = config.gblocksize,
        dropout = config.gdrop,
    ).cuda().eval()

    generator.load_state_dict(torch.load(config.gcheckpoints))

    # Initialize datasets
    #for phase in ['tr', 'dt', 'et']:

    max_ch = 6 if config.phase == 'tr' else 1

    count = 0
    for ch in range(max_ch):
        dataset = wav_dataset(config, config.phase, ch)

        with torch.no_grad():
            for example in dataset:
                data = np.squeeze(generator(example['noisy'].cuda()).cpu().detach().numpy())
                fname = make_filename(config, ch, example['id'])
                with sf.SoundFile(fname, 'w', 16000, 1) as w:
                    w.write(data)

                if count % 1000 == 0:
                    print("finished #%d" % count)
                count += 1

def make_filename(config, channel, id):
    args = [config.output_dir, config.phase, id + '.wav']
    if config.phase == 'tr':
        args[-1] = id + '.ch%d' % channel + '.wav'
    return os.path.join(*args)

def main():
    config = parse_args()
    run_test(config)

if __name__=='__main__':
    main()

