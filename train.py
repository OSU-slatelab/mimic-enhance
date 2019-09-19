import argparse
import torch
import time
import os

from aecnn import AECNN
from resnet import ResNet
from data_io import wav_dataset, mag
from torch import autograd

import torch.nn.functional as F

def run_training(config):
    """ Define our model and train it """

    load_generator = config.gpretrain is not None
    train_generator = config.gcheckpoints is not None

    load_mimic = config.mpretrain is not None
    train_mimic = config.mcheckpoints is not None

    models = {}

    # Build enhancement model
    if load_generator or train_generator:
        models['generator'] = AECNN(
            channel_counts = config.gchan,
            kernel_size = config.gkernel,
            block_size = config.gblocksize,
            dropout = config.gdrop,
            training = train_generator,
        ).cuda()

        models['generator'].requires_grad = train_generator

        if load_generator:
            models['generator'].load_state_dict(torch.load(config.gpretrain))

    # Build acoustic model
    if load_mimic or train_mimic:

        models['mimic'] = ResNet(
            input_dim = 256,
            output_dim = config.moutdim,
            channel_counts = config.mchan,
            dropout = config.mdrop,
            training = train_mimic,
        ).cuda()

        models['mimic'].requires_grad = train_mimic

        if load_mimic:
            models['mimic'].load_state_dict(torch.load(config.mpretrain))

            if config.mimic_weight > 0 or any(config.texture_weights) and train_mimic:
                models['teacher'] = ResNet(
                    input_dim = 256,
                    output_dim = config.moutdim,
                    channel_counts = config.mchan,
                    dropout = 0,
                    training = False,
                ).cuda()

                models['teacher'].requires_grad = False
                models['teacher'].load_state_dict(torch.load(config.mpretrain))

    # Initialize datasets
    tr_dataset = wav_dataset(config, 'tr')
    dt_dataset = wav_dataset(config, 'dt', 4)

    # Initialize optimizer
    params = []
    if train_generator:
        params.append({'params': models['generator'].parameters()})
    if train_mimic:
        params.append({'params': models['mimic'].parameters()})
    optimizer = torch.optim.Adam(params, lr = config.learn_rate)
    optimizer.zero_grad()

    # Reduce learning rate if we're not improving dev loss
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience = config.patience,
        factor = config.lr_decay,
        verbose = True,
    )

    # Run the training
    best_dev_loss = float('inf')
    for epoch in range(config.epochs):
        print("Starting epoch %d" % epoch)

        # Train for one epoch
        start_time = time.time()
        samples = 0
        for sample in tr_dataset:
            loss = generate_loss(sample, config, models)
            loss.backward()
            samples += 1

            # only do the update if we've reached the batch size
            if samples % config.batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()

        total_time = time.time() - start_time

        print("Completed epoch %d in %d seconds" % (epoch, int(total_time)))

        # Compute validation loss
        dev_loss = 0
        with torch.no_grad():
            for sample in dt_dataset:
                dev_loss += generate_loss(sample, config, models) / len(dt_dataset)

        print("Dev loss: %f" % dev_loss)
        scheduler.step(dev_loss)

        # Save our model
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            if train_mimic:
                mfile = os.path.join(config.mcheckpoints, config.mfile)
                torch.save(models['mimic'].state_dict(), mfile)
            if train_generator:
                gfile = os.path.join(config.gcheckpoints, config.gfile)
                torch.save(models['generator'].state_dict(), gfile)

# Compute loss, using weights for each type of loss
def generate_loss(sample, config, models):

    # Acoustic model training
    if 'generator' not in models:

        # Generate spectrogram
        inputs = mag(sample['clean'].cuda(), truncate = True)
        target = sample['senone'].cuda()
        
        # Ensure equal length
        newlen = min(inputs.shape[3], target.shape[1])
        newlen -= newlen % 16
        inputs = inputs[:, :, :, :newlen]
        target = target[:, :newlen]

        # Make prediction and evaluate
        prediction = models['mimic'](inputs)[-1]
        return config.ce_weight * F.cross_entropy(prediction, target)

    # Enhancement model training
    else:
        prediction = models['generator'](sample['noisy'].cuda())
        target = sample['clean'].cuda()
        loss = 0

        # We need STFT if any loss other than time-domain is used.
        if any([config.sm_weight, config.mimic_weight, config.ce_weight] + config.texture_weights):
            denoised_mag = mag(prediction, truncate = True)
            clean_mag = mag(target, truncate = True)

            # generate outputs/targets for mimic training
            if any([config.mimic_weight] + config.texture_weights):
                predictions = models['mimic'](denoised_mag)
                if 'teacher' in models:
                    targets = models['teacher'](clean_mag)
                else:
                    targets = models['mimic'](clean_mag)

        # Time-domain loss
        if config.l1_weight > 0:
            loss += config.l1_weight * F.l1_loss(prediction, target)

        # Spectral mapping loss
        if config.sm_weight > 0:
            loss += config.sm_weight * F.l1_loss(denoised_mag, clean_mag)

        # Mimic loss (perceptual loss)
        if config.mimic_weight > 0:
            loss += config.mimic_weight * F.l1_loss(predictions[-1], targets[-1])

        # Texture loss at each convolutional block
        if any(config.texture_weights):
            for index in range(len(predictions) - 1):
                if config.texture_weights[index] > 0:
                    prediction = get_gram_matrix(predictions[index])
                    target = get_gram_matrix(targets[index])
                    loss += config.texture_weights[index] * F.l1_loss(prediction, target)

        # Cross-entropy loss (for joint training?)
        if config.ce_weight > 0:
            loss += config.ce_weight * F.cross_entropy(predictions[-1], example['senone'].cuda())

        return loss

def get_gram_matrix(x):
    feature_maps = x.shape[1]
    x = x.view(feature_maps, -1)
    x = (x - torch.mean(x)) / torch.std(x)

    mat = torch.mm(x, x.t())

    return mat

def parse_args():
    parser = argparse.ArgumentParser()

    file_args = {
        'base_dir': None, 'clean_flist': None, 'noise_flist': None, 'noisy_flist': None, 'senone_file': None,
        'gpretrain': None, 'gcheckpoints': None, 'mpretrain': None, 'mcheckpoints': None,
        'mfile': 'model.pt', 'gfile': 'model.pt', 'output_dir': None, 'phase': None,
    }
    train_args = {
        'learn_rate': 2e-4, 'lr_decay': 0.5, 'patience': 1, 'epochs': 25, 'batch_size': 4,
        'l1_weight': 0., 'sm_weight': 0., 'mimic_weight': 0., 'ce_weight': 0., 'texture_weights': [0., 0., 0., 0.],
    }
    gen_args = {
        'gmodel': 'aecnn', 'gchan': [64, 128, 256], 'gblocksize': 3, 'gdrop': 0.2, 'gkernel': 11,
    }
    mim_args = {
        'mmodel': 'resnet', 'mchan': [64, 128, 256, 512], 'mdrop': 0.2, 'moutdim': 2023,
    }

    for arg_list in [file_args, train_args, gen_args, mim_args]:
        for arg, default in arg_list.items():
            if default is None:
                parser.add_argument(f"--{arg}")
            elif type(default) == list:
                parser.add_argument(f"--{arg}", default=default, nargs="+", type=type(default[0]))
            else:
                parser.add_argument(f"--{arg}", default=default, type=type(default))

    return parser.parse_args()


def main():
    config = parse_args()
    run_training(config)

if __name__=='__main__':
    main()

