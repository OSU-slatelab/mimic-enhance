import argparse
import torch
import time
import os

from aecnn import AECNN
from resnet import ResNet
from discriminator import Discriminator
from trainer import Trainer
from data_io import wav_dataset
from torch import autograd

def run_training(config):
    """ Define our model and train it """

    load_generator = config.gpretrain is not None
    train_generator = config.gcheckpoints is not None

    load_mimic = config.mpretrain is not None
    train_mimic = config.mcheckpoints is not None

    if torch.cuda.is_available():
        config.device = torch.device('cuda')
    else:
        config.device = torch.device('cpu')

    models = {}

    # Build enhancement model
    if load_generator or train_generator:
        models['generator'] = AECNN(
            channel_counts = config.gchan,
            kernel_size = config.gkernel,
            block_size = config.gblocksize,
            dropout = config.gdrop,
            training = train_generator,
        ).to(config.device)

        models['generator'].requires_grad = train_generator

        if load_generator:
            models['generator'].load_state_dict(torch.load(config.gpretrain, map_location=config.device))

    # Build acoustic model
    if load_mimic or train_mimic:

        if config.mact == 'rrelu':
            activation = lambda x: torch.nn.functional.rrelu(x, training = train_mimic)
        else:
            activation = lambda x: torch.nn.functional.leaky_relu(x, negative_slope = 0.3)

        models['mimic'] = ResNet(
            input_dim = 256,
            output_dim = config.moutdim,
            channel_counts = config.mchan,
            dropout = config.mdrop,
            training = train_mimic,
            activation = activation,
        ).to(config.device)

        models['mimic'].requires_grad = train_mimic

        if load_mimic:
            models['mimic'].load_state_dict(torch.load(config.mpretrain, map_location=config.device))

            if config.mimic_weight > 0 or any(config.texture_weights) and train_mimic:
                models['teacher'] = ResNet(
                    input_dim = 256,
                    output_dim = config.moutdim,
                    channel_counts = config.mchan,
                    dropout = 0,
                    training = False,
                ).to(config.device)

                models['teacher'].requires_grad = False
                models['teacher'].load_state_dict(torch.load(config.mpretrain, map_location=config.device))

    if config.gan_weight > 0:
        models['discriminator'] = Discriminator(
            channel_counts = config.gchan,
            kernel_size = config.gkernel,
            block_size = config.gblocksize,
            dropout = config.gdrop,
            training = True,
        ).to(config.device)

    # Initialize datasets
    tr_dataset = wav_dataset(config, 'tr')
    dt_dataset = wav_dataset(config, 'dt', 4)

    trainer = Trainer(config, models)

    # Run the training
    best_dev_loss = float('inf')
    for epoch in range(config.epochs):
        print("Starting epoch %d" % epoch)

        # Train for one epoch
        start_time = time.time()
        trainer.run_epoch(tr_dataset, training = True)
        total_time = time.time() - start_time

        print("Completed epoch %d in %d seconds" % (epoch, int(total_time)))

        dev_loss, dev_losses = trainer.run_epoch(dt_dataset, training = False)

        print("Dev loss: %f" % dev_loss)
        for key in dev_losses:
            print("%s loss: %f" % (key, dev_losses[key]))

        # Save our model
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            if train_mimic:
                mfile = os.path.join(config.mcheckpoints, config.mfile)
                torch.save(models['mimic'].state_dict(), mfile)
            if train_generator:
                gfile = os.path.join(config.gcheckpoints, config.gfile)
                torch.save(models['generator'].state_dict(), gfile)

        if train_mimic:
            mfile = os.path.join(config.mcheckpoints, config.mfile)
            torch.save(models['mimic'].state_dict(), mfile + "-epoch%d" % epoch)

def parse_args():
    parser = argparse.ArgumentParser()

    file_args = {
        'base_dir': None, 'clean_flist': None, 'noise_flist': None, 'noisy_flist': None, 'senone_file': None,
        'gpretrain': None, 'gcheckpoints': None, 'mpretrain': None, 'mcheckpoints': None,
        'mfile': 'model.pt', 'gfile': 'model.pt', 'output_dir': None, 'phase': None,
    }
    train_args = {
        'learn_rate': 2e-4, 'lr_decay': 0.5, 'patience': 1, 'epochs': 25, 'batch_size': 4,
        'l1_weight': 0., 'sm_weight': 0., 'mimic_weight': 0., 'ce_weight': 0.,
        'texture_weights': [0., 0., 0., 0.], 'gan_weight': 0.,
    }
    gen_args = {
        'gmodel': 'aecnn', 'gchan': [64, 128, 256], 'gblocksize': 3, 'gdrop': 0.2, 'gkernel': 11,
    }
    mim_args = {
        'mmodel': 'resnet', 'mchan': [64, 128, 256, 512], 'mdrop': 0.2, 'moutdim': 2023, 'mact': 'lrelu',
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

