import torch
import torch.nn.functional as F
from data_io import mag

class Trainer:
    def __init__(self, config, models):

        self.config = config
        self.models = models

        # Initialize optimizer
        params = []
        if self.config.gcheckpoints:
            params.append({'params': models['generator'].parameters()})
        if self.config.mcheckpoints:
            params.append({'params': models['mimic'].parameters()})
            
        self.optimizer = torch.optim.Adam(params, lr = self.config.learn_rate)
        self.optimizer.zero_grad()

        # Reduce learning rate if we're not improving dev loss
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            patience = self.config.patience,
            factor = self.config.lr_decay,
            verbose = True,
        )

    def run_epoch(self, dataset, training = False):

        if training:
            samples = 0
            for sample in dataset:
                loss = self.generate_loss(sample)
                loss.backward()

                samples += 1
                if samples % self.config.batch_size == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

        else:
            loss = 0
            with torch.no_grad():
                for sample in dataset:
                    loss += self.generate_loss(sample) / len(dataset)
            
            self.scheduler.step(loss)

            return loss

    # Compute loss, using weights for each type of loss
    def generate_loss(self, sample):

        # Acoustic model training
        if 'generator' not in self.models:

            # Generate spectrogram
            inputs, target = normalize(mag(sample['clean'].cuda(), truncate = True), sample['senone'].cuda())

            # Make prediction and evaluate
            prediction = self.models['mimic'](inputs)[-1]
            return self.config.ce_weight * F.cross_entropy(prediction, target)

        # Enhancement model training
        else:
            prediction = self.models['generator'](sample['noisy'].cuda())
            target = sample['clean'].cuda()
            loss = 0

            # We need STFT if any loss other than time-domain is used.
            if any([self.config.sm_weight, self.config.mimic_weight, self.config.ce_weight]
                    + self.config.texture_weights):
                denoised_mag = mag(prediction, truncate = True)
                clean_mag = mag(target, truncate = True)

                # generate outputs/targets for mimic training
                if any([self.config.mimic_weight] + self.config.texture_weights):
                    predictions = self.models['mimic'](denoised_mag)
                    if 'teacher' in self.models:
                        targets = self.models['teacher'](clean_mag)
                    else:
                        targets = self.models['mimic'](clean_mag)

            # Time-domain loss
            if self.config.l1_weight > 0:
                loss += self.config.l1_weight * F.l1_loss(prediction, target)

            # Spectral mapping loss
            if self.config.sm_weight > 0:
                loss += self.config.sm_weight * F.l1_loss(denoised_mag, clean_mag)

            # Mimic loss (perceptual loss)
            if self.config.mimic_weight > 0:
                loss += self.config.mimic_weight * F.l1_loss(predictions[-1], targets[-1])

            # Texture loss at each convolutional block
            if any(self.config.texture_weights):
                for index in range(len(predictions) - 1):
                    if self.config.texture_weights[index] > 0:
                        prediction = predictions[index]
                        target = targets[index]
                        loss += self.config.texture_weights[index] * F.l1_loss(prediction, target)

            # Cross-entropy loss (for joint training?)
            if self.config.ce_weight > 0:
                inputs, targets = normalize(denoised_mag, sample['senone'].cuda())
                loss += self.config.ce_weight * F.cross_entropy(self.models['mimic'](inputs)[-1], targets)

            return loss

def normalize(inputs, target, factor = 16):
    
    # Ensure equal length
    newlen = min(inputs.shape[3], target.shape[1])
    newlen -= newlen % factor
    inputs = inputs[:, :, :, :newlen]
    target = target[:, :newlen]

    return inputs, target

def get_gram_matrix(x):
    feature_maps = x.shape[1]
    x = x.view(feature_maps, -1)
    x = (x - torch.mean(x)) / torch.std(x)

    mat = torch.mm(x, x.t())

    return mat
