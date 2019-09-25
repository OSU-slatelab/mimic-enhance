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
        if self.config.gan_weight > 0:
            self.optimizerD = torch.optim.Adam(models['discriminator'].parameters(), lr = self.config.learn_rate)
            self.optimizerD.zero_grad()
            
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
                samples += 1

                if self.config.gan_weight > 0:
                    outputs = self.forward(sample)
                    d_loss = self.discriminate_loss(outputs)
                    d_loss.backward()

                    if samples % self.config.batch_size == 0:
                        self.optimizerD.step()
                        self.optimizerD.zero_grad()

                outputs = self.forward(sample)
                loss = self.generate_loss(outputs)
                loss.backward()

                if samples % self.config.batch_size == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

        else:
            loss = 0
            with torch.no_grad():
                for sample in dataset:
                    outputs = self.forward(sample)
                    loss += self.generate_loss(outputs) / len(dataset)
            
            self.scheduler.step(loss)

            return loss

    def forward(self, sample):

        if 'generator' not in self.models:
            outputs = normalize(mag(sample['clean'].cuda(), tuncate = True), sample['senone'].cuda())
            outputs['mimic'] = self.models['mimic'](outputs['clean_mag'])[-1]
        else:
            outputs = {
                'generator': self.models['generator'](sample['noisy'].cuda()),
                'clean_wav': sample['clean'].cuda(),
            }

            if self.config.sm_weight or 'mimic' in self.models:
                outputs['denoised_mag'] = mag(outputs['generator'], truncate = True)
                outputs['clean_mag'] = mag(outputs['clean_wav'], truncate = True)

                if 'mimic' in self.models:
                    outputs['mimic'] = self.models['mimic'](outputs['denoised_mag'])

                    if 'teacher' in self.models:
                        outputs['soft_label'] = self.models['teacher'](outputs['clean_mag'])
                    else:
                        outputs['soft_label'] = self.models['mimic'](outputs['clean_mag'])

                    if 'discriminator' in self.models:
                        outputs['d_real'] = self.models['discriminator'](outputs['soft_label'][-1].transpose(1, 2)).mean()
                        outputs['d_fake'] = self.models['discriminator'](outputs['mimic'][-1].transpose(1, 2)).mean()

                if 'senone' in sample:
                    outputs['senone'] = sample['senone'].cuda()

        return outputs

    def discriminate_loss(self, outputs):

        #print("Discrim real error: %f" % outputs['d_real'])
        #print("Discrim fake error: %f" % outputs['d_fake'])

        return self.config.gan_weight * (outputs['d_real'] - outputs['d_fake'])

    # Compute loss, using weights for each type of loss
    def generate_loss(self, outputs):

        # Acoustic model training
        if 'generator' not in self.models:
            return self.config.ce_weight * F.cross_entropy(outputs['mimic'], outputs['senone'])

        # Enhancement model training
        else:
            loss = 0

            # Time-domain loss
            if self.config.l1_weight > 0:
                loss += self.config.l1_weight * F.l1_loss(outputs['generator'], outputs['clean_wav'])

            # Spectral mapping loss
            if self.config.sm_weight > 0:
                loss += self.config.sm_weight * F.l1_loss(outputs['denoised_mag'], outputs['clean_mag'])

            # Mimic loss (perceptual loss)
            if self.config.mimic_weight > 0:
                loss += self.config.mimic_weight * F.l1_loss(outputs['mimic'][-1], outputs['soft_label'][-1])

            # Texture loss at each convolutional block
            if any(self.config.texture_weights):
                for index in range(len(predictions) - 1):
                    if self.config.texture_weights[index] > 0:
                        prediction = outputs['mimic'][index]
                        target = outputs['soft_label'][index]
                        loss += self.config.texture_weights[index] * F.l1_loss(prediction, target)

            # Cross-entropy loss (for joint training?)
            if self.config.ce_weight > 0:
                inputs, targets = normalize(outputs['denoised_mag'], outputs['senone'])
                loss += self.config.ce_weight * F.cross_entropy(self.models['mimic'](inputs)[-1], targets)

            if self.config.gan_weight > 0:
                #print("Generator error: %f" % outputs['d_fake'])
                if outputs['d_fake'] > 0.2:
                    loss += self.config.gan_weight * outputs['d_fake']

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
