import torch
import torch.nn.functional as F
from data_io import mag
from collections import defaultdict

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
            self.schedulerD = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizerD,
                patience = self.config.patience,
                factor = self.config.lr_decay,
                verbose = True,
            )
            
        self.optimizer = torch.optim.Adam(params, lr = self.config.learn_rate)
        self.optimizer.zero_grad()

        # Reduce learning rate if we're not improving dev loss
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            patience = self.config.patience,
            factor = self.config.lr_decay,
            verbose = True,
        )

    def run_epoch(self, dataset, training = False, real = False):

        if training:
            samples = 0
            for sample in dataset:
                samples += 1

                outputs = self.forward(sample)
                if self.config.gan_weight > 0:# and outputs['d_fake'] > 0.2:
                    d_loss = self.discriminate_loss(outputs)
                    d_loss.backward()

                    if samples % self.config.batch_size == 0:
                        self.optimizerD.step()
                        self.optimizerD.zero_grad()

                    outputs = self.forward(sample)

                loss, losses = self.generate_loss(outputs, training, real)
                if loss != 0:
                    loss.backward()

                if samples % self.config.batch_size == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

        else:
            dev_loss = 0
            dev_losses = defaultdict(lambda: 0)
            with torch.no_grad():
                for sample in dataset:
                    outputs = self.forward(sample)
                    loss, losses = self.generate_loss(outputs, training)
                    dev_loss += loss / len(dataset)
                    for key in losses:
                        dev_losses[key] += losses[key] / len(dataset)
            
            self.scheduler.step(dev_loss)
            if self.config.gan_weight > 0:
                self.schedulerD.step(dev_loss)

            return dev_loss, dev_losses

    def forward(self, sample):

        device = self.config.device

        if 'generator' not in self.models:
            outputs = normalize(mag(sample['clean'].to(device), truncate = True), sample['senone'].to(device))
            outputs['mimic'] = self.models['mimic'](outputs['clean_mag'])
        else:
            outputs = {
                'generator': self.models['generator'](sample['noisy'].to(device)),
            }
            if 'clean' in sample:
                outputs['clean_wav'] = sample['clean'].to(device)

            if self.config.sm_weight or 'mimic' in self.models:
                outputs['denoised_mag'] = mag(outputs['generator'], truncate = True)

                if 'clean' in sample:
                    outputs['clean_mag'] = mag(outputs['clean_wav'], truncate = True)

                if 'mimic' in self.models:
                    outputs['mimic'] = self.models['mimic'](outputs['denoised_mag'])

                    mimic_losses = self.config.texture_weights + \
                            [self.config.mimic_weight, self.config.soft_senone_weight]

                    if 'teacher' in self.models and 'clean' in sample:
                        outputs['soft_label'] = self.models['teacher'](outputs['clean_mag'])
                    elif any(mimic_losses) and 'clean' in sample:
                        outputs['soft_label'] = self.models['mimic'](outputs['clean_mag'])
                    

            if 'discriminator' in self.models:
                outputs['d_real'] = self.models['discriminator'](outputs['clean_wav'])
                outputs['d_fake'] = self.models['discriminator'](outputs['generator'])

            if 'senone' in sample:
                outputs['senone'] = sample['senone'].to(device)

            if self.config.soft_senone_weight:
                outputs['embedding'] = self.models['embedding'](outputs['senone']).transpose(1, 2)

        return outputs

    def discriminate_loss(self, outputs):

        #print("Discrim real error: %f" % outputs['d_real'].mean())
        #print("Discrim fake error: %f" % outputs['d_fake'].mean())

        target_real = torch.ones_like(outputs['d_real'])
        loss_real = F.l1_loss(outputs['d_real'], target_real)

        target_fake = torch.zeros_like(outputs['d_fake'])
        loss_fake = F.l1_loss(outputs['d_fake'], target_fake)

        return self.config.gan_weight * (loss_real + loss_fake)

    # Compute loss, using weights for each type of loss
    def generate_loss(self, outputs, training = False, real = False):

        # Acoustic model training
        if 'generator' not in self.models or real:
            loss = self.config.ce_weight * truncate_and_ce(outputs['mimic'][-1], outputs['senone'])
            losses = {'ce': truncate_and_ce(outputs['mimic'][-1], outputs['senone'])}

        # Enhancement model training
        else:
            loss = 0
            losses = {}

            # Time-domain loss
            if self.config.l1_weight > 0:
                loss += self.config.l1_weight * F.l1_loss(outputs['generator'], outputs['clean_wav'])
                losses['l1'] = F.l1_loss(outputs['generator'], outputs['clean_wav']).detach()

            # Spectral mapping loss
            if self.config.sm_weight > 0:
                loss += self.config.sm_weight * F.l1_loss(outputs['denoised_mag'], outputs['clean_mag'])
                losses['sm'] = F.l1_loss(outputs['denoised_mag'], outputs['clean_mag']).detach()

            # Mimic loss (perceptual loss)
            if self.config.mimic_weight > 0:
                loss += self.config.mimic_weight * F.l1_loss(outputs['mimic'][-1], outputs['soft_label'][-1])
                losses['mimic'] = F.l1_loss(outputs['mimic'][-1], outputs['soft_label'][-1]).detach()

            # Texture loss at each convolutional block
            if any(self.config.texture_weights):
                for index in range(len(outputs['mimic']) - 1):
                    if self.config.texture_weights[index] > 0:
                        prediction = outputs['mimic'][index]
                        target = outputs['soft_label'][index]
                        loss += self.config.texture_weights[index] * F.l1_loss(prediction, target)
                        losses['texture%d' % index] = F.l1_loss(prediction, target).detach()

            # Cross-entropy loss (for joint training?)
            if self.config.ce_weight > 0:
                #norm = normalize(outputs['denoised_mag'], outputs['senone'])
                #outputs = self.models['mimic'](norm['clean_mag'])[-1]
                #targets = norm['senone']
                loss += self.config.ce_weight * truncate_and_ce(outputs['mimic'], outputs['senone'])
                losses['ce'] = truncate_and_ce(outputs['mimic'], outputs['senone']).detach()

            if self.config.gan_weight > 0:
                target = torch.ones_like(outputs['d_fake'])
                losses['generator'] = F.mse_loss(outputs['d_fake'], target)
                #print("Generator prediction: %f" % outputs['d_fake'].mean())
                #print("Generator loss: %f" % losses['generator'])

                if training:#outputs['d_fake'].mean() < 0.4 and training:
                    loss += self.config.gan_weight * F.l1_loss(outputs['d_fake'], target)

            if self.config.soft_senone_weight > 0:
                losses['soft_senone'] = truncate_and_l1(outputs['mimic'][-1], outputs['embedding']).detach()
                loss += self.config.soft_senone_weight * truncate_and_l1(outputs['mimic'][-1], outputs['embedding'])

        return loss, losses

def normalize(inputs, target, factor = 16):
    
    # Ensure equal length
    newlen = min(inputs.shape[3], target.shape[1])
    newlen -= newlen % factor
    inputs = inputs[:, :, :, :newlen]
    target = target[:, :newlen]

    return {'clean_mag': inputs, 'senone': target}

def get_gram_matrix(x):
    feature_maps = x.shape[1]
    x = x.view(feature_maps, -1)
    x = (x - torch.mean(x)) / torch.std(x)

    mat = torch.mm(x, x.t())

    return mat

def truncate_and_l1(inputs, target):
    newlen = min(inputs.shape[-1], target.shape[-1])

    inputs = inputs[:, :, :newlen]
    target = target[:, :, :newlen]

    return F.l1_loss(inputs, target)
    
def truncate_and_ce(inputs, target):
    newlen = min(inputs.shape[-1], target.shape[-1])

    inputs = inputs[:, :, :newlen]
    target = target[:, :newlen]

    return F.cross_entropy(inputs, target)
