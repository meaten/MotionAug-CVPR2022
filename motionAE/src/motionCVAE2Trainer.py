from motionAE.src.models import lstmCVAE2
from torch.distributions.kl import kl_divergence
from torch.distributions.normal import Normal
import torch
from motionAE.src.motionCVAETrainer import motionCVAETrainer

import numpy as np


class motionCVAE2Trainer(motionCVAETrainer):

    def load_param(self, arg_parser, **kwargs):
        super().load_param(arg_parser, **kwargs)

    def build_model(self):
        if self.architecture == 'lstmCVAE2':
            self.model = lstmCVAE2(
                self.input_length, self.dim_pose, self.dim_z, len(self.all_classes))
        else:
            raise(ValueError)

        self.model = self.model.cuda()

    def sample(self, batch_size=20, used_class=None, gpu=True):
        if used_class is not None:
            class_vector = self.one_hot_encoder.transform([used_class])
        else:
            class_vector = self.one_hot_encoder.transform([self.used_class])
        class_vector = np.tile(class_vector, (batch_size, 1))

        z_sample_shape = [batch_size]
        z_sample_shape.extend([self.dim_z])
        z_sample = np.random.normal(size=z_sample_shape)

        self.model.decoder.eval()

        if gpu:
            z_sample = self._to_torch(z_sample)
            class_vector = self._to_torch(class_vector)

        else:
            z_sample = torch.from_numpy(z_sample.astype(np.float32))
            class_vector = torch.from_numpy(class_vector.astype(np.float32))

        with torch.no_grad():
            mu_c, log_var_c = self.model.encoder_class(class_vector)
            std = torch.exp(0.5 * log_var_c)
            z_sample = z_sample * std + mu_c
            motions = self.model.decoder(z_sample, class_vector)

        if gpu:
            motions = self._to_numpy(motions)
        else:
            motions = motions.numpy()

        self.model.decoder.train()
        return motions

    def kld_loss(self, *args):
        mu = args[3]
        log_var = args[4]

        mu_c = args[6]
        log_var_c = args[7]

        q = Normal(mu, torch.exp(0.5 * log_var))
        pi = Normal(mu_c, torch.exp(0.5 * log_var_c))
        return torch.mean(torch.sum(kl_divergence(q, pi), dim=1))

        # sigma1 : mu, log_var
        # sigma2 : mu_c, log_var_c
        # return torch.mean(-0.5 * torch.sum(1 - log_var_c + log_var - ((mu - mu_c) ** 2 + log_var.exp()) / log_var_c.exp() , dim=1), dim=0).cuda()
