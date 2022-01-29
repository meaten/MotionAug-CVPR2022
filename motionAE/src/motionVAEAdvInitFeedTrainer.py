from motionAE.src.motionVAETrainer import motionVAETrainer
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from motionAE.src.models import lstmVAE_initfeed, Discriminator_frame, Discriminator_seq


class motionVAEAdvInitFeedTrainer(motionVAETrainer):

    def build_model(self, gpu=True):
        if self.architecture == 'lstmVAE_adv_initfeed':
            self.model = lstmVAE_initfeed(self.input_length, self.dim_pose, self.dim_z)
            self.dis_f = Discriminator_frame(self.dim_pose, self.dim_z)
            self.dis_s = Discriminator_seq(
                self.input_length, self.dim_pose, self.dim_z)
        else:
            raise(ValueError)

        if gpu is True:
            self.model = self.model.cuda()
            self.dis_f = self.dis_f.cuda()
            self.dis_s = self.dis_s.cuda()

    def train(self):
        num_epochs = self.num_epochs

        optimizer_vae = optim.SGD(
            self.model.parameters(), lr=self.learning_rate)
        # optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        optimizer_dis_f = optim.SGD(
            self.dis_f.parameters(), lr=self.learning_rate)
        optimizer_dis_s = optim.SGD(
            self.dis_s.parameters(), lr=self.learning_rate)

        # scheduler_vae = optim.lr_scheduler.ReduceLROnPlateau(optimizer_vae, 'min', factor=self.decay_factor,
        #                                                      patience=int(num_epochs / 20), cooldown=int(num_epochs / 15))
        scheduler_vae = optim.lr_scheduler.StepLR(
            optimizer_vae, step_size=int(num_epochs / 5), gamma=self.decay_factor)
        scheduler_dis_f = optim.lr_scheduler.StepLR(
            optimizer_dis_f, step_size=int(num_epochs / 5), gamma=self.decay_factor)
        scheduler_dis_s = optim.lr_scheduler.StepLR(
            optimizer_dis_s, step_size=int(num_epochs / 5), gamma=self.decay_factor)
        criterion = self.loss

        with tqdm(range(num_epochs), ncols=150) as pbar:
            for i, epoch in enumerate(pbar):
                inputs = self.sample_motions(
                    batch_size=self.batch_size)  # [motions, lengths]
                # [inputs, recons, lengths, mu, log_var, z]
                result = self.model(*inputs)

                fake = result[1]
                pred_lengths = self.input_length - result[6]
                pred_lengths = torch.clamp(pred_lengths, 2, self.input_length)
                pred_lengths = torch.squeeze(torch.round(pred_lengths)).type(torch.IntTensor)
                
                # vae
                optimizer_vae.zero_grad()
                loss_vae = criterion(*result) \
                    - self.dis_f(fake).mean() \
                    - self.dis_s(fake, pred_lengths).mean()
                loss_vae.backward(retain_graph=True)
                optimizer_vae.step()
                # scheduler_vae.step(loss_vae)
                scheduler_vae.step()

                # discriminator
                optimizer_dis_f.zero_grad()
                optimizer_dis_s.zero_grad()
                real_loss_dis_f = torch.nn.ReLU()(
                    1.0 - self.dis_f(result[0])).mean()
                fake_loss_dis_f = torch.nn.ReLU()(
                    1.0 + self.dis_f(result[1].detach())).mean()
                real_loss_dis_s = torch.nn.ReLU()(
                    1.0 - self.dis_s(result[0], result[2])).mean()
                fake_loss_dis_s = torch.nn.ReLU()(
                    1.0 + self.dis_s(result[1].detach(), result[2])).mean()

                loss_dis_f = (real_loss_dis_f + fake_loss_dis_f) / 2
                loss_dis_s = (real_loss_dis_s + fake_loss_dis_s) / 2

                loss_dis_f.backward()
                loss_dis_s.backward()

                optimizer_dis_f.step()
                optimizer_dis_s.step()

                scheduler_dis_f.step()
                scheduler_dis_s.step()

                np.set_printoptions(
                    precision=3, floatmode='maxprec', suppress=True)

                pbar.set_postfix(
                    dict(
                        loss=np.asarray(
                            [loss_vae.item(), loss_dis_s.item(), loss_dis_f.item()]),
                        lr=[optimizer_vae.param_groups[0]['lr'],
                            optimizer_dis_f.param_groups[0]['lr'],
                            optimizer_dis_s.param_groups[0]['lr']]
                    ))

        self.save_model()
        print('end training')
    """
    def sample(self, batch_size=20, gpu=True):
        z_sample_shape = [batch_size]
        z_sample_shape.extend([self.dim_z])
        z_sample = np.random.normal(size=z_sample_shape)
        if gpu:
            z_sample = self._to_torch(z_sample)
        else:
            z_sample = torch.from_numpy(z_sample.astype(np.float32))
        self.model.decoder.eval()
        pose = self.sample_motions(batch_size=batch_size)[0][:, 0, :] 
        with torch.no_grad():
            motions = self.model.decoder(z_sample, pose)
            lengths = self.input_length - self.model.estimator_length(z_sample)
        self.model.decoder.train()
        if gpu:
            motions = self._to_numpy(motions)
        else:
            motions = motions.numpy()
        return motions
    """
    def sample(self, batch_size=20, gpu=True):
        z_sample = self.sample_z(batch_size)

        if gpu:
            z_sample = self._to_torch(z_sample)
        else:
            z_sample = torch.from_numpy(z_sample.astype(np.float32))
        self.model.decoder.eval()
        pose = self.sample_motions(batch_size=batch_size)[0][:, 0, :]
        with torch.no_grad():
            motions = self.model.decoder(z_sample, pose)
            lengths = self.input_length - self.model.estimator_length(z_sample)
        self.model.decoder.train()
        if gpu:
            motions = self._to_numpy(motions)
            lengths = self._to_numpy(lengths)
        else:
            motions = motions.numpy()
            lengths = lengths.numpy()

        lengths = np.round(lengths).squeeze(axis=1).astype('int32')
        motions = [motion[:length] for motion, length in zip(motions, lengths)]

        return motions