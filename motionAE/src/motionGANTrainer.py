from motionAE.src.motionTrainer import motionTrainer

import os
from tqdm import tqdm
import shutil
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable

from util.arg_parser import ArgParser
from motionAE.src.models import lstmAE_wo_Norm, lstmDecoder, Discriminator_frame, Discriminator_seq, lstmAE, Estimator_length
from motionAE.src.MMD import MMD_loss


class motionGANTrainer(motionTrainer):
    required_attributes = ['G', 'D_s', 'D_f']

    def load_param(self, arg_parser: ArgParser, **kwargs):
        super().load_param(arg_parser, **kwargs)

        self.num_epochs = arg_parser.parse_int('num_epochs')
        self.batch_size = arg_parser.parse_int('batch_size')

        self.learning_rate = arg_parser.parse_float('learning_rate')
        self.decay_factor = arg_parser.parse_float('decay_factor')
        self.decay_interval = arg_parser.parse_int('decay_interval')

        self.dim_z = arg_parser.parse_int('dim_z')

        self.smooth_coef = arg_parser.parse_float('smooth_coef')

        # viable in _feedback, _initfeed architecture
        self.residual = arg_parser.parse_bool('residual')

        self.path = os.path.join(arg_parser.parse_string('output_path'),
                                 self.architecture,
                                 os.path.basename(self.dataset),
                                 # "geodesic_" + str(self.use_geodesic),
                                 "residual_" + str(self.residual),
                                 "_".join(self.used_class),
                                 "_".join(self.used_subject)
                                 )

        # for MMD calculation
        self.extractor_path = os.path.join(
            arg_parser.parse_string("output_path"), "ext", "model.pth")

    def build_model(self, gpu: bool = True) -> None:
        if self.architecture == 'GAN':
            self.G = lstmDecoder(self.input_length, self.dim_pose, self.dim_z)
            self.estimator_length = Estimator_length(self.dim_z)
            self.D_f = Discriminator_frame(self.dim_pose, self.dim_z)
            self.D_s = Discriminator_seq(
                self.input_length, self.dim_pose, self.dim_z)
        else:
            raise(ValueError)

        if gpu is True:
            self.G = self.G.cuda()
            self.estimator_length = self.estimator_length.cuda()
            self.D_f = self.D_f.cuda()
            self.D_s = self.D_s.cuda()

    def save_model(self):
        self.recreate_path()
        path = os.path.join(self.path, "G.pth")
        torch.save(self.G.state_dict(), path)

        path = os.path.join(self.path, "D_f.pth")
        torch.save(self.D_f.state_dict(), path)

        path = os.path.join(self.path, "D_s.pth")
        torch.save(self.D_s.state_dict(), path)
        
        path = os.path.join(self.path, "len_ext.pth")
        torch.save(self.estimator_length.state_dict(), path)

    def load_model(self, gpu=True):
        if gpu is False:
            path = os.path.join(self.path, "G.pth")
            self.G.load_state_dict(torch.load(
                path, map_location=torch.device('cpu')))
            self.G.to('cpu')
            path = os.path.join(self.path, "len_ext.pth")
            self.estimator_length.load_state_dict(torch.load(
                path, map_location=torch.device('cpu')
            ))
            self.estimator_length.to('cpu')
        else:
            path = os.path.join(self.path, "G.pth")
            self.G.load_state_dict(torch.load(path))
            path = os.path.join(self.path, "len_ext.pth")
            self.estimator_length.load_state_dict(torch.load(path))
            
    def train(self):
        num_epochs = self.num_epochs

        optimizer_G = optim.Adam(
            self.G.parameters(), lr=self.learning_rate)
        optimizer_D_f = optim.Adam(
            self.D_f.parameters(), lr=self.learning_rate)
        optimizer_D_s = optim.Adam(
            self.D_s.parameters(), lr=self.learning_rate)
        optimizer_est = optim.Adam(
            self.estimator_length.parameters(), lr=self.learning_rate)

        scheduler_G = optim.lr_scheduler.StepLR(
            optimizer_G, step_size=int(
                num_epochs / 5), gamma=self.decay_factor)
        scheduler_D_f = optim.lr_scheduler.StepLR(
            optimizer_D_f, step_size=int(
                num_epochs / 5), gamma=self.decay_factor)
        scheduler_D_s = optim.lr_scheduler.StepLR(
            optimizer_D_s, step_size=int(
                num_epochs / 5), gamma=self.decay_factor)
        scheduler_est = optim.lr_scheduler.StepLR(
            optimizer_est, step_size=int(
                num_epochs / 5), gamma=self.decay_factor)
        
        FloatTensor = torch.cuda.FloatTensor if self.device != 'cpu' else torch.FloatTensor

        with tqdm(range(num_epochs), ncols=150) as pbar:
            for i, epoch in enumerate(pbar):
                real, length = self.sample_motions(
                    batch_size=self.batch_size)  # [motions, lengths]
                z = Variable(
                    FloatTensor(
                        np.random.normal(
                            0, 1, (self.batch_size, self.dim_z))))
                fake = self.G(z)
                fake_length = self.input_length - torch.clamp(self.estimator_length(z), 0, self.input_length)
                fake_length = torch.squeeze(torch.round(fake_length)).type(torch.IntTensor)
                
                # generator training
                optimizer_G.zero_grad()
                optimizer_est.zero_grad()
                loss_G = - self.D_f(fake).mean() \
                    - self.D_s(fake, fake_length).mean() \
                    # + self.smooth_coef * self.smoothness_loss(fake, length)
                loss_G.backward(retain_graph=True)
                optimizer_G.step()
                optimizer_est.step()
                scheduler_G.step()
                scheduler_est.step()

                # discriminator training
                optimizer_D_f.zero_grad()
                optimizer_D_s.zero_grad()
                optimizer_est.zero_grad()
                real_loss_D_f = torch.nn.ReLU()(
                    1.0 - self.D_f(real).mean())
                fake_loss_D_f = torch.nn.ReLU()(
                    1.0 + self.D_f(fake.detach())).mean()
                real_loss_D_s = torch.nn.ReLU()(
                    1.0 - self.D_s(real, length)).mean()
                fake_loss_D_s = torch.nn.ReLU()(
                    1.0 + self.D_s(fake.detach(), fake_length.detach())).mean()

                loss_D_f = (real_loss_D_f + fake_loss_D_f) / 2
                loss_D_s = (real_loss_D_s + fake_loss_D_s) / 2

                loss_D_f.backward()
                loss_D_s.backward()

                optimizer_D_f.step()
                optimizer_D_s.step()

                scheduler_D_f.step()
                scheduler_D_s.step()

                np.set_printoptions(
                    precision=3, floatmode='maxprec', suppress=True)

                pbar.set_postfix(
                    dict(
                        loss=np.asarray(
                            [loss_G.item(), loss_D_s.item(), loss_D_f.item()]),
                        lr=[optimizer_G.param_groups[0]['lr'],
                            optimizer_D_f.param_groups[0]['lr'],
                            optimizer_D_s.param_groups[0]['lr']]
                    ))

        self.save_model()
        print('end training')

    def test(self):
        #self.sample_from_latent()
        #self.calc_MMD()
        self.calc_MMD(test_set=True)

    def sample_from_latent(self):
        path_sample = os.path.join(self.path, 'sample/')

        for path in [path_sample]:
            try:
                shutil.rmtree(path)
            except FileNotFoundError:
                pass
            os.makedirs(path)

        batch_size = 20

        sampled_motions = self.sample(batch_size)
        names = [str(i).zfill(4) for i in range(batch_size)]

        self.write_bvhs(sampled_motions, names, 1 / self.fps, path_sample)

    def sample(self, batch_size=20, gpu=True):
        z_sample_shape = [batch_size]
        z_sample_shape.extend([self.dim_z])
        z_sample = np.random.normal(size=z_sample_shape)
        if gpu:
            z_sample = self._to_torch(z_sample)
        else:
            z_sample = torch.from_numpy(z_sample.astype(np.float32))
        self.G.eval()
        with torch.no_grad():
            motions = self.G(z_sample)
            lengths = self.input_length - self.estimator_length(z_sample)
        self.G.train()
        if gpu:
            motions = self._to_numpy(motions)
            lengths = self._to_numpy(lengths)
        else:
            motions = motions.numpy()
            lengths = lengths.numpy()

        lengths = np.round(lengths).squeeze(axis=1).astype('int32')
        motions = [motion[:length] for motion, length in zip(motions, lengths)]

        return motions

    def smoothness_loss(self, motions, lengths):
        loss = torch.zeros(self.batch_size, self.input_length - 2).cuda()
        if not self.omit_root_pos:
            pos = motions[:, :, :3]

            pos_vel = pos[:, 1:] - pos[:, :-1]
            loss += torch.sum((pos_vel[:, 1:] - pos_vel[:, :-1])**2, 2)

            motions = motions[:, :, 3:]

        if self.rep in ['quat', 'expmap', 'ortho6d']:
            motions = self.compute_rotation_matrix(motions)

            ang_vel = motions[:, 1:] - motions[:, :-1]
            loss += torch.sum((ang_vel[:, 1:] - ang_vel[:, :-1])**2, [2, 3, 4])

        mask = self.get_mask(lengths).cuda().type(torch.float)[:, 2:]
        loss = torch.sum(loss * mask, 1) / torch.sum(mask, 1)

        return loss.mean()

    def calc_MMD(self, ntrial=10, test_set=False):
        if test_set:
            print('Test Set')
        else:
            print('Train Set')
        
        inputs = self.sample_motions(test=test_set)

        self.G.eval()

        extractor = lstmAE_wo_Norm(self.max_len, self.dim_pose, self.dim_z).cuda()
        extractor.load_state_dict(torch.load(self.extractor_path))
        extractor.eval()

        motions, lengths = inputs
        batch_size, _, _ = motions.shape
        motions_real_pad = torch.zeros(
            [batch_size, self.max_len, self.dim_pose]).cuda()
        motions_real_pad[:, :self.input_length, :] = motions
        with torch.no_grad():
            feature_real = extractor.encoder(motions_real_pad, lengths)

        mmds = []
        for _ in range(ntrial):
            z_sample_shape = [batch_size]
            z_sample_shape.extend([self.dim_z])
            z_sample = np.random.normal(size=z_sample_shape)
            z_sample = self._to_torch(z_sample)
            with torch.no_grad():
                motions = self.G(z_sample)

                motions_fake_pad = torch.zeros(
                    [batch_size, self.max_len, self.dim_pose]).cuda()
                motions_fake_pad[:, :self.input_length, :] = motions
                
                lengths = self._to_numpy(
                    self.input_length - self.estimator_length(z_sample)).squeeze().astype('int32')
                
                feature_fake = extractor.encoder(motions_fake_pad, lengths)
                mmd = MMD_loss()(feature_real, feature_fake)
            mmds = np.append(mmds, self._to_numpy(mmd))

        self.G.train()

        print(f"mmd: {mmds.mean()}"
              f"min: {mmds.min()}"
              f"max: {mmds.max()}")
        
        suffix = "_test" if test_set else "_train"
        np.save(os.path.join(self.path, "mmd" + suffix + ".npy"), mmds)
