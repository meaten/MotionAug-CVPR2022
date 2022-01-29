from motionAE.src.motionVAETrainer import motionVAETrainer
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from motionAE.src.models import lstmVAE, Discriminator_frame, Discriminator_seq


class motionVAEAdvTrainer(motionVAETrainer):

    def build_model(self, gpu=True):
        if self.architecture == 'lstmVAE_adversarial':
            self.model = lstmVAE(self.input_length, self.dim_pose, self.dim_z)
            self.dis_f = Discriminator_frame(self.dim_pose, self.dim_z)
            self.dis_s = Discriminator_seq(
                self.input_length, self.dim_pose, self.dim_z)

        elif self.architecture == 'lstmVAE_adversarial_frame':
            self.model = lstmVAE(self.input_length, self.dim_pose, self.dim_z)
            self.dis_f = Discriminator_frame(self.dim_pose, self.dim_z)
            self.dis_s = None
                
                
        elif self.architecture == 'lstmVAE_adversarial_sequence':
            self.model = lstmVAE(self.input_length, self.dim_pose, self.dim_z)
            self.dis_f = None
            self.dis_s = Discriminator_seq(self.input_length, self.dim_pose, self.dim_z)
                
        else:
            raise(ValueError)


        if gpu is True:
            self.model = self.model.cuda()
            self.dis_f = self.dis_f.cuda() if self.dis_f is not None else self.dis_f
            self.dis_s = self.dis_s.cuda() if self.dis_s is not None else self.dis_s
    def train(self):
        num_epochs = self.num_epochs

        optimizer_vae = optim.SGD(
            self.model.parameters(), lr=self.learning_rate)
        scheduler_vae = optim.lr_scheduler.StepLR(
            optimizer_vae, step_size=int(num_epochs / 5), gamma=self.decay_factor)

        if self.dis_f is not None:
            optimizer_dis_f = optim.SGD(
                self.dis_f.parameters(), lr=self.learning_rate)
            scheduler_dis_f = optim.lr_scheduler.StepLR(
                optimizer_dis_f, step_size=int(num_epochs / 5), gamma=self.decay_factor)
            
        if self.dis_s is not None:
            optimizer_dis_s = optim.SGD(
                self.dis_s.parameters(), lr=self.learning_rate)
            scheduler_dis_s = optim.lr_scheduler.StepLR(
                optimizer_dis_s, step_size=int(num_epochs / 5), gamma=self.decay_factor)
            
        criterion = self.loss

        with tqdm(range(num_epochs), ncols=100) as pbar:
            for i, epoch in enumerate(pbar):
                loss_list = []
                lr_list = []
                
                inputs = self.sample_motions(
                    batch_size=self.batch_size)  # [motions, lengths]
                # [inputs, recons, lengths, mu, log_var, z, pred_lengths]
                result = self.model(*inputs)
                
                fake = result[1]
                
                # pred_lengths = result[2]
                pred_lengths = self.input_length - result[6]
                pred_lengths = torch.clamp(pred_lengths, 1, self.input_length)
                pred_lengths = torch.squeeze(torch.round(pred_lengths)).type(torch.IntTensor)
                
                # vae
                optimizer_vae.zero_grad()

                loss_vae = criterion(*result)
                if self.dis_f is not None:
                    loss_vae -= self.dis_f(fake).mean()
                if self.dis_s is not None:
                    loss_vae -= self.dis_s(fake, pred_lengths).mean()

                loss_vae.backward(retain_graph=True)
                optimizer_vae.step()
                scheduler_vae.step()

                loss_list.append(loss_vae.item())
                lr_list.append(optimizer_vae.param_groups[0]['lr'])
                
                # discriminator
                if self.dis_f is not None:
                    optimizer_dis_f.zero_grad()
                    real_loss_dis_f = torch.nn.ReLU()(
                        1.0 - self.dis_f(result[0])).mean()
                    fake_loss_dis_f = torch.nn.ReLU()(
                        1.0 + self.dis_f(fake.detach())).mean()
                    loss_dis_f = (real_loss_dis_f + fake_loss_dis_f) / 2
                    loss_dis_f.backward()
                    optimizer_dis_f.step()
                    scheduler_dis_f.step()
                    loss_list.append(loss_dis_f.item())
                    lr_list.append(optimizer_dis_f.param_groups[0]['lr'])
                

                if self.dis_s is not None:    
                    optimizer_dis_s.zero_grad()
                    real_loss_dis_s = torch.nn.ReLU()(
                        1.0 - self.dis_s(result[0], result[2])).mean()
                    fake_loss_dis_s = torch.nn.ReLU()(
                        1.0 + self.dis_s(fake.detach(), pred_lengths.detach())).mean()
                    loss_dis_s = (real_loss_dis_s + fake_loss_dis_s) / 2
                    loss_dis_s.backward()
                    optimizer_dis_s.step()
                    scheduler_dis_s.step()
                    loss_list.append(loss_dis_s.item())
                    lr_list.append(optimizer_dis_s.param_groups[0]['lr'])
                

                np.set_printoptions(
                    precision=3, floatmode='maxprec', suppress=True)

                pbar.set_postfix(
                    dict(
                        loss=np.array(loss_list),
                        lr=np.array(lr_list)
                    ))

        self.save_model()
        print('end training')
