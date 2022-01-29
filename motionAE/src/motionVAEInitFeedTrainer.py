from motionAE.src.motionVAETrainer import motionVAETrainer
import numpy as np
import torch
from motionAE.src.models import lstmVAE_initfeed


class motionVAEInitFeedTrainer(motionVAETrainer):
    
    def build_model(self, gpu=True):
        if self.architecture == 'lstmVAE_initfeed':
            self.model = lstmVAE_initfeed(self.input_length, self.dim_pose, self.dim_z)
        else:
            raise(ValueError)
            
        if gpu is True:
            self.model = self.model.cuda()
        
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
        self.model.decoder.train()
        if gpu:
            motions = self._to_numpy(motions)
        else:
            motions = motions.numpy()
        return motions
