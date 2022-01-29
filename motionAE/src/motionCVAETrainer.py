from motionAE.src.motionVAETrainer import motionVAETrainer

import os
import shutil
import numpy as np

import torch

from motionAE.src.models import lstmCVAE


class motionCVAETrainer(motionVAETrainer):

    def load_param(self, arg_parser, **kwargs):
        super().load_param(arg_parser, **kwargs)

    def choose_data(self, used_class):
        classes = self.one_hot_encoder.inverse_transform(self.one_hot_labels)
        print(np.unique(classes))
        self.all_classes = np.unique(classes)
        self.used_class = used_class  # used only for sampling motions
        # do not remove other classes in training

    def build_model(self, gpu=True):
        if self.architecture == 'lstmCVAE':
            self.model = lstmCVAE(
                self.input_length, self.dim_pose, self.dim_z, len(self.all_classes))
        else:
            raise(ValueError)

        if gpu is True:
            self.model = self.model.cuda()

    def sample_motions(self, idx=[], batch_size=None, return_name=False):
        if len(idx) != 0:
            pass
        elif batch_size is not None:
            idx = np.random.randint(0, len(self.motions), size=batch_size)
        else:
            raise(ValueError)

        motions = np.asarray([self._sample_motion(motion, self.input_length, frame_time)
                              for motion, frame_time in zip(self.motions[idx], self.frame_times[idx])])
        motions = self._to_torch(motions)
        # add class_vector to inputs
        class_vector = self._to_torch(self.one_hot_labels[idx])

        ret = [motions, self._calc_lengths(idx), class_vector]

        if return_name:
            return ret, self.names[idx]

        return ret

    def sample(self, batch_size=20, used_class=None, gpu=True):
        z_sample_shape = [batch_size]
        z_sample_shape.extend([self.dim_z])
        z_sample = np.random.normal(size=z_sample_shape)
        if used_class is not None:
            class_vector = self.one_hot_encoder.transform([used_class])
        else:
            class_vector = self.one_hot_encoder.transform([self.used_class])
        class_vector = np.tile(class_vector, (batch_size, 1))

        if gpu:
            z_sample = self._to_torch(z_sample)
            class_vector = self._to_torch(class_vector)
        else:
            z_sample = torch.from_numpy(z_sample.astype(np.float32))
            class_vector = torch.from_numpy(class_vector.astype(np.float32))

        self.model.decoder.eval()
        with torch.no_grad():
            motions = self.model.decoder(z_sample, class_vector)
        self.model.decoder.train()

        if gpu:
            motions = self._to_numpy(motions)
        else:
            motions = motions.numpy()
        return motions

    def sample_from_latent(self):
        path_sample = os.path.join(self.path, 'sample/')

        for path in [path_sample]:
            try:
                shutil.rmtree(path)
            except FileNotFoundError:
                pass
            os.makedirs(path)

        if self.used_class == 'all':
            classes = self.all_classes
        else:
            classes = [self.used_class]

        batch_size = 20

        for cla in classes:
            sampled_motions = self.sample(batch_size, used_class=cla)
            names = [cla + str(i).zfill(4) for i in range(batch_size)]

            self.write_bvhs(sampled_motions, names, 1 / self.fps, path_sample)
