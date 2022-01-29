import os
import shutil
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.spatial.transform import Rotation as R
import torch
import torch.optim as optim
# from torchviz import make_dot

from util.arg_parser import ArgParser

from motionAE.src.motionTrainer import motionTrainer
from motionAE.src.models import lstmAE, lstmAE_feedback, lstmAE_wo_Norm


class motionAETrainer(motionTrainer):
    required_attributes = ['model']

    def load_param(self, arg_parser: ArgParser, **kwargs) -> None:
        super().load_param(arg_parser, **kwargs)

        self.num_epochs = arg_parser.parse_int('num_epochs')
        self.batch_size = arg_parser.parse_int('batch_size')

        self.learning_rate = arg_parser.parse_float('learning_rate')
        self.decay_factor = arg_parser.parse_float('decay_factor')

        self.dim_z = arg_parser.parse_int('dim_z')

        self.smooth_coef = arg_parser.parse_float('smooth_coef')

        # viable in _feedback, _initfeed architecture
        self.residual = arg_parser.parse_bool('residual')
        # calc recon loss as geodesic loss, not recommended
        self.use_geodesic = arg_parser.parse_bool('use_geodesic')
        # use DTW recon loss, not recommended
        self.use_DTW = arg_parser.parse_bool('use_DTW')

        self.path = os.path.join(arg_parser.parse_string('output_path'),
                                 self.architecture,
                                 os.path.basename(self.dataset),
                                 # "geodesic_" + str(self.use_geodesic),
                                 "residual_" + str(self.residual),
                                 "_".join(self.used_class),
                                 "_".join(self.used_subject)
                                 )
        
    def build_model(self, gpu: bool = True) -> None:
        if self.architecture == 'lstmAE':
            self.model = lstmAE(self.input_length, self.dim_pose, self.dim_z)
        elif self.architecture == 'lstmAE_feedback':
            self.model = lstmAE_feedback(
                self.input_length,
                self.dim_pose,
                self.dim_z,
                residual=self.residual)
        elif self.architecture == 'lstmAE_wo_Norm':
            self.model = lstmAE_wo_Norm(self.input_length, self.dim_pose, self.dim_z)
        else:
            raise(ValueError)

        if gpu is True:
            self.model = self.model.cuda()

    def loss(self, *args):
        return self.recon_loss(*args) + self.smooth_coef * \
            self.smoothness_loss(*args)

    def recon_loss(self, *args, separate=False):
        inputs = args[0]
        recons = args[1]
        lengths = args[2]

        if separate:
            dist_pos = self.distance_pos(inputs, recons)
            dist_angle = self.distance_angle(inputs, recons)

            mask = self.get_mask(lengths).cuda().type(torch.float)

            dist_pos = torch.sum(dist_pos * mask, 1) / torch.sum(mask, 1)
            dist_angle = torch.sum(dist_angle * mask, 1) / torch.sum(mask, 1)

            return dist_pos, dist_angle

        if self.use_DTW:
            inputs_dtw = torch.zeros_like(inputs)
            recons_dtw = torch.zeros_like(recons)

            paths = self.DTW(self._to_numpy(inputs), self._to_numpy(recons))
            for i in range(len(paths)):
                inputs_dtw[i] = inputs[i][paths[i, :, 0]]
                recons_dtw[i] = recons[i][paths[i, :, 1]]

            loss = self.distance_framewise(inputs_dtw, recons_dtw)
        else:
            loss = self.distance_framewise(inputs, recons)

        mask = self.get_mask(lengths).cuda().type(torch.float)
        loss = torch.sum(loss * mask, 1) / torch.sum(mask, 1)

        return loss.mean()

    def distance_pos(self, seq1, seq2):
        batch_size, input_length, _ = seq1.shape
        dist = torch.zeros(batch_size, input_length).cuda()

        if not self.omit_root_pos:
            pos_seq1 = seq1[:, :, :3]
            pos_seq2 = seq2[:, :, :3]

            dist += torch.sum((pos_seq1 - pos_seq2)**2, 2)

        return dist

    def distance_angle(self, seq1, seq2):
        batch_size, input_length, _ = seq1.shape
        dist = torch.zeros(batch_size, input_length).cuda()

        if not self.omit_root_pos:

            seq1 = seq1[:, :, 3:]
            seq2 = seq2[:, :, 3:]

        if self.rep in ['quat', 'expmap', 'ortho6d']:
            # batch * seq * (dim_pose - 3) : batch * seq * joint_num * 3 * 3
            seq1 = self.compute_rotation_matrix(seq1)
            seq2 = self.compute_rotation_matrix(seq2)

            if self.use_geodesic:
                dist += torch.sum(self.compute_geodesic_distance(seq1,
                                                                 seq2), dim=2)
            else:
                dist += torch.sum((seq1 - seq2)**2, [2, 3, 4])

        else:
            raise(ValueError)

        return dist

    def smoothness_loss(self, *args):
        recons = args[1]
        lengths = args[2]

        loss = torch.zeros(self.batch_size, self.input_length - 2).cuda()
        if not self.omit_root_pos:
            pos = recons[:, :, :3]

            pos_vel = pos[:, 1:] - pos[:, :-1]
            loss += torch.sum((pos_vel[:, 1:] - pos_vel[:, :-1])**2, 2)

            recons = recons[:, :, 3:]

        if self.rep in ['quat', 'expmap', 'ortho6d']:
            recons = self.compute_rotation_matrix(recons)

            ang_vel = recons[:, 1:] - recons[:, :-1]
            loss += torch.sum((ang_vel[:, 1:] - ang_vel[:, :-1])**2, [2, 3, 4])

        mask = self.get_mask(lengths).cuda().type(torch.float)[:, 2:]
        loss = torch.sum(loss * mask, 1) / torch.sum(mask, 1)

        return loss.mean()

    def distance_framewise(self, seq1, seq2):
        return self.distance_pos(seq1, seq2) + self.distance_angle(seq1, seq2)

    def DTW(self, inputs, recons):
        from fastdtw import fastdtw
        # paths = np.array([fastdtw(inp, rec, dist=self.compute_dist_numpy)[1] for inp, rec in zip(inputs, recons)])
        # approximate ditance for simple MSE
        dists_and_paths = [Parallel(n_jobs=16)(delayed(fastdtw)(
            inp, rec, dist=2) for inp, rec in zip(inputs, recons))]

        paths = np.array([[path for dist, path in dist_and_path]
                          for dist_and_path in dists_and_paths]).squeeze()

        print(np.sum(paths[:, :, 0] - range(self.input_length)))
        print(np.sum(paths[:, :, 1] - range(self.input_length)))

        return paths

    def compute_dist_numpy(self, pose0, pose1):
        dist = 0
        if not self.omit_root_pos:
            pos0 = pose0[:3]
            pos1 = pose1[:3]

            dist += np.sum((pos0 - pos1)**2)

            pose0 = pose0[3:]
            pose1 = pose1[3:]

        if self.rep in ['quat', 'expmap']:
            dist += np.sum((pose0 - pose1)**2)
        elif self.rep == 'ortho6d':
            dim = len(pose0)
            pose0 = np.reshape(pose0, [int(dim / 6), 6])
            pose1 = np.reshape(pose1, [int(dim / 6), 6])
            if self.use_geodesic:
                dist += np.sum([(ortho6dToR(ortho0) * ortho6dToR(ortho1).inv()).magnitude()
                                for ortho0, ortho1 in zip(pose0, pose1)])
            else:
                dist += np.sum((self.ortho6dToMatrix_numpy(pose0) -
                                self.ortho6dToMatrix_numpy(pose1))**2)
        else:
            raise(ValueError)
        print(dist)

        return dist

    def compute_geodesic_distance(self, inputs, recons):
        batch_size, input_len, num_joints, _, _ = inputs.size()
        inputs = inputs.reshape(-1, 3, 3)
        recons = recons.reshape(-1, 3, 3)

        m = torch.bmm(inputs, recons.transpose(1, 2))

        cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
        cos = torch.min(cos, torch.autograd.Variable(
            torch.ones(batch_size * input_len * num_joints).cuda()))
        cos = torch.max(cos, torch.autograd.Variable(
            torch.ones(batch_size * input_len * num_joints).cuda()) * -1)

        theta = torch.acos(cos)

        return theta.reshape([batch_size, input_len, num_joints])

    def train(self):
        num_epochs = self.num_epochs

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=int(num_epochs / 5), gamma=self.decay_factor)
        criterion = self.loss

        with tqdm(range(num_epochs), ncols=100) as pbar:
            # with tqdm(range(2), ncols=100) as pbar:
            for i, epoch in enumerate(pbar):
                optimizer.zero_grad()

                inputs = self.sample_motions(batch_size=self.batch_size)
                result = self.model(*inputs)

                loss = criterion(*result)

                loss.backward()
                optimizer.step()

                scheduler.step()

                """
                if i == 0:
                    dot = make_dot(loss, params=dict(self.model.named_parameters()))
                    dot.format = 'png'
                    dot.render('graph_image')
                """

                pbar.set_postfix(dict(loss=np.asarray(
                    [loss.item()]), lr=optimizer.param_groups[0]['lr']))

        self.save_model()
        print('end training')

    def test(self) -> None:
        self.reconst_motion()
        self.calc_recon_error()
        self.calc_recon_error(test_set=True)

    def encode(self, batch):
        batch = self._to_torch(batch)
        return self.model.encoder(batch)

    def reconst_motion(self):
        path_input = os.path.join(self.path, 'input/')
        path_reconst = os.path.join(self.path, 'reconst/')

        for path in [path_input, path_reconst]:
            try:
                shutil.rmtree(path)
            except FileNotFoundError:
                pass
            os.makedirs(path)

        size = 20

        inputs, names_batch = self.sample_motions(
            batch_size=size, return_name=True)
        self.model.eval()
        result = self.model(*inputs)
        X_batch = self._to_numpy(result[0])
        X_batch_reconst = self._to_numpy(result[1])
        # X_length = self._to_numpy(result[2])

        self.write_bvhs(X_batch, names_batch, 1 / self.fps, path_input)
        self.write_bvhs(X_batch_reconst, names_batch,
                        1 / self.fps, path_reconst)

    def calc_recon_error(self, ntrial=10, test_set=False):
        if test_set:
            print('Test Set')
        else:
            print('Train Set')

        inputs = self.sample_motions(test=test_set)

        self.model.eval()

        err_pos = []
        err_ang = []
        for _ in range(ntrial):
            with torch.no_grad():
                result = self.model(*inputs)
            dist_pos, _ = self.recon_loss(*result, separate=True)
            dist_pos = torch.sqrt(dist_pos)
            
            seq_input = self.compute_rotation_matrix(result[0][:, :, 3:])
            seq_recon = self.compute_rotation_matrix(result[1][:, :, 3:])
            
            b, n, j, _, _ = seq_input.data.shape
            
            seq_input = rotmat2euler_torch(seq_input.view(-1, 3, 3)).view(b, n, j, 3)
            seq_recon = rotmat2euler_torch(seq_recon.view(-1, 3, 3)).view(b, n, j, 3)
            
            dist_ang = torch.sum((seq_input - seq_recon) ** 2, axis=(2, 3))
            dist_ang = torch.sqrt(dist_ang)
            
            mask = self.get_mask(result[2]).cuda().type(torch.float)
            
            dist_ang = torch.sum(dist_ang * mask, axis=1) / torch.sum(mask, axis=1)
            
            dist_pos = self._to_numpy(dist_pos.mean())
            dist_ang = self._to_numpy(dist_ang.mean())
            err_pos = np.append(err_pos, dist_pos)
            err_ang = np.append(err_ang, dist_ang)

        self.model.train()

        print(f"pos err: {err_pos.mean()} "
              f"min: {err_pos.min()} "
              f"max: {err_pos.max()} ")
        print(f"ang err: {err_ang.mean()} "
              f"min: {err_ang.min()} "
              f"max: {err_ang.max()} ")
        
        suffix = "_test" if test_set else "_train"
        np.save(os.path.join(self.path, "err_pos" + suffix + ".npy"), err_pos)
        np.save(os.path.join(self.path, "err_ang" + suffix + ".npy"), err_ang)

    def seq_matrix2Euler(self, seq):
        # seq: batch * len * 3 * 3
        shape = seq.shape
        seq = np.reshape(seq, [-1, 3, 3])
        from joblib import Parallel, delayed
        seq = np.array(Parallel(n_jobs=10)([delayed(matrix2euler)(rm) for rm in seq]))
        return np.reshape(seq, np.concatenate([shape[:-2], [3]]))


def ortho6dToR(ortho6d, return_matrix=False):
    assert len(ortho6d) == 6

    x_raw = ortho6d[0:3]
    y_raw = ortho6d[3:6]

    x = x_raw / np.linalg.norm(x_raw)
    z = np.cross(x, y_raw)
    z = z / np.linalg.norm(z)
    y = np.cross(z, x)
    if return_matrix:
        return np.array([x, y, z])
    else:
        return R.from_matrix([x, y, z])
    
    
def matrix2euler(rm):
    return R.from_matrix(rm).as_euler('ZYX')


def rotmat2euler_torch(R):
    """
    Converts a rotation matrix to euler angles
    batch pytorch version ported from the corresponding numpy method above

    :param R:N*3*3
    :return: N*3
    """
    from torch.autograd import Variable
    n = R.data.shape[0]
    eul = Variable(torch.zeros(n, 3).float()).cuda()
    idx_spec1 = (R[:, 0, 2] == 1).nonzero(
    ).cpu().data.numpy().reshape(-1).tolist()
    idx_spec2 = (R[:, 0, 2] == -
                 1).nonzero().cpu().data.numpy().reshape(-1).tolist()
    if len(idx_spec1) > 0:
        R_spec1 = R[idx_spec1, :, :]
        eul_spec1 = Variable(torch.zeros(len(idx_spec1), 3).float()).cuda()
        eul_spec1[:, 2] = 0
        eul_spec1[:, 1] = -np.pi / 2
        delta = torch.atan2(R_spec1[:, 0, 1], R_spec1[:, 0, 2])
        eul_spec1[:, 0] = delta
        eul[idx_spec1, :] = eul_spec1

    if len(idx_spec2) > 0:
        R_spec2 = R[idx_spec2, :, :]
        eul_spec2 = Variable(torch.zeros(len(idx_spec2), 3).float()).cuda()
        eul_spec2[:, 2] = 0
        eul_spec2[:, 1] = np.pi / 2
        delta = torch.atan2(R_spec2[:, 0, 1], R_spec2[:, 0, 2])
        eul_spec2[:, 0] = delta
        eul[idx_spec2] = eul_spec2

    idx_remain = np.arange(0, n)
    idx_remain = np.setdiff1d(np.setdiff1d(
        idx_remain, idx_spec1), idx_spec2).tolist()
    if len(idx_remain) > 0:
        R_remain = R[idx_remain, :, :]
        eul_remain = Variable(torch.zeros(len(idx_remain), 3).float()).cuda()
        eul_remain[:, 1] = -torch.asin(R_remain[:, 0, 2])
        eul_remain[:, 0] = torch.atan2(R_remain[:, 1, 2] / torch.cos(eul_remain[:, 1]),
                                       R_remain[:, 2, 2] / torch.cos(eul_remain[:, 1]))
        eul_remain[:, 2] = torch.atan2(R_remain[:, 0, 1] / torch.cos(eul_remain[:, 1]),
                                       R_remain[:, 0, 0] / torch.cos(eul_remain[:, 1]))
        eul[idx_remain, :] = eul_remain

    return eul
