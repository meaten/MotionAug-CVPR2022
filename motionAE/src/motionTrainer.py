from abc import ABCMeta

import os
import sys
import re
import shutil
import numpy as np
from scipy.spatial.transform import Rotation as R
from sklearn.preprocessing import LabelBinarizer
from joblib import Parallel, delayed
import torch
# from torchviz import make_dot

from util.arg_parser import ArgParser
from motionAE.src.util.util_bvh import loadBvh, writeBvh, getRotationOrderAndChannels
from motionAE.src.util.bvh import offsetEulerAngle, loadSetting


class MyMeta(ABCMeta):
    required_attributes = []

    def __call__(self, *args, **kwargs):
        obj = super(MyMeta, self).__call__(*args, **kwargs)
        for attr_name in obj.required_attributes:
            if not hasattr(obj, attr_name):
                raise ValueError('required attribute (%s) not set' % attr_name)
        return obj


class motionTrainer(object, metaclass=MyMeta):

    def __init__(self, arg_parser: ArgParser, **kwargs):
        self.load_param(arg_parser, **kwargs)
        self.build_model(**kwargs)

    def load_param(self, arg_parser: ArgParser, gpu: bool = True):
        self.architecture = arg_parser.parse_string('architecture')
        os.environ['CUDA_VISIBLE_DEVICES'] = arg_parser.parse_string(
            'gpu', '0') if gpu else '-1'
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() and gpu else "cpu")

        self.training = arg_parser.parse_bool('training', False)

        self.data_path = arg_parser.parse_string('data_path')
        self.dataset = arg_parser.parse_string('dataset')
        # should be set as denominator of dataset fps
        self.fps = arg_parser.parse_int('fps')

        print(f"dataset: {self.dataset}")
        with np.load(self.dataset, allow_pickle=True) as dataset:
            self.motions = dataset['motions']
            self.frame_times = dataset['frame_times']
            self.names = dataset['names']
            self.subjects = dataset['subjects']
            self.dim_pose = int(dataset['dim_pose'])
            self.rep = str(dataset['rep'])
            self.omit_root_pos = bool(dataset['omit_root_pos'])
            self.abs_angle = bool(dataset['abs_angle'])

            # HACK
            self.input_length = -1
            self.max_len = np.max(self._calc_lengths(range(len(self.motions)))) + 1

        self.build_labels()
        self.pick_data(
            used_class=arg_parser.parse_strings('class', 'all'),
            used_subject=arg_parser.parse_strings('subject', ['all']))
        self.pick_test_data()
        self.calc_input_length(window_size=arg_parser.parse_int("window_size"))

        self.settings = loadSetting(self.data_path)
        self.used_joints = [joint[0]
                            for joint in self.settings['jointAssignments']]

        self.dim_dict = {
            'quat': 4,
            'expmap': 3,
            'ortho6d': 6
        }

    def pick_data(self, used_class: list, used_subject: list) -> None:
        self.used_class = used_class
        self.used_subject = used_subject
        if 'all' not in self.used_class:
            classes = self.one_hot_encoder.inverse_transform(
                self.one_hot_labels)
            self.all_classes = np.unique(classes)
            idx = np.isin(classes, self.used_class)
            self.pick_index(idx)

        if 'all' not in self.used_subject:
            idx = np.isin(self.subjects, self.used_subject)
            self.pick_index(idx)
        print(f"class: {self.used_class}")
        print(f"subject: {self.used_subject}")
        print(f"#motion: {len(self.motions)}")
        
    def pick_test_data(self) -> None:
        self.test_motions = []
        self.test_frame_times = []
        self.test_names = []
        self.test_subjects = []
        self.test_one_hot_labels = []
        
        with np.load("motionAE/dataset/dataset_ortho6d_rel_rp_rr_mirror_split.npz", allow_pickle=True) as dataset:
            motions = dataset['motions']
            frame_times = dataset['frame_times']
            names = dataset['names']
            subjects = dataset['subjects']
            descs = [name[7:-8] for name in names]
            labels = np.array([re.findall('([a-z]+)', desc)[0] for desc in descs])
            one_hot_labels = self.one_hot_encoder.transform(labels)
            
        if 'all' not in self.used_class:
            idx = np.isin(labels, self.used_class)
            motions = motions[idx]
            frame_times = frame_times[idx]
            names = names[idx]
            subjects = subjects[idx]
            one_hot_labels = one_hot_labels[idx]
            
        if "all" not in self.used_subject:
            idx = np.isin(subjects, self.used_subject)
            ex_idx = np.logical_not(idx)
            self.test_motions = motions[ex_idx]
            self.test_frame_times = frame_times[ex_idx]
            self.test_names = names[ex_idx]
            self.test_subjects = subjects[ex_idx]
            self.test_one_hot_labels = one_hot_labels[ex_idx]
            
        print(f"test class: {self.used_class}")
        print(f"test subject: {np.unique(self.test_subjects)}")
        print(f"#test motion: {len(self.test_motions)}")
            
    def pick_index(self, idx: np.ndarray) -> None:
        self.names = self.names[idx]
        self.frame_times = self.frame_times[idx]
        self.motions = self.motions[idx]
        self.one_hot_labels = self.one_hot_labels[idx]
        self.subjects = self.subjects[idx]

    def build_labels(self):
        descs = [name[7:-8] for name in self.names]
        labels = np.array([re.findall('([a-z]+)', desc)[0] for desc in descs])

        one_hot_encoder = LabelBinarizer()
        one_hot_labels = one_hot_encoder.fit_transform(labels)

        self.one_hot_encoder = one_hot_encoder
        self.one_hot_labels = one_hot_labels

    def calc_input_length(self, window_size):
        if window_size > 0:
            self.input_length = window_size
        else:
            self.input_length = -1
            self.motion_lengths = self._calc_lengths(range(len(self.motions)))
            self.input_length = np.max(self.motion_lengths)

        print(f"input sequence length: {self.input_length}")

    def recreate_path(self):
        try:
            shutil.rmtree(self.path)
        except FileNotFoundError:
            pass

        os.makedirs(self.path)

    def train(self):
        pass

    def test(self):
        pass

    def save_model(self):
        self.recreate_path()
        path = os.path.join(self.path, "model.pth")
        torch.save(self.model.state_dict(), path)

    def load_model(self, gpu=True):
        path = os.path.join(self.path, "model.pth")
        if gpu is False:
            self.model.load_state_dict(torch.load(
                path, map_location=torch.device('cpu')))
            self.model.to('cpu')
        else:
            self.model.load_state_dict(torch.load(path))

    def compute_rotation_matrix(self, motions):
        batch_size, input_len, dim_pose = motions.size()
        motions = motions.reshape(-1, self.dim_dict[self.rep])
        if self.rep == 'ortho6d':
            motions = self.ortho6dToMatrix(motions)
        elif self.rep == 'quat':
            motions = self.quatToMatrix(motions)
        elif self.rep == 'expmap':
            motions = self.rodToMatrix(motions)
        else:
            raise(ValueError)
        motions = motions.reshape(batch_size, input_len, int(
            dim_pose / self.dim_dict[self.rep]), 3, 3)
        return motions

    def ortho6dToMatrix_numpy(self, ortho6d):
        assert ortho6d.shape()[-1] == 6

        x_raw = ortho6d[:, 0:3]
        y_raw = ortho6d[:, 3:6]

        x = normalize_vector_numpy(x_raw)
        z = np.cross(x, y_raw)
        z = normalize_vector_numpy(z)
        y = np.cross(z, x)

        x = np.reshape(x, [-1, 1, 3])
        y = np.reshape(y, [-1, 1, 3])
        z = np.reshape(z, [-1, 1, 3])

        matrix = np.concatenate([x, y, z], axis=1)
        return matrix

    def ortho6dToMatrix(self, ortho6d):
        # ortho6d: -1 * 6
        assert ortho6d.size()[-1] == 6

        x_raw = ortho6d[:, 0:3]  # batch*3
        y_raw = ortho6d[:, 3:6]  # batch*3

        x = normalize_vector(x_raw, device=self.device)  # batch*3
        z = cross_product(x, y_raw)  # batch*3
        z = normalize_vector(z, device=self.device)  # batch*3
        y = cross_product(z, x)  # batch*3

        x = x.view(-1, 1, 3)
        y = y.view(-1, 1, 3)
        z = z.view(-1, 1, 3)
        matrix = torch.cat((x, y, z), 1)  # batch*3*3
        return matrix

    def quatToMatrix(self, quaternion):
        # quaternion: -1*4 xyzw
        assert quaternion.size()[-1] == 4

        quat = normalize_vector(quaternion, device=self.device).contiguous()

        qx = quat[..., 0].contiguous().view(-1, 1)
        qy = quat[..., 1].contiguous().view(-1, 1)
        qz = quat[..., 2].contiguous().view(-1, 1)
        qw = quat[..., 3].contiguous().view(-1, 1)

        # Unit quaternion rotation matrices computatation
        xx = qx * qx
        yy = qy * qy
        zz = qz * qz
        xy = qx * qy
        xz = qx * qz
        yz = qy * qz
        xw = qx * qw
        yw = qy * qw
        zw = qz * qw

        row0 = torch.cat((1 - 2 * yy - 2 * zz,
                          2 * xy - 2 * zw,
                          2 * xz + 2 * yw), 1)  # batch*3
        row1 = torch.cat((2 * xy + 2 * zw,
                          1 - 2 * xx - 2 * zz,
                          2 * yz - 2 * xw), 1)  # batch*3
        row2 = torch.cat((2 * xz - 2 * yw,
                          2 * yz + 2 * xw,
                          1 - 2 * xx - 2 * yy), 1)  # batch*3

        matrix = torch.cat(
            (row0.view(-1, 1, 3), row1.view(-1, 1, 3), row2.view(-1, 1, 3)), 1)  # batch*3*3

        return matrix

    def rodToMatrix(self, rod):
        # axisAngle: -1*3 (x,y,z)*theta
        assert rod.size()[-1] == 3

        axis, theta = normalize_vector(
            rod, return_mag=True, device=self.device)

        sin = torch.sin(theta / 2)

        qw = torch.cos(theta / 2)
        qx = axis[:, 0] * sin
        qy = axis[:, 1] * sin
        qz = axis[:, 2] * sin

        # Unit quaternion rotation matrices computatation
        xx = (qx * qx).view(-1, 1)
        yy = (qy * qy).view(-1, 1)
        zz = (qz * qz).view(-1, 1)
        xy = (qx * qy).view(-1, 1)
        xz = (qx * qz).view(-1, 1)
        yz = (qy * qz).view(-1, 1)
        xw = (qx * qw).view(-1, 1)
        yw = (qy * qw).view(-1, 1)
        zw = (qz * qw).view(-1, 1)

        row0 = torch.cat((1 - 2 * yy - 2 * zz,
                          2 * xy - 2 * zw,
                          2 * xz + 2 * yw), 1)  # batch*3
        row1 = torch.cat((2 * xy + 2 * zw,
                          1 - 2 * xx - 2 * zz,
                          2 * yz - 2 * xw), 1)  # batch*3
        row2 = torch.cat((2 * xz - 2 * yw,
                          2 * yz + 2 * xw,
                          1 - 2 * xx - 2 * yy), 1)  # batch*3

        matrix = torch.cat(
            (row0.view(-1, 1, 3), row1.view(-1, 1, 3), row2.view(-1, 1, 3)), 1)  # batch*3*3

        return matrix

    def get_mask(self, lengths, dtype=torch.bool):
        maxlen = self.input_length
        lengths = torch.from_numpy(lengths).type(torch.float)
        mask = ~(torch.ones((len(lengths), maxlen)
                            ).cumsum(dim=1).t() > lengths).t()
        mask.type(dtype)
        return mask

    def write_bvhs(
            self,
            motions,
            names,
            frame_times,
            outpath,
            use_parallel=False,
            **kwargs):
        if isinstance(frame_times, float):
            frame_times = [frame_times for _ in range(len(motions))]

        positions, motions = self.shaping(motions)

        if use_parallel:
            Parallel(
                n_jobs=8)(
                delayed(
                    self._write_bvh)(
                    position,
                    motion,
                    name,
                    frame_time,
                    outpath) for position,
                motion,
                name,
                frame_time in zip(
                    positions,
                    motions,
                    names,
                    frame_times))
        else:
            return [
                self._write_bvh(
                    position,
                    motion,
                    name,
                    frame_time,
                    outpath,
                    **kwargs) for position,
                motion,
                name,
                frame_time in zip(
                    positions,
                    motions,
                    names,
                    frame_times)]

    def shaping(self, inputs):
        positions = []
        motions = []
        for motion in inputs:
            if self.omit_root_pos:
                position = np.zeros([len(motion), 3, 1])
            else:
                position = motion[:, :3]
                motion = motion[:, 3:]

            length, dim = np.shape(motion)

            dim_feature = self.dim_dict[self.rep]
            motion = np.reshape(
                motion, [length, int(dim / dim_feature), dim_feature])

            positions.append(position)
            motions.append(motion)

        return positions, motions

    def repTransfer_reverse(self, rot_joint):
        if self.rep == 'quat':
            return [R.from_quat(normalize_vec(quat)) for quat in rot_joint]
        elif self.rep == 'expmap':
            return [R.from_rotvec(rotvec) for rotvec in rot_joint]
        elif self.rep == 'ortho6d':
            return [ortho6dToR(ortho6d) for ortho6d in rot_joint]
        else:
            print("unknown 3D rotation representation")
            sys.exit(1)

    def _write_bvh(
            self,
            position,
            motion,
            name,
            frame_time,
            outpath,
            **kwargs):
        bvhpath = os.path.join(
            self.data_path,
            'bvh/hdm05/HDM_bd_cartwheelLHandStart1Reps_001_120.bvh')
        mocap = loadBvh(bvhpath)

        rot_dict = {}
        for i, joint_name in enumerate(self.used_joints):
            rot_dict[joint_name] = self.repTransfer_reverse(motion[:, i])

        rot_dict = self._add_bvh_joint(mocap, rot_dict)

        if self.abs_angle:
            rot_dict = self._without_offset(mocap, rot_dict)
            rot_dict = self._abs_to_rel(mocap, rot_dict)

        frames = self._rot_dict_to_frames(mocap, rot_dict)
        frameNum = len(motion)

        frames = np.concatenate(frames, axis=1)
        frames = np.reshape(frames, [frameNum, -1])
        frames = np.concatenate([np.squeeze(position), frames], axis=1)

        return writeBvh(
            bvhpath,
            os.path.join(
                outpath,
                name + '.bvh'),
            frames,
            frameTime=frame_time,
            frameNum=frameNum,
            **kwargs)

    def _add_bvh_joint(self, mocap, rot_dict):
        for bvhjoint in mocap.get_joints_names():
            if bvhjoint in self.used_joints:
                pass
            else:
                parent_joint = mocap.joint_parent(bvhjoint).name
                if not self.abs_angle:
                    rot_dict[bvhjoint] = [R.identity()
                                          for rp in rot_dict[parent_joint]]
                else:
                    rot_dict[bvhjoint] = [rp for rp in rot_dict[parent_joint]]

        return rot_dict

    def _rot_dict_to_frames(self, mocap, rot_dict):
        frames = []
        for bvhjoint in mocap.get_joints_names():
            euler_angle = [rotationToBvhEuler(r) for r in rot_dict[bvhjoint]]
            frames.append(euler_angle)
        return frames

    def _without_offset(self, mocap, rot_dict):
        offset_dict = self._get_offset(mocap)
        for bvhjoint in mocap.get_joints_names():
            if bvhjoint in self.used_joints:
                rot_dict[bvhjoint] = [r * offset_dict[bvhjoint].inv()
                                      for r in rot_dict[bvhjoint]]

        return rot_dict

    def _abs_to_rel(self, mocap, rot_dict):
        rot_dict_rel = {}
        for bvhjoint in mocap.get_joints_names():
            parent_id = mocap.joint_parent_index(bvhjoint)
            if parent_id == -1:
                parent_joint = None
                rot_dict_rel[bvhjoint] = [r for r in rot_dict[bvhjoint]]
            elif parent_id == 0:
                parent_joint = mocap.joint_parent(bvhjoint).name
                rot_dict_rel[bvhjoint] = [
                    rp.inv() * r for rp,
                    r in zip(
                        rot_dict[parent_joint],
                        rot_dict[bvhjoint])]
            elif bvhjoint in self.used_joints:
                parent_joint = mocap.joint_parent(bvhjoint).name
                rot_dict_rel[bvhjoint] = [
                    rp.inv() * r for rp,
                    r in zip(
                        rot_dict[parent_joint],
                        rot_dict[bvhjoint])]
            else:
                parent_joint = mocap.joint_parent(bvhjoint).name
                rot_dict_rel[bvhjoint] = [rp.inv()
                                          for rp in rot_dict[parent_joint]]

        return rot_dict_rel

    def _get_offset(self, mocap):
        offset_dict = {}
        for bvhjoint in mocap.get_joints_names():
            parent_id = mocap.joint_parent_index(bvhjoint)
            if parent_id == -1:
                parent_joint = None
                offset_dict[bvhjoint] = R.identity()
            else:
                parent_joint = mocap.joint_parent(bvhjoint).name

                if bvhjoint in [
                    "LeftArm",
                    "RightArm",
                    "LeftUpLeg",
                        "RightUpLeg"]:
                    r_offset = offsetEulerAngle(
                        mocap, bvhjoint, self.settings, typeR=True)
                else:
                    r_offset = R.identity()

                if bvhjoint in ["LeftHand", "RightHand"]:  # fixed joint
                    offset_dict[bvhjoint] = r_offset
                elif bvhjoint in ["LeftFoot", "RightFoot"]:
                    # offset_dict[bvhjoint] = offset_dict[parent_joint] * R.from_euler('ZYX', [0, -90, 0])
                    offset_dict[bvhjoint] = offset_dict[parent_joint] * \
                        r_offset * R.from_euler('ZYX', [0, -90, 0])
                    # print(bvhjoint, offset_dict[bvhjoint].as_euler("ZYX", degrees=True))
                else:
                    offset_dict[bvhjoint] = offset_dict[parent_joint] * r_offset

        return offset_dict

    def sample_motions(
            self,
            idx=None,
            batch_size=None,
            return_name=False,
            test=False):
        if not test:
            motions = self.motions
            names = self.names
            frame_times = self.frame_times
        else:
            motions = self.test_motions
            names = self.test_names
            frame_times = self.test_frame_times

        if idx is not None:
            pass
        elif batch_size is not None:
            idx = np.random.randint(0, len(motions), size=batch_size)
        else:
            idx = range(len(motions))

        motions = np.asarray([self._sample_motion(motion,
                                                  self.input_length,
                                                  frame_time)
                              for motion, frame_time
                              in zip(motions[idx], frame_times[idx])])

        if self.device.type != 'cpu':
            motions = self._to_torch(motions)
        else:
            motions = torch.Tensor(motions)

        ret = [motions, self._calc_lengths(idx, test=test)]

        if return_name:
            return ret, names[idx]

        return ret

    def _sample_motion(self, motion, input_length, frame_time):
        fps_bvh = int(1 / frame_time)
        sampling_rate = int(fps_bvh / self.fps)

        motion = self._frame_sample(motion, sampling_rate)

        len_diff = len(motion) - input_length
        if len_diff < 0:
            len_diff = abs(len_diff)
            motion = np.pad(motion, pad_width=[
                            [0, len_diff], [0, 0]], mode='edge')
            return motion
        elif len_diff == 0:
            return motion
        else:
            idx = np.random.randint(0, len_diff)
            return motion[idx:idx + input_length]

    def _calc_lengths(self, idx: list = None, test=False):
        if not test:
            motions = self.motions
            frame_times = self.frame_times
        else:
            motions = self.test_motions
            frame_times = self.test_frame_times

        lengths = np.asarray([self._calc_length(len(motion),
                                                frame_time) for motion,
                              frame_time in zip(motions[idx],
                                                frame_times[idx])],
                             dtype=np.int32)
        return lengths

    def _calc_length(self, length, frame_time):
        fps_bvh = int(1 / frame_time)
        sampling_rate = int(fps_bvh / self.fps)
        length_motion = int(length / sampling_rate)
        if self.input_length > 0 and length_motion > self.input_length:
            return self.input_length
        else:
            return length_motion

    def _frame_sample(self, motion, sampling_rate):
        idx = np.random.randint(0, sampling_rate)
        motion = motion[idx:]
        return motion[::sampling_rate]

    def _to_numpy(self, x):
        return x.to('cpu').detach().numpy().copy()

    def _to_torch(self, x):
        return torch.from_numpy(x.astype(np.float32)).clone().cuda()


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


def quat2bvheuler(quat, mocap, joint_name):
    order, _ = getRotationOrderAndChannels(mocap, joint_name)
    return R.from_quat(quat).as_euler(order, degrees=True)


def rotationToBvhEuler(rotation):
    euler = rotation.as_euler('ZYX', degrees=True)  # extrinsic euler
    return [euler[0], euler[1], euler[2]]


def normalize_vec(vec):
    mag = np.linalg.norm(vec)
    return vec / mag


def normalize_vector_numpy(v, return_mag=False):
    v_mag = np.linalg.norm(v, axis=1)
    v_mag = np.maximum(v_mag, [1e-8] * len(v_mag))
    v = v / v_mag
    if return_mag:
        return v, v_mag[:, 0]
    else:
        return v


def normalize_vector(v, return_mag=False, device=None):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    v_mag = torch.max(v_mag, torch.autograd.Variable(
        torch.FloatTensor([1e-8]).to(device)))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v / v_mag
    if(return_mag is True):
        return v, v_mag[:, 0]
    else:
        return v


# u, v batch*n
def cross_product(u, v):
    batch = u.shape[0]
    # print (u.shape)
    # print (v.shape)
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    out = torch.cat((i.view(batch, 1), j.view(batch, 1),
                     k.view(batch, 1)), 1)  # batch*3

    return out
