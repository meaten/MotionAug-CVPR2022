import os
import argparse
from numpy.lib.function_base import append
from numpy.lib.npyio import load
from tqdm import tqdm
from abc import ABCMeta, abstractmethod
from bvh import Bvh

import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.sampler import BatchSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_optimizer import RAdam
from torch.optim import SGD

from default_param import _C as cfg
from tools import *


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--debiaser_type",
        default="Linear",
        choices=["Linear", "NN"])
    parser.add_argument(
        "--config_file",
        default="")
    parser.add_argument(
        "--orig_data_npz",
        type=str,
        default="../dataset/dataset_split.npz")
    parser.add_argument(
        "--phys_data_npz",
        type=str,
        default="../dataset/dataset_Fixed_phys_kick.npz")
    parser.add_argument(
        "--aug_data_npz",
        type=str,
        default="../dataset/dataset_IK_phys_kick.npz")
    parser.add_argument(
        "--joint_subset",
        type=str,
        choices=["all", "ankle"])
    parser.add_argument(
        "--rep_in",
        type=str,
        default='expmap')
    parser.add_argument(
        "--rep_out",
        type=str,
        default='expmap')
    parser.add_argument(
        "--show_diff",
        action="store_true")
    parser.add_argument(
        "--act_class",
        default="kick")
    parser.add_argument(
        "--debug",
        action="store_true")
    return parser.parse_args()


def load_npz(path):
    with np.load(path, allow_pickle=True) as dataset:
        sort = np.argsort(dataset["names"])
        motions = dataset["motions"][sort]
        names = dataset["names"][sort]
        return motions, names


def search_by_str(query, key):
    idx = []
    for q in query:
        idx.append(np.char.find(key, q) != -1)

    return np.array(idx)


def offset_motion(orig_motion, phys_motion, aug_motion, **kwargs):
    len_orig = len(orig_motion)
    len_phys = len(phys_motion)
    len_aug = len(aug_motion)
    len_min = min(len_orig, len_phys)

    diff = []
    for i, aug_pose in enumerate(aug_motion):
        idx = int(i / len_aug * len_min)
        aug_motion[i], diff_pose = offset_pose(
            orig_motion[idx], phys_motion[idx], aug_pose, **kwargs)
        diff.append(diff_pose)

    return aug_motion, np.mean(diff, axis=0)


def offset_pose(
        orig_pose,
        phys_pose,
        aug_pose,
        joint_subset="all",
        rep_in='euler',
        rep_out='euler',
        ret_diff=False):
    aug_pose[:3] += orig_pose[:3] - phys_pose[:3]
    rot_orig = pose2rot(orig_pose[3:], rep=rep_in)
    rot_phys = pose2rot(phys_pose[3:], rep=rep_in)
    rot_aug = pose2rot(aug_pose[3:], rep=rep_in)
    rot_diff = rot_orig * rot_phys.inv()

    if joint_subset == "ankle":
        joint_id = [9, 15]
        rot_diff = rot_mask(rot_diff, joint_id)

    rot_aug = rot_diff * rot_aug
    aug_pose[3:] = rot2pose(rot_aug, rep=rep_out)

    if ret_diff:
        return aug_pose, rot2pose(rot_diff, rep=rep_out)
    else:
        return aug_pose, np.full(np.size(aug_pose), np.inf)


def rot_mask(rot, ids):
    expmap = rot.as_rotvec()
    expmap_mask = [v if i in ids else [0, 0, 0] for i, v in enumerate(expmap)]
    return R.from_rotvec(expmap_mask)


def pose2rot(pose, rep='euler'):
    pose = np.reshape(pose, [-1, 3])
    if rep == 'euler':
        rot = R.from_euler('ZYX', pose, degrees=True)
    elif rep == 'expmap':
        rot = R.from_rotvec(pose)
    else:
        raise ValueError

    return rot


def rot2pose(rot, rep='euler'):
    if rep == 'euler':
        pose = rot.as_euler('ZYX', degrees=True)
    elif rep == 'expmap':
        pose = rot.as_rotvec()
    else:
        raise ValueError

    return pose.flatten()


def select_debiaser(debiaser_type):
    if debiaser_type == "Linear":
        return Linear_Debiaser
    if debiaser_type == "NN":
        return NN_Debiaser


class Debiaser(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, orig_data_npz, phys_data_npz, **kwargs):
        pass

    @abstractmethod
    def type(self):
        return ""

    @abstractmethod
    def process(self, aug_data_npz, **kwargs):
        pass


class Linear_Debiaser(Debiaser):
    def __init__(self, orig_data_npz, phys_data_npz, **kwargs):
        self.orig_motions, self.orig_names = load_npz(orig_data_npz)
        self.phys_motions, self.phys_names = load_npz(phys_data_npz)

        idx_matched_motions = search_by_str(self.orig_names, self.phys_names)
        self.orig_motions = self.orig_motions[np.any(
            idx_matched_motions, axis=1)]
        self.orig_names = self.orig_names[np.any(idx_matched_motions, axis=1)]

        self.kwargs = kwargs

    def type(self):
        return "Linear"

    def process(self, aug_data_npz):
        aug_motions, aug_names = load_npz(aug_data_npz)

        idx_matched_motions = search_by_str(self.orig_names, aug_names)
        diffs = []
        for idxs, orig_motion, phys_motion in tqdm(zip(
                idx_matched_motions, self.orig_motions, self.phys_motions)):
            for i, aug_motion in enumerate(aug_motions[idxs]):
                # assert orig_names[j] in aug_names[idxs][i]
                aug_motions[idxs][i], diff = offset_motion(
                    orig_motion, phys_motion, aug_motion, **self.kwargs)
                diffs.append(diff)

        self.save_npz(aug_data_npz, aug_motions)

        return diffs

    def save_npz(self, aug_data_npz, aug_motions):
        with np.load(aug_data_npz, allow_pickle=True) as dataset:
            np.savez(self.get_outpath(aug_data_npz),
                     motions=aug_motions,
                     names=dataset["names"],
                     subjects=dataset["subjects"],
                     classes=dataset["classes"],
                     max_len=dataset["max_len"],
                     min_len=dataset["min_len"],
                     dim_pose=dataset["dim_pose"],
                     frame_times=dataset["frame_times"],
                     omit_root_pos=dataset["omit_root_pos"],
                     rep=dataset["rep"],
                     abs_angle=dataset["abs_angle"],
                     allow_pickle=True)

    def get_outpath(self, aug_data_npz):
        split_name = os.path.splitext(aug_data_npz)
        save_path = f"{split_name[0]}_offset_{self.type()}{split_name[1]}"
        return save_path


class NN_Debiaser(Linear_Debiaser):
    def __init__(self, orig_data_npz, phys_data_npz, **kwargs):
        super().__init__(orig_data_npz, phys_data_npz, **kwargs)

        self.cfg = cfg

        load_config(self.cfg, kwargs['config_file'], kwargs['act_class'], "phys" if "phys" in phys_data_npz else "kin")
        angle_dim = 3 if self.cfg.INPUT.REP == "expmap" else 3
        input_size = (3 + self.cfg.INPUT.NUM_JOINT * angle_dim) * len(self.cfg.INPUT.TYPE)
        output_size = 3 + 6 * self.cfg.INPUT.NUM_JOINT
        self.model = ModelWithLoss(self.cfg, input_size, output_size).cuda()
        path = os.path.join(self.cfg.OUTPUT_DIR, "model.pth")
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
        else:
            self.train()

    def train(self):
        dataloader = self.build_dataloader()
        if cfg.SOLVER.OPTIMIZER == "radam":
            optimizer = RAdam(
                filter(
                    lambda p: p.requires_grad,
                    self.model.parameters()),
                lr=cfg.SOLVER.LR,
                weight_decay=cfg.MODEL.REG_WEIGHT)
        elif cfg.SOLVER.OPTIMIZER == "sgd":
            optimizer = SGD(
                filter(
                    lambda p: p.requires_grad,
                    self.model.parameters()),
                lr=cfg.SOLVER.LR,
                momentum=0.9)
        scheduler = ReduceLROnPlateau(optimizer,
                                      factor=cfg.SOLVER.DECAY_FACTOR,
                                      patience=10)

        with tqdm(range(cfg.SOLVER.ITER), ncols=100) as pbar:
            for i in pbar:
                self.model.train()
                logging_loss = 0
                for data_dict in dataloader:
                    optimizer.zero_grad()
                    loss = self.model(data_dict)
                    loss = loss.mean()

                    loss.backward()

                    optimizer.step()
                    logging_loss += loss.item()
                scheduler.step(logging_loss)
                logging_loss /= len(dataloader)

                np.set_printoptions(
                    precision=4, floatmode='maxprec', suppress=True)

                pbar.set_postfix(
                    dict(
                        loss=np.asarray(
                            [logging_loss]),
                        lr=[optimizer.param_groups[0]['lr']]
                    ))

                if optimizer.param_groups[0]['lr'] < cfg.SOLVER.LR * 1e-3:
                    print("early stopping")
                    break

        torch.save(
            self.model.state_dict(),
            os.path.join(
                self.cfg.OUTPUT_DIR,
                "model.pth"))

    def build_dataset(self):
        x, y = [], []
        for i in range(len(self.orig_motions)):
            orig_motion = self.orig_motions[i]
            phys_motion = self.phys_motions[i]
            length = min(len(orig_motion), len(phys_motion))
            phys_motion = phys_motion[:length]
            orig_motion = orig_motion[:length]

            phys_motion = self.preprocess(phys_motion)
            
            x.append(phys_motion)
            y.append(orig_motion)
        x = np.concatenate(x, axis=0)
        y = np.concatenate(y, axis=0)
        assert len(x) == len(y)
        return SimpleDataset(x=x, y=y)

    def preprocess(self, motion):
        if "vel" in self.cfg.INPUT.TYPE:
            vel = motion[1:, :] - motion[:-1, :]
            try:
                vel = np.pad(vel, [[0,1], [0,0]], mode='edge')                
                motion = np.concatenate([motion, vel], axis=1)
            except ValueError:
                motion = np.concatenate([motion, np.zeros_like(motion)], axis=1)
        return motion

    def build_dataloader(self):
        dataset = self.build_dataset()
        sampler = sampler = RandomSampler(dataset)
        batch_sampler = BatchSampler(sampler=sampler,
                                     batch_size=cfg.SOLVER.BATCH_SIZE,
                                     drop_last=False)
        loader = DataLoader(
            dataset,
            num_workers=cfg.SOLVER.NUM_WORKERS,
            batch_sampler=batch_sampler)
        return loader

    def type(self):
        return "NN"

    def process(self, aug_data_npz, **kwargs):
        self.model.eval()
        aug_motions, aug_names = load_npz(aug_data_npz)

        diffs = []
        with torch.no_grad():
            for i in tqdm(range(len(aug_motions))):
                aug_motions[i], diff = self.model.predict(
                    {'input': torch.Tensor(self.preprocess(aug_motions[i]))})
                diffs.append(diff)
        self.save_npz(aug_data_npz, aug_motions)

        return np.concatenate(diffs, axis=0)


class ModelWithLoss(nn.Module):
    def __init__(self, cfg, input_size, output_size):
        super(ModelWithLoss, self).__init__()

        if cfg.MODEL.TYPE == "mlp":
            self.model = MLP(
                cfg,
                input_size=input_size,
                output_size=output_size)

        self.build_loss(cfg)

    def forward(self, data_dict):
        ret = self.pred(data_dict)
        loss = self.loss(ret)

        return loss

    def pred(self, data_dict):
        x = data_dict["input"].float().cuda()
        pred = self.model(x)

        pred_pos = x[:, :3] + pred[:, :3]
        res = compute_rotation_matrix_from_Rodriguez(x[:, 3:3+16*3].reshape(-1, 3))
        pred_diff = compute_rotation_matrix_from_ortho6d(
            pred[:, 3:].reshape(-1, 6))
        pred_rot = torch.matmul(res, pred_diff)
        data_dict["pred_pos"] = pred_pos
        data_dict["pred_rot"] = pred_rot
        data_dict["pred_diff"] = pred_diff

        return data_dict

    def predict(self, data_dict):
        shape = data_dict["input"].shape
        ret_dict = self.pred(data_dict)
        pred_pos = ret_dict['pred_pos'].cpu().numpy()
        pred_rot = R.from_matrix(ret_dict['pred_rot'].cpu(
        ).numpy()).as_rotvec().reshape([shape[0], -1])
        pred_diff = R.from_matrix(ret_dict['pred_diff'].cpu(
        ).numpy()).as_rotvec().reshape([shape[0], -1])
        return np.concatenate([pred_pos, pred_rot], axis=1), pred_diff

    def loss(self, data_dict):
        loss = 0
        gt_rot = compute_rotation_matrix_from_Rodriguez(
            data_dict["gt"][:, 3:].float().cuda().reshape(-1, 3))
        loss += self.pred_loss(data_dict["pred_rot"].squeeze(),
                               gt_rot)
        loss += self.pred_loss(data_dict["pred_pos"].squeeze(),
                               data_dict["gt"][:, :3].float().cuda())

        for name, param in self.model.named_parameters():
            if 'weight' in name:
                loss += torch.norm(param, self.reg_norm) * self.reg_weight

        return loss

    def build_loss(self, cfg):
        self.pred_loss = choose_loss(cfg.MODEL.PRED_LOSS)
        self.reg_norm = 1 if cfg.MODEL.REG_TYPE == "l1" else 2
        self.reg_weight = cfg.MODEL.REG_WEIGHT


class MLP(nn.Module):
    def __init__(self, cfg, input_size, output_size):
        super(MLP, self).__init__()

        dim_h = cfg.MODEL.DIM_HIDDEN
        num_l = cfg.MODEL.NUM_HIDDEN
        dim_o = output_size
        coef = cfg.MODEL.DIM_HIDDEN_FACT

        self.h_size = [input_size]
        for _ in range(num_l):
            self.h_size.append(dim_h)
            dim_h = int(dim_h * coef)
        self.h_size.append(dim_o)

        self.layers = nn.ModuleList([nn.Linear(
            self.h_size[k], self.h_size[k + 1]) for k in range(len(self.h_size) - 1)])
        self.norm = nn.BatchNorm1d(self.h_size[0])
        self.drop = nn.Dropout(p=cfg.MODEL.P_DROPOUT)
        self.act = nn.LeakyReLU(0.01)

    def forward(self, x):
        x = self.drop(x)

        for i in range(0, len(self.layers) - 1):
            x = self.act(self.layers[i](x))

        x = self.layers[-1](x)
        return x


class SimpleDataset(Dataset):
    def __init__(self, x, y=[]):

        self.x = x
        self.y = y
        self.test = False
        if len(y) == 0:
            self.test = True

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx] if not self.test else -1.0
        return {"input": x, "gt": y}


def choose_loss(loss_name):
    if loss_name == "l1":
        return nn.L1Loss()
    elif loss_name == "mse":
        return nn.MSELoss()
    elif loss_name == "bce":
        return nn.BCELoss()
    else:
        raise ValueError(f"unknown loss function {loss_name}")


def load_config(cfg, conf_path, act_class, phys_or_kin):
    if os.path.isfile(conf_path):
        print(f"Configuration file loaded from {conf_path}.")
        cfg.merge_from_file(conf_path)
        cfg.OUTPUT_DIR = os.path.join(
            cfg.OUTPUT_DIR, os.path.splitext(
                os.path.basename(conf_path))[0], act_class, phys_or_kin)
    else:
        print("Use default configuration.")
        cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, "default", act_class, phys_or_kin)

    cfg.freeze()

    print(f"output dirname: {cfg.OUTPUT_DIR}")
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    if os.path.isfile(conf_path):
        import shutil
        shutil.copy2(conf_path, os.path.join(cfg.OUTPUT_DIR, 'config.yaml'))


def visualize_diff(diffs, type):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    diffs = np.array(diffs)
    shape = diffs.shape
    diffs = np.mean(np.reshape(diffs, [-1, int(shape[1] / 3), 3]), axis=2)
    bp = ax.boxplot(diffs)

    plt.title('offset diff on each joint')
    plt.grid()
    plt.savefig(f"diff_{type}.png")


def main():
    args = parse_args()

    kwargs = {'joint_subset': args.joint_subset,
              'rep_in': args.rep_in,
              'rep_out': args.rep_out,
              'ret_diff': args.show_diff}

    if args.debiaser_type != "Linear":
        kwargs['config_file'] = args.config_file
        kwargs['act_class'] = args.act_class

    debiaser = select_debiaser(
        args.debiaser_type)(
        args.orig_data_npz,
        args.phys_data_npz,
        **kwargs)
    diffs = debiaser.process(args.aug_data_npz)
    if args.show_diff:
        visualize_diff(diffs, args.debiaser_type)

    if args.debug and args.rep_out == "expmap":
        data_path = '../../data/bvh/hdm05/'
        bvhpath = os.path.join(
            data_path, 'HDM_bd_cartwheelLHandStart1Reps_001_120.bvh')

        outpath = "debug"
        if os.path.exists(outpath):
            import shutil
            shutil.rmtree(outpath)
        os.makedirs(outpath, exist_ok=True)

        dataset = np.load(debiaser.get_outpath(args.aug_data_npz), allow_pickle=True)
        #dataset = np.load('../dataset/dataset_Fixed_phys_kick.npz', allow_pickle=True)
        motions = dataset['motions']
        names = dataset['names']
        frame_times = dataset['frame_times']

        from joblib import Parallel, delayed
        Parallel(n_jobs=12)(delayed(saveBvh)(motion, name, frame_time, outpath) for motion, name, frame_time in tqdm(zip(motions, names, frame_times)))
            

def loadBvh(bvhpath):
    with open(bvhpath) as f:
        return Bvh(f.read())


def repTransfer_reverse(rep, rot_joint):
    if rep == 'quat':
        return [R.from_quat(quat) for quat in rot_joint]
    elif rep == 'expmap':
        return [R.from_rotvec(rotvec) for rotvec in rot_joint]
    elif rep == 'ortho6d':
        return [ortho6dToR(ortho6d) for ortho6d in rot_joint]
    elif rep == 'euler':
        rot_joint = [r.from_euler('ZYX', degrees=True) for r in rot_joint]
    else:
        print("unknown 3D rotation representation")


def ortho6dToR(ortho6d):
    assert len(ortho6d) == 6

    x_raw = ortho6d[0:3]
    y_raw = ortho6d[3:6]

    x = x_raw / np.linalg.norm(x_raw)
    z = np.cross(x, y_raw)
    z = z / np.linalg.norm(z)
    y = np.cross(z, x)

    return R.from_matrix([x, y, z])


def rot_dict_to_frames(mocap, rot_dict):
    frames = []
    for bvhjoint in mocap.get_joints_names():
        euler_angle = [rotationToBvhEuler(r) for r in rot_dict[bvhjoint]]
        frames.append(euler_angle)
    return frames


def rotationToBvhEuler(rotation):
    euler = rotation.as_euler('ZYX', degrees=True)  # extrinsic euler
    return [euler[0], euler[1], euler[2]]


def add_bvh_joint(used_joints, mocap, rot_dict):

    for bvhjoint in mocap.get_joints_names():
        if bvhjoint in used_joints:
            pass
        else:
            parent_joint = mocap.joint_parent(bvhjoint).name
            rot_dict[bvhjoint] = [rp for rp in rot_dict[parent_joint]]

    return rot_dict


def saveBvh(motion, name, frame_time, outpath):
    data_path = '../../data/bvh/hdm05/'
    bvhpath = os.path.join(
        data_path, 'HDM_bd_cartwheelLHandStart1Reps_001_120.bvh')
    
    position = motion[:, :3]
    motion = np.reshape(motion[:, 3:], [-1, 16, 3])

    mocap = loadBvh(bvhpath)

    used_joints = ["Hips",
                    "Spine", "Spine1", "Neck",
                    "RightUpLeg", "RightLeg", "RightFoot",
                    "RightArm", "RightForeArm", "RightHand",
                    "LeftUpLeg", "LeftLeg", "LeftFoot",
                    "LeftArm", "LeftForeArm", "LeftHand"]

    rot_dict = {}
    for i, joint_name in enumerate(used_joints):
        rot_dict[joint_name] = repTransfer_reverse("expmap", motion[:, i])

    rot_dict = add_bvh_joint(used_joints, mocap, rot_dict)
    frames = rot_dict_to_frames(mocap, rot_dict)
    frameNum = len(motion)
    frames = np.concatenate(frames, axis=1)
    frames = np.reshape(frames, [frameNum, -1])
    frames = np.concatenate([np.squeeze(position), frames], axis=1)
    writeBvh(
        bvhpath,
        os.path.join(
            outpath,
            name + '_offset.bvh'),
        frames,
        frameTime=frame_time,
        frameNum=frameNum)


def writeBvh(bvhpath, outpath, frames, frameTime=None, frameNum=None):
    print("saving bvh file to {}".format(outpath))
    with open(outpath, 'w') as fw:
        with open(bvhpath, 'r') as fr:
            while True:
                line = fr.readline()

                if 'Frame Time' in line:
                    if frameTime is None:
                        pass
                    else:
                        line = "Frame Time: {}\n".format(frameTime)
                    fw.write(line)
                    break
                elif 'Frames' in line:
                    if frameNum is None:
                        frameNum = len(frames)
                    else:
                        pass
                    line = "Frames: {}\n".format(frameNum)
                    fw.write(line)
                else:
                    fw.write(line)
            for vector in frames:
                line_aligned = ' '.join(vector.astype(str)) + '\n'
                fw.write(line_aligned)


if __name__ == '__main__':
    main()
