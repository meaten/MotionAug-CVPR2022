import os
import sys
import argparse
import numpy as np
import json
import glob
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

sys.path.append(".")
from util import bvh
from data.bvh.util_bvh import loadBvh, getRotationOrderAndChannels
from data.bvh.util_bvh import writeBvh

from joblib import Parallel, delayed


def parse_args():
    parser = argparse.ArgumentParser(description='enbed bvh motions to npz format.')

    parser.add_argument('--dirpath', help='path to bvh dir',
                        default='./data/bvh/hdm05_aligned_split/')

    parser.add_argument('--outpath', help='path to output .npz file',
                        default='./data/bvh/dataset_aligned_split.npz')

    parser.add_argument('--rep', help='3D rotation representation',
                        choices=['euler', 'quat', 'expmap', 'ortho6d'], default='quat')
    parser.add_argument('--abs_angle', help='represent each joint angel as abs angle from ROOT',
                        action='store_true')
    parser.add_argument('--omit_root_pos', help='add 3d position info',
                        action='store_true')
    parser.add_argument('--omit_root_rot', help='add root_rotation position info',
                        action='store_true')
    
    parser.add_argument('--setting_file_path', help='path to setting.json',
                        default='./data/bvh/settings.json')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--parallel', default=12, type=int)
    args = parser.parse_args()

    return args


class BvhToNp(object):
    def __init__(self, args):
        self.dirpath = args.dirpath
        self.outpath = args.outpath
        self.rep = args.rep
        self.omit_root_pos = args.omit_root_pos
        self.omit_root_rot = args.omit_root_rot
        self.abs_angle = args.abs_angle
        self.debug = args.debug
        
        self.settings = self.loadSetting(args.setting_file_path)
        self.used_joints = [joint[0] for joint in self.settings['jointAssignments']]
        
    def getMotion(self, bvhpath):
        mocap = loadBvh(bvhpath)
        
        frame_time = mocap.frame_time
    
        if self.abs_angle:
            motion = self.getAbsPoseWithOffset(mocap)
        else:
            try:
                motion = self.getRelPose(mocap)
            except IndexError:
                print(bvhpath)
                return [], '', 0.0
        motion = np.asarray(motion)
        
        motion = np.transpose(np.array(motion), [1, 0, 2])  # frame, joint, dim
        motion = np.reshape(motion, [len(motion), -1])
        
        if self.omit_root_pos:
            pass
        else:
            position = np.asarray(self.getPosition(mocap))
            motion = np.concatenate([position, motion], axis=1)
         
        name = os.path.splitext(os.path.basename(bvhpath))[0]
        
        return motion, name, frame_time
    
    def getPosition(self, mocap):
        
        root = mocap.get_joints_names()[0]
        channels = mocap.joint_channels(root)[:3]
        
        return mocap.frames_joint_channels(root, channels)
    
    def getRelPose(self, mocap):
        rot_dict = {}
        for joint_name in mocap.get_joints_names():
            order, channels = getRotationOrderAndChannels(mocap, joint_name)
            
            if joint_name == self.used_joints[0] and self.omit_root_rot:
                rot_dict[joint_name] = [R.from_euler(order, [0, 0, 0], degrees=True)
                                        for angle in mocap.frames_joint_channels(joint_name, channels)]
            elif joint_name == self.used_joints[0]:
                rot_dict[joint_name] = [R.from_euler(order, angle, degrees=True)
                                        for angle in mocap.frames_joint_channels(joint_name, channels)]
            else:
                rot_dict[joint_name] = [R.from_euler(order, angle, degrees=True)
                                        for angle in mocap.frames_joint_channels(joint_name, channels)]
                
        frames = [self.postProcess(self.repTransfer(rot_dict[joint_name])) for joint_name in self.used_joints]
        
        return frames

    def getAbsPoseWithOffset(self, mocap):
        # transfer bvh pose to internal state pose representation
        
        rot_dict = {}
        for joint_name in mocap.get_joints_names():
            order, channels = getRotationOrderAndChannels(mocap, joint_name)
            parent = ''
            if joint_name == self.used_joints[0] and self.omit_root_rot:
                rot_dict[joint_name] = [R.from_euler(order, [0, 0, 0], degrees=True)
                                        for angle in mocap.frames_joint_channels(joint_name, channels)]
            elif joint_name == self.used_joints[0]:
                rot_dict[joint_name] = [R.from_euler(order, angle, degrees=True)
                                        for angle in mocap.frames_joint_channels(joint_name, channels)]
            else:
                parent = mocap.joint_parent(joint_name).name
                rot = [R.from_euler(order, angle, degrees=True)
                       for angle in mocap.frames_joint_channels(joint_name, channels)]
                rot_dict[joint_name] = [rp * r for rp, r in zip(rot_dict[parent], rot)]
        
        # import pdb;pdb.set_trace()
        
        offset_dict = self.getOffset(mocap)
        for joint_name in mocap.get_joints_names():
            if joint_name in self.used_joints:
                rot_dict[joint_name] = [r * offset_dict[joint_name] for r in rot_dict[joint_name]]
                    
        frames = [self.postProcess(self.repTransfer(rot_dict[joint_name])) for joint_name in self.used_joints]
        
        return frames

    def repTransfer(self, rot_joint):
        if self.rep == 'quat':
            rot_joint = [r.as_quat() for r in rot_joint]  # xyzw
            rot_joint = np.array(rot_joint)
            # rot_joint = rot_joint[:,[3,0,1,2]]  #wxyz
        elif self.rep == 'expmap':
            rot_joint = [r.as_rotvec() for r in rot_joint]
        elif self.rep == 'ortho6d':
            rot_joint = [r.as_matrix().flatten()[0:6] for r in rot_joint]
        elif self.rep == 'euler':
            rot_joint = [r.as_euler('ZYX', degrees=True) for r in rot_joint]
        else:
            print("unknown 3D rotation representation")
            sys.exit(1)

        return rot_joint
    
    def repTransfer_reverse(self, rot_joint):
        if self.rep == 'quat':
            return [R.from_quat(quat) for quat in rot_joint]
        elif self.rep == 'expmap':
            return [R.from_rotvec(rotvec) for rotvec in rot_joint]
        elif self.rep == 'ortho6d':
            return [self.ortho6dToR(ortho6d) for ortho6d in rot_joint]
        elif self.rep == 'euler':
            rot_joint = [r.from_euler('ZYX', degrees=True) for r in rot_joint]
        else:
            print("unknown 3D rotation representation")
            sys.exit(1)
            
    def ortho6dToR(self, ortho6d):
        assert len(ortho6d) == 6
        
        x_raw = ortho6d[0:3]
        y_raw = ortho6d[3:6]
        
        x = x_raw / np.linalg.norm(x_raw)
        z = np.cross(x, y_raw)
        z = z / np.linalg.norm(z)
        y = np.cross(z, x)
        
        return R.from_matrix([x, y, z])
            
    def postProcess(self, angles_joint):
        angles_joint_smoothed = []
        if self.rep == 'quat':
            angle_prev = angles_joint[0]
            for angle in angles_joint:
                angle_equal = self.equal_quat(angle)
                sim = np.abs(np.inner(angle, angle_prev))
                sim_equal = np.abs(np.inner(angle_equal, angle_prev))
                if sim > sim_equal:
                    angles_joint_smoothed.append(angle)
                else:
                    angles_joint_smoothed.append(angle_equal)
                    
                angle_prev = angle
                
        elif self.rep == 'expmap':
            angle_prev = angles_joint[0]
            for angle in angles_joint:
                if np.linalg.norm(angle) == 0:
                    angles_joint_smoothed.append(angle)
                    continue
                angle_equal = self.equal_expmap(angle)
                dis = np.linalg.norm(angle - angle_prev)
                dis_equal = np.linalg.norm(angle_equal - angle_prev)
                if dis < dis_equal:
                    angles_joint_smoothed.append(angle)
                else:
                    angles_joint_smoothed.append(angle_equal)
                    
                angle_prev = angle
        elif self.rep == 'ortho6d' or self.rep == 'euler':
            angles_joint_smoothed = angles_joint
        else:
            print("unknown 3D rotation representation")
            sys.exit(1)
            
        return angles_joint_smoothed
            
    def equal_quat(self, quat):
        return -quat

    def equal_expmap(self, expmap):
        theta = np.linalg.norm(expmap)
        vec = expmap / theta
        
        if theta > 2 * np.pi:
            sys.exit(1)
            
        return - vec * (2 * np.pi - theta)

    def euler2quat(self, euler, order):
        quat = R.from_euler(order, euler, degrees=True).as_quat()
        if quat[3] < 0:
            return -quat
        else:
            return quat

    def getOffset(self, mocap):
        offset_dict = {}
        for bvhjoint in mocap.get_joints_names():
            parent_id = mocap.joint_parent_index(bvhjoint)
            if parent_id == -1:
                parent_joint = None
                offset_dict[bvhjoint] = R.identity()
            else:
                parent_joint = mocap.joint_parent(bvhjoint).name
                
                if bvhjoint in ["LeftArm", "RightArm", "LeftUpLeg", "RightUpLeg"]:
                    r_offset = bvh.offsetEulerAngle(mocap, bvhjoint, self.settings, typeR=True)
                else:
                    r_offset = R.identity()

                if bvhjoint in ["LeftHand", "RightHand"]:  # fixed joint
                    offset_dict[bvhjoint] = r_offset
                elif bvhjoint in ["LeftFoot", "RightFoot"]:
                    # offset_dict[bvhjoint] = offset_dict[parent_joint] * R.from_euler('ZYX', [0, -90, 0])
                    offset_dict[bvhjoint] = offset_dict[parent_joint] * r_offset * R.from_euler('ZYX', [0, -90, 0])
                    # print(bvhjoint, offset_dict[bvhjoint].as_euler("ZYX", degrees=True))
                else:
                    offset_dict[bvhjoint] = offset_dict[parent_joint] * r_offset
                    
        return offset_dict

    def loadSetting(self, setting_file_path):
        path_settings = os.path.join(setting_file_path)
        with open(path_settings) as f:
            settings = json.loads(f.read())

        return settings

    def write_bvh(self, position, motion, name, frame_time, outpath):
        data_path = '../interaction/data/bvh/hdm05/'
        bvhpath = os.path.join(data_path, 'HDM_bd_cartwheelLHandStart1Reps_001_120.bvh')
        mocap = loadBvh(bvhpath)

        rot_dict = {}
        for i, joint_name in enumerate(self.used_joints):
            rot_dict[joint_name] = self.repTransfer_reverse(motion[:, i])
                
        rot_dict = self.add_bvh_joint(mocap, rot_dict)
        
        if self.abs_angle:
            rot_dict = self.without_offset(mocap, rot_dict)
            rot_dict = self.abs_to_rel(mocap, rot_dict)
                
        frames = self.rot_dict_to_frames(mocap, rot_dict)
        frameNum = len(motion)
        
        frames = np.concatenate(frames, axis=1)
        frames = np.reshape(frames, [frameNum, -1])
        frames = np.concatenate([np.squeeze(position), frames], axis=1)

        writeBvh(bvhpath, os.path.join(outpath, name + '.bvh'), frames, frameTime=frame_time, frameNum=frameNum)
        
    def add_bvh_joint(self, mocap, rot_dict):
        
        for bvhjoint in mocap.get_joints_names():
            if bvhjoint in self.used_joints:
                pass
            else:
                parent_joint = mocap.joint_parent(bvhjoint).name
                if not self.abs_angle:
                    rot_dict[bvhjoint] = [R.identity() for rp in rot_dict[parent_joint]]
                else:
                    rot_dict[bvhjoint] = [rp for rp in rot_dict[parent_joint]]
                    
        return rot_dict
    
    def rot_dict_to_frames(self, mocap, rot_dict):
        frames = []
        for bvhjoint in mocap.get_joints_names():
            euler_angle = [self.rotationToBvhEuler(r) for r in rot_dict[bvhjoint]]
            frames.append(euler_angle)
        return frames
    
    def without_offset(self, mocap, rot_dict):
        offset_dict = self.getOffset(mocap)
        for bvhjoint in mocap.get_joints_names():
            if bvhjoint in self.used_joints:
                rot_dict[bvhjoint] = [r * offset_dict[bvhjoint].inv() for r in rot_dict[bvhjoint]]
        
        return rot_dict
    
    def abs_to_rel(self, mocap, rot_dict):
        rot_dict_rel = {}
        for bvhjoint in mocap.get_joints_names():
            parent_id = mocap.joint_parent_index(bvhjoint)
            if parent_id == -1:
                parent_joint = None
                rot_dict_rel[bvhjoint] = [r for r in rot_dict[bvhjoint]]
            elif parent_id == 0:
                parent_joint = mocap.joint_parent(bvhjoint).name
                rot_dict_rel[bvhjoint] = [rp.inv() * r
                                          for rp, r in zip(rot_dict[parent_joint], rot_dict[bvhjoint])]
            elif bvhjoint in self.used_joints:
                parent_joint = mocap.joint_parent(bvhjoint).name
                rot_dict_rel[bvhjoint] = [rp.inv() * r
                                          for rp, r in zip(rot_dict[parent_joint], rot_dict[bvhjoint])]
            else:
                parent_joint = mocap.joint_parent(bvhjoint).name
                rot_dict_rel[bvhjoint] = [rp.inv()
                                          for rp in rot_dict[parent_joint]]
                
        return rot_dict_rel
    
    def rotationToBvhEuler(self, rotation):
        euler = rotation.as_euler('ZYX', degrees=True)  # extrinsic euler
        return [euler[0], euler[1], euler[2]]


def main():
    args = parse_args()
    
    bvhtonp = BvhToNp(args)

    outdir = os.path.dirname(args.outpath)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    names = []
    motions = []
    if args.dirpath is not None:
        import re
        print('enbedding all bvh files in {} to {}'.format(args.dirpath, args.outpath))
        bvhpaths = glob.glob(os.path.join(args.dirpath, '*.bvh'))
        
        dataset = Parallel(n_jobs=args.parallel)(delayed(bvhtonp.getMotion)(path) for path in tqdm(bvhpaths))
        # dataset = [bvhtonp.getMotion(path) for path in tqdm(bvhpaths)]
        motions = [data[0] for data in dataset]
        names = [data[1] for data in dataset]
        subjects = [re.findall('([a-z]+)', name)[0] for name in names]
        classes = [re.findall('([a-z]+)', name)[1] for name in names]
        frame_times = [data[2] for data in dataset]
        list_len = [len(motion) for motion in motions]
        min_len = min(list_len)
        max_len = max(list_len)
        dim_pose = np.shape(motions[0])[1]
        np.savez(args.outpath,
                 motions=motions,
                 names=names,
                 subjects=subjects,
                 classes=classes,
                 max_len=max_len,
                 min_len=min_len,
                 dim_pose=dim_pose,
                 frame_times=frame_times,
                 omit_root_pos=args.omit_root_pos,
                 rep=args.rep,
                 abs_angle=args.abs_angle,
                 allow_pickle=True)
             
        if args.debug:
            dataset = np.load(args.outpath, allow_pickle=True)
            motions = dataset['motions']
            names = dataset['names']
            frame_times = dataset['frame_times']
            
            outdebugdir = os.path.join(outdir, 'debug')
            if not os.path.exists(outdebugdir):
                os.makedirs(outdebugdir)
            
            for motion, name, frame_time in zip(motions, names, frame_times):
                position = motion[:, :3]
                motion = motion[:, 3:]
                dim_dict = {
                    'quat': 4,
                    'expmap': 3,
                    'ortho6d': 6
                }
                dim = dim_dict[args.rep]
                length, _ = motion.shape
                motion = np.reshape(motion, [length, -1, dim])
                bvhtonp.write_bvh(position, motion, name, frame_time, outdebugdir)
            
    else:
        print("No bvh files specified.")
        sys.exit(1)

        
if __name__ == '__main__':
    main()
