import os
import sys
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R
import shutil
from tqdm import tqdm
from util_bvh import loadBvh, FK

from joblib import Parallel, delayed


def parse_args():
    parser = argparse.ArgumentParser(description='align bvh motion(s) to ground for DM. skeleton needs LeftToeBase and RightToeBase joints.')

    parser.add_argument('--bvhpath', help='path to bvh file for single alignment',
                        default=None)
    parser.add_argument('--dirpath', help='path to bvh dir  for multi alignment',
                        default=None)

    parser.add_argument('--outpath', help='path to output the aligned bvh motion',
                        default='./output/')
    parser.add_argument('--split_config',
                        default=None)
    
    args = parser.parse_args()

    return args


def load_split_config(split_config: str) -> dict:
    if split_config is not None:
        arr = np.loadtxt(split_config, delimiter=" ", dtype=object)
        return dict(zip(arr[:, 0], arr[:, 1]))
    else:
        return {}


def vector_align_rotation(a, b):
    """
    return Rotation that transform vector a to vector b

    input
    a : np.array(3)
    b : np.array(3)

    return
    rot : scipy.spatial.transform.Rotation
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    assert norm_a != 0 and norm_b != 0

    a = a / norm_a
    b = b / norm_b

    cross = np.cross(a, b)
    norm_cross = np.linalg.norm(cross)
    dot = np.dot(a, b)
    if norm_cross < 1e-8 and dot > 0:
        '''same direction, no rotation a == b'''
        return R.from_quat([0, 0, 0, 1])
    elif norm_cross < 1e-8 and dot < 0:
        '''opposite direction a == -b'''
        c = np.eye(3)[np.argmax(np.linalg.norm(np.eye(3) - a, axis=1))]
        cross = np.cross(a, c)
        norm_cross = np.linalg.norm(cross)
        cross = cross / norm_cross

        return R.from_rotvec(cross * np.pi)
    
    cross = cross / norm_cross
    rot = R.from_rotvec(cross * np.arctan2(norm_cross, dot))
    
    assert np.linalg.norm(rot.apply(a) - b) < 1e-7
    return rot


def align(bvhpath, outpath, split_dict={}):
    mocap = loadBvh(bvhpath)
    
    basename = os.path.basename(bvhpath)
    frames = [1]
    if basename in split_dict:
        frames += [int(split_dict[basename])]
    
    midpoint_foots = []
    rots = []
    for frame in frames:
        pos_LF = FK(mocap, frame, 'LeftToeBase')
        pos_RF = FK(mocap, frame, 'RightToeBase')
        midpoint_foot = (pos_LF + pos_RF) / 2
        midpoint_foot[1] = min(pos_LF[1], pos_RF[1])
        
        offset = np.linalg.norm(mocap.joint_offset('LeftToeBase')) + \
            np.linalg.norm(mocap.joint_offset('RightToeBase'))
        offset /= 4
        # midpoint_foot -= offset
        midpoint_foots.append(midpoint_foot)
        
        pos_LS = FK(mocap, frame, 'LeftArm')
        pos_RS = FK(mocap, frame, 'RightArm')
    
        vec_shoulder = pos_RS - pos_LS
        vec_shoulder[1] = 0  # index 1 is vertical axis value in bvh file
        x_axis = np.array([-1, 0, 0])

        rot = vector_align_rotation(vec_shoulder, x_axis)
        rots.append(rot)

    mirror = NeedMirror(bvhpath)

    writeAlignedBvh(bvhpath, outpath, mocap, frames, midpoint_foots, rots, mirror=mirror)

    
def writeAlignedBvh(bvhpath, outpath, mocap, frames, translations, rotations, mirror=False):
    for i, (frame, translation, rotation) in enumerate(zip(frames, translations, rotations)):
        frame_num = mocap.nframes
        frame_start = int(frame)
        try:
            frame_end = frames[i + 1]
        except IndexError:
            frame_end = frame_num
            
        outpath_ = outpath
        if mirror:
            base, ext = os.path.splitext(outpath)
            outpath_ = base + '_mirror' + ext
        if len(frames) > 1:
            base, ext = os.path.splitext(outpath_)
            outpath_ = base + f'_{i}' + ext
        
        with open(outpath_, 'w') as fw:
            with open(bvhpath, 'r') as fr:
                while True:
                    line = fr.readline()
                    if "Frames:" in line and len(frames) > 1:
                        line = f"Frames: {frame_end - frame_start + 1}\n"
                    fw.write(line)
                    if 'Frame Time' in line:
                        break

                i = 0
                while True:
                    line = fr.readline()
                    i += 1
                    if len(line) == 0:
                        break
                    if i < frame_start or frame_end < i:
                        continue
                    
                    vector = np.fromstring(line, dtype=float, sep=' ')
                    vector[:3] -= translation
                    
                    order = 'ZYX'
                    rot_original = R.from_euler(order, vector[3:6], degrees=True)
                    rot = rotation * rot_original
                    vector[3:6] = rot.as_euler(order, degrees=True)
            
                    vector[:3] = rotation.apply(vector[:3])
                        
                    if mirror:
                        # mirror pose respect to y-z plane
                        vector[0] = -vector[0]
                        vector[3:] = mirror_pose(vector[3:], mocap)
                    # vector[6:] = addGaussianNoise(vector[6:], var=0.10)
                                            
                    line_aligned = ' '.join(vector.astype(str)) + '\n'
                    fw.write(line_aligned)
                    
                
def NeedMirror(bvhpath):
    import re
    bvhname = os.path.basename(bvhpath)[7:-12]
    cla = re.findall('([a-z]+)', bvhname)[0]
    mirror = False
    if cla in ['cartwheel', 'hit']:  # L/R Hand
        mirror = ('LHand' in bvhname)
        
    elif cla in ['elbow']:  # L/R elbow
        mirror = ('Lelbow' in bvhname)
        
    elif cla in ['deposit', 'grab', 'throw']:  # L/R last character
        mirror = ('L' == bvhname[-1])
        
    elif cla in ['kick', 'punch', 'hop', 'turn']:  # L/R/Both right after class name
        mirror = ('L' == bvhname[len(cla)])
        
    elif cla in ['rotate']:  # L/R/Both after rotateArms
        mirror = ('L' == bvhname[len('rotateArms')])
        
    elif cla in ['jog', 'run', 'shuffle', 'skier', 'sneak', 'staircase', 'walk']:  # L/R start
        mirror = ('LStart' in bvhname or 'Lstart' in bvhname)
        
    elif cla in ['clap', 'jump', 'jumping', 'lie', 'sit', 'squat', 'stand']:
        mirror = False
    else:
        raise(ValueError)
    
    return mirror


def mirror_pose(pose, mocap):
    order = 'ZYX'
    joint_names = mocap.get_joints_names()
    pose = np.reshape(pose, [-1, 3])
    assert len(pose) == len(joint_names)
    pose = [R.from_euler(order, euler, degrees=True) for euler in pose]
    pose = [mirror_rot(rot) for rot in pose]
    pose = [rot.as_euler(order, degrees=True) for rot in pose]
    
    idx = [joint_names.index(mirror_joint(joint)) for joint in joint_names]
    out_pose = [pose[i] for i in idx]
    out_pose = np.reshape(out_pose, [-1])
    
    return out_pose


def mirror_rot(rot):
    quat = rot.as_quat()
    quat[1] = -quat[1]
    quat[2] = -quat[2]
    return R.from_quat(quat)


def mirror_joint(joint_name):
    if 'Right' in joint_name:
        return 'Left' + joint_name[5:]
    elif 'Left' in joint_name:
        return 'Right' + joint_name[4:]
    else:
        return joint_name

                
def addGaussianNoise(pose, var=0.03):
    order = 'ZYX'
    
    vec = np.reshape(pose, [-1, 3])
    vec = [(R.from_euler(order, euler, degrees=True) * randGaussRotation(var)).as_euler(order, degrees=True)
           for euler in vec]
    vec = np.reshape(vec, [-1])
    
    return vec

                    
def randGaussRotation(var, degrees=False):
    order = 'ZYX'
    euler = np.random.normal(0, scale=var, size=3)
    return R.from_euler(order, euler, degrees=degrees)


def unit_vector(vec):
    return vec / np.linalg.norm(vec)

                
def align_single_file(file, dirpath, outpath, **kwargs):
    base, ext = os.path.splitext(file)
    if ext == '.bvh':
        bvhpath = os.path.join(dirpath, file)
        path = os.path.join(outpath, base + '_aligned' + ext)
        align(bvhpath, path, **kwargs)
        
        
def main():
    args = parse_args()

    bvhpath = args.bvhpath
    dirpath = args.dirpath
    outpath = args.outpath
    
    split_config = args.split_config
    split_dict = load_split_config(split_config)
    
    os.makedirs(outpath, exist_ok=True)
    
    if bvhpath is not None:
        print('aligning {} to {}'.format(bvhpath, outpath))
        file = os.path.basename(bvhpath)
        dirpath = os.path.dirname(bvhpath)
        align_single_file(file, dirpath, outpath)
    elif dirpath is not None:
        print('aligning all bvh files in {} to {}'.format(dirpath, outpath))
        Parallel(n_jobs=12)(delayed(align_single_file)(file, dirpath, outpath, split_dict=split_dict) for file in tqdm(os.listdir(dirpath)))
        #[align_single_file(path, dirpath, outpath, split_dict=split_dict) for path in tqdm(os.listdir(dirpath))]
        # arr = Parallel(n_jobs=8)(delayed(align_single_file)(path, dirpath, outpath) for path in tqdm(os.listdir(dirpath)))
        # arr = np.unique(arr, axis=0)
    else:
        print("No bvh file(s) specified.")
        sys.exit(1)

        
if __name__ == '__main__':
    main()
