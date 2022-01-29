import numpy as np
from env.env import Env
from scipy.spatial.transform import Rotation as R
from data.bvh import util_bvh
from util import bvh


class Path(object):
    def __init__(self):
        self.clear()
        return

    def pathlength(self):
        return len(self.actions)

    def is_valid(self):
        valid = True
        length = self.pathlength()
        valid &= len(self.states) == length + 1
        valid &= len(self.goals) == length + 1
        valid &= len(self.actions) == length
        valid &= len(self.logps) == length
        valid &= len(self.rewards) == length
        valid &= len(self.flags) == length
        valid &= len(self.poses) == length + 1

        return valid

    def check_vals(self):
        for vals in [self.states, self.goals, self.actions, self.logps, self.rewards]:
            for v in vals:
                if not np.isfinite(v).all():
                    return False
        return True

    def clear(self):
        self.states = []
        self.goals = []
        self.actions = []
        self.logps = []
        self.rewards = []
        self.flags = []
        self.poses = []
        self.terminate = Env.Terminate.Null
        return

    def get_pathlen(self):
        return len(self.rewards)

    def calc_return(self):
        return sum(self.rewards)
    
    def calc_return_mean(self):
        return sum(self.rewards) / len(self.rewards)
    
    def calc_hacked_normalized_return(self):
        # HACK we can calc normalized return by phase and rewards count on test (initialized on time=0 and holdendframe=0)
        return sum(self.rewards) / len(self.rewards) * (self.states[-3][0] - self.states[0][0])
    
    def write_bvh(self, bvhpath, outpath, reward_threshold=0.7, record_kin=False):
        if np.mean(self.rewards) < reward_threshold:
            return False
        
        mocap = util_bvh.loadBvh(bvhpath)
        path_data = "data"
        settings = bvh.loadSetting(path_data)
        
        joints = [
            "Hips",
            "Spine", "Spine1", "Neck",
            "RightUpLeg", "RightLeg", "RightFoot",
            "RightArm", "RightForeArm", "RightHand",
            "LeftUpLeg", "LeftLeg", "LeftFoot",
            "LeftArm", "LeftForeArm", "LeftHand"
        ]
        # joints_fixed = ["RightHand", "LeftHand"]
        joints_fixed = []
        num_ang = len(joints)
        frameNum, pose_dim = np.shape(self.poses)
        
        poses = np.reshape(self.poses, [-1, 2, int(pose_dim / 2)])
        if not record_kin:
            poses = poses[:, 0]
        else:
            poses = poses[:, 1]
        
        rootPos = poses[:, 0:3] / settings["scale"]
        poses = np.reshape(poses[:, 3:], [-1, num_ang, 4])
        rot = []
        for i in range(num_ang):
            # world rotation. need to get relative rotation by multiplying inv to child
            quaternion_dm = poses[:, i]
            rot.append([DMToRotation(q_dm) for q_dm in quaternion_dm])
            
        offset_dict = {}
        for bvhjoint in mocap.get_joints_names():
            if bvhjoint in ["LeftArm", "RightArm", "LeftUpLeg", "RightUpLeg"]:
                offset_dict[bvhjoint] = bvh.offsetEulerAngle(mocap, bvhjoint, settings, typeR=True)
            else:
                offset_dict[bvhjoint] = R.identity()

        rot_dict = {}
        for bvhjoint in mocap.get_joints_names():
            if bvhjoint in joints:
                id = joints.index(bvhjoint)
                rot_dict[bvhjoint] = [offset_dict[bvhjoint].inv() * r for r in rot[id]]

        frames = []
        for bvhjoint in mocap.get_joints_names():
            if bvhjoint in joints:
                euler_joint = [rotationToBvhEuler(r)
                               for r in rot_dict[bvhjoint]]
            else:
                euler_joint = [[0, 0, 0]
                               for i in range(frameNum)]

            frames.append(euler_joint)

        frames = np.concatenate(frames, axis=1)
        frames = np.reshape(frames, [frameNum, -1])
        frames = np.concatenate([rootPos, frames], axis=1)

        util_bvh.writeBvh(bvhpath, outpath, frames, frameTime=1 / 120, frameNum=frameNum)
        return True
        
        
def DMToQuat(dm):
    return[
        dm[1],
        dm[2],
        dm[3],
        dm[0]
    ]


def rotationToBvhEuler(rotation):
    euler = rotation.as_euler('ZYX', degrees=True)  # extrinsic euler
    return [euler[0], euler[1], euler[2]]


def DMToRotation(dm):
    return R.from_quat(DMToQuat(dm))
