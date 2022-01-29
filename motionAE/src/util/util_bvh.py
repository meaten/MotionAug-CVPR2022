from bvh import Bvh
import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R

def loadBvh(bvhpath):
    with open(bvhpath) as f:
        return Bvh(f.read())

def writeBvh(bvhpath, outpath, frames, frameTime = None, frameNum=None, supress=False, return_string=False):
    if not supress:
        print("saving bvh file to {}".format(outpath))
    if not return_string:
        with open(outpath, 'w') as fw:
            with open(bvhpath, 'r') as fr:
                while True:
                    line = fr.readline()
                        
                    if 'Frame Time' in line:
                        if frameTime == None:
                            pass
                        else:
                            line = "Frame Time: {}\n".format(frameTime)
                        fw.write(line)
                        break
                    elif 'Frames' in line:
                        if frameNum == None:
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
    else:
        with open(bvhpath, 'r') as fr:
            bvh_string = ''
            while True:
                line = fr.readline()    
                if 'Frame Time' in line:
                    if frameTime == None:
                        pass
                    else:
                        line = "Frame Time: {}\n".format(frameTime)
                    bvh_string += line
                    break
                elif 'Frames' in line:
                    if frameNum == None:
                        frameNum = len(frames)
                    else:
                        pass
                    line = "Frames: {}\n".format(frameNum)
                    bvh_string += line
                else:
                    bvh_string += line
            for vector in frames:
                line_aligned = ' '.join(vector.astype(str)) + '\n'
                bvh_string += line_aligned
                
        return bvh_string
def FK(mocap, frame, joint_name):
    assert(joint_name in mocap.get_joints_names())

    offsets = []
    rotations = []

    indicator = ''
    while(indicator != 'ROOT'):
        offsets.append(getOffset(mocap, joint_name))
        rotations.append(getRotation(mocap, frame, joint_name))
        indicator, joint_name = mocap.joint_parent(joint_name).value

    offset_root = getTranslation_root(mocap, frame, joint_name)
    rotation_root = getRotation(mocap, frame, joint_name)

    offsets.reverse()
    rotations.reverse()

    pos = offset_root
    r_accumulated = rotation_root

    for offset, rotation in zip(offsets, rotations):
        pos += r_accumulated.apply(offset)
        r_accumulated  = r_accumulated * rotation

    return pos


def getRotation(mocap, frame, joint_name):

    order, channels = getRotationOrderAndChannels(mocap, joint_name)
    euler_angle = mocap.frame_joint_channels(frame, joint_name, channels)
    r = R.from_euler(order, euler_angle, degrees=True)

    return r


def getOffset(mocap, joint_name):
    return np.array(mocap.joint_offset(joint_name))

def getTranslation_root(mocap, frame, joint_name='Hips'):
    channels = ['Xposition', 'Yposition', 'Zposition']
    translation = mocap.frame_joint_channels(frame, joint_name, channels)

    offset_root = np.array(mocap.joint_offset(joint_name)) + translation

    return offset_root

def getRotationOrderAndChannels(mocap, joint_name):
    channels = mocap.joint_channels(joint_name)
    
    order = ''
    channels_rot = []
    for channel in channels:
        if 'rotation' in channel:
            order += channel[0] #Extrinsic Euler Angle 
            channels_rot.append(channel)

    return order, channels_rot
        
def main():
    print("test")

if __name__ == "__main__":
    main()
