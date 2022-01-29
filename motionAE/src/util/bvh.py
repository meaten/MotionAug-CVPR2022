import os
import numpy as np
import json
from bvh import Bvh
from scipy.spatial.transform import Rotation as R
from copy import deepcopy


def bvhToMimic(bvhfile, output_path, **kwargs):
    print(bvhfile)
    with open(bvhfile) as f:
        bvh = Bvh(f.read())
    path_data = "data/"
    settings = loadSetting(path_data)
    bvhToModel(bvh, path_data, output_path, settings)
    bvhToControl(bvh, path_data, output_path, settings, **kwargs)
    bvhToMotion(bvh, bvhfile, path_data, output_path, settings)


def bvhToModel(bvh, path, output_path, settings):
    modelfile = path + "characters/humanoid3d.txt"
    with open(modelfile) as f:
        mimicmodel = json.loads(f.read())

    joints = mimicmodel["Skeleton"]["Joints"]
    bodydefs = mimicmodel["BodyDefs"]
    drawshapedefs = mimicmodel["DrawShapeDefs"]
    mimic_offset_norm = {}
    joint_list = []
    for i, assign in enumerate(settings["jointAssignments"]):
        bvh_name = assign[0]
        dm_name = assign[1]
        for dic in joints:
            if dic["Name"] == dm_name:
                dic = deepcopy(dic)
                mimic_offset_norm[dm_name] = np.linalg.norm([dic["AttachX"], dic["AttachY"], dic["AttachZ"]])
                dic["Name"] = bvh_name
                dic["ID"] = i
                dic["Parent"] = settings["Parents"][bvh_name]
                offset = np.array(bvh.joint_offset(bvh_name)) * settings["scale"]
                
                if "TorqueLim" in dic:
                    dic["TorqueLim"] *= 1.5
                
                dic["AttachX"] = offset[0]
                dic["AttachY"] = offset[1]
                dic["AttachZ"] = offset[2]

                if bvh_name in ["LeftUpLeg", "RightUpLeg"]:

                    euler, dim_order = offsetEulerAngle(bvh, bvh_name, settings, degrees=False)
                    
                    dic["LimLow0"] = -2.57 + euler[0]
                    dic["LimHigh0"] = 1.0 + euler[0]
                    dic["LimLow1"] = -1.0 + euler[1]
                    dic["LimHigh1"] = 1.0 + euler[1]
                    dic["LimLow2"] = -1.0 + euler[2]
                    dic["LimHigh2"] = 1.0 + euler[2]

                    dic["LimLow0"] += euler[0]
                    dic["LimHigh0"] += euler[0]
                    dic["LimLow1"] += euler[1]
                    dic["LimHigh1"] += euler[1]
                    dic["LimLow2"] += euler[2]
                    dic["LimHigh2"] += euler[2]

                    dic["AttachThetaX"] = -euler[0]
                    dic["AttachThetaY"] = -euler[1]
                    dic["AttachThetaZ"] = -euler[2]
 
                if bvh_name in ["LeftLeg", "RightLeg"]:
                    dic["Type"] = "spherical"
                    dic["LimLow0"] = 0
                    dic["LimHigh0"] = 3.14
                    dic["LimLow1"] = -0.1
                    dic["LimHigh1"] = 0.1
                    dic["LimLow2"] = -0.1
                    dic["LimHigh2"] = 0.1

                if bvh_name in ["RightArm", "LeftArm"]:
                    
                    euler, dim_order = offsetEulerAngle(bvh, bvh_name, settings, degrees=False)
                    
                    dic["LimLow0"] = -1.8
                    dic["LimHigh0"] = 1.8
                    if bvh_name == "RightArm":
                        dic["LimLow1"] = -1.3
                        dic["LimHigh1"] = 2.0
                    elif bvh_name == "LeftArm":
                        dic["LimLow1"] = -2.0
                        dic["LimHigh1"] = 1.3
                    dic["LimLow2"] = -1.8
                    dic["LimHigh2"] = 1.8

                    dic["LimLow0"] += euler[0]
                    dic["LimHigh0"] += euler[0]
                    dic["LimLow1"] += euler[1]
                    dic["LimHigh1"] += euler[1]
                    dic["LimLow2"] += euler[2]
                    dic["LimHigh2"] += euler[2]
                    
                    dic["AttachThetaX"] = -euler[0]
                    dic["AttachThetaY"] = -euler[1]
                    dic["AttachThetaZ"] = -euler[2]
                    
                if bvh_name in ["RightForeArm", "LeftForeArm"]:
                    dic["Type"] = "spherical"
                    dic["LimLow0"] = -0.1
                    dic["LimHigh0"] = 0.1
                    if bvh_name == "RightForeArm":
                        dic["LimLow1"] = 0
                        dic["LimHigh1"] = 2.57
                    if bvh_name == "LeftForeArm":
                        dic["LimLow1"] = -2.57
                        dic["LimHigh1"] = 0
                    dic["LimLow2"] = -0.1
                    dic["LimHigh2"] = 0.1

                if bvh_name in ["RightHand", "LeftHand"]:
                    dic["Type"] = "spherical"
                    dic["DiffWeight"] = 0.1
                    dic["TorqueLim"] = 30

                joint_list.append(dic)
                break

    body_list = []
    
    for i, assign in enumerate(settings["jointAssignments"]):
        bvh_name = assign[0]
        dm_name = assign[1]
        for dic in bodydefs:
            
            if dic["Name"] == dm_name:
                dic = deepcopy(dic)
                dic["Name"] = bvh_name
                dic["ID"] = i
                
                coef = 0.75
                dic["Param0"] *= coef
                dic["Param1"] *= coef
                dic["Param2"] *= coef
                
                assignlist = [assign[1] for assign
                              in settings["jointAssignments"]]
                count = assignlist.count(dm_name)
                if count > 1:
                    dic["Mass"] /= count
                if dic["Shape"] in ["capsule", "box"]:
                    euler, dim_order = offsetEulerAngle(bvh, bvh_name, settings, degrees=False)
                    offset = childOffset(bvh, bvh_name, settings)
                    attach = offset / 2
                    dic["AttachX"] = attach[0]
                    dic["AttachY"] = attach[1]
                    dic["AttachZ"] = attach[2]

                    dic["AttachThetaX"] = euler[0]
                    dic["AttachThetaY"] = euler[1]
                    dic["AttachThetaZ"] = euler[2]

                    if dic["Name"] == "LeftFoot":
                        dic["AttachThetaX"] += 33 / 180 * np.pi
                        dic["AttachThetaY"] += -30 / 180 * np.pi
                        dic["AttachThetaZ"] += 0 / 180 * np.pi
                    if dic["Name"] == "RightFoot":
                        dic["AttachThetaX"] += -10 / 180 * np.pi
                        dic["AttachThetaY"] += -30 / 180 * np.pi
                        dic["AttachThetaZ"] += 0 / 180 * np.pi
                else:
                    if bvh_name in ["Hips", "Spine", "Spine1"]:
                        dic["AttachX"] = 0
                        dic["AttachY"] = 0
                        dic["AttachZ"] = 0
                    if bvh_name == "Neck":
                        dic["AttachY"] *= 1 / 2

                body_list.append(dic)
                break
            
    shape_list = []
    
    for i, assign in enumerate(settings["jointAssignments"]):
        bvh_name = assign[0]
        dm_name = assign[1]
        for dic in drawshapedefs:
            if dic["Name"] == dm_name:
                dic = deepcopy(dic)
                dic["Name"] = bvh_name
                dic["ID"] = i
                dic["ParentJoint"] = i
                
                coef = 0.75
                dic["Param0"] *= coef
                dic["Param1"] *= coef
                dic["Param2"] *= coef
                
                if dic["Shape"] in ["capsule", "box"]:
                    euler, dim_order = offsetEulerAngle(bvh, bvh_name, settings, degrees=False)
                    offset = childOffset(bvh, bvh_name, settings)
                    attach = offset / 2
                    dic["AttachX"] = attach[0]
                    dic["AttachY"] = attach[1]
                    dic["AttachZ"] = attach[2]

                    dic["AttachThetaX"] = euler[0]
                    dic["AttachThetaY"] = euler[1]
                    dic["AttachThetaZ"] = euler[2]
                    
                    if dic["Name"] == "LeftFoot":
                        dic["AttachThetaX"] += 33 / 180 * np.pi
                        dic["AttachThetaY"] += -30 / 180 * np.pi
                        dic["AttachThetaZ"] += 0 / 180 * np.pi
                    if dic["Name"] == "RightFoot":
                        dic["AttachThetaX"] += -10 / 180 * np.pi
                        dic["AttachThetaY"] += -30 / 180 * np.pi
                        dic["AttachThetaZ"] += 0 / 180 * np.pi
                    
                else:
                    if bvh_name in ["Hips", "Spine", "Spine1"]:
                        dic["AttachX"] = 0
                        dic["AttachY"] = 0
                        dic["AttachZ"] = 0
                    if bvh_name == "Neck":
                        dic["AttachY"] *= 1 / 2
                
                shape_list.append(dic)
                break
    # body_list = []
    # shape_list =[]
    skeleton = {"Joints": joint_list}
    outputDict = {
        "Skeleton": skeleton,
        "BodyDefs": body_list,
        "DrawShapeDefs": shape_list
    }
    if output_path == "":
        with open(path + "bvh/character.txt", "w") as f:
            f.write(json.dumps(outputDict, indent=4))
    else:
        with open(os.path.join(output_path, "character.txt"), "w") as f:
            f.write(json.dumps(outputDict, indent=4))


def bvhToControl(bvh, path, output_path, settings, resforcetype="rootPD_weight_1"):
    ctrlfile = path + "controllers/humanoid3d_ctrl.txt"
    with open(ctrlfile) as f:
        outputDict = json.loads(f.read())

    outputDict["GoalPosInput"] = True
    outputDict["RecordWorldRootPos"] = True

    outputDict["EnablePhaseInput"] = True
    
    outputDict["ResForceType"] = resforcetype
    
    outputDict["RightJoints"] = settings["RightJoints"]
    outputDict["LeftJoints"] = settings["LeftJoints"]
    joint_ctrl_list = []
    for i, assign in enumerate(settings["jointAssignments"]):
        bvh_name = assign[0]
        dm_name = assign[1]
        for dic in outputDict["PDControllers"]:
            if dm_name in dic.values():
                dic = deepcopy(dic)
                dic["Name"] = bvh_name
                dic["ID"] = i
                dic["Kp"] *= 1
                    
                joint_ctrl_list.append(dic)
                break
    outputDict["PDControllers"] = joint_ctrl_list
    
    if output_path == "":
        with open(path + "bvh/ctrl.txt", "w") as f:
            f.write(json.dumps(outputDict, indent=4))
    else:
        with open(os.path.join(output_path, "ctrl.txt"), "w") as f:
            f.write(json.dumps(outputDict, indent=4))


def bvhToMotion(bvh, bvhfile, path, output_path, settings, return_string=False):
    loopText = "none"
    frames = calcFrames(bvh, settings)
    outputDict = {
        "Loop": loopText,  # "none" or "wrap"
        "Frames": frames
    }
    if return_string:
        return json.dumps(outputDict, indent=4, separators=(',', ': '))
    
    if output_path == "":
        with open(os.path.join(path + "bvh/motion.txt"), "w") as f:
            f.write(json.dumps(outputDict, indent=4, separators=(',', ': ')))
    else:
        with open(os.path.join(output_path, "motion.txt"), "w") as f:
            f.write(json.dumps(outputDict, indent=4, separators=(',', ': ')))


def calcFrames(bvh, settings):
    joints = settings["joints"]
    frames = []
    for i, joint in enumerate(joints):
        dim = settings["jointDimensions"][i]
        if joint == "second" and dim == 1:
            frames.append(np.reshape([bvh.frame_time] * bvh.nframes, [-1, 1]))
            continue
        elif joint == "Hips" and dim == 3:
            pos = [[pos[0], pos[1], pos[2]] for pos in bvh.frames_joint_channels(joint, settings["positionChannelNames"])]
            frames.append(np.array(pos) * settings["scale"])
        elif dim == 4:
            channels = settings["rotationChannelNames"]
            if 1:
                
                if joint in ["LeftArm", "RightArm", "LeftUpLeg", "RightUpLeg"]:
                    r_offset = offsetEulerAngle(bvh, joint, settings, typeR=True)
                    quat = [rotationToDM(r_offset * bvhEulerToRotation(euler))
                            for euler in bvh.frames_joint_channels(joint, channels)]
                else:
                    quat = [bvhEulerToDM(euler)
                            for euler in bvh.frames_joint_channels(joint, channels)]
                
                # quat = [bvhEulerToDM(euler), 'xyz')
                #        for euler in bvh.frames_joint_channels(joint, channels)]
                    
            else:
                """
                if joint in ["LeftArm", "RightArm", "LeftUpLeg", "RightUpLeg"]:
                    r_offset = offsetEulerAngle(bvh, joint, settings,typeR=True)
                    quat = [rotationToDM(r_offset)
                            for euler in bvh.frames_joint_channels(joint, channels)]
                else:
                    quat = [bvhEulerToDM([0.0,0.0,0])] * bvh.nframes
                """
                quat = [bvhEulerToDM([0.0, 0.0, 0])] * bvh.nframes
            frames.append(np.array(quat))
    frames = np.concatenate(frames, 1)
    assert frames.shape[1] == np.sum(settings["jointDimensions"])
    return frames.tolist()


def bvhEulerToDM(euler):
    return quatToDM(bvhEulerToRotation(euler).as_quat())


def bvhEulerToRotation(euler):
    # (z,x,y) -> (x, y, z)
    euler = [euler[1], euler[2], euler[0]]
    return R.from_euler('xyz', euler, degrees=True)


def rotationToDM(rot):
    return quatToDM(rot.as_quat())


def quatToDM(quaternion):
    # quaternion (x,y,z,w) -> (w,x,y,z)
    return [
        quaternion[3],
        quaternion[0],
        quaternion[1],
        quaternion[2]
    ]


def vectorAlignEulerAngle(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    assert norm_a != 0 and norm_b != 0

    a = a / norm_a
    b = b / norm_b

    cross = np.cross(a, b)
    norm_cross = np.linalg.norm(cross)
    cross = cross / norm_cross
    dot = np.dot(a, b)

    if norm_cross == 0 and dot > 0:
        return R.from_quat([0, 0, 0, 1])
    elif norm_cross == 0 and dot < 0:
        return R.from_quat([0, 0, 0, -1])
    
    rot = R.from_rotvec(cross * np.arcsin(norm_cross))
    return rot


def offsetEulerAngle(bvh, bvh_name, settings, degrees=False, typeR=False):
    offset_child = childOffset(bvh, bvh_name, settings)
    zero_rot_vec = settings["zeroRotationVectors"][bvh_name]
    dim_order = 'xyz'
    rot = vectorAlignEulerAngle(zero_rot_vec, offset_child)
    if typeR:
        return rot
    euler = rot.as_euler(dim_order, degrees=degrees)
    # import pdb;pdb.set_trace()
    return euler, dim_order


def childOffset(bvh, bvh_name, settings):
    for item in bvh.get_joint(bvh_name).children:
        if item in bvh.get_joints():
            child_name = item.value[-1]
            offset_child = np.array(bvh.joint_offset(child_name))
            offset_child *= settings["scale"]
            return offset_child


def applyOffset(euler_offset, euler, dim_order):
    r = R.from_euler(dim_order, euler, degrees=True)
    r_offset = R.from_euler(dim_order, euler_offset, degrees=True)

    if np.allclose(np.zeros(3), euler_offset):
        pass
    else:
        r = r * r_offset.inv()
    euler = r.as_euler(dim_order, degrees=True)
    return [euler[1], euler[2], euler[0]]
    
    
def loadSetting(path_data):
    path_settings = os.path.join(path_data, "bvh/settings.json")
    with open(path_settings) as f:
        settings = json.loads(f.read())

    return settings
