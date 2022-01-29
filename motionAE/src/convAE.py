import os
import sys
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import shutil

import warnings
warnings.simplefilter('ignore')

import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Reshape
from keras.models import Model, Sequential, load_model
from keras.regularizers import l1
from keras.optimizers import SGD, Adam
from keras import backend as K
from keras.backend import tensorflow_backend
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)

sys.path.append('./')
from util.util_bvh   import loadBvh, writeBvh, getRotationOrderAndChannels
from util.arg_parser import ArgParser
from util.bvh import offsetEulerAngle, loadSetting

#from util.exit_notify import exit_notify;exit_notify()

class ConvAE():
    def __init__(self, args):
        
        arg_parser = build_arg_parser(args)
        args = arg_parser._table
        
        self.window_size  = int(args['window_size'][0])
        self.dim_pose     = int(args['dim_pose'][0])
        self.dim_channels = int(args['dim_channels'][0])
        self.strides_temp = int(args['strides_temp'][0])
        self.strides_dim  = int(args['strides_dim'][0])
        self.pool_size    = int(args['pool_size'][0])
        self.param_l1     = float(args['param_l1'][0])
        self.w_quat_reg   = float(args['w_quat_reg'][0])
        
        self.from_npz  = bool(args['from_npz'][0])
        self.data_rep  = args['data_rep'][0]
        self.data_path = args['data_path'][0]
        self.dataset   = args['dataset'][0]
        self.abs_angle = bool(args['abs_angle'][0])
        self.fps       = int(args['fps'][0])
        self.training     = args['training'][0]
        self.save_path = args['save_path'][0]
        
        
        if self.from_npz != True:
            #not implemented 
            pass    
        dataset = np.load(self.dataset, allow_pickle=True)
        self.motions = dataset['motions']
        self.names   = dataset['names']
        self.seq_length = max([self.window_size, dataset['min_len']])
        
        self.encoder = self.build_encoder()
        
        if self.training != 'true':
            self.encoder.load_weights(self.save_path)
            del(self.motions)
            del(self.names)
            return
        
        self.decoder = self.build_decoder()    
        
        optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
        
        motion = Input(shape=(self.seq_length, self.dim_pose, 1))
        feature = self.encoder(motion)
        motion_recon = self.decoder(feature)
        
        self.combined = Model(motion, motion_recon)
        self.combined.compile(loss=quat_loss, optimizer=optimizer)
        
        
    def build_encoder(self):
        # x:(batch_size, seq_length, dim=dim_pose, channels=1)
        motion_shape = (self.seq_length, self.dim_pose, 1)
        print('motion shape:', end='');print(motion_shape)
        
        model = Sequential(name='encoder')
        
        model.add(Conv2D(filters=self.dim_channels,
                         kernel_size=(self.window_size, self.strides_dim), 
                         strides=(self.strides_temp, self.strides_dim), 
                         activation='relu', 
                         padding='same', 
                         data_format='channels_last',
                         input_shape=motion_shape,
                         kernel_regularizer=l1(self.param_l1)
                         )
                  )

        # encoded: (batch_size, seq_length / pool_size, dim=dim_pose, channels=dim_channels)
        model.add(MaxPooling2D(pool_size=(self.pool_size, 1), 
                               strides=None, 
                               data_format='channels_last',
                               padding='same',
                               name='feature'))
        
        model.summary()
        
        motion = Input(name='motion',shape=motion_shape)
        feature = model(motion)
        
        return Model(motion, feature)
    
    def build_decoder(self):
        
        feature_shape = (int(self.seq_length / self.pool_size) + self.seq_length % self.pool_size,
                         int(self.dim_pose / self.strides_dim), self.dim_channels)
        print('\n\nfeature shape:', end='');print(feature_shape)
        model = Sequential(name='decoder')
        
        # x: (batch_size,seq_length, dim=dim_pose, channels=dim_channels)
        model.add(UpSampling2D(size=(self.pool_size, 1), 
                               data_format='channels_last'))

        # decoded: (batch_size, seq_length, dim=dim_pose, channels=1)
        model.add(Conv2D(filters=self.strides_dim,
                         kernel_size=(self.window_size, 1),
                         strides=(self.strides_temp, 1),
                         padding='same',
                         data_format='channels_last',
                         kernel_regularizer=l1(self.param_l1)
                         )
                  )
        
        model.add(Reshape(target_shape=(self.seq_length, self.dim_pose, 1)))
        
        feature = Input(shape=feature_shape)
        motion = model(feature)
        
        model.summary()
        
        return Model(feature, motion)
    
    def train(self, epochs=10000, batch_size=128, save_interval=1000):
        
        motions = self.motions
        for epoch in range(epochs):
            print("epoch: {0}".format(epoch), end='')
            idx = np.random.randint(0, len(motions), size=batch_size)
            X_batch = np.array([sample_motion(motion, self.seq_length) for motion in motions[idx]])
            X_batch = X_batch[:,:,:,None]
            loss = self.combined.train_on_batch(X_batch, X_batch)
            print(" loss: {0}".format(loss))
            
        print('end training')
        
    def save_encoder(self):
        self.encoder.save(self.save_path)
        """
        model = load_model("data/bvh/keras_models/encoder.h5")
        print(model.outputs)
        frozen_graph = freeze_session(K.get_session(),
                                      output_names=[out.op.name for out in model.outputs])
        
        tf.train.write_graph(frozen_graph, "data/bvh/tf_models", "tf_model.pb", as_text=False)
        """
        
    def predict(self, motions):
        return self.encoder.predict(motions)
    
    def feature_diff(self, motion1, motion2):
        
        motion1 = np.reshape(motion1, [1, self.seq_length, self.dim_pose, 1])
        motion2 = np.reshape(motion2, [1, self.seq_length, self.dim_pose, 1])
        feature1 = self.encoder.predict(motion1)
        feature2 = self.encoder.predict(motion2)
        
        diff = float(np.linalg.norm(feature1 - feature2))
        
        del(motion1); del(motion2); del(feature1); del(feature2)
        
        return diff 
        
    def reconst_motion(self):
        import pdb;pdb.set_trace()
        path = './motion_recon/'
        
        try:
            shutil.rmtree(path)
        except FileNotFoundError:
            pass
        os.makedirs(path)
        os.makedirs(os.path.join(path, 'input'))
        os.makedirs(os.path.join(path, 'reconst'))
        
        motions = self.motions
        
        idx = np.random.randint(0, len(motions), size=20)
        X_batch = np.array([sample_motion(motion, self.seq_length) for motion in motions[idx]])
        names_batch = self.names[idx]
        X_batch = X_batch[:,:,:,None]
        X_batch_reconst = self.combined.predict(X_batch)
        
        batch, length, dim, _ = np.shape(X_batch)
        dim_feature = 4 if self.data_rep == 'quaternion' else 3
        X_batch = np.reshape(X_batch, [batch, length, int(dim / dim_feature), dim_feature])
        X_batch_reconst = np.reshape(X_batch_reconst, [batch, length, int(dim / dim_feature), dim_feature])
        if self.data_rep == 'quaternion':
            norm_X_batch_reconst = np.linalg.norm(X_batch_reconst, axis=3)
            X_batch_reconst = X_batch_reconst / norm_X_batch_reconst[:,:,:,None]
            
            X_batch = X_batch[:,:,:,[1,2,3,0]] #xyzw
            X_batch_reconst = X_batch_reconst[:,:,:,[1,2,3,0]] #xyzw
            
        self.write_bvh(X_batch, names_batch, path+'input/')
        self.write_bvh(X_batch_reconst, names_batch, path+'reconst/')
        
    def write_bvh(self, motions, names, outpath):
        joints = [
        "Hips",
        "Spine", "Spine1", "Neck",
        "RightUpLeg", "RightLeg", "RightFoot",
        "RightArm", "RightForeArm", "RightHand",
        "LeftUpLeg", "LeftLeg", "LeftFoot",
        "LeftArm", "LeftForeArm", "LeftHand"
        ]
        
        data_path = './../interaction/data/'
        settings = loadSetting(data_path)
        
        bvhpath = os.path.join(self.data_path, 'HDM_bd_cartwheelLHandStart1Reps_001_120.bvh')
        mocap = loadBvh(bvhpath)
        
        for motion, name in zip(motions, names):
            
            offset_dict = {}
            for bvhjoint in mocap.get_joints_names():
                parent_id = mocap.joint_parent_index(bvhjoint)
                if parent_id == -1:
                    parent_joint = None
                    offset_dict[bvhjoint] = R.identity()
                else:
                    parent_joint = mocap.joint_parent(bvhjoint).name

                    if bvhjoint in ["LeftArm", "RightArm", "LeftUpLeg", "RightUpLeg"]:
                        r_offset = offsetEulerAngle(mocap, bvhjoint, settings, typeR=True)
                    else:
                        r_offset = R.identity()

                    if bvhjoint in ["LeftHand", "RightHand"]: #fixed joint
                        offset_dict[bvhjoint] = r_offset
                    elif bvhjoint in ["LeftFoot", "RightFoot"]:
                        #offset_dict[bvhjoint] = offset_dict[parent_joint] * R.from_euler('ZYX', [0, -90, 0])
                        offset_dict[bvhjoint] = offset_dict[parent_joint] * r_offset * R.from_euler('ZYX', [0, -90, 0])
                        #print(bvhjoint, offset_dict[bvhjoint].as_euler("ZYX", degrees=True))
                    else:
                        offset_dict[bvhjoint] = offset_dict[parent_joint] * r_offset
            
            
            rot_dict = {}
            if self.data_rep == 'quaternion':
                for i, joint_name in enumerate(joints):
                    if i == 0:
                        rot_dict[joint_name] = [R.from_quat([0,0,0,1]) for i in range(len(motion)) ]
                    else:
                        rot_dict[joint_name] = [R.from_quat(quat) for quat in motion[:,i-1]]
            else:
                print('unknown data representation')
                sys.exit(0)
                
            for bvhjoint in mocap.get_joints_names():
                if bvhjoint in joints:
                    rot_dict[bvhjoint] = [ r * offset_dict[bvhjoint].inv() for r in rot_dict[bvhjoint]]
                    
            frames = []
            for bvhjoint in mocap.get_joints_names():
                parent_id = mocap.joint_parent_index(bvhjoint)
                if parent_id == -1:
                    parent_joint = None
                    euler_joint = [rotationToBvhEuler(r) for r in rot_dict[bvhjoint]]
                elif parent_id == 0:
                    parent_joint = mocap.joint_parent(bvhjoint).name
                    euler_joint = [rotationToBvhEuler(rp * r)
                                   for rp, r in zip(rot_dict[parent_joint], rot_dict[bvhjoint])]
                elif bvhjoint in joints:
                    parent_joint = mocap.joint_parent(bvhjoint).name
                    euler_joint = [rotationToBvhEuler(rp.inv() * r)
                                   for rp, r in zip(rot_dict[parent_joint], rot_dict[bvhjoint])]
                else:
                    parent_joint = mocap.joint_parent(bvhjoint).name
                    euler_joint = [rotationToBvhEuler(rp.inv())
                                   for rp in rot_dict[parent_joint]]
                    
                frames.append(euler_joint)

            frameNum = len(motion)
            rootPos = np.zeros([frameNum, 3])
            
            frames = np.concatenate(frames, axis=1)
            frames = np.reshape(frames, [frameNum, -1])
            frames = np.concatenate([rootPos, frames], axis=1)

            writeBvh(bvhpath, outpath+name+'.bvh', frames, frameTime=1/30, frameNum=frameNum)
            
            

def quat2bvheuler(quat, mocap, joint_name):
    order, _ = getRotationOrderAndChannels(mocap, joint_name)
    return R.from_quat(quat).as_euler(order, degrees=True)


def rotationToBvhEuler(rotation):
    euler = rotation.as_euler('ZYX', degrees=True) #extrinsic euler 
    return [euler[0], euler[1], euler[2]]

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph

def quat_loss(y_true, y_pred): # y_true (batch_size, row, col, ch)
    _, seq_length, dim_pose, _ = y_pred.get_shape().as_list()
    
    prod = y_pred * y_pred
    prod = K.reshape(prod, shape=(-1, seq_length, int(dim_pose/4), 4))
    
    norm = K.sqrt(K.sum(prod, axis=3, keepdims=True))
    
    reg_loss = K.abs(K.sum(K.ones(K.shape(norm)) - norm, axis=[1,2,3]))
    
    prod = y_true * y_pred
    prod = K.reshape(prod, shape=(-1, seq_length, int(dim_pose/4), 4)) / norm
    prod = K.abs(K.sum(prod, axis=3))
    
    pred_loss = K.sum(K.ones(K.shape(prod)) - prod, axis=[1,2])
      
    return pred_loss + reg_loss * 0.01
        
def sample_motion(motion, seq_length):
    len_diff = len(motion) - seq_length
    if len_diff < 0:
        len_diff = abs(len_diff)
        odd = (len_diff % 2 == 1)
        motion = np.pad(motion, pad_width=[[int(len_diff/2)+odd, int(len_diff/2)],[0,0]], mode='reflect')
        return motion
    elif len_diff == 0:
        return motion
    else:
        idx = np.random.randint(0, len_diff)
        return motion[idx:idx+seq_length]
        

def build_arg_parser(args):
    arg_parser = ArgParser()
    arg_parser.load_args(args)
    
    arg_file = arg_parser.parse_string('arg_file', '')
    if (arg_file != ''):
        succ = arg_parser.load_file(arg_file)
        assert succ
        
    return arg_parser
    
def main():
    args = sys.argv[1:]
    
    autoencoder = ConvAE(args)
    if autoencoder.training == 'true':
        autoencoder.train()
        autoencoder.save_encoder()
        autoencoder.reconst_motion()
    
    
if __name__ == '__main__':
    main()
        
        
        
        
