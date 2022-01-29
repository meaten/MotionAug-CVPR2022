from VAE import VAE

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
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Reshape, Lambda
from keras.models import Model, Sequential, load_model
from keras.regularizers import l1
from keras.optimizers import SGD, Adam
from keras import backend as K
from keras.backend import tensorflow_backend

class convVAE(VAE):
    def __init__(self, args):
        
        arg_parser = self.build_arg_parser(args)
        args = arg_parser._table
        
        self.architecture = args['architecture'][0]
        
        self.window_size  = int(args['window_size'][0])
        self.dim_temp     = int(args['dim_temp'][0])
        self.dim_pose     = int(args['dim_pose'][0])
        self.dim_channels = int(args['dim_channels'][0])
        self.strides_temp = int(args['strides_temp'][0])
        self.strides_dim  = int(args['strides_dim'][0])
        self.pool_size    = int(args['pool_size'][0])
        
        self.param_l1     = float(args['param_l1'][0])
        self.w_quat_reg   = float(args['w_quat_reg'][0])
        self.learning_rate = float(args['learning_rate'][0])
        self.KL_coef = float(args['KL_coef'][0])
        
        self.from_npz  = bool(args['from_npz'][0])
        self.data_rep  = args['data_rep'][0]
        self.data_path = args['data_path'][0]
        self.dataset   = args['dataset'][0]
        self.abs_angle = bool(args['abs_angle'][0])
        self.fps       = int(args['fps'][0])
        self.training  = args['training'][0]
        self.save_path = args['save_path'][0]
        
        
        if self.from_npz != True:
            #not implemented 
            pass    
        dataset = np.load(self.dataset, allow_pickle=True)
        self.motions = dataset['motions']
        self.seq_length = max([self.window_size, dataset['min_len']])
        
        self.names   = dataset['names']
        self.build_labels()
        
        classes = self.one_hot_encoder.inverse_transform(self.one_hot_labels)
        print(np.unique(classes))
        used_class = 'punch'
        self.names = self.names[np.where(classes==used_class)]
        self.motions = self.motions[np.where(classes==used_class)]
        self.one_hot_labels = self.one_hot_labels[np.where(classes==used_class)]
        
        self.encoder = self.build_encoder()
        
        self.decoder = self.build_decoder()    
        
        #optimizer = Adam(learning_rate=0.001, beta_1=0.5)
        optimizer = SGD(learning_rate=self.learning_rate)
        
        motion = Input(shape=(self.seq_length, self.dim_pose, 1))
        z_mean, z_log_sigma = self.encoder(motion)
        
        self.z_mean = z_mean
        self.z_log_sigma = z_log_sigma
    
        z = Lambda(self.sampling)([z_mean, z_log_sigma])
        
        motion_recon = self.decoder(z)
        
        self.combined = Model(motion, motion_recon)
        self.combined.compile(loss=self.loss, optimizer=optimizer, metrics=[self.KL_loss])
        self.combined.summary()
        
        if self.training != 'true':
            self.combined.load_weights(self.save_path)
            del(self.motions)
            del(self.names)
            return
        
    def build_encoder(self):
        # x:(batch_size, seq_length, dim=dim_pose, channels=1)
        motion_shape = (self.seq_length, self.dim_pose, 1)
        print('motion shape:', end='');print(motion_shape)
        
        motion = Input(name='motion',shape=motion_shape)
        
        h = Conv2D(filters= self.dim_channels,
                   kernel_size=(self.dim_temp, self.strides_dim), 
                   strides=(self.strides_temp, self.strides_dim), 
                   activation='relu', 
                   padding='same', 
                   data_format='channels_last',
                   input_shape=motion_shape)(motion)

        h = MaxPooling2D(pool_size=(self.pool_size, 1), 
                         strides=None, 
                         data_format='channels_last',
                         padding='same',
                         name='h1_encoder')(h)
        
        z_mean = Conv2D(filters=self.dim_channels * self.dim_channels,
                        kernel_size=(self.dim_temp, self.strides_dim), 
                        strides=(self.strides_temp, self.strides_temp), 
                        activation='relu', 
                        padding='same', 
                        data_format='channels_last',
                        input_shape=motion_shape)(h)
        
        z_mean = MaxPooling2D(pool_size=(self.pool_size, 1), 
                              strides=None, 
                              data_format='channels_last',
                              padding='same',
                              name='z_mean')(z_mean)
        
        z_log_sigma = Conv2D(filters=self.dim_channels * self.dim_channels,
                       kernel_size=(self.dim_temp, self.strides_dim), 
                       strides=(self.strides_temp, self.strides_temp), 
                       activation='relu', 
                       padding='same', 
                       data_format='channels_last',
                       input_shape=motion_shape)(h)
        
        z_log_sigma = MaxPooling2D(pool_size=(self.pool_size, 1), 
                             strides=None, 
                             data_format='channels_last',
                             padding='same',
                             name='z_log_sigma')(z_log_sigma)
        
        model = Model(motion, [z_mean, z_log_sigma])
        model.summary()
        return model
    
    def build_decoder(self):
        
        self.z_shape = (int(self.seq_length / self.pool_size / self.pool_size),
                        int(self.dim_pose / self.strides_dim), self.dim_channels * self.dim_channels)
        print('\n\nfeature shape:', end='');print(self.z_shape)
        
        z = Input(shape=self.z_shape)
        
        h = UpSampling2D(size=(self.pool_size, 1), 
                         data_format='channels_last')(z)
        
        print(h.shape)
        
        h = Conv2D(filters=self.dim_channels,
                   kernel_size=(self.dim_temp, self.strides_dim),
                   strides=(self.strides_temp, self.strides_temp),
                   activation='relu',
                   padding='same',
                   data_format='channels_last')(h)
        
        print(h.shape)
        
        
        motion_recon = UpSampling2D(size=(self.pool_size, 1), 
                                    data_format='channels_last')(h)

        motion_recon = Conv2D(filters=self.strides_dim,
                              kernel_size=(self.dim_temp, self.strides_dim),
                              strides=(1, 1),
                              padding='same',
                              data_format='channels_last')(motion_recon)
        
        print(motion_recon.shape)
        
        motion_recon = Reshape(target_shape=(self.seq_length, self.dim_pose, 1))(motion_recon)
        model = Model(z, motion_recon)
        model.summary()
        return model
    
    
        
        
        