from VAE import VAE

import os
import sys
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import shutil

import warnings
warnings.simplefilter('ignore')

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)
from tensorflow import keras
from keras.layers import Input, Dense, RNN, LSTM, LSTMCell, Bidirectional, TimeDistributed, AveragePooling2D, Reshape, Lambda, Concatenate, Add
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential, load_model
from keras.regularizers import l1
from keras.optimizers import SGD, Adam
import tensorflow.python.keras.backend as K

class lstmVAE_feedback(VAE):
    def __init__(self, args):
        
        super().__init__(args)
        
        self.residual = bool(args['residual'][0])
        
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()    
        
        optimizer = Adam(learning_rate=self.learning_rate)
        
        motion = Input(shape=(self.seq_length, self.dim_pose, 1))
        z_mean, z_log_sigma, motion_pass = self.encoder(motion)
        
        self.z_mean = z_mean
        self.z_log_sigma = z_log_sigma
    
        z = Lambda(self.sampling)([z_mean, z_log_sigma])
        
        motion_recon = self.decoder([z, motion_pass])
        
        self.combined = Model(motion, motion_recon)
        self.combined.compile(loss=self.loss, optimizer=optimizer, metrics=[self.KL_loss])
        self.combined.summary()
        
        if self.training != 'true':
            self.combined.load_weights(self.save_path)    
        
            
        
    def build_encoder(self):
        # x:(batch_size, seq_length, dim=dim_pose, channels=1)
        self.motion_shape = (self.seq_length, self.dim_pose, 1)
        print('motion shape:', end='');print(self.motion_shape)
        
        motion = Input(name='motion',shape=self.motion_shape)
        motion_ = Lambda(lambda x: K.identity(x))(motion)
        
        motion_reshape = Reshape(target_shape=(self.seq_length, self.dim_pose))(motion)
        
        motion_reshape = BatchNormalization()(motion_reshape)
        
        h = Bidirectional(LSTM(self.z_dim * 2, 
                 dropout=0.2, recurrent_dropout=0.2,
                 return_sequences=True
                 ), merge_mode='concat')(motion_reshape)
        
        h = Lambda(lambda x: K.mean(x, axis=1))(h)
        
        h = Dense(self.z_dim * 2, 
                  activation='relu'
                  )(h)
        
        z_mean = Dense(self.z_dim, 
                  activation=None
                  )(h)
        
        z_log_sigma = Dense(self.z_dim, 
                  activation=None
                  )(h)
        
        model = Model(motion, [z_mean, z_log_sigma, motion_])
        model.summary()
        return model
    
    def build_decoder(self):
        self.z_shape = [self.z_dim]
        print('\n\nfeature shape:', end='');print(self.z_shape)
        
        self.decoder_latent_dim = self.z_dim * 2
        
        z = Input(shape=self.z_shape)
        motion = Input(shape=self.motion_shape) #for teacher forcing
        
        motion_reshape = Reshape(target_shape=(self.seq_length, self.dim_pose))(motion)
        
        initial_pose = Dense(self.dim_pose, activation=None)(z)
        initial_pose = Reshape(target_shape=[1, self.dim_pose])(initial_pose)
        
        """
        self.decoder_lstm_cell = LSTMCell(self.decoder_latent_dim,
                                          dropout=0.2, recurrent_dropout=0.2)
        self.decoder_lstm = RNN(self.decoder_lstm_cell,
                 return_sequences=True,
                 return_state=True)
        """
        self.decoder_lstm = LSTM(self.decoder_latent_dim,
                                 dropout=0.2, recurrent_dropout=0.2,
                                 return_sequences=True,
                                 return_state=True)
        decoder_dense = Dense(self.z_dim, activation='relu')
        decoder_dense2 = Dense(self.dim_pose, activation=None)
        
        self.initial_pose_inference = Model(
            z, initial_pose
        )
        
        decoder_inputs_previous_pose = Input(shape=(1, self.dim_pose))
        decoder_inputs_z =Input(shape=(1, self.z_dim))
        decoder_inputs_previous_step = Concatenate(axis=2)([decoder_inputs_previous_pose, decoder_inputs_z])
        decoder_state_input_h = Input(shape=(self.decoder_latent_dim,))
        decoder_state_input_c = Input(shape=(self.decoder_latent_dim,))
        decoder_state_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = self.decoder_lstm(
            decoder_inputs_previous_step, initial_state=decoder_state_inputs
        )
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_outputs = decoder_dense2(decoder_outputs)
        if self.residual:
            decoder_outputs = Add()([decoder_outputs, decoder_inputs_previous_pose])
        self.decoder_inference = Model(
            [decoder_inputs_previous_pose, decoder_inputs_z] + decoder_state_inputs,
            [decoder_outputs] + decoder_states
        )
        
        output_seq = []
        initial_state = Lambda(lambda x: K.zeros_like(x))(z)
        initial_state = Concatenate(axis=1)([initial_state, initial_state])
        state_value = [initial_state, initial_state]
        
        output_pose = initial_pose
        z_reshape = Lambda(lambda x: K.expand_dims(x, axis=1))(z)
        
        for _ in range(self.seq_length):
            output_pose, h, c = self.decoder_inference(
                [output_pose, z_reshape] + state_value
            )
            
            output_seq.append(output_pose)
            
            state_value = [h, c]
            
        motion_recon = Concatenate(axis=1)(output_seq)
        
        motion_recon = Reshape(target_shape=(self.seq_length, self.dim_pose, 1))(motion_recon)
        model = Model([z, motion], motion_recon)
        model.summary()
        
        return model
    
    def encode_mean_log_sigma(self, batch):
        z_mean, z_log_sigma, motion = self.encoder.predict(batch)
        return z_mean, z_log_sigma
        
    def sample(self, batch_size=20):
        z_sample_shape = [batch_size]
        z_sample_shape.extend(self.z_shape)
        z_sample = np.random.normal(size=z_sample_shape)
        
        return self.decode_sequence(z_sample)
    
    def decode_sequence(self, z_sample):
        batch_size = z_sample.shape[0]
        
        initial_pose = self.initial_pose_inference.predict(z_sample)
        
        output_seq = []
        state_h = np.zeros([batch_size, self.decoder_latent_dim])
        state_c = np.zeros([batch_size, self.decoder_latent_dim])
        state_value = [state_h, state_c]
        output_pose = initial_pose
        for _ in range(self.seq_length):
            output_pose, h, c = self.decoder_inference.predict(
                [output_pose, z_sample[:, None, :]] + state_value
            )
            
            output_seq.append(output_pose)
            
            state_value = [h, c]
            
        output_seq = np.concatenate(output_seq, axis=1)
        
        return output_seq[:, :, :, None]
    

class lstmVAE(VAE):
    def __init__(self, args):
        super().__init__(args)
    
        self.encoder = self.build_encoder()
        
        self.decoder = self.build_decoder()    
        
        optimizer = Adam(learning_rate=self.learning_rate)
        
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
            
    def build_encoder(self):
        # x:(batch_size, seq_length, dim=dim_pose, channels=1)
        self.motion_shape = (self.seq_length, self.dim_pose, 1)
        print('motion shape:', end='');print(self.motion_shape)
        
        motion = Input(name='motion',shape=self.motion_shape)
        motion_ = Lambda(lambda x: K.identity(x))(motion)
        
        motion_reshape = Reshape(target_shape=(self.seq_length, self.dim_pose))(motion)
        
        motion_reshape = BatchNormalization()(motion_reshape)
        
        h = Bidirectional(LSTM(self.z_dim * 2, 
                 dropout=0.2, recurrent_dropout=0.2,
                 return_sequences=True
                 ), merge_mode='concat')(motion_reshape)
        
        h = Lambda(lambda x: K.mean(x, axis=1))(h)
        
        h = Dense(self.z_dim * 2, 
                  activation='relu'
                  )(h)
        
        z_mean = Dense(self.z_dim, 
                       activation=None
                       )(h)

        z_log_sigma = Dense(self.z_dim, 
                       activation=None
                       )(h)

        
        model = Model(motion, [z_mean, z_log_sigma])
        model.summary()
        return model
    
    def build_decoder(self):
        self.z_shape = [self.z_dim]
        print('\n\nfeature shape:', end='');print(self.z_shape)
        
        z = Input(shape=self.z_shape)
        
        z_reshape = Lambda(lambda x: K.tile(x, [1,self.seq_length]))(z)
        z_reshape = Reshape(target_shape=(self.seq_length, self.z_dim))(z_reshape)
        
        self.decoder_lstm = LSTM(self.z_dim * 2,
                 dropout=0.2, recurrent_dropout=0.2,
                 return_sequences=True)
        
        decoder_dense = TimeDistributed(Dense(self.z_dim, activation=None))
        decoder_dense2 = TimeDistributed(Dense(self.dim_pose, activation=None))
        
        lstm_out = self.decoder_lstm(z_reshape)
        lstm_out = decoder_dense(lstm_out)
        motion_recon = decoder_dense2(lstm_out)
        
        motion_recon = Reshape(target_shape=(self.seq_length, self.dim_pose, 1))(motion_recon)
        model = Model(z, motion_recon)
        model.summary()
        
        return model