""""
Tied Variational Autoencoder with 2 latent variables:
 - y: variable tied across all the samples in the segment
 - z: untied variable,  it has a different value for each frame

Factorization of the posterior:
   q(y_i,Z_i)=q(y_i) \prod_j q(z_{ij})

The parameters \phi of all the  variational distributions are given by 
a unique NN.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import time

import numpy as np

from keras import backend as K
from keras import optimizers
from keras import objectives
from keras.layers import Input, Lambda, Merge
from keras.models import Model, load_model, model_from_json

from ..layers.sampling import *
from .. import objectives as hyp_obj
from ..keras_utils import *

from .vae import VAE

class TiedVAE_qYqZ(VAE):

    def __init__(self, encoder_net, decoder_net, x_distribution):
        super(TiedVAE_qYqZ, self).__init__(encoder_net, decoder_net, x_distribution)
        self.y_dim = 0
        self.max_seq_length = 0

    def build(self, max_seq_length=None):
        self.x_dim = self.encoder_net.internal_input_shapes[0][-1]
        self.y_dim = self.decoder_net.internal_input_shapes[0][-1]
        self.z_dim = self.decoder_net.internal_input_shapes[1][-1]
        if max_seq_length is None:
            self.max_seq_length = self.encoder_net.internal_input_shapes[0][-2]
        else:
            self.max_seq_length = max_seq_length
        self._build_model()
        self._build_loss()

        
    def _build_model(self):
        x=Input(shape=(self.max_seq_length, self.x_dim,))
        yz_param=self.encoder_net(x)
        self.y_param=yz_param[:2]
        self.z_param=yz_param[2:]

        z = DiagNormalSampler()(self.z_param)
        y = DiagNormalSamplerFromSeqLevel(self.max_seq_length)(self.y_param)
        #y = DiagNormalSampler()(self.z_param)
        x_dec_param=self.decoder_net([y, z])
        # hack for keras to work
        if self.x_distribution != 'bernoulli':
            x_dec_param=Merge(mode='concat', concat_axis=-1)(x_dec_param)

        self.model=Model(x, x_dec_param)

        
    def _build_loss(self):
        if self.x_distribution == 'bernoulli':
            self.loss=lambda x, y : self._get_loss_bernoulli(
                x, y, self.y_param, self.z_param)
        else:
            self.loss=lambda x, y : self._get_loss_normal(
                x, y, self.y_param, self.z_param)

            
    def _compile(self, optimizer=None):
        if optimizer is None:
            optimizer=optimizers.Adam(lr=0.001)
        self.model.compile(optimizer=optimizer, loss=self.loss,
                           sample_weight_mode='temporal')
        self.is_compiled=True

            
    def compute_qyz_x(self, x, batch_size):
        return self.encoder_net.predict(x, batch_size=batch_size)

    
    def compute_px_yz(self, y, z, batch_size):
        return self.decoder_net.predict([y, z], batch_size=batch_size)

    
    def decode_yz(self, y, z, batch_size, sample_x=True):
        if y.ndim == 2:
            y=np.expand_dims(y,axis=1)
        if y.shape[1]==1:
            y=np.tile(y,(1,z.shape[1],1))

        if not(sample_x):
            x_param=self.decoder_net.predict([y, z], batch_size=batch_size)
            if self.x_distribution=='bernoulli':
                return x_param
            return x_param[:,:,:self.x_dim]

        y_input=Input(shape=(self.max_seq_length, self.y_dim,))
        z_input=Input(shape=(self.max_seq_length, self.z_dim,))
        x_param=self.decoder_net([y_input, z_input])
        if self.x_distribution == 'bernoulli' :
            x_sampled = BernoulliSampler()(x_param)
        else:
            x_sampled = DiagNormalSampler()(x_param)
        generator = Model([y_input, z_input], x_sampled)
        return generator.predict([y, z],batch_size=batch_size)


    def sample(self, n_signals, n_samples, batch_size,sample_x=True):
        y=np.random.normal(loc=0., scale=1., size=(n_signals, 1, self.y_dim))
        z=np.random.normal(loc=0., scale=1., size=(n_signals, n_samples,self.z_dim))
        return self.decode_yz(y, z, batch_size, sample_x)


    def sample_x_g_y(self, y, n_samples, batch_size, sample_x=True):
        n_signals=y.shape[0]
        z=np.random.normal(loc=0., scale=1., size=(n_signals, n_samples, self.z_dim))
        return self.decode_yz(y, z, batch_size, sample_x)

            
    def elbo(self, x, nb_samples, batch_size=None, mask_value=0):
        if not self.is_compiled:
            self._compile()

        if self.elbo_function is None:
            self.elbo_function = make_eval_function(self.model, self.loss)

        if batch_size is None:
            batch_size = x.shape[0]

        sw = np.any(np.not_equal(x, mask_value),
                    axis=-1, keepdims=False).astype('float32')

        elbo = - eval_loss(self.model, self.elbo_function, x, x, batch_size=batch_size, sample_weight=sw)
        for i in xrange(1, nb_samples):
            elbo -= eval_loss(self.model, self.elbo_function, x, x, batch_size=batch_size, sample_weight=sw)

        return elbo/nb_samples

    
    def eval_llr_1vs1(self, x1, x2, score_mask=None, method='elbo', nb_samples=1):
        if method == 'elbo':
            return self.eval_llr_1vs1_elbo(x1, x2, score_mask, nb_samples)
        if method == 'cand':
            return self.eval_llr_1vs1_cand(x1, x2, score_mask)
        if method == 'qscr':
            return self.eval_llr_1vs1_qscr(x1, x2, score_mask)

        
    def eval_llr_1vs1_elbo(self, x1, x2, score_mask=None, nb_samples=1):
        # x1 = np.expand_dims(x1, axis=1)
        # elbo_1 = self.elbo(x1, nb_samples)
        # x2 = np.expand_dims(x2, axis=1)
        # elbo_2 = self.elbo(x2, nb_samples)
        # print(elbo_1.shape)
        # print(elbo_2.shape)
        # scores = - (np.expand_dims(elbo_1, axis=-1) +
        #             np.expand_dims(elbo_2, axis=-1).T)

        # xx_shape = (x1.shape[0], 2, x1.shape[2])
        # xx = np.zeros(xx_shape, 'float32')
        # xx[:,1,:] = x2
        # for i in xrange(x1.shape[0]):
        #     xx[:,0,:] = x1[i,0,:]
        #     elbo_3 = np.expand_dims(self.elbo(xx, nb_samples), axis=-1)
        #     scores[i,:]= elbo_3
        # return scores
        
        xx_shape = (x1.shape[0], self.max_seq_length, x1.shape[1])
        xx = np.zeros(xx_shape, 'float32')
        xx[:,0,:] = x1
        elbo_1 = self.elbo(xx, nb_samples)

        xx_shape = (x2.shape[0], self.max_seq_length, x2.shape[1])
        xx = np.zeros(xx_shape, 'float32')
        xx[:,0,:] = x2
        elbo_2 = self.elbo(xx, nb_samples)
        
        print(elbo_1.shape)
        print(elbo_2.shape)
        scores = - (np.expand_dims(elbo_1, axis=-1) +
                    np.expand_dims(elbo_2, axis=-1).T)

        for i in xrange(x1.shape[0]):
            xx[:,1,:] = x1[i,:]
            elbo_3 = self.elbo(xx, nb_samples)
            scores[i,:] += elbo_3
        return scores

    
    def eval_llr_1vs1_cand(self, x1, x2, score_mask=None):
        xx_shape = (x1.shape[0], self.max_seq_length, x1.shape[1])
        xx = np.zeros(xx_shape, 'float32')
        xx[:,0,:] = x1
        y_mean, y_logvar, _, _ = self.compute_qyz_x(xx, batch_size=x1.shape[0])
        logq_1 = self._eval_logqy_eq_0(y_mean, y_logvar)
        
        xx_shape = (x2.shape[0], self.max_seq_length, x2.shape[1])
        xx = np.zeros(xx_shape, 'float32')
        xx[:,0,:] = x2
        y_mean, y_logvar, _, _ = self.compute_qyz_x(xx, batch_size=x2.shape[0])
        logq_2 = self._eval_logqy_eq_0(y_mean, y_logvar)

        print(logq_1.shape)
        print(logq_2.shape)
        scores = np.expand_dims(logq_1, axis=-1) + np.expand_dims(logq_2, axis=-1).T

        for i in xrange(x1.shape[0]):
            xx[:,1,:] = x1[i,:]
            y_mean, y_logvar, _, _ = self.compute_qyz_x(xx, batch_size=x2.shape[0])
            scores[i,:] -= self._eval_logqy_eq_0(y_mean, y_logvar)
        return scores

    
    @staticmethod
    def _eval_logqy_eq_0(mu, logvar):
        var = np.exp(logvar)
        return -0.5*np.sum(logvar + mu**2/var, axis=-1)
        

    def eval_llr_1vs1_qscr(self, x1, x2, score_mask=None):
        xx_shape = (x1.shape[0], self.max_seq_length, x1.shape[1])
        xx = np.zeros(xx_shape, 'float32')
        xx[:,0,:] = x1
        y1_mean, y1_logvar, _, _ = self.compute_qyz_x(xx, batch_size=x1.shape[0])
        y1_p = np.exp(-y1_logvar)
        r1 = y1_p*y1_mean
        logq_1 = -0.5*np.sum(y1_logvar + r1**2/y1_p, axis=-1)
        
        xx_shape = (x2.shape[0], self.max_seq_length, x2.shape[1])
        xx = np.zeros(xx_shape, 'float32')
        xx[:,0,:] = x2
        y2_mean, y2_logvar, _, _ = self.compute_qyz_x(xx, batch_size=x2.shape[0])
        y2_p = np.exp(-y2_logvar)
        r2 = y2_p*y2_mean
        logq_2 = -0.5*np.sum(y2_logvar + r2**2/y2_p, axis=-1)

        scores = np.expand_dims(logq_1, axis=-1) + np.expand_dims(logq_2, axis=-1).T

        for i in xrange(x1.shape[0]):
            p_3 = y1_p[i,:] + y2_p - np.ones_like(y2_p)
            r3 = r1[i,:] + r2
            scores[i,:] += 0.5*np.sum(-np.log(p_3) + r3**2/p_3, axis=-1)
        return scores
    
    # @staticmethod
    # def _sample_normal3D(params,seq_length):
    #     mu, logvar = params
    #     mu=K.expand_dims(mu,dim=1)
    #     logvar=K.expand_dims(logvar,dim=1)
    #     shape=list(K.shape(mu))
    #     shape[1]=seq_length
    #     epsilon = K.random_normal(shape=shape, mean=0., std=1.)
    #     return mu + K.exp(logvar / 2) * epsilon

    # @staticmethod
    # def _sample_normal3D_2(params,seq_length):
    #     mu, logvar = params
    #     mu=K.expand_dims(mu,dim=1)
    #     logvar=K.expand_dims(logvar,dim=1)
    #     epsilon = K.random_normal(shape=K.shape(mu), mean=0., std=1.)
    #     y=mu + K.exp(logvar / 2) * epsilon
    #     return K.tile(y, (1, seq_length, 1))

    def get_config(self):
        qy_config = self.qy_net.get_config()
        qz_config = self.qz_net.get_config()
        dec_config = self.decoder_net.get_config()
        config = {
            'class_name': self.__class__.__name__,
            'qy_net': qy_config,
            'qz_net': qy_config,
            'decoder_net': dec_config,
            'x_distribution': self.x_distribution }
        return config

        
    def save(self, file_path):
        file_model = '%s.json' % (file_path)
        with open(file_model, 'w') as f:
            f.write(self.to_json())
        
        file_model = '%s.qy.h5' % (file_path)
        self.qy_net.save(file_model)
        file_model = '%s.qz.h5' % (file_path)
        self.qz_net.save(file_model)
        file_model = '%s.dec.h5' % (file_path)
        self.decoder_net.save(file_model)

    @classmethod
    def load(cls, file_path):
        file_config = '%s.json' % (file_path)
        with open(file_config,'r') as f:
            config=VAE.load_config_from_json(f.read())

        file_model = '%s.qy.h5' % (file_path)
        qy_net = load_model(file_model, custom_objects=get_keras_custom_obj())
        file_model = '%s.qz.h5' % (file_path)
        qz_net = load_model(file_model, custom_objects=get_keras_custom_obj())
        file_model = '%s.dec.h5' % (file_path)
        decoder_net = load_model(file_model, custom_objects=get_keras_custom_obj())

        return cls(qy_net, qz_net, decoder_net, config['x_distribution'])

        
    @staticmethod
    def _get_loss_bernoulli(x, x_dec_param, y_param, z_param):
        n_samples=K.sum(K.cast(K.any(K.not_equal(x, 0), axis=-1),
                               K.floatx()), axis=-1)
        logPx_g_z = hyp_obj.bernoulli(x, x_dec_param)
        kl_y = K.expand_dims(
            hyp_obj.kl_diag_normal_vs_std_normal(y_param)/n_samples, dim=1)
        kl_z = hyp_obj.kl_diag_normal_vs_std_normal(z_param)
        return logPx_g_z + kl_y + kl_z

    

    @staticmethod
    def _get_loss_normal(x, x_dec_param, y_param, z_param):
        n_samples=K.sum(K.cast(K.any(K.not_equal(x, 0), axis=-1),
                               K.floatx()), axis=-1)
        x_dim=K.cast(K.shape(x)[-1],'int32')
        x_dec_param = [x_dec_param[:,:,:x_dim], x_dec_param[:,:,x_dim:]]
        logPx_g_z = hyp_obj.diag_normal(x, x_dec_param)
        kl_y = K.expand_dims(
            hyp_obj.kl_diag_normal_vs_std_normal(y_param)/n_samples, dim=1)
        kl_z = hyp_obj.kl_diag_normal_vs_std_normal(z_param)
        return logPx_g_z + kl_y + kl_z

        # # x_decoded_mean, x_decoded_logvar = x_decoded
        # y_mean, y_logvar=y_param
        # z_mean, z_logvar=z_param
        # n_samples=K.sum(K.cast(K.any(K.not_equal(x, 0), axis=-1),K.floatx()),axis=-1)
        # x_dim=K.cast(K.shape(x)[-1],'int32')
        # max_seq_length=K.cast(K.shape(x)[-2],'float32')
        # log2pi=np.log(2*np.pi).astype('float32')
        # x_mean=x_dec_param[:,:,:x_dim]
        # x_logvar=x_dec_param[:,:,x_dim:]
        # x_var=K.exp(x_logvar)
        # dist2=K.square(x-x_mean)/x_var
        # logPx_g_z_loss=0.5*K.sum(log2pi+x_logvar+dist2, axis=-1)
        # kl_z_loss=-0.5*K.sum(1+z_logvar-K.square(z_mean)-K.exp(z_logvar), axis=-1)
        # kl_y_loss=-0.5*K.expand_dims(
        #     K.sum(1 + y_logvar - K.square(y_mean) - K.exp(y_logvar), axis=-1)/n_samples, dim=1)
        # return logPx_g_z_loss + kl_y_loss + kl_z_loss

    
