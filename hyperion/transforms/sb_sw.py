"""
Estimate between and within class matrices
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import numpy as np
import h5py

import scipy.linalg as la

from ..hyp_model import HypModel

class SbSw(HypModel):

    def __init__(self, Sb=None, Sw=None, mu=None, nb_classes=0, **kwargs):
        super(SbSw, self).__init__(**kwargs)
        self.Sb = None
        self.Sw = None
        self.mu = None
        self.nb_classes = nb_classes

    def fit(self, x, class_ids, normalize=True):
        dim = x.shape[1]
        self.Sb = np.zeros((dim, dim))
        self.Sw = np.zeros((dim, dim))
        self.mu = np.zeros((dim,))

        u_ids = np.unique(class_ids)
        self.nb_classes = len(u_ids)

        for i in u_ids:
            idx = (class_ids==i)
            N_i = np.sum(idx)
            mu_i = np.mean(x[idx,:], axis=0)
            self.mu += mu_i
            x_i = x[idx, :] - mu_i
            self.Sb += np.outer(mu_i, mu_i)
            self.Sw += np.dot(x_i.T, x_i)/N_i

        if normalize:
            self.normalize()


    def normalize(self):
        self.mu /= self.nb_classes
        self.Sb = self.Sb/self.nb_classes - np.outer(self.mu, self.mu)
        self.Sw /= self.nb_classes

        
    @classmethod
    def accum_stats(cls, stats):
        mu = np.zeros_like(stats[0].mu)
        Sb = np.zeros_like(stats[0].Sb)
        Sw = np.zeros_like(stats[0].Sw)
        nb_classes = 0
        for s in stats:
            mu += s.mu
            Sb += s.Sb
            Sw += s.Sw
            nb_classes += s.nb_classes

            
    def save_params(self, f):
        params = {'mu': self.mu,
                  'Sb': self.Sb,
                  'Sw': self.Sw,
                  'nb_classes': self.nb_classes}
        self._save_params_from_dict(f, params)

    
    @classmethod
    def load(cls, file_path):
        with h5py.File(file_path,'r') as f:
            config = self.load_config_from_json(f['config'])
            param_list = ['mu', 'Sb', 'Sw', 'nb_classes']
            params = cls._load_params_to_dict(f, config['name'], param_list)
            nb_classes = int(params['nb_classes'])
            return cls(Sb=params['Sb'], Sw=params['Sw'], mu=params['mu'],
                       nb_classes=nb_classes, name=config['name'])
        
        
            
