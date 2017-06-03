import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

class VariationalLayer(object):

    def __init__(self, rng, sample=True):
        self.theano_rng = RandomStreams(rng.randint(2 ** 30))
        self.sample = sample
        self.params = []
        
    def __call__(self, input):
        if self.sample:
            mu, sg = input[:,0::2], input[:,1::2]
            eps = self.theano_rng.normal(mu.shape, dtype=theano.config.floatX)
            eta = 1

            return mu + T.exp(sg * eta) * eps
        else:
            return input[:,0::2] 
    
    def inv(self, output): pass
    def load(self, filename): pass
    def save(self, filename): pass
    def reset(self): pass
