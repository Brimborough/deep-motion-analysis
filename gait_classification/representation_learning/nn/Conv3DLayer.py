import os
import sys
import random
import numpy as np
import theano
import theano.tensor.nnet.conv3d2d
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

class Conv3DLayer:
    
    def __init__(self, rng, input_shape, filter_shape, W=None, b=None):
        
        self.input_shape = input_shape
        self.filter_shape = filter_shape
        
        self.output_shape = (
            input_shape[0],input_shape[1],
            filter_shape[0],input_shape[3],
            input_shape[4])
        # Not counting the bachsize
        self.input_units = self.input_shape[1:]
        self.output_units = self.output_shape[1:]
        
        self.theano_rng = RandomStreams(rng.randint(2 ** 30))
        
        fan_in = np.prod(filter_shape[1:])
        fan_out = filter_shape[0] * np.prod(filter_shape[3:] + filter_shape[1:2])
        
        if W is None:
            W_bound = np.sqrt(6.0 / (fan_in + fan_out))
            W = np.asarray(
                    rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                    dtype=theano.config.floatX)
        
        if b is None:
            b = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
            
        self.W = theano.shared(value=W)
        self.b = theano.shared(value=b)
        self.params = [self.W, self.b]
        
    def pad(self, input, i):
        f = self.filter_shape
        rzeros = T.basic.zeros((i[0], (f[1]-1)//2, i[2], i[3], i[4]), dtype=theano.config.floatX)
        czeros = T.basic.zeros((i[0], i[1] + (f[1]-1), i[2], (f[3]-1)//2, i[4]), dtype=theano.config.floatX)
        tzeros = T.basic.zeros((i[0], i[1] + (f[1]-1), i[2], i[3] + (f[3]-1), (f[4]-1)//2), dtype=theano.config.floatX)
        input = T.concatenate([rzeros, input, rzeros], axis=1)
        input = T.concatenate([czeros, input, czeros], axis=3)
        input = T.concatenate([tzeros, input, tzeros], axis=4)
        return input
        
    def __call__(self, input):
        input = self.pad(input, self.input_shape)
        input = T.nnet.conv3d2d.conv3d(
            signals=input,
            filters=self.W,
            border_mode='valid')
        return input + self.b.dimshuffle('x', 'x', 0, 'x', 'x')
        
    
    def inv(self, output):
        output = output - self.b.dimshuffle('x', 'x', 0, 'x', 'x')
        output = self.pad(output, self.output_shape)
        output = T.nnet.conv3d2d.conv3d(
            signals=output,
            filters=self.W[:,::-1,:,::-1,::-1].dimshuffle(2,1,0,3,4),
            border_mode='valid')
        return output
        
    def load(self, filename):
        if filename is None: return
        if not filename.endswith('.npz'): filename+='.npz'
        data = np.load(filename)
        self.W = theano.shared(value=data['W'].astype(theano.config.floatX), borrow=True)
        self.b = theano.shared(value=data['b'].astype(theano.config.floatX), borrow=True)
        self.params = [self.W, self.b]
    
    def save(self, filename):
        if filename is None: return
        np.savez_compressed(filename,
            W=np.array(self.W.eval()),
            b=np.array(self.b.eval()))
    
