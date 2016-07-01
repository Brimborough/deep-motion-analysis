import os
import sys
import random
import numpy as np
import theano
import theano.tensor.nnet.conv3d2d
import theano.tensor as T
T.tensor5 = T.TensorType(theano.config.floatX, (False,False,False,False,False))
from theano.tensor.shared_randomstreams import RandomStreams

class Conv3DLayer:
    
    def __init__(self, rng, input_shape, pool_shape=(2,2,2), pooler=T.max, depooler='random'):
        self.input_shape = input_shape
        self.pool_shape = pool_shape
        self.theano_rng = RandomStreams(rng.randint(2 ** 30))
        self.params = []
        self.pooler = pooler        
        self.depooler = depooler
        
    def __call__(self, input):
        
        input = input.reshape((
            self.input_shape[0], self.input_shape[1], 
            self.input_shape[2]//self.pool_shape[0], self.pool_shape[0],
            self.input_shape[3]//self.pool_shape[1], self.pool_shape[1],
            self.input_shape[4]//self.pool_shape[2], self.pool_shape[2]))
            
        input = self.pooler(input, axis=7)
        input = self.pooler(input, axis=5)
        input = self.pooler(input, axis=3)
        
        return input
    
    def inv(self, output):
        
        output = (output.dimshuffle(0,1,2,'x',3,'x',4,'x')
            .repeat(self.pool_shape[0], axis=7),
            .repeat(self.pool_shape[1], axis=5),
            .repeat(self.pool_shape[2], axis=3))
        
        if self.depooler == 'random':
            unpooled = (
                self.input_shape[0], self.input_shape[1], 
                self.input_shape[2]//self.pool_shape[0], self.pool_shape[0],
                self.input_shape[3]//self.pool_shape[1], self.pool_shape[1],
                self.input_shape[4]//self.pool_shape[2], self.pool_shape[2])
            
            pooled = (
                self.input_shape[0], self.input_shape[1], 
                self.input_shape[2]//self.pool_shape[0], 1,
                self.input_shape[3]//self.pool_shape[1], 1,
                self.input_shape[4]//self.pool_shape[2], 1)
            
            output_mask = self.theano_rng.uniform(size=unpooled, dtype=theano.config.floatX)
            output_mask = output_mask / output_mask.max(axis=7).max(axis=5).max(axis=3).dimshuffle(0,1,2,'x',3,'x',4,'x')
            output_mask = T.floor(output_mask)
            
            return (output_mask * output).reshape(self.input_shape)
        else:
            output = self.depooler(output, axis=3)
        
    def load(self, filename): pass
    def save(self, filename): pass
    def reset(self): pass
