import os
import sys
import numpy as np
import theano
import theano.tensor as T

from theano.tensor.shared_randomstreams import RandomStreams

class Pool1DLayer:
    
    def __init__(self, rng, pool_shape, input_shape, pooler=T.max, depooler='random'):
        """
        Allocate a Pool1DLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type pool_shape: tuple or list of length 4
        :param pool_shape: (pool height, pool width)

        :type input_shape: tuple or list of length 3
        :param input_shape: (num input feature maps, number of joints, number of time frames)

        :type pooler: theano.tensor method
        :param pooler: Type of pooling operation (by deafult max pooling)

        :type depooler: string
        :param depooler: Type of deepoling operation (random or first)
        """
        
        self.pool_shape = pool_shape
        self.input_shape = input_shape
        self.output_shape = (input_shape[0], input_shape[1], input_shape[2]//self.pool_shape[0])
        self.theano_rng = RandomStreams(rng.randint(2 ** 30))
        self.pooler = pooler        
        self.depooler = depooler
        self.params = []
        
    def __call__(self, input):
        
        return self.pooler(input.reshape((
            self.input_shape[0], self.input_shape[1], 
            self.input_shape[2]//self.pool_shape[0],
            self.pool_shape[0])), axis=3)
        
    def inv(self, output):
        
        output = output.dimshuffle(0,1,2,'x').repeat(self.pool_shape[0], axis=3)
        
        if self.depooler == 'random':
            mask = self.theano_rng.uniform(size=output.shape, dtype=theano.config.floatX) 
            mask = T.floor(mask / mask.max(axis=3).dimshuffle(0,1,2,'x'))
            output = mask * output
        elif self.depooler == 'first':
            mask_np = np.zeros(self.pool_shape, dtype=theano.config.floatX)
            mask_np[0] = 1.0
            mask = theano.shared(mask_np, borrow=True).dimshuffle('x','x','x',0)
            output = mask * output
        else:
            output = self.depooler(output, axis=3)
        
        return output.reshape(self.input_shape)
        
    def load(self, filename): pass
    def save(self, filename): pass
    def reset(self): pass
