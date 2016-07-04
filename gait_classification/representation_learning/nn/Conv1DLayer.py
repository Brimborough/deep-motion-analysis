import numpy as np
import theano
import theano.tensor as T

from Param import Param
from theano.tensor.nnet import conv

class Conv1DLayer(object):
    def __init__(self, rng, filter_shape, input_shape, scale=1.0):
        """
        Allocate a Conv1DLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type filter_shape: tuple or list of length 3
        :param filter_shape: (number of filters, filter height, filter width)

        :type input_shape: tuple or list of length 3
        :param input_shape: (num input feature maps, number of joints, number of time frames)

        :type scale: float
        :param scale: Scale for weight initalisation
        """

        self.filter_shape = filter_shape
        self.input_shape = input_shape
        self.output_shape = (input_shape[0], filter_shape[0], input_shape[2])

        # Not counting the bachsize
        self.input_units = self.input_shape[1:]
        self.output_units = self.output_shape[1:]
        
        # Number of inputs to a hidden unit.
        fan_in = np.prod(filter_shape[1:])
        fan_out = filter_shape[0] * np.prod(filter_shape[2:])

        self.rng = rng
        
        self.W_bound = scale * np.sqrt(6. / (fan_in + fan_out))
        W = np.asarray(
                self.rng.uniform(low=-self.W_bound, high=self.W_bound, size=self.filter_shape),
                dtype=theano.config.floatX)
        
        b = np.zeros((self.filter_shape[0],), dtype=theano.config.floatX)
        
        self.W = theano.shared(value=W, borrow=True)
        self.b = theano.shared(value=b, borrow=True)
        self.params = [Param(self.W, True), Param(self.b, False)]
    
    def __call__(self, input):
        
        s, f = self.input_shape, self.filter_shape
        zeros = T.basic.zeros((s[0], s[1], (f[2]-1)//2), dtype=theano.config.floatX)
        input = T.concatenate([zeros, input, zeros], axis=2)
        input = conv.conv2d(
            input=input.dimshuffle(0,1,2,'x'),
            filters=self.W.dimshuffle(0,1,2,'x'),
            border_mode='valid')[:,:,:,0]
        
        return input + self.b.dimshuffle('x', 0, 'x')
    
    def inv(self, output):
        
        output = output - self.b.dimshuffle('x', 0, 'x')
        
        s, f = self.output_shape, self.filter_shape
        zeros = T.basic.zeros((s[0], s[1], (f[2]-1)//2), dtype=theano.config.floatX)
        output = T.concatenate([zeros, output, zeros], axis=2)
        output = conv.conv2d(
            input=output.dimshuffle(0,1,2,'x'),
            filters=self.W.dimshuffle(1,0,2,'x')[:,:,::-1],
            border_mode='valid')[:,:,:,0]
        
        return output
    
    def load(self, filename):
        if filename is None: return
        data = np.load(filename)
        self.W = theano.shared(value=data['W'].astype(theano.config.floatX), borrow=True)
        self.b = theano.shared(value=data['b'].astype(theano.config.floatX), borrow=True)
        self.params = [self.W, self.b]
    
    def save(self, filename):
        if filename is None: return
        np.savez_compressed(filename,
            W=np.array(self.W.eval()),
            b=np.array(self.b.eval()))

    def reset(self):
        W = np.asarray(
                self.rng.uniform(low=-self.W_bound, high=self.W_bound, size=self.filter_shape),
                dtype=theano.config.floatX)
        
        b = np.zeros((self.filter_shape[0],), dtype=theano.config.floatX)
        
        self.W = theano.shared(value=W, borrow=True)
        self.b = theano.shared(value=b, borrow=True)
        self.params = [self.W, self.b]
