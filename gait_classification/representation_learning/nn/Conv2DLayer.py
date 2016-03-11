import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.shared_randomstreams import RandomStreams

class Conv2DLayer(object):

    def __init__(self, rng, filter_shape, input_shape, scale=1.0):

        self.filter_shape = filter_shape
        self.input_shape = input_shape
        self.output_shape = (input_shape[0], filter_shape[0], input_shape[2], input_shape[3])
        self.theano_rng = RandomStreams(rng.randint(2 ** 30))
        
        fan_in = np.prod(filter_shape[1:])
        fan_out = filter_shape[0] * np.prod(filter_shape[2:])
        
        W_bound = scale * np.sqrt(6. / (fan_in + fan_out))
        W = np.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX)
        
        b = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        
        self.W = theano.shared(value=W, borrow=True)
        self.b = theano.shared(value=b, borrow=True)
        self.params = [self.W, self.b]
    
    def __call__(self, input):
        
        s, f = self.input_shape, self.filter_shape
        hzeros = T.basic.zeros((s[0], s[1], (f[2]-1)//2, s[3]), dtype=theano.config.floatX)
        vzeros = T.basic.zeros((s[0], s[1], s[2] + (f[2]-1), (f[3]-1)//2), dtype=theano.config.floatX)
        input = T.concatenate([hzeros, input, hzeros], axis=2)
        input = T.concatenate([vzeros, input, vzeros], axis=3)
        input = conv.conv2d(
            input=input,
            filters=self.W,
            border_mode='valid')
        
        return input + self.b.dimshuffle('x', 0, 'x', 'x')
    
    def inv(self, output):
        
        output = output - self.b.dimshuffle('x', 0, 'x', 'x')
        
        s, f = self.output_shape, self.filter_shape
        hzeros = T.basic.zeros((s[0], s[1], (f[2]-1)//2, s[3]), dtype=theano.config.floatX)
        vzeros = T.basic.zeros((s[0], s[1], s[2] + (f[2]-1), (f[3]-1)//2), dtype=theano.config.floatX)
        output = T.concatenate([hzeros, output, hzeros], axis=2)
        output = T.concatenate([vzeros, output, vzeros], axis=3)
        output = conv.conv2d(
            input=output.dimshuffle(0,1,2,3),
            filters=self.W.dimshuffle(1,0,2,3)[:,:,::-1,::-1],
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
        
        