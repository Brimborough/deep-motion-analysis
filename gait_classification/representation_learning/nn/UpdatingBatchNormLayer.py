import numpy as np
import theano
import theano.tensor as T

from theano.ifelse import ifelse

class BatchNormLayer(object):
    """
    This layer implements batch normalisation, a technique presented in [1]
    which helps to accelerate learning. This is done by transforming the input 
    to have 0 mean and a std. dev. of one. Note that through the trainable parameters beta (shifting)
    and gamma (scaling), the network can learn to undo this normalisation. 

    Notes
    ---------- This layer should be inserted between a linear transformation (e.g. Conv2DLayer or HiddenLayer) and the
    respective ActivationLayer.
    ----------
    References
    ----------
       [1] Ioffe, Sergey and Szegedy, Christian (2015):
           Batch Normalization: Accelerating Deep Network Training by Reducing
           Internal Covariate Shift. http://arxiv.org/abs/1502.03167.
    """

    def __init__(self, rng, shape, axes=(0,), epsilon=1e-10, alpha=1e-4, update_average=False):
        self.axes = axes
        self.shape = [(1 if si in axes else s) for si,s in enumerate(shape)]
        self.alpha = alpha
        self.beta = theano.shared(value = np.zeros(self.shape, dtype=theano.config.floatX), name='beta')
        self.gamma = theano.shared(value = np.ones(self.shape, dtype=theano.config.floatX), name='gamma')
        self.epsilon = epsilon
        self.params = [self.beta, self.gamma]
        self.mean = theano.shared(value = np.zeros(self.shape, dtype=theano.config.floatX), name='mean')
        self.std = theano.shared(value = np.ones(self.shape, dtype=theano.config.floatX), name='std')
        self.update_average = update_average
        
    def __call__(self, input): 
        mean = input.mean(self.axes, keepdims=True) 
        std = input.std(self.axes, keepdims=True) + self.epsilon 

        # Don't batchnormalise a single data point
        mean = ifelse(T.gt(input.shape[0], 1), mean, T.zeros(mean.shape, dtype=mean.dtype))
        std  = ifelse(T.gt(input.shape[0], 1), std, T.ones(std.shape, dtype=std.dtype))

        if self.update_average:
            running_mean = theano.clone(self.mean, share_inputs=False)
            running_std  = theano.clone(self.std, share_inputs=False)

            running_mean.default_update = ((1 - self.alpha) * running_mean + self.alpha * mean)
            running_std.default_update  = ((1 - self.alpha) / std + self.alpha / std)

        return (input - mean) * T.addbroadcast((self.gamma / std) + self.beta, *self.axes)

    def eval(self, input):
        if self.update_average:
            mean = self.mean
            std = self.std
        else:
            mean = input.mean(self.axes, keepdims=True)
            std = input.std(self.axes, keepdims=True) + self.epsilon

        # Don't batchnoramlise a single data point
        mean = ifelse(T.gt(input.shape[0], 1), mean, T.zeros(mean.shape, dtype=mean.dtype))
        std  = ifelse(T.gt(input.shape[0], 1), std, T.ones(std.shape, dtype=std.dtype))

        return (input - mean) * T.addbroadcast((self.gamma / std) + self.beta, *self.axes)

    def inv(self, output): 
        return output
        
    def load(self, filename): pass
    def save(self, filename): pass
