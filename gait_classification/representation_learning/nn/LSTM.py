import numpy as np
import theano
import theano.tensor as T
from theano.compat.python2x import OrderedDict
from theano.tensor.shared_randomstreams import RandomStreams

dtype=theano.config.floatX

# Function to help keep track of layer parameters, returns string
def _pN(param, layer_name):
    return '%s_%s' % (param, layer_name)


class GradClip(theano.compile.ViewOp):
    """
    Here we clip the gradients as Alex Graves does in his
    recurrent neural networks. In particular this prevents
    explosion of gradients during backpropagation.
    The original poster of this code was Alex Lamb,
    [here](https://groups.google.com/forum/#!topic/theano-dev/GaJwGw6emK0).
    """

    def __init__(self, clip_lower_bound, clip_upper_bound):
        self.clip_lower_bound = clip_lower_bound
        self.clip_upper_bound = clip_upper_bound
        assert(self.clip_upper_bound >= self.clip_lower_bound)

    def grad(self, args, g_outs):
        return [T.clip(g_out, self.clip_lower_bound, self.clip_upper_bound) for g_out in g_outs]


def clip_gradient(x, bound):
    grad_clip = GradClip(-bound, bound)
    try:
        T.opt.register_canonicalize(theano.gof.OpRemove(grad_clip), name='grad_clip_%.1f' % (bound))
    except ValueError:
        pass
    return grad_clip(x)

# Change this to Xavier init
def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(T.config.floatX)

#Alter this.
def param_init_lstm(options, params, prefix='lstm'):
    """
    Init the LSTM parameter:

    :see: init_params
    """

    # Weight matrix for multiplying with input
    W = np.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params[_pN(prefix, 'W')] = W
    # Weight matrix for multiplying with hidden activations
    U = np.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params[_pN(prefix, 'U')] = U
    # 4 Bias since we will use matrix multiplication to make it a lot nicer.
    b = np.zeros((4 * options['dim_proj'],))
    params[_pN(prefix, 'b')] = b.astype(T.config.floatX)

    return params

# Creates shared variables in order
def init_params(params):
    tparams = OrderedDict()
    for kk, pp in params.items():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams

class LSTM(object):


    def __init__(self, shape, options,rng,drop = 0, zone = 0, batch_size = 1, prefix="lstm", clip_gradients=False, mask=None):
        self.nsteps = shape
        self.n_samples = shape
        self.mask = mask
        self.clip_gradients = clip_gradients
        self.params = init_params(param_init_lstm(options=options, prefix=prefix))
        self.options = options
        self.Dropout = drop
        self.Zoneout = zone
        self.theano_rng = RandomStreams(rng.randint(2 ** 30))
        # No need to create the hidden and memory cells, used through recursion.

    def __call__(self, input):

        # Clip grads here since we will use the same input later on everywhere.
        if self.clip_gradients is not False:
            input = clip_gradient(input, self.clip_gradients)

            # Helper coder to split
            def _slice(_x, n, dim):
                if _x.ndim == 3:
                    return _x[:, :, n * dim:(n + 1) * dim]
                return _x[:, n * dim:(n + 1) * dim]

            # Step function for each part of the project
            # TODO: zoneout
            def _step(m_, x_, h_, c_):
                # Initial dot product saving on computation
                preact = T.dot(h_, self.params[_pN(self.prefix, 'U')])
                # Add
                preact += x_

                i = T.nnet.sigmoid(_slice(preact, 0, self.options['dim_proj']))
                f = T.nnet.sigmoid(_slice(preact, 1, self.options['dim_proj']))
                o = T.nnet.sigmoid(_slice(preact, 2, self.options['dim_proj']))
                c = T.tanh(_slice(preact, 3, self.options['dim_proj']))
                if self.Dropout > 0:
                    '''
                        Implemented dropout like:
                        @author Stanislau Semeniuta et al.
                        @paper https://arxiv.org/pdf/1603.05118.pdf
                        @year 2016
                    '''
                    d = (i * c * self.theano_rng.binomial(
                        size=x_.shape, n=1, p=(1 - self.Dropout),
                        dtype=theano.config.floatX))
                    c = f * c_ + d
                else:
                    c = f * c_ + i * c

                #Multiply by mask for different size inputs
                c = m_[:, None] * c + (1. - m_)[:, None] * c_
                h = o * T.tanh(c)
                h = m_[:, None] * h + (1. - m_)[:, None] * h_
                # RECURSIVE SO THESE WILL GET PASSED THROUGH
                return h, c
        # Perform the actions.
        self.rval, self.updates = theano.scan(_step,
                                              sequences=[self.mask, input],
                                              outputs_info=[T.alloc(np.asarray((0.), dtype=T.config.floatX),
                                                                    self.n_samples,
                                                                    self.options['dim_proj']),
                                                            T.alloc(np.asarray((0.), dtype=T.config.floatX),
                                                                    self.n_samples,
                                                                    self.options['dim_proj'])],
                                              name=_pN(self.prefix, '_layers'),
                                              n_steps=self.nsteps)


        # Can get output by doing something with rval[0]

        #How to return the output here????
        return input + self.b.dimshuffle('x', 0, 'x', 'x')

    #TODO: implement outputs
    #TODO: implement zoneout!
    #TODO: implement stacked