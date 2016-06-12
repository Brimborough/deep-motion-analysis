import numpy as np
import theano
import theano.tensor as T
from BatchNormLayer import BatchNormLayer
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
        - Look into
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

    '''
        TODO:
         - Stacked should just work, but might need another layer..
         - Xavier init - W = [input, hidden] U = [hidden,hidden] b = [hidden]
         - Mini-batches, check matrix multiplications and shapes of BNs
         - Add possibility of peepholes?
    '''
    def __init__(self, options, shape, rng, drop=0, zone_hidden=0, zone_cell=0, prefix="lstm",
                 bn=False, clip_gradients=False, mask=None):

        self.nsteps = shape
        self.mask = mask if mask is None else '' #TODO: Make mask
        self.prefix = prefix
        #TODO: Replace options and update the step function
        self.options = options
        self.clip_gradients = clip_gradients
        self.params = init_params(param_init_lstm(options=options, params=[], prefix=prefix))
        #TODO: Sort shapes, can have input,hidden for W, U = hidden,hidden, b = hidden
        # Saves upon changing code lots below.
        self.bninput = BatchNormLayer(None, shape) if bn else lambda x: x
        self.bnhidden = BatchNormLayer(None, shape) if bn else lambda x: x
        self.bncell = BatchNormLayer(None, shape) if bn else lambda x: x
        # Add BN params to layer (for SGD)
        if bn:
            self.params += self.bnhidden.params + self.bninput.params + self.bncell.params
        self.dropout = drop
        self.zoneout = {'h': zone_hidden, 'c': zone_cell}
        self.theano_rng = RandomStreams(rng.randint(2 ** 30))

    def __call__(self, input):

        # Clip grads here since we will use the same input later on everywhere.
        if self.clip_gradients is not False:
            input = clip_gradient(input, self.clip_gradients)

        # Helper coder to split
        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n * dim:(n + 1) * dim]
            return _x[:, n * dim:(n + 1) * dim]

        '''
            Step function for each part of the project
            Order of params: Sequences, Returned Values, Non-sequence data

            Implemented BN for LSTM like:
            @author Tim Cooijmans et al.
            @paper https://arxiv.org/pdf/1603.09025.pdf
            @year 2016

            Implemented dropout like:
            @author Stanislau Semeniuta et al.
            @paper https://arxiv.org/pdf/1603.05118.pdf
            @year 2016

            Implemented zoneout like:
            @author David Krueger et al.
            @paper http://arxiv.org/abs/1606.01305.pdf
            @year 2016

        '''
        def _step(m_, x_, h_, c_, dropout, zoneout, rng, prefix, bnh, bnc):
            # Initial dot product saving on computation
            preact = bnh(T.dot(h_, self.params[_pN(prefix, 'U')]))
            preact += x_

            i = T.nnet.sigmoid(_slice(preact, 0, self.options['dim_proj']))
            f = T.nnet.sigmoid(_slice(preact, 1, self.options['dim_proj']))
            o = T.nnet.sigmoid(_slice(preact, 2, self.options['dim_proj']))
            c = T.tanh(_slice(preact, 3, self.options['dim_proj']))
            if dropout > 0:
                d = (i * c * rng.binomial(
                    size=x_.shape, n=1, p=(1 - dropout),
                    dtype=theano.config.floatX))
                c = f * c_ + d
            # No reg..
            elif zoneout['c'] > 0:
                d = rng.binomial(
                    size=x_.shape, n=1, p=(1 - zoneout['c']),
                    dtype=theano.config.floatX)
                c = (d * c_) + ((1-d) * (f * c_ + i * c))
            else:
                c = f * c_ + i * c

            # Multiply by mask for different size inputs
            c = m_[:, None] * c + (1. - m_)[:, None] * c_

            if zoneout['h'] > 0:
                d = rng.binomial(
                    size=x_.shape, n=1, p=(1 - zoneout['h']),
                    dtype=theano.config.floatX)
                h = (h_*d) + ((1-d) * (o * T.tanh(bnc(c))))

            h = m_[:, None] * h + (1. - m_)[:, None] * h_
            # These get passed in the middle due to recursion. (IDK why in the middle)

            return h, c

        # Initial transform.
        input = (self.bninput(T.dot(input, self.params[_pN(self.prefix, 'W')])) +
                       self.params[_pN(self.prefix, 'b')])


        #TODO: Add n_samples which is the batch size!

        # Perform the actions - lots of non_sequences (hoping they save on overhead transfers but unsure...)
        hidden_outputs, updates = theano.scan(_step,
                                              sequences=[self.mask, input],
                                              outputs_info=[T.alloc(np.asarray((0.), dtype=T.config.floatX),
                                                                    self.n_samples,
                                                                    self.options['dim_proj']),
                                                            T.alloc(np.asarray((0.), dtype=T.config.floatX),
                                                                    self.n_samples,
                                                                    self.options['dim_proj'])],
                                              non_sequences=[self.dropout, self.zoneout,
                                                             self.theano_rng, self.prefix,
                                                             self.bnhidden, self.bncell],
                                              name=_pN(self.prefix, '_layers'),
                                              n_steps=self.nsteps)


        # outputs = hiddens.
        return hidden_outputs
