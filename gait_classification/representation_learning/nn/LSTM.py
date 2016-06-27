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

# Change this to Xavier init
def xavier_weight(in_dim, out_dim, rng):
    W_bound = np.sqrt(6. / (in_dim + out_dim))
    weights = np.asarray(
        rng.uniform(low=-W_bound, high=W_bound, size=(in_dim, out_dim)),
        dtype=theano.config.floatX)
    return weights.astype(T.config.floatX)

#Alter this.
def param_init_lstm(input, hidden, params, rng, prefix='lstm'):
    """
    Init the LSTM parameter, multiply some by 4 due to results presented in [Xavier10] suggest that you
                should use 4 times larger initial weights for sigmoid compared to tanh.
    :see: init_params
    """

    # Weight matrix for multiplying with input
    W = np.concatenate([4*xavier_weight(input, hidden, rng),
                        4*xavier_weight(input, hidden, rng),
                        4*xavier_weight(input, hidden, rng),
                        xavier_weight(input, hidden, rng)], axis=1)
    params[_pN(prefix, 'W')] = W
    # Weight matrix for multiplying with hidden activations
    U = np.concatenate([4*xavier_weight(hidden, hidden, rng),
                        4*xavier_weight(hidden, hidden, rng),
                        4*xavier_weight(hidden, hidden, rng),
                        xavier_weight(hidden, hidden, rng)], axis=1)
    params[_pN(prefix, 'U')] = U
    # 4 Bias since we will use matrix multiplication to make it a lot nicer.
    b = np.zeros((4 * hidden,))
    params[_pN(prefix, 'b')] = b.astype(T.config.floatX)

    return params

# Creates shared variables in order
def init_params(params, clip_gradients):
    tparams = OrderedDict()
    for kk, pp in params.items():
        tparams[kk] = theano.shared(params[kk], name=kk)
    # Extra node to allow for clipping of params, which ensures the params are clipped in comp graph.
    # Do it here so when we re-load params it's ok.
    clipped = OrderedDict()
    if clip_gradients:
        for k, p in tparams.items():
            clipped[k] = theano.gradient.grad_clip(tparams[k], -clip_gradients, clip_gradients)
    else:
        clipped = tparams
    return tparams, clipped

# Used for getting latest theano variables and saving them.
def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.items():
        new_params[kk] = vv.get_value()
    return new_params

class LSTM(object):

    """ TODO:
         - Mini-batches, check matrix multiplications and shapes of BNs
         - Masks
         - Add options for BN, then do if not None, that way we can solve them
         - Make self.params a func, less heavy on memory?
         - Toy tests of scan and backprop.
    """

    def __init__(self, input_shape, hidden_shape, rng, batch_size=1, drop=0, zone_hidden=0, zone_cell=0, prefix="lstm",
                 bn=None, clip_gradients=False, mask=None, backwards=False, return_seq=False, truncated=-1):
        '''
            Initilisation method, with lots of parameters because it is a small network in itself.

            :param input_shape: The shape for which the 1st dimension of the input weights will take.
            :type input_shape: int

            :param hidden_shape: The shape the rest of the weights will take.
            :type hidden_shape: int

            :param rng: a random number generator used to initialize weights
            :type rng: numpy.random.RandomState

            :param batch_size: The number of instances in a mini-batch
            :type batch_size: int

            :param drop: The dropout rate
            :type drop: float

            :param zone_cell: The zone-out rate for the memory cell
            :type zone_cell: float

            :param zone_hidden: The zone-out rate for the hidden inputs
            :type zone_hidden: float

            :param prefix: The prefix for which to fix to weights in the ordered dict
            :type prefix: string

            :param bn: The dictionary of set able options, i.e. epsilon for the batch norm layers.
                        In the form of layer_option, with layer being input, hidden or cell.
                        { layer_axes: type of list, layer_epsilon: float}
            :type bn: dictionary

            :param clip_gradients: To clip gradients
            :type clip_gradients: float

            :param mask: Mask for which to pad the inputs with (all ones in all same length)
            :type mask: Matrix conforming to {batchsize,max_seq}

            :param backwards: For reversing the LSTM to make the backward part of a LSTM
            :type backwards: Bool

            :param return_seq: Determines whether we return a sequence or just the final output
            :type return_seq: Bool

            :param truncated: A value for the truncated_gradients in the scan function
            :type truncated: signed int
        '''

        def createBN(bn, batch_size, input_shape, hidden_shape):
            data = OrderedDict()
            for i in ['input', 'hidden', 'cell']:
                if bn is not None:
                    axes = False
                    epsilon = False

                    if i == 'input':
                        shape = input_shape
                    else:
                        shape = hidden_shape

                    if i + '_axes' in bn:
                        axes = bn[i + '_axes']
                    if i + '_epsilon' in bn:
                        epsilon = bn[i + '_epsilon']

                    if axes and epsilon:
                        data[i] = BatchNormLayer(None, [batch_size, shape], axes=axes, epsilon=epsilon)
                    elif axes:
                        print axes
                        data[i] = BatchNormLayer(None, [batch_size, shape], axes=axes)
                    elif epsilon:
                        data[i] = BatchNormLayer(None, [batch_size, shape], epsilon=epsilon)
                    else:
                        data[i] = BatchNormLayer(None, [batch_size, shape])

                else:
                    data[i] = lambda x: x

            return data.values()

        def return_params(params, bn):
            if bn is not None:
                return list(params.values()) + self.bnhidden.params \
                                + self.bninput.params + self.bncell.params
            else:
                return list(params.values())

        self.backwards = backwards #BLSTM
        self.return_seq = return_seq #Return sequences
        self.mask = '' if mask is None else mask #TODO: Ensure people make masks....
        self.prefix = prefix
        self.truncated = truncated
        self.clip_gradients = clip_gradients
        self.batch_size = batch_size
        self.input_shape = input_shape #TODO Should this be batch size, since we shuffle dimensions
        self.bn = bn
        self.hidden_shape = hidden_shape
        self.shared_params, self.clipped_params = \
            init_params(param_init_lstm(input_shape, hidden_shape, {}, rng, prefix=prefix), clip_gradients)

        # Shapes, can have input,hidden for W, U = hidden,hidden, b = hidden
        # Use function to allow for the options to be set, makes saving and loading easier.
        self.bninput, self.bnhidden, self.bncell = createBN(bn, batch_size, input_shape, hidden_shape)

        # Function to save space in memory, otherwise we keep 2 parts.
        self.params = return_params(self.shared_params, bn)
        self.dropout = drop
        self.zoneout = {'h': zone_hidden, 'c': zone_cell}
        self.theano_rng = RandomStreams(rng.randint(2 ** 30))


    '''
        thoughts: - should we swap axes if coming from a conv? SHAPE: TS,BATCH,Vector Length
        - shape for BN
    '''
    def __call__(self, input):

        n_steps = input.shape[0] #Since the number of steps is always the first shape

        # For BLSTM
        if self.backwards:
            input = input[::-1]
            self.mask = self.mask[::-1] # Should move the mask backwards also.

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
        def _step(m_, x_, h_, c_, dropout, zoneout, rng, prefix, bnh, bnc, clipped):
            # Initial dot product saving on computation
            preact = bnh(T.dot(h_, clipped[_pN(prefix, 'U')]))
            preact += x_

            i = T.nnet.sigmoid(_slice(preact, 0, h_.shape[1]))  # Should be the size of the hidden nodes, hence shape[1]
            f = T.nnet.sigmoid(_slice(preact, 1, h_.shape[1]))
            o = T.nnet.sigmoid(_slice(preact, 2, h_.shape[1]))
            c = T.tanh(_slice(preact, 3, h_.shape[1]))
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

            # Multiply by mask for different size inputs, use old memory cells otherwise????
            c = m_[:, None] * c + (1. - m_)[:, None] * c_

            if zoneout['h'] > 0:
                d = rng.binomial(
                    size=x_.shape, n=1, p=(1 - zoneout['h']),
                    dtype=theano.config.floatX)
                h = (h_*d) + ((1-d) * (o * T.tanh(bnc(c))))
            else:
                o * T.tanh(bnc(c))

            h = m_[:, None] * h + (1. - m_)[:, None] * h_
            # These get passed in the middle due to recursion. (IDK why in the middle)

            return h, c

        # Initial transform.
        input = (self.bninput(T.dot(input, self.clipped_params[_pN(self.prefix, 'W')])) +
                 self.clipped_params[_pN(self.prefix, 'b')])


        # Perform the actions - lots of non_sequences (hoping they save on overhead transfers but unsure...)
        hidden_outputs, updates = theano.scan(_step,
                                              sequences=[self.mask, input],
                                              outputs_info=[T.alloc(np.asarray((0.), dtype=T.config.floatX),
                                                                    self.batch_size,
                                                                    self.hidden_shape),
                                                            T.alloc(np.asarray((0.), dtype=T.config.floatX),
                                                                    self.batch_size,
                                                                    self.hidden_shape)],
                                              non_sequences=[self.dropout, self.zoneout,
                                                             self.theano_rng, self.prefix,
                                                             self.bnhidden, self.bncell,
                                                             self.clipped_params],
                                              name=_pN(self.prefix, '_layers'),
                                              n_steps=n_steps,
                                              truncate_gradient=self.truncated)

        # For BLSTM
        if self.backwards:
            hidden_outputs = hidden_outputs[::-1]

        if not self.return_seq:
            hidden_outputs = hidden_outputs[-1]

        # outputs = hiddens.
        return hidden_outputs

    def inv(self, output): # Implement if you want, better to have Encoder-Decoder network though
        return output

    def load(self, filename):
        if filename is None: pass
        if not filename.endswith('.npz'): filename += '.npz'
        #Ensures all the expected params are there, dummy create
        params = param_init_lstm(1, 1, {}, np.random.RandomState(23455))
        pp = np.load(filename)
        for kk, vv in params.items():
            if kk not in pp:
                raise Warning('%s is not in the archive' % kk)
            params[kk] = pp[kk]
        #Make shared variables again
        init_params(params, self.clip_gradients)
        # load the params into the sub layers
        if self.bn:
            fn = filename + 'BN'
            self.bninput.load(fn + 'input')
            self.bncell.load(fn + 'cell')
            self.bnhidden.load(fn + 'hidden')

    def save(self, filename):
        if filename is None: pass
        params = unzip(self.shared_params)
        np.savez_compressed('filename', **params)
        # Save the params in the batch norm layer.
        if self.bn:
            fn = filename + 'BN'
            self.bninput.save(fn+'input')
            self.bncell.save(fn+'cell')
            self.bnhidden.save(fn+'hidden')



