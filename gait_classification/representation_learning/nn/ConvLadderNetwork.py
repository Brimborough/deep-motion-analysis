import numpy as np
import operator
import sys
import theano
import theano.tensor as T

from ActivationLayer import ActivationLayer
from Conv2DLayer import Conv2DLayer
from HiddenLayer import HiddenLayer
from Param import Param

from theano.ifelse import ifelse
from theano.tensor.shared_randomstreams import RandomStreams

class ConvLadderNetwork(object):
    """Implementation of the LadderNetwork as introduced in [1].
       Partially based on: https://github.com/rinuboney/ladder/
    
    References:
        [1] Rasmus, Antti, et al. 
        "Semi-Supervised Learning with Ladder Networks." 
        Advances in Neural Information Processing Systems. 2015.
        [2] Ioffe, Sergey, and Christian Szegedy. 
        "Batch normalization: Accelerating deep network training by reducing internal covariate shift." 
        arXiv preprint arXiv:1502.03167 (2015)."""

    def __init__(self, encoding_layers, decoding_layers, rng, sigma):

        if (0 == len(encoding_layers)):
            raise ValueError('Need to supply encoding layers')
        self.encoding_layers = encoding_layers 

        if (0 == len(decoding_layers)):
            raise ValueError('Need to supply decoding layers')
        self.decoding_layers = decoding_layers 

        if (0.0 >= sigma):
            raise ValueError('Received sigma <= 0.0')
        self.sigma = sigma

        # Used to add noise
        self.theano_rng = RandomStreams(rng.randint(2 ** 30))

        # Trainable parameters
        self.params = sum([e_layer.params for e_layer in self.encoding_layers], [])
        self.params += sum([d_layer.params for d_layer in self.decoding_layers], [])

        # Used to seperate between labeled from unlabeled examples
        self.labeled     = lambda x, y: x[T.nonzero(y)[0]]
        self.unlabeled   = lambda x, y: x[T.nonzero(1.-T.sum(y, axis=1))]
#        self.unlabeled_y = lambda y: y[T.nonzero(1.-T.sum(y, axis=1))]
        self.split_data  = lambda x, y: [self.labeled(x, y), self.unlabeled(x, y)]
        # Where x, x2 must be of the same type
        self.join        = lambda x, x2: T.concatenate([x, x2], axis=0)

        # Classification predictions
        self.predictions = None

        # Used to calculate the unsupervised cost
        self.clean_z         = []
        self.reconstructions = []

        self.n_units = [l.output_units for l in self.encoding_layers if (hasattr(l, 'output_units'))]
        # Dimensionality of the input
        self.n_units.insert(0, self.encoding_layers[0].input_units)

        def setToOne(a, idx):
            a[..., idx] = 1.
            return theano.shared(value=a, borrow=True)

        # Concatentates a tuple and an integer, returns a list
        concat = lambda tup, i: list(tup) + [i]

        # Parameters needed to learn an optimal denoising function
        self.A       = [Param(setToOne(np.zeros(concat(i[1:], 10), dtype=theano.config.floatX), [0, 1, 6]), True) for i in self.n_units[::-1]]

        # Used during the computation of the optimal denoising function
        self.expand  = lambda A, o: A[:,:,:,o].dimshuffle('x', 0, 1, 2)

        # The dimensions used to calculate the moments for different layers
        self.mom_axes = [((0, 2, 3) if type(l) is Conv2DLayer else (0,)) for l in self.encoding_layers if (hasattr(l, 'output_units'))]

        # Parameters of trainable batchnorm layers
#        gamma_beta_shapes = [[(1 if si in self.mom_axes[shape_id] else s) for si, s in enumerate(shape)] for shape_id, shape in enumerate(self.n_units[1:])]
        gamma_beta_shapes = []

        for id, shape in enumerate(self.n_units[1:]):
            gamma_beta_shapes.append([(1 if si in self.mom_axes[id] else s) for si,s in enumerate(shape)])
                    
        # Parameters of trainable batchnorm layers
        self.gamma = [Param(theano.shared(value=np.ones(s, dtype=theano.config.floatX), borrow=True), True) for s in gamma_beta_shapes]
        self.beta  = [Param(theano.shared(value=np.zeros(s, dtype=theano.config.floatX), borrow=True), True) for s in gamma_beta_shapes]

        self.params += self.A
        self.params += self.gamma
        self.params += self.beta

    def __call__(self, input, output):
        """Calculates both a clean/noisy version of the encoder as well as the 
           corresponding denoising decoder. We pass both input and output to be able
           to split between labeled and unlabeled data points."""
        # 123, 127
        # Returns classification predictions and layerwise data
        noisy_y, noisy = self.fprop(input=input, output=output, sigma=self.sigma)

        # sigma = 0.0 -> no noise. Used to supply denoising targets
        self.predictions, clean = self.fprop(input=input, output=output, sigma=0.0)

        # Decoder part of the denoising autoencoder, pass only unlabeled data
        self.inv(self.unlabeled(noisy_y, output), noisy['unlabeled'], clean['unlabeled'])

        # Used to calculate the unsupervised cost
        self.clean_z = clean['unlabeled']['z_pre'].values()
        self.reconstructions.reverse()

        # predict with the clean version, calculate supervised cost with the noisy version
        return noisy_y

    def add_noise(self, input, sigma=0.0):
        # Where sigma=0.0 is equivalent to a clean pass
        if (sigma > 0.0):
            gaussian_noise = sigma * self.theano_rng.normal(size=input.shape, dtype=theano.config.floatX)
            input += gaussian_noise

        return input

    def batchnorm(self, input, axes=None, mean=None, std=None, eps=1e-10):
        """
            Performs batch-normalisation as proposed in [2]. Does not implement
            the trainable part including beta, gamma. This is done in beta_gamma.

            Parameters
            ----------
            input : Tensor 
                Minibatch to be normalised 
            mean : tensor
                std dev. of the incoming batch. Will be caclulated if not provided.
            std : tensor
                mean of the incoming batch. Will be caclulated if not provided.
            axes : tuple
                The axes along which moments of the input are calulated
            eps : float
                A constant used to avoid divisions by zero
        """
        # TODO: eval/training
        if (None == mean or None == std):
            if (None == axes):
                raise ValueError('No axes provided')

            mean = input.mean(axes, keepdims=True)
            std  = input.std(axes, keepdims=True)

        # Don't batchnoramlise a single data point
        mean = ifelse(T.gt(input.shape[0], 1), mean, T.zeros(mean.shape, dtype=mean.dtype))
        std  = ifelse(T.gt(input.shape[0], 1), std, T.ones(std.shape, dtype=std.dtype))

        return (input - mean) / (std + eps)

    def beta_gamma(self, input, p_index):
        """
            p_index : int
                Used as an index for the trainable parameters beta, gamma.
                If p_index = -1, we implement non-traininable bn
        """
        if (p_index > -1):
            # Trainable bn
            bn = self.gamma[p_index].value * (input + self.beta[p_index].value)

        return input

    def fprop(self, input, output, sigma=0.0):
        # Used to store pre-activations, activations, mean and std. dev of each layer
        d = {}
        d['labeled']   = {'z_pre': {}, 'mean': {}, 'std': {}}
        d['unlabeled'] = {'z_pre': {}, 'mean': {}, 'std': {}}

        input = self.add_noise(input, sigma)
        # 78
        d['labeled']['z_pre'][0], d['unlabeled']['z_pre'][0] = self.split_data(input, output)
        _, d['unlabeled']['z_pre'][0] = self.split_data(input, output)

        # Hidden layer id
        h_id = 0
        for layer in self.encoding_layers:
            if (type(layer) is ActivationLayer):
                # labeled & unlabeled input
                l_input, u_input = self.split_data(input, output)

                # 82
                u_mean = u_input.mean(self.mom_axes[h_id], keepdims=True)
                u_std  = u_input.std(self.mom_axes[h_id], keepdims=True)

                # Batch-normalise unlabeled and labeled examples seperatly 110/84-96
                # 90-91
                l_input = self.batchnorm(l_input, axes=self.mom_axes[h_id])
                u_input = self.batchnorm(u_input, axes=self.mom_axes[h_id], mean=u_mean, std=u_std)

                # 91
                input = self.add_noise(self.join(l_input, u_input), sigma)
                # 114: TODO: ReLUs don't need beta, gamma
                input = self.beta_gamma(input, h_id)

                # Re-used during decoding (118-119)
                d['labeled']['z_pre'][h_id+1]   = l_input 
                d['unlabeled']['z_pre'][h_id+1] = u_input
                # statistics only for unlabeled examples 
                d['unlabeled']['mean'][h_id+1] = u_mean
                d['unlabeled']['std'][h_id+1]  = u_std

                h_id += 1

            input = layer(input)

        return input, d

    def compute_mu(self, input, A, offset=0, conv=False):
        # Implements eq. (2) in [1] 
        o = offset

        if (conv):
            rval = self.expand(A, 0+o) * T.nnet.sigmoid(input * self.expand(A, 1+o) + self.expand(A, 2+o)) + \
                   input * self.expand(A, 3+o) + self.expand(A, 4+o)
        else:
            rval = A[:,0+o] * T.nnet.sigmoid(input*A[:,1+o] + A[:,2+o]) + input*A[:,3+o] + A[:,4+o]

        return rval

    def compute_v(self, input, A, conv=False):
        return self.compute_mu(input, A, offset=5, conv=conv)
    
    def inv(self, noisy_y, u_noisy, u_clean):
        # Non batch-normalised estimates/resconstructions
        z_est = {}

        u = noisy_y
        n_g = np.sum([1 for l in self.decoding_layers if (type(l) is Conv2DLayer) or (type(l) is HiddenLayer)])
        l_id = 0

        # decoding
        for d_index in xrange(0, len(self.decoding_layers) + 1):

            if(d_index > 0):
#                u = d_layer.inv(z_est[d_index-1])
                d_layer = self.decoding_layers[d_index-1]
                u = d_layer.inv(u)

            if ((d_index == 0) or (type(d_layer) is Conv2DLayer) or 
                                  (type(d_layer) is HiddenLayer)):

                if (d_index == 0):
                    axes = (0,)
                else:
                    axes = self.mom_axes[n_g]

                u = self.batchnorm(u, axes)
                conv = True if (len(axes) > 1) else False

                # Computations performed in the encoder
                noisy_z = u_noisy['z_pre'][n_g]
                mean    = u_clean['mean'].get(n_g, None)
                std     = u_clean['std'].get(n_g, None)

                # Add pre-activations from corresponding layer in the encoder
                z_est[l_id] = self.skip_connect(u, noisy_z, l_id, conv)
                u = z_est[l_id]

#                self.tmp = tmp

                # Used to calculate the denoising cost
                self.reconstructions.append(self.batchnorm(z_est[l_id], axes=axes, mean=mean, std=std))
                l_id += 1
                n_g  -= 1

    def skip_connect(self, u, noisy_z, d_index, conv):
        """ Implements the skip connections (g-function) in [1] eq. (2).
            These skip-connections release the pressure for higher layer to represent
            all information necessary for decoding.
        """
        mu = self.compute_mu(u, self.A[d_index].value, conv)
        v  = self.compute_v(u, self.A[d_index].value, conv)

#        z_est = (noisy_z - mu) * v + mu

        return noisy_z
#        return z_est
    
    def save(self, filename):
        if filename is None: return
        for fname, layer in zip(filename, self.encoding_layers + self.decoding_layers):
            layer.save(fname)
        
    def load(self, filename):
        if filename is None: return
        for fname, layer in zip(filename, self.encoding_layers + self.decoding_layers):
            layer.load(fname)
