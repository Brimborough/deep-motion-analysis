import numpy as np
import sys
import theano
import theano.tensor as T

from BatchNormLayer import BatchNormLayer, InverseBatchNormLayer
from NoiseLayer import GaussianNoiseLayer

class LadderNetwork(object):
    """Implementation of the LadderNetwork as intoroduced in [1].
    # Example archtiecture with encoder and decoder
    # '->' symbolises a skip-connection
    #
    # Encoder:     Decoder:
    #
    # Activation -> Batchnorm
    #            -> g()
    # Noise       | Hiddenlayer
    # Batchnorm   | Batchnorm
    # Hiddenlayer | 
    # Activation  | 
    #            -> g()
    # Noise       | Hiddenlayer
    # Batchnorm   | Batchnorm
    # HiddenLayer | 
    # Noise      -> g()
    # Input       | 
    
       References:
           [1] Rasmus, Antti, et al. "Semi-Supervised Learning with Ladder Networks." 
           Advances in Neural Information Processing Systems. 2015."""
    def __init__(self, **kw):

        if kw.get('encoding_layers', None) is None:
            raise ValueError('Need to supply encoding layers')
        else:
            self.encoding_layers = kw['encoding_layers']

        if kw.get('decoding_layers', None) is None:
            raise ValueError('Need to supply decoding layers')
        else:
            self.decoding_layers = kw['decoding_layers']

        if kw.get('params', None) is None:
            self.params = sum([e_layer.params for e_layer in self.encoding_layers], [])
            self.params += sum([d_layer.params for d_layer in self.decoding_layers], [])
        else:
            self.params = kw.get('params', None)

        # Needed to calculate the reconstruction error.
        # z is the weighted input
        self.clean_z = []
        self.reconstructions = []

        # Used to pass on information through skip-connections
        self.noisy_z = []

        # classification
        self.predict = lambda y_pred: T.argmax(y_pred, axis=1)

        # As only hidden layers or conv layers encode units, we can simply
        # loop through all such layers and save the number of units
        n_units =  [l.output_units for l in self.decoding_layers if (hasattr(l, 'output_units'))]
        # Dimensionality of an input example 
        n_units.append(self.encoding_layers[1].input_units)

        # Parameters needed to learn an optimal denoising function
        self.As = [theano.shared(value=np.ones((i, 10), dtype=theano.config.floatX), borrow=True) for i in n_units]

        self.params += self.As

#        # U is the input from a downward pass in the decoder
#        U = T.dmatrix('U')
#        # Index for self.As
#        i = T.iscalar('i')
#        # Where o is used to switch between computation of mu and v
#        o = T.iscalar('offset')
#
#        # This is used to calculate the result of the mu and v functions used in eq. (2)
#        self.compute_mu_v = []
#
#        # TODO: At the moment, we have to create a function for each layer, fix this
#        for i in range(len(n_units)):
#            Y = self.As[i][:,0+o] * T.nnet.sigmoid(U*self.As[i][:,1+o] + self.As[i][:,2+o]) + U*self.As[i][:,3+o] + self.As[i][:,4+o]
#            self.compute_mu_v.append(theano.function([U, theano.In(o, value=0)], Y))

        # Implements a skip-connection
#        noisy_Z = T.dmatrix('noisy_Z')
#        mu_of_U = T.dmatrix('mu_of_U')
#        v_of_U  = T.dmatrix('v_of_U')
#        Z_reconstruction = (noisy_Z - mu_of_U) * v_of_U + mu_of_U
#        self.g = theano.function([noisy_Z, mu_of_U, v_of_U], Z_reconstruction)

        # Used to batch-normalise before running the decoder
        self.bn = BatchNormLayer((1, n_units[0]))
        self.params += self.bn.params

    def __call__(self, input):
        # Encoder part of the denoising autoencoder
        noisy_pass = self.noisy_fprop(input)
        # To supply denoising targets and classification inputs
        clean_pass = self.clean_fprop(input)
        # TODO: the original paper uses the noisy pass here - why?
        self.predictions = self.predict(clean_pass)

        # Decoder part of the denoising autoencoder
        self.inv(self.bn(noisy_pass))

        return clean_pass

    def clean_fprop(self, input):
        self.clean_z = []
        for layer in self.encoding_layers:
            if (type(layer) is GaussianNoiseLayer):
                # Skip noise layer, add denoising target
                self.clean_z.append(input)
            else:
                input = layer(input)

        return input

    def noisy_fprop(self, input):
        self.noisy_z = []

        for layer in self.encoding_layers: 
            input = layer(input)
            # Assuming noise layers come directly before activations
            if (type(layer) is GaussianNoiseLayer):
                self.noisy_z.append(input)

        return input

    def compute_mu(self, input, A, offset=0):
        # Implements eq. (2) in [1] 
        o = offset
        return A[:,0+o] * T.nnet.sigmoid(input*A[:,1+o] + A[:,2+o]) + input*A[:,3+o] + A[:,4+o]

    def compute_v(self, input, A):
        return self.compute_mu(input, A, offset=5)
    
    def inv(self, output):

        # Example archtiecture with encoder and decoder
        # '->' symbolises a skip-connection
        #
        # Encoder:     Decoder:
        #
        # Activation -> Batchnorm
        #            -> g()
        # Noise       | Conv2D
        # Batchnorm   | Batchnorm
        # Conv2D      | 
        # Activation  | 
        #            -> g()
        # Noise       | Conv2D
        # Batchnorm   | Batchnorm
        # Conv2D      | 
        # Noise      -> g()
        # Input       | 

        self.reconstructions = []
        decoding_index = 0

        # decoding
        for layer_index, d_layer in enumerate(self.decoding_layers): 
            output = d_layer.inv(output)

            if (type(d_layer) is BatchNormLayer):
                # Apply skip connections after batchnorm layers
                output = self.skip_connect(output, decoding_index)
                decoding_index += 1

        # Final skip-connection
        _ = self.skip_connect(output, decoding_index)

    def skip_connect(self, input, layer_index):
        if ([] == self.noisy_z):
            raise ValueError('Error: noisy_z is an empty list')

        MU = self.compute_mu(input, self.As[layer_index])
        V  = self.compute_v(input, self.As[layer_index])

        reconstruction = (self.noisy_z[-1] - MU) * V + MU

        # Non-trainable BN
        reconstruction = (reconstruction - reconstruction.mean(0)) / reconstruction.std(0) 
        # To caluclate the reconstruction error later
        self.reconstructions.append(reconstruction)
        self.noisy_z = self.noisy_z[0:-1]

        return reconstruction
    
    def save(self, filename):
        if filename is None: return
        for fname, layer in zip(filename, self.encoding_layers + self.decoding_layers):
            layer.save(fname)
        
    def load(self, filename):
        if filename is None: return
        for fname, layer in zip(filename, self.encoding_layers + self.decoding_layers):
            layer.load(fname)
