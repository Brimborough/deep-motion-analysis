import numpy
import theano
import theano.tensor as T

from BatchNormLayer import BatchNormLayer
from BatchNormLayer import InverseBatchNormLayer

class LadderNetwork(object):
    """Implementation of the LadderNetwork as intoroduced in [1].
    
       References:
           [1] Rasmus, Antti, et al. "Semi-Supervised Learning with Ladder Networks." 
           Advances in Neural Information Processing Systems. 2015."""
    def __init__(self, **kw):

        if kw.get('encoding_layers', None) is None:
            self.encoding_layers = kw['encoding_layers']
        else:
            raise ValueError('Need to supply encoding layers')

        if kw.get('decoding_layers', None) is None:
            self.decoding_layers = kw['decoding_layers']
        else:
            raise ValueError('Need to supply decoding layers')

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

        # In order to batch-normalise in the decoder part
        self.bn = BatchNormLayer(shape=clean_pass.shape)
        self.params.append(self.bn.params)

        # classification
        self.predict = lambda y_pred: T.argmax(y_pred, axis=1)

        # As only hidden layers or conv layers encode units, we can simply
        # loop through all such layers and save the number of units
        n_units =  [l.output_units for l in self.decoding_layers if (l.output_units)]

        A = np.array([np.ones((i, 10), dtype=theano.config.floatX) for i in n_units])

        # Skip connection of the noisy input
        A.append(np.ones((self.encoding_layers[0].input_units, 10), dtype=theano.config.floatX))

        self.A = theano.shared(value=np.array(A), borrow=True)
        self.params.append(self.A)

        # See [1] eq. (2)
        # U is a tensor for MLPs, TODO: ConvNets
        U = T.dmatrix('U')
        Y = self.A * T.nnet.sigmoid(self.A*U.T + self.A) + self.A*U.T + self.A
        # This simultanously calculates the result of the mu and v functions used in eq. (2)
        self.compute_mu_v = theano.function([U], Y)

        # Implements the skip-connections
        noisy_Z = T.dmatrix('noisy_Z')
        mu_of_U = T.dmatrix('mu_of_U')
        v_of_U  = T.dmatrix('v_of_U')
        Z_reconstruction = (noisy_Z - mu_of_U) * v_of_U + mu_of_U
        self.g = theano.function([noisy_Z, mu_of_U, v_of_U], Z_reconstruction)

    def __call__(self, input):
        # Encoder part of the denoising autoencoder
        noisy_pass = self.noisy_prop(input)
        # To supply denoising targets and classification inputs
        clean_pass = self.clean_prop(input)
        self.predictions = self.predict(clean_pass)

        # Decoder part of the denoising autoencoder
        self.decode(noisy_pass)

        return clean_pass

    def clean_fprop(self, input):
        self.clean_z = []
        for layer in self.layers: 
            if (type(layer) is GaussianNoiseLayer):
                # Skip noise layer, add denoising target
                self.clean_z.append(clean_pass)
            else:
                clean_pass = layer(clean_pass)

    def noisy_fprop(self, input):
        self.noisy_z = []
        for layer in self.layers: 
            input = layer(input)
            # Assuming noise layers come directly before activations
            if (type(layer) is GaussianNoiseLayer):
                self.noisy_z.append(input)
        return input
    
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
        output = self.bn(output)

        # decoding
        for d_layer in self.decoding_layers[::-1]: 
            if (None is not d_layer.input_units):
                output = skip_connect(output)

            output = d_layer.inv(output)

        # Final skip-connection
        _ = skip_connect(output)

    def skip_connect(self, input):
        if ([] == noisy_Z):
            raise ValueError('Error: noisy_Z is an empty list')

        # Add input from the  skip-connection
        mu_v  = self.compute_mu_v(input)
        MU, V = np.split(mu_v, 5)

        reconstruction = self.g([self.noisy_z[0]], MU, V)
        # Non-trainable BN
        reconstruction = (reconstruction - np.mean(reconstruction, 0)) / np.std(reconstruction, 0) 
        # To caluclate the reconstruction error later
        self.reconstructions.append(reconstruction)
        self.noisy_z = self.noisy_z[1:]

        return reconstruction
    
    def save(self, filename):
        if filename is None: return
        for fname, layer in zip(filename, self.layers):
            layer.save(fname)
        
    def load(self, filename):
        if filename is None: return
        for fname, layer in zip(filename, self.layers):
            layer.load(fname)
