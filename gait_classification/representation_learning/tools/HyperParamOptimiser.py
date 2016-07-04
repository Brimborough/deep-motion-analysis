import numpy as np
import sys
#import theano
#import theano.tensor as T
import timeit

from copy import deepcopy

class HyperParamOptimiser(object):
    """Implements an optimisation method for hyperparameter optimisation
       using random search as explained in [1]
    
    References:
        [1] Bergstra, James, and Yoshua Bengio. 
        "Random search for hyper-parameter optimization." 
        Journal of Machine Learning Research 13.Feb (2012): 281-305. """

    def __init__(self, rng, iterations):
        self.rng = rng
        self.iterations = iterations

        # Initalise hyperparam ranges
        self.set_range()

    def set_range(self, alpha_range=[1e-5, 1e-5], beta1_range=[0.9, 0.9], 
                        beta2_range=[0.999, 0.999], l1_range=[0.0, 0.0], l2_range=[0.0, 0.0]):

        self.alpha_range = alpha_range
        self.beta1_range = beta1_range
        self.beta2_range = beta2_range
        self.l1_range    = l1_range   
        self.l2_range    = l2_range   

    def optimise(self, network, trainer, train_input, train_output,
                       valid_input=None, valid_output=None,
                       test_input=None, test_output=None, filename=None, logging=False):

        # TODO: The optimiser ought to ensure that the weights are intialised exactly
        # the same way for each iteration so as to not tempt the user to draw false conclusions

        size = (self.iterations,)

        sys.stdout.write('\r... sampling parameters\n')

        alpha_samples = self.rng.uniform(low=self.alpha_range[0], high=self.alpha_range[1], size=size)
        beta1_samples = self.rng.uniform(low=self.beta1_range[0], high=self.beta1_range[1], size=size)
        beta2_samples = self.rng.uniform(low=self.beta2_range[0], high=self.beta2_range[1], size=size)
        l1_samples    = self.rng.uniform(low=self.l1_range[0], high=self.l1_range[1], size=size)
        l2_samples    = self.rng.uniform(low=self.l2_range[0], high=self.l2_range[1], size=size)

        best_model_params = {}

        start_time = timeit.default_timer()
        best_model_performance = 1.

        for i in xrange(self.iterations):
            network_copy = deepcopy(network)
            trainer.set_params(alpha=alpha_samples[i], beta1=beta1_samples[i], 
                               beta2=beta2_samples[i], l1_weight=l1_samples[i], 
                               l2_weight=l2_samples[i])

            model_performance = trainer.train(network=network_copy, train_input=train_input, train_output=train_output,
                                              valid_input=valid_input, valid_output=valid_output,
                                              test_input=test_input, test_output=test_output, logging=logging)
            
            sys.stdout.write('\r[Model %i] performance: %.5f\n' % (i+1, model_performance))
            sys.stdout.flush()

            if (model_performance < best_model_performance):
                best_model_params['alpha'] = alpha_samples[i] 
                best_model_params['beta1'] = beta1_samples[i]
                best_model_params['beta2'] = beta2_samples[i] 
                best_model_params['l1'] = l1_samples[i]
                best_model_params['l2'] = l2_samples[i]

                best_model_performance = model_performance

                # Only keep the best model
#                network.save(filename)

#            TODO: resetting doesn't work for some reason, fix this
#            network.reset()

        end_time = timeit.default_timer()

        sys.stdout.write('Optimisation complete. Best model achieves a performance of %.2f %%\n' % (best_model_performance))
        sys.stdout.write(('Optimisation took %.2fm\n' % ((end_time - start_time) / 60.)))
        sys.stdout.flush()
