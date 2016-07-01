import numpy as np
import operator
import sys
import theano
import theano.tensor as T
import timeit

from nn.AdamTrainer import AdamTrainer
from theano.ifelse import ifelse
from theano.tensor.shared_randomstreams import RandomStreams

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

    def set_range(self, alpha_range=[1e-10, 1.], beta1_range=[0.5, 1.], 
                        beta2_range=[0.5, 1.], l1_range=[0., 10.], l2_range=[0., 10.]):

        self.alpha_range = alpha_range
        self.beta1_range = beta1_range
        self.beta2_range = beta2_range
        self.l1_range    = l1_range   
        self.l2_range    = l2_range   

    def optimise(self, network, trainer, train_input, train_output,
                       valid_input=None, valid_output=None,
                       test_input=None, test_output=None, filename=None):

        size = (self.iterations,)

        sys.stdout.write('\r... sampling parameters\n')

        alpha_samples = self.rng.uniform(low=self.alpha_range[1], high=self.alpha_range[1], size=size)
        beta1_samples = self.rng.uniform(low=self.beta1_range[1], high=self.beta1_range[1], size=size)
        beta2_samples = self.rng.uniform(low=self.beta2_range[1], high=self.beta2_range[1], size=size)
        l1_samples    = self.rng.uniform(low=self.l1_range[1], high=self.l1_range[1], size=size)
        l2_samples    = self.rng.uniform(low=self.l2_range[1], high=self.l2_range[1], size=size)

        best_model_params = {}

        start_time = timeit.default_timer()
        best_model_performance = 1.

        for i in xrange(self.iterations):
            trainer.set_params(alpha=alpha_samples[i], beta1=beta1_samples[i], 
                               beta2=beta2_samples[i], l1_weight=l1_samples[i], 
                               l2_weight=l2_samples[i])

#            trainer.train(network=network, train_input=train_input, train_output=train_output,
#                                           valid_input=valid_input, valid_output=valid_output,
#                                           test_input=test_input, test_output=test_output)
            
            model_performance = self.rng.normal()
            sys.stdout.write('\r[Model %i] performance: %.5f\n' % (i+1, model_performance))
            sys.stdout.flush()

            if (model_performance < best_model_performance):
                best_model_params['alpha'] = alpha_samples[i] 
                best_model_params['beta1'] = beta1_samples[i]
                best_model_params['beta2'] = beta2_samples[i] 
                best_model_params['l1'] = l1_samples[i]
                best_model_params['l1'] = l2_samples[i]

                best_model_performance = model_performance

                # Only keep the best model
#                network.save(filename)

#            network.reset()

        end_time = timeit.default_timer()

        sys.stdout.write('Optimization complete. Best model achieves a performance of %.2f %%\n' % (best_model_performance))
        sys.stdout.write(('Training took %.2fm\n' % ((end_time - start_time) / 60.)))
        sys.stdout.flush()
