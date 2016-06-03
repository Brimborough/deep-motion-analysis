import sys
import numpy as np
import timeit
import theano
import theano.tensor as T

from theano.tensor.shared_randomstreams import RandomStreams
from datetime import datetime

class AdamTrainer(object):
    
    def __init__(self, rng, batchsize, epochs=100, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-08, gamma=0.1, cost='mse'):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.gamma = gamma
        self.rng = rng
        self.theano_rng = RandomStreams(rng.randint(2 ** 30))
        self.epochs = epochs
        self.batchsize = batchsize
        if   cost == 'mse':
            self.cost = lambda network, x, y: T.mean((network(x) - y)**2)
        elif cost == 'binary_cross_entropy':
            self.y_pred = lambda network, x: network(x)
            self.error = lambda network, y_pred, y: T.mean(T.neq(T.argmax(y_pred, axis=1), T.argmax(y, axis=1)))
            self.cost   = lambda network, y_pred, y: T.nnet.binary_crossentropy(y_pred, y).mean()
        elif cost == 'cross_entropy':
            self.y_pred = lambda network, x: network(x)
            self.error = lambda network, y_pred, y: T.mean(T.neq(T.argmax(y_pred, axis=1), T.argmax(y, axis=1)))
            self.cost   = lambda network, y_pred, y: T.nnet.categorical_crossentropy(y_pred, y).mean()
        else:
            self.cost = cost
        
    def regularization(self, network, target=0.0):
        # L1 regularisation
        return sum([T.mean(abs(p - target)) for p in network.params]) / len(network.params)
        
    def get_cost_updates(self, network, input, output):
        
        y_pred = self.y_pred(network, input)
        cost = self.cost(network, y_pred, output) + self.gamma * self.regularization(network)
        error = None

        if (self.error):
            # Only meaningful in classification
            error = self.error(network, y_pred, output)
        
        gparams = T.grad(cost, self.params)
        m0params = [self.beta1 * m0p + (1-self.beta1) *  gp     for m0p, gp in zip(self.m0params, gparams)]
        m1params = [self.beta2 * m1p + (1-self.beta2) * (gp*gp) for m1p, gp in zip(self.m1params, gparams)]
        params = [p - (self.alpha) * 
                  ((m0p/(1-(self.beta1**self.t[0]))) /
            (T.sqrt(m1p/(1-(self.beta2**self.t[0]))) + self.eps))
            for p, m0p, m1p in zip(self.params, m0params, m1params)]
        
        updates = ([( p,  pn) for  p,  pn in zip(self.params, params)] +
                   [(m0, m0n) for m0, m0n in zip(self.m0params, m0params)] +
                   [(m1, m1n) for m1, m1n in zip(self.m1params, m1params)] +
                   [(self.t, self.t+1)])

        return (cost, updates, error)
        
    def train(self, network, train_input, train_output,
                             valid_input=None, valid_output=None,
                             test_input=None, test_output=None, filename=None):

        """
        Conventions: For training examples with labels, pass a one-hot vector, otherwise a numpy array with zero values.
        """
        
        input = train_input.type()
        output = train_output.type()
        # Match batch index
        index = T.lscalar()
        
        self.params = network.params
        self.m0params = [theano.shared(np.zeros(p.shape.eval(), dtype=theano.config.floatX), borrow=True) for p in self.params]
        self.m1params = [theano.shared(np.zeros(p.shape.eval(), dtype=theano.config.floatX), borrow=True) for p in self.params]
        self.t = theano.shared(np.array([1], dtype=theano.config.floatX))
        
        cost, updates, error = self.get_cost_updates(network, input, output)

        train_func = theano.function(inputs=[index], 
                                     outputs=[cost, error], 
                                     updates=updates, 
                                     givens={input:train_input[index*self.batchsize:(index+1)*self.batchsize],
                                             output:train_output[index*self.batchsize:(index+1)*self.batchsize],}, 
                                     allow_input_downcast=True)

        valid_func = None
        if (valid_input):
            # Full batch evaluation
#            valid_batchsize = valid_input.get_value().shape[0]
            valid_batchsize = self.batchsize

            valid_func = theano.function(inputs=[index],
                                         outputs=[cost, error],
                                         givens={input:valid_input[index*valid_batchsize:(index+1)*valid_batchsize],
                                                 output:valid_output[index*valid_batchsize:(index+1)*valid_batchsize],},
                                         allow_input_downcast=True)

        test_func = None
        if (test_input):
            # Full batch evaluation
#            test_batchsize = test_input.get_value().shape[0]
            test_batchsize = self.batchsize

            test_func = theano.function(inputs=[index],
                                        outputs=[cost, error],
                                        givens={input:test_input[index*test_batchsize:(index+1)*test_batchsize],
                                                output:test_output[index*test_batchsize:(index+1)*test_batchsize],},
                                        allow_input_downcast=True)

        ###############
        # TRAIN MODEL #
        ###############
        print('... training')
        
        best_valid_error = 1.
        best_epoch = 0

        last_tr_mean = 0.

        start_time = timeit.default_timer()

        for epoch in range(self.epochs):
            
            train_batchinds = np.arange(train_input.shape.eval()[0] // self.batchsize)
            self.rng.shuffle(train_batchinds)
            
            sys.stdout.write('\n')
            
            tr_costs  = []
            tr_errors = []
            for bii, bi in enumerate(train_batchinds):
                tr_cost, tr_error = train_func(bi)
                tr_costs.append(tr_cost)
                tr_errors.append(tr_error)
                if np.isnan(tr_costs[-1]): return
                if bii % (int(len(train_batchinds) / 1000) + 1) == 0:
                    sys.stdout.write('\r[Epoch %i]  %0.1f%% mean training error: %.5f' % (epoch, 100 * float(bii)/len(train_batchinds), np.mean(tr_errors) * 100.))
                    sys.stdout.flush()

            curr_tr_mean = np.mean(tr_errors)
            diff_tr_mean, last_tr_mean = curr_tr_mean-last_tr_mean, curr_tr_mean

            valid_batchinds = np.arange(valid_input.shape.eval()[0] // self.batchsize)

            vl_errors = []
            for bii, bi in enumerate(valid_batchinds):
                vl_cost, vl_error = valid_func(bi)
                vl_errors.append(vl_error)

            valid_error = np.mean(vl_errors)
            valid_diff = valid_error - best_valid_error

            sys.stdout.write('\r[Epoch %i] 100.0%% mean training error: %.5f training diff: %.5f validation error: %.5f validation diff: %.5f %s\n' % 
                (epoch, curr_tr_mean * 100., diff_tr_mean * 100., valid_error * 100., valid_diff * 100., str(datetime.now())[11:19]))
            sys.stdout.flush()

            # if we got the best validation score until now
            if valid_error < best_valid_error:
                best_valid_error = valid_error
                best_epoch = epoch

                # Only save the model if the validation error improved
                # TODO: Don't add time needed to save model to training time
                network.save(filename)

        end_time = timeit.default_timer()

        ####################
        # Final Validation #
        ####################

        test_batchinds = np.arange(test_input.shape.eval()[0] // self.batchsize)

        ts_errors = []
        for bii, bi in enumerate(test_batchinds):
            ts_cost, ts_error = test_func(bi)
            ts_errors.append(ts_error)

        test_error = np.mean(ts_errors)

        sys.stdout.write(('Optimization complete. Best validation score of %f %% '
                          'obtained at epoch %i, with test performance %f %%\n') %
                         (best_valid_error * 100., best_epoch + 1, test_error * 100.))
        sys.stdout.flush()

        sys.stdout.write(('Training took %.2fm\n' % ((end_time - start_time) / 60.)))
        sys.stdout.flush()

class LadderAdamTrainer(AdamTrainer):
    """
    AdamTrainer for ladder networks. This seperation into two classes is necessary due
    to the two-fold cost objective of the ladder network. 

    References:
        [1] Rasmus, Antti, et al. "Semi-Supervised Learning with Ladder Networks." 
        Advances in Neural Information Processing Systems. 2015."""

    def __init__(self, rng, batchsize, epochs=100, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-08, supervised_gamma=0.1, unsupervised_gamma=0.1, supervised_cost='cross_entropy'): 

        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.supervised_gamma   = supervised_gamma
        self.unsupervised_gamma = unsupervised_gamma
        self.rng = rng
        self.theano_rng = RandomStreams(rng.randint(2 ** 30))
        self.epochs = epochs
        self.batchsize = batchsize

        # This will not be callable outside of the constructor
#        def create_cost(self, cost, var_name):
#            if   cost == 'mse':
#                exec('self.%s_cost = lambda network, x, y: T.mean((network(x) - y)**2)'% (var_name))
#            elif cost == 'binary_cross_entropy':
#                exec('self.%s_y_pred = lambda network, x: network(x)'% (var_name))
#                exec('self.%s_error  = lambda network, y_pred, y: T.mean(T.neq(T.argmax(y_pred, axis=1), y))'% (var_name))
#                exec('self.%s_cost   = lambda network, y_pred, y: T.nnet.binary_crossentropy(y_pred, y).mean()'% (var_name))
#            elif cost == 'cross_entropy':
#                exec('self.%s_y_pred = lambda network, x: network(x)'% (var_name))
#                exec('self.%s_error  = lambda network, y_pred, y: T.mean(T.neq(T.argmax(y_pred, axis=1), y))'% (var_name))
#                exec('self.%s_cost   = lambda network, y_pred, y: T.nnet.categorical_crossentropy(y_pred, y).mean()'% (var_name))
#            else:
#                exec('self.%_cost = cost)'% (var_name))
#
#        create_cost(supervised_cost, 'supervised')
#        create_cost(unsupervised_cost, 'unsupervised')

        # Create the supervised cost functions. The term with T.nonzero selects only the training examples with a label.
        if   supervised_cost == 'mse':
            self.s_cost = lambda network, x, y: T.mean((network(x)[T.nonzero(y)] - y[T.nonzero(y)]**2))
        elif supervised_cost == 'binary_cross_entropy':
            self.y_pred = lambda network, x: network(x)
            self.error  = lambda network, y_pred, y: T.mean(T.neq(T.argmax(y_pred, axis=1), y))
            self.s_cost   = lambda network, y_pred, y: T.nnet.binary_crossentropy(y_pred[T.nonzero(y)], y[T.nonzero(y)]).mean()
        elif supervised_cost == 'cross_entropy':
            self.y_pred = lambda network, x: network(x)
            self.error  = lambda network, y_pred, y: T.mean(T.neq(T.argmax(y_pred, axis=1), y))
            self.s_cost   = lambda network, y_pred, y: T.nnet.categorical_crossentropy(y_pred[T.nonzero(y)], y[T.nonzero(y)]).mean()
        else:
            self.s_cost = supervised_cost
        
        # Will be used to calculate the unsupervised cost
        self.mse = lambda network, x, y: T.mean((network(x) - y)**2)

    def us_cost(self, network, lambdas):
        """
        Unsupervised cost is the sum of a weighted layerwise MSE. The weights are given by lambdas.
        C = \sum_{l=0}^L \lambda_i ||\mathbf{z}^{(l)} - \hat{\mathbf{z}}^{(l)}_{\mathbf{BN}}||^2 
        
        """

        layerwise_mse = 0.
        # TODO: Theano's scan?
        for pair in zip(lambdas, network.clean_z[::-1], network.reconstructions):
            layerwise_mse += pair[0] * self.mse(pair[1], pair[2])

        return layerwise_mse

    def get_cost_updates(self, network, input, output):
        
        y_pred = self.y_pred(network, input)
        # Only add this if input comes with a label
        cost = self.s_cost(network, y_pred, output) + self.supervised_gamma * super(LadderAdamTrainer, self).regularization(network)
        cost += self.us_cost(network) + self.unsupervised_gamma * super(LadderAdamTrainer, self).regularization(network)
        error = None

        if (self.error):
            # Only meaningful in classification
            error = self.error(network, y_pred, output)
        
        gparams = T.grad(cost, self.params)
        m0params = [self.beta1 * m0p + (1-self.beta1) *  gp     for m0p, gp in zip(self.m0params, gparams)]
        m1params = [self.beta2 * m1p + (1-self.beta2) * (gp*gp) for m1p, gp in zip(self.m1params, gparams)]
        params = [p - (self.alpha) * 
                  ((m0p/(1-(self.beta1**self.t[0]))) /
            (T.sqrt(m1p/(1-(self.beta2**self.t[0]))) + self.eps))
            for p, m0p, m1p in zip(self.params, m0params, m1params)]
        
        updates = ([( p,  pn) for  p,  pn in zip(self.params, params)] +
                   [(m0, m0n) for m0, m0n in zip(self.m0params, m0params)] +
                   [(m1, m1n) for m1, m1n in zip(self.m1params, m1params)] +
                   [(self.t, self.t+1)])

        return (s_cost, us_cost, updates, error)
        
    def train(self, network, train_input, train_output,
                             valid_input=None, valid_output=None,
                             test_input=None, test_output=None, filename=None):

        """
        Conventions: For training examples with labels, pass a one-hot vector, otherwise a numpy array with zero values.
        """
        super(LadderAdamTrainer, self).train(network, train_input, train_output, valid_input, valid_output, test_input, test_output, filename)
