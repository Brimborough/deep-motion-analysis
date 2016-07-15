import numpy as np
import timeit
import theano
import theano.tensor as T
import sys
import timeit

from ActivationLayer import ActivationLayer
from Network import AutoEncodingNetwork
from LadderNetwork import LadderNetwork
from Param import Param

from datetime import datetime

# To split between labeld & unlabeled examples
labeled    = lambda X, Y: X[T.nonzero(Y)[0]]
unlabeled  = lambda X, Y: X[T.nonzero(1.-T.sum(Y, axis=1))]
split_data = lambda X, Y: [labeled(X, Y), unlabeled(X, Y)]
join       = lambda X, Y: T.concatenate([X, Y], axis=0)

# Classification predictions
pred       = lambda Y: T.argmax(Y, axis=1)

class AdamTrainer(object):
    
    def __init__(self, rng, batchsize, epochs=100, alpha=0.000001, beta1=0.9, beta2=0.999, 
                 eps=1e-08, l1_weight=0.0, l2_weight=0.0, cost='mse'):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.rng = rng
        self.epochs = epochs
        self.batchsize = batchsize

        self.y_pred = lambda network, x: network(x)
        self.error, self.cost = self.create_cost_functions(cost=cost)

    def create_cost_functions(self, cost='mse'):
        # Where cost is always the cost which is minimised in supervised training
        # the T.nonzero term ensures that the cost is only calculated for examples with a label
        #
        # Convetion: We mark unlabelled examples with a vector of zeros in lieu of a one-hot vector

        if   cost == 'mse':
            error  = lambda network, x, y: T.mean((x - y)**2)
            cost   = lambda network, x, y: T.mean((x - y)**2)
        elif cost == 'binary_cross_entropy':
            cost   = lambda network, y_pred, y: T.nnet.binary_crossentropy(labeled(y_pred, y), labeled(y, y)).mean()
            # classification error (taking into account only training examples with labels)
            error  = lambda network, y_pred, y: T.mean(T.neq(pred(labeled(y_pred, y)), pred(labeled(y, y))))
        elif cost == 'cross_entropy':
            cost   = lambda network, y_pred, y: T.nnet.categorical_crossentropy(labeled(y_pred, y), labeled(y, y)).mean()
            # classification error (taking into account only training examples with labels)
            error  = lambda network, y_pred, y: T.mean(T.neq(pred(labeled(y_pred, y)), pred(labeled(y, y))))
        else:
            error  = lambda network, y_pred, y: T.zeros((1,))
            cost   = cost

        return error, cost

    def l1_regularization(self, network, target=0.0):
        return sum([T.mean(abs(p.value - target)) for p in network.params if (p.regularisable == True)])

    def l2_regularization(self, network, target=0.0):
        return sum([T.mean((p.value - target)**2) for p in network.params if (p.regularisable == True)])
        
    def get_cost_updates(self, network, input, output):
        
        y_pred = self.y_pred(network, input)
        cost = self.cost(network, y_pred, output) + self.l1_weight * self.l1_regularization(network) + \
                                                    self.l2_weight * self.l2_regularization(network)

        #overall_cost, repr_cost = self.cost(network, y_pred, output)
        #cost = overall_cost + self.l1_weight * self.l1_regularization(network) + \
        #                                            self.l2_weight * self.l2_regularization(network)
        error = None

        if (self.error):
            # Only meaningful in classification
            error = self.error(network, y_pred, output)

        updates = self.get_grad_updates(cost)

        return (cost, updates, error)

    def get_grad_updates(self, cost):
        param_values = [p.value for p in self.params]

        gparams = T.grad(cost, param_values)
        m0params = [self.beta1 * m0p + (1-self.beta1) *  gp     for m0p, gp in zip(self.m0params, gparams)]
        m1params = [self.beta2 * m1p + (1-self.beta2) * (gp*gp) for m1p, gp in zip(self.m1params, gparams)]
        params = [p - self.alpha * 
                  ((m0p/(1-(self.beta1**self.t[0]))) /
            (T.sqrt(m1p/(1-(self.beta2**self.t[0]))) + self.eps))
            for p, m0p, m1p in zip(param_values, m0params, m1params)]
        
        updates = ([( p,  pn) for  p,  pn in zip(param_values, params)] +
                   [(m0, m0n) for m0, m0n in zip(self.m0params, m0params)] +
                   [(m1, m1n) for m1, m1n in zip(self.m1params, m1params)] +
                   [(self.t, self.t+1)])

        return updates


    def get_eval_cost_error(self, network, input, output):
        
        y_pred = self.y_pred(network, input)
        cost   = self.cost(network, y_pred, output) + \
                 self.l1_weight * self.l1_regularization(network) + \
                 self.l2_weight * self.l2_regularization(network)

        error = None

        if (self.error):
            # Only meaningful in classification
            error = self.error(network, y_pred, output)
        
        return (cost, error)

    def get_hidden(self, network, input, depth):

        return network.get_hidden(input, depth)

    def get_predictions(self, network, input):

        return pred(self.y_pred(network, input)) + 1

    def get_representation(self, network, rep_input, depth):
        input = rep_input.type()
        index = T.lscalar()
        
        rep = self.get_hidden(network, input, depth)

        rep_func = theano.function(inputs=[index],
                                   outputs=[rep],
                                   givens={input:rep_input[index*self.batchsize:(index+1)*self.batchsize]},
                                   allow_input_downcast=True)

        #############################################
        # Calculating the representation at layer n #
        #############################################

        rep_batchinds = np.arange(rep_input.shape.eval()[0] // self.batchsize)

        # Will store the hidden representation
        rep_tensor = [rep_func(0)[0]]

        for bi in xrange(1, len(rep_batchinds)):
            rep_tensor.append(rep_func(bi)[0])

        #rep_tensor = np.array(rep_tensor)
        rep_tensor = np.squeeze(np.array(rep_tensor), axis=(1,))

        return rep_tensor

    def create_eval_func(self, network=None, eval_input=None, eval_output=None):
        if (None in [network, eval_input]):
            # Equivalent to not defining the function
            return None

        # Match batch index
        index  = T.lscalar()
        input  = eval_input.type()
        output = eval_output.type()

        cost, error = self.get_eval_cost_error(network, input, output)

        func = theano.function(inputs=[index],
                               outputs=[cost, error],
                               givens={input:eval_input[index*self.batchsize:(index+1)*self.batchsize],
                                       output:eval_output[index*self.batchsize:(index+1)*self.batchsize],},
                               allow_input_downcast=True)

        return func

    def create_train_func(self, network=None, train_input=None, train_output=None):
        if (None in [network, train_input, train_output]):
            # Equivalent to not defining the function
            return None

        # Match batch index
        index  = T.lscalar()
        input  = train_input.type()
        output = train_output.type()

        cost, updates, error = self.get_cost_updates(network, input, output)

        func = theano.function(inputs=[index], 
                               outputs=[cost, error], 
                               updates=updates, 
                               givens={input:train_input[index*self.batchsize:(index+1)*self.batchsize],
                                       output:train_output[index*self.batchsize:(index+1)*self.batchsize],}, 
                               allow_input_downcast=True)

        return func

    def eval(self, network, eval_input, eval_output, filename):

        eval_func = self.create_eval_func(network=network, eval_input=eval_input, eval_output=eval_output)

        # Resetting to the parameters with the best validation performance
        network.load(filename)

        ##############
        # Validation #
        ##############

        sys.stdout.write('... evaluating the model\n')

        eval_error, eval_cost = self.run_func(eval_func, eval_input, logging=False, epoch=0)

        sys.stdout.write(('Test set performance: %.2f %%\n\n') % (eval_error * 100.))
        sys.stdout.flush()

        return eval_error

    def run_func(self, func, func_input, logging=True, epoch=0):

        if (None == func):
            return np.inf, np.inf

        func_batchinds = np.arange(func_input.shape.eval()[0] // self.batchsize)
        self.rng.shuffle(func_batchinds)
        
        if (logging):
            sys.stdout.write('\n')
        
        tr_costs  = []
        tr_errors = []

        for bii, bi in enumerate(func_batchinds):
            tr_cost, tr_error = func(bi)

            # tr_error might be nan for a batch without labels in semi-supervised learning
            if np.isnan(tr_error):
                raise ValueError('Most recent error is nan')

            if np.isnan(tr_cost): 
                raise ValueError('Most recent cost is nan')

            tr_costs.append(tr_cost)
            tr_errors.append(tr_error)

            if (logging and (bii % (int(len(func_batchinds) / 1000) + 1) == 0)):
                sys.stdout.write('\r[Epoch %i]  %0.1f%% mean training error: %.5f, cost: %.5f' % (epoch, 100 * float(bii)/len(func_batchinds), np.mean(tr_errors), tr_cost))
                sys.stdout.flush()

        error_mean = np.mean(tr_errors)
        cost_mean = np.mean(tr_costs)

        return error_mean, cost_mean
        
    def train(self, network, train_input, train_output, valid_input=None, valid_output=None,
                             filename=None, logging=True):

        """ Conventions: For training examples with labels, pass a one-hot vector, 
                         otherwise a numpy array with zero values.
        """

        self.params = network.params
        param_values = [p.value for p in self.params]

        self.m0params = [theano.shared(np.zeros(p.shape.eval(), 
                         dtype=theano.config.floatX), borrow=True) for p in param_values]
        self.m1params = [theano.shared(np.zeros(p.shape.eval(), 
                         dtype=theano.config.floatX), borrow=True) for p in param_values]

        self.t = theano.shared(np.array([1], dtype=theano.config.floatX))


        train_func = self.create_train_func(network=network, train_input=train_input, 
                                            train_output=train_output)
        valid_func = self.create_eval_func(network=network, eval_input=valid_input, 
                                           eval_output=valid_output)

        ###############
        # TRAIN MODEL #
        ###############
        if (logging):
            sys.stdout.write('... training\n')
        
        best_epoch = 0

        best_train_cost  = np.inf
        best_train_error = np.inf
        best_valid_cost  = np.inf
        best_valid_error = np.inf

        last_train_error = 0.

        start_time = timeit.default_timer()

        for epoch in range(1, self.epochs+1):

            curr_train_error, train_cost = self.run_func(train_func, train_input, logging=True, epoch=epoch)
            diff_train_error, last_train_error = curr_train_error-last_train_error, curr_train_error

            output_str = '\r[Epoch %i] 100.0%% mean training error: %.5f training diff: %.5f ' % \
                         (epoch, curr_train_error, diff_train_error)

            valid_error, valid_cost = self.run_func(valid_func, valid_input, logging=False)

            if (valid_error is not np.nan):
                valid_diff = valid_error - best_valid_error
                output_str += ' validation error: %.5f validation diff: %.5f' % \
                              (valid_error, valid_diff)
                pass

            output_str += ' %s\n' % (str(datetime.now())[11:19])

            if (logging):
                sys.stdout.write(output_str)
                sys.stdout.flush()

            # Early stopping
            if (valid_func):
                if ((valid_error < best_valid_error) or
                    (valid_error == best_valid_error and valid_cost < best_valid_cost)):
                    best_valid_error = valid_error
                    best_valid_cost  = valid_cost
                    r_val = best_valid_error
                    best_epoch = epoch

                    network.save(filename)
                    result_str = 'Optimization complete. Best validation error of %.5f %% obtained at epoch %i\n' % (best_valid_error, best_epoch)

            elif ((curr_train_error < best_train_error) or 
                 (curr_train_error == best_train_error and train_cost < best_train_cost)):
                best_train_error = curr_train_error
                best_train_cost  = train_cost
                r_val = best_train_error
                best_epoch = epoch

                network.save(filename)
                result_str = 'Optimization complete. Best train error of %.5f %% obtained at epoch %i\n' % (best_train_error, best_epoch)

            else:
                pass

        end_time = timeit.default_timer()

        if (logging):
            sys.stdout.write(result_str)
            sys.stdout.write(('Training took %.2fm\n\n' % ((end_time - start_time) / 60.)))
            sys.stdout.flush()

        return r_val

    def predict(self, network, test_input, filename):
        input = test_input.type()
        index = T.lscalar()
        
        predictions = self.get_predictions(network, input)

        pred_func = theano.function(inputs=[index],
                                    outputs=[predictions],
                                    givens={input:test_input[index*self.batchsize:(index+1)*self.batchsize]},
                                    allow_input_downcast=True)

        #####################
        # Predicting labels #
        #####################

        sys.stdout.write('... predicting for new input\n')

        pred_batchinds = np.arange(test_input.shape.eval()[0] // self.batchsize)

        test_output = []
        for bii, bi in enumerate(pred_batchinds):
            test_output.append(pred_func(bi))

        np.savez_compressed(filename, test_output=np.array(test_output).flatten())

    def set_params(self, alpha=0.001, beta1=0.9, beta2=0.999, l1_weight=0.0, l2_weight=0.1):
        self.alpha=alpha; self.beta1=beta1; self.beta2=beta2; self.l1_weight=l1_weight
        self.l2_weight=l2_weight

class PreTrainer(AdamTrainer):
    """Implements greedy layerwise pre-training as discussed in [1].
    May be used for stacked autoencoders by setting input=output and
    defining an appropriate cost function.

    References:
        [1] Goodfellow, Ian et al. 
        "Deep Learning." 
        MIT Press, 2016
    """

    def __init__(self, rng, batchsize, epochs=100, alpha=0.001, 
                       beta1=0.9, beta2=0.999, eps=1e-08, 
                       l1_weight=0.0, l2_weight=0.1, cost='mse'):

        super(PreTrainer, self).__init__(rng, batchsize, epochs=epochs, alpha=alpha, 
                                         beta1=beta1, beta2=beta2, eps=eps, 
                                         l1_weight=l1_weight, l2_weight=l2_weight, 
                                         cost=cost)

    def pretrain(self, network=None, pretrain_input=None, filename=None, logging=False):

        if (None in [network, input]):
            raise ValueError('Received incorrect parameters')

        ###########################
        # Pretraining the network #
        ###########################

        activation_idx = [-1] + [i for i in xrange(len(network.layers)) if type(network.layers[i]) is ActivationLayer]

        # The layers leading to the last activation are not pretrained
        iterations = len(activation_idx) - 2

        finetuning_layers = list(network.layers[(activation_idx[-2]+1):])
        layer_stack = list(network.layers[:(activation_idx[-2]+1)])
        pretrained_layers = []

        sys.stdout.write('... pretraining\n\n')
        iteration = 0

        start_time = timeit.default_timer()

        for i in xrange(1, (iterations+1)):
            network.set_layers(layer_stack[activation_idx[i-1]+1:(activation_idx[i]+1)])

            inner_start_time = timeit.default_timer()

            cost = self.train(AutoEncodingNetwork(network), train_input=pretrain_input, 
                              train_output=pretrain_input, filename=None, logging=logging)

            inner_end_time = timeit.default_timer()

            sys.stdout.write('\r[Layer %i] 100.0%% training error: %.5f\n' % (iteration, cost))
            sys.stdout.write('\r[Layer %i] Training took: %.2fm\n' % (iteration, (inner_end_time - inner_start_time) / 60.))

            pretrained_layers += network.layers
            pretrain_input = self.get_representation(network, rep_input=pretrain_input, depth=len(network.layers)-1)
            pretrain_input = theano.shared(value=pretrain_input.astype(theano.config.floatX), borrow=True)
            iteration += 1

        end_time = timeit.default_timer()

        network.set_layers(pretrained_layers + finetuning_layers)
        network.save(filename)
        sys.stdout.write('Pretraining complete. Took %.2fm\n\n' % ((end_time - start_time) / 60.))
        sys.stdout.flush()
