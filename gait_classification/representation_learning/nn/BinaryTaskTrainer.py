import numpy as np
import timeit
import theano
import theano.tensor as T
import sys
import timeit

from AdamTrainer import AdamTrainer
from Network import AutoEncodingNetwork
from LadderNetwork import LadderNetwork
from Param import Param
from datetime import datetime

class BinaryTaskTrainer(AdamTrainer):
    """Implements a training algorithm for multi-task learning. While
    the optimisation algorithm is equivalent, the algorithm differs in
    the sense that the optimisation objective may be the sum of multiple
    individul cost functions.
    """

    def __init__(self, rng, batchsize, epochs=100, alpha=0.001, 
                       beta1=0.9, beta2=0.999, eps=1e-08, 
                       task1_weight=1., task2_weight=1., 
                       l1_weight=0.0, l2_weight=0.0, costs=[]):

        if (0 == len(costs)):
            raise ValueError('Must specify at least two cost functions.')

        super(BinaryTaskTrainer, self).__init__(rng, batchsize, epochs=epochs, alpha=alpha, 
                                         beta1=beta1, beta2=beta2, eps=eps, 
                                         l1_weight=l1_weight, l2_weight=l2_weight, 
                                         cost=costs[0])

        # Creates cost functions for each type of cost, tuples of (error, cost)
        # The first cost functions will be used during validation
        self.cost_funcs = [(self.error, self.cost)]
        self.cost_funcs.append(self.create_cost_functions(costs[1]))

        self.task1_weight = task1_weight
        self.task2_weight = task2_weight

    def get_cost_updates(self, network, input, output1, output2):

        network_outputs = self.y_pred(network, input)

        cost1 = self.task1_weight * self.cost_funcs[0][1](network, network_outputs[0], output1)
        cost2 = self.task2_weight * self.cost_funcs[1][1](network, network_outputs[1], output2)

        cost =  cost1 + cost2 + self.l1_weight * self.l1_regularization(network) + \
                self.l2_weight * self.l2_regularization(network)

        error = self.cost_funcs[0][0](network, network_outputs[0], output1)
        updates = self.get_grad_updates(cost)

        return (cost1, cost2, updates, error)

    def get_eval_cost_error(self, network, input, output):
        
        y_pred = self.y_pred(network, input)[0]
        cost   = self.cost(network, y_pred, output) + \
                 self.l1_weight * self.l1_regularization(network) + \
                 self.l2_weight * self.l2_regularization(network)

        error = None

        if (self.error):
            # Only meaningful in classification
            error = self.error(network, y_pred, output)
        
        return (cost, error)

    def create_train_func(self, network=None, train_input=None, train_outputs=None):
        if (None in [network, train_input, train_outputs]):
            # Equivalent to not defining the function
            return None

        # Match batch index
        index  = T.lscalar()
        input  = train_input.type()

        output1 = train_outputs[0].type()
        output2 = train_outputs[1].type()

        cost1, cost2, updates, error = self.get_cost_updates(network, input, output1, output2)

        func = theano.function(inputs=[index], 
                               outputs=[cost1, cost2, error], 
                               updates=updates, 
                               givens={input:train_input[index*self.batchsize:(index+1)*self.batchsize],
                                       output1:train_outputs[0][index*self.batchsize:(index+1)*self.batchsize],
                                       output2:train_outputs[1][index*self.batchsize:(index+1)*self.batchsize],}, 
                               allow_input_downcast=True)

        return func

    def train(self, network, train_input, train_outputs, 
                    valid_input=None, valid_output=None, filename=None, logging=True):

        """ Conventions: For training examples with labels, pass a one-hot vector, otherwise a numpy array with zero values.
        """
        
        self.params = network.params
        param_values = [p.value for p in self.params]

        self.m0params = [theano.shared(np.zeros(p.shape.eval(), 
                         dtype=theano.config.floatX), borrow=True) for p in param_values]
        self.m1params = [theano.shared(np.zeros(p.shape.eval(), 
                         dtype=theano.config.floatX), borrow=True) for p in param_values]

        self.t = theano.shared(np.array([1], dtype=theano.config.floatX))


        train_func = self.create_train_func(network=network, train_input=train_input, 
                                            train_outputs=train_outputs)
        valid_func = self.create_eval_func(network=network, eval_input=valid_input, 
                                           eval_output=valid_output)

        ###############
        # TRAIN MODEL #
        ###############
        if (logging):
            sys.stdout.write('... training\n')
        
        best_epoch = 0
        best_train_error = np.inf
        best_valid_error = np.inf

        last_tr_mean = 0.

        start_time = timeit.default_timer()

        for epoch in range(self.epochs):
            
            train_batchinds = np.arange(train_input.shape.eval()[0] // self.batchsize)
            self.rng.shuffle(train_batchinds)
            
            if (logging):
                sys.stdout.write('\n')
            
            tr_costs1 = []
            tr_costs2 = []

            tr_errors = []

            for bii, bi in enumerate(train_batchinds):
                tr_cost1, tr_cost2, tr_error = train_func(bi)

                if not np.isnan(tr_error):
                    tr_errors.append(tr_error)

                if (np.nan in [tr_cost1, tr_cost2]):
                    raise ValueError('Most recent training cost is nan')

                tr_costs1.append(tr_cost1)
                tr_costs2.append(tr_cost2)

                if (logging and (bii % (int(len(train_batchinds) / 1000) + 1) == 0)):
                    sys.stdout.write('\r[Epoch %i]  %0.1f%% mean training error: %.5f cost1: %.5f cost2: %.5f' % \
                                    (epoch, 100 * float(bii)/len(train_batchinds), np.mean(tr_errors), np.mean(tr_costs1), np.mean(tr_costs2)))
                    sys.stdout.flush()

            curr_tr_mean = np.mean(tr_errors)
            diff_tr_mean, last_tr_mean = curr_tr_mean-last_tr_mean, curr_tr_mean

            output_str = '\r[Epoch %i] 100.0%% mean training error: %.5f cost1: %.5f cost2: %.5f' % (epoch, np.mean(tr_errors), np.mean(tr_costs1), np.mean(tr_costs2))

            valid_error, valid_cost = self.run_func(valid_func, valid_input, logging=False)

            if (valid_error is not np.nan):
                valid_diff = valid_error - best_valid_error
                output_str += ' validation error: %.5f validation diff: %.5f' % \
                              (valid_error, valid_diff)

            output_str += ' %s\n' % (str(datetime.now())[11:19])

            if (logging):
                sys.stdout.write(output_str)
                sys.stdout.flush()

#            # Early stopping
#            if (valid_func and (valid_error < best_valid_error)):
#                best_valid_error = valid_error
#                r_val = best_valid_error
#                best_epoch = epoch
#
#                # TODO: Don't add time needed to save model to training time
#                network.save(filename)
#
#                result_str = 'Optimization complete. Best validation error of %.5f %% obtained at epoch %i\n' % (best_valid_error, best_epoch)
#            elif (curr_tr_mean < best_train_error):
#                best_train_error = curr_tr_mean
#                r_val = best_train_error
#                best_epoch = epoch
#
#                network.save(filename)
#                result_str = 'Optimization complete. Best train error of %.4f %% obtained at epoch %i\n' % (best_train_error, best_epoch)
#            else:
#                pass
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
