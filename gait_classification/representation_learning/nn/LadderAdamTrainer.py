import sys
import numpy as np
import timeit
import theano
import theano.tensor as T

from AdamTrainer import AdamTrainer
from ConvLadderNetwork import ConvLadderNetwork
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

class LadderAdamTrainer(AdamTrainer):
    """
    AdamTrainer for ladder networks (see [1]). This seperation into two classes is necessary due
    to the two-fold cost objective of the ladder network. This imlements a basic semi-supervised
    cost.

    References:
        [1] Rasmus, Antti, et al. "Semi-Supervised Learning with Ladder Networks." 
        Advances in Neural Information Processing Systems. 2015."""

    def __init__(self, rng, batchsize, epochs=100, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-08, 
                 l1_weight=0.0, l2_weight=0.1, supervised_cost='cross_entropy'): 

        # Initialises all components needed to calulate the supervised cost of the network
        super(LadderAdamTrainer, self).__init__(rng=rng, batchsize=batchsize, epochs=epochs,
                                                alpha=alpha, beta1=beta1, beta2=beta2, eps=eps, 
                                                l1_weight=l1_weight, l2_weight=l2_weight, cost=supervised_cost)

        self.lambdas = None
        
        # Will be used to calculate the unsupervised cost
        self.mse = lambda x, y: T.mean((x - y)**2)

    def unsupervised_cost(self, network):
        """
        Unsupervised cost is the sum of a weighted layerwise MSE.
        C = \sum_{l=0}^L \lambda_i ||\mathbf{z}^{(l)} - \hat{\mathbf{z}}^{(l)}_{\mathbf{BN}}||^2 
        """
        layerwise_mse = 0.
        # TODO: Theano's scan?
        # As the network's reconstructions are collected during the downward pass,
        # we must iterate through the list in revese to obtain the matching pairs

        for pair in zip(self.lambdas, network.clean_z, network.reconstructions):
            if (0.0 < pair[0]):
#               TODO: research effect of n_units
#                layerwise_mse += (pair[0] / pair[1]) * self.mse(pair[2], pair[3])
                layerwise_mse += pair[0] * self.mse(pair[1], pair[2])

        return layerwise_mse

    def get_cost_updates(self, network, input, output):

        if (type(network) is not LadderNetwork and (type(network) is not ConvLadderNetwork)):
            raise ValueError('Invalid argument: parameter network must be of type LadderNetwork')

        # predict with the clean version, calculate supervised cost with the noisy version
        coding_dist = network(input, output)
        y_pred      = network.predictions

        # supervised cost + regularisation
        cost = self.cost(network, coding_dist, output) + self.l1_weight * self.l1_regularization(network) + \
                                                         self.l2_weight * self.l2_regularization(network)

        # unsupervised cost
        us = self.unsupervised_cost(network)
        cost += us

        error = None

        if (self.error):
            # Only meaningful in classification
            error = self.error(network, y_pred, output)
        
        updates = self.get_grad_updates(cost)

        return (cost, updates, error, us)

    def train(self, network, lambdas, labeled_train_input, labeled_train_output,
                                      unlabeled_train_input, unlabeled_train_output,
                                      valid_input=None, valid_output=None,
                                      test_input=None, test_output=None, filename=None):

        """
        lambdas: Weight corresponding to the contribution of the mse for each pair of layers, given
                 bottom-to-top, i.e. lambdas[0] is the weight of the reconstruction error for the first
                 encoder and last decoder layer
        """

        self.lambdas = lambdas

        input = labeled_train_input.type()
        output = labeled_train_output.type()

        # Match batch index
        index = T.lscalar()
        
        self.params = network.params

        param_values = [p.value for p in self.params]
        self.m0params = [theano.shared(np.zeros(p.shape.eval(), 
                         dtype=theano.config.floatX), borrow=True) for p in param_values]
        self.m1params = [theano.shared(np.zeros(p.shape.eval(), 
                         dtype=theano.config.floatX), borrow=True) for p in param_values]

        self.t = theano.shared(np.array([1], dtype=theano.config.floatX))
        
        cost, updates, error, us = self.get_cost_updates(network, input, output)

        train_func = theano.function(inputs=[index], 
                                     outputs=[cost, error, us], 
                                     updates=updates, 
                                     givens={input:join(unlabeled_train_input[index*self.batchsize:(index+1)*self.batchsize], labeled_train_input),
                                             output:join(unlabeled_train_output[index*self.batchsize:(index+1)*self.batchsize], labeled_train_output),}, 
                                     allow_input_downcast=True)

#        train_func = theano.function(inputs=[index], 
#                                     outputs=[cost, error, us], 
#                                     updates=updates, 
#                                     givens={input:join(unlabeled_train_input[index*self.batchsize:(index+1)*self.batchsize], unlabeled_train_input[index*self.batchsize:(index+1)*self.batchsize]),
#                                             output:join(unlabeled_train_output[index*self.batchsize:(index+1)*self.batchsize], unlabeled_train_output[index*self.batchsize:(index+1)*self.batchsize])}, 
#                                     allow_input_downcast=True)

#        valid_func = None
#        if (valid_input):
#            # Full batch evaluation
#            valid_func = theano.function(inputs=[],
#                                         outputs=[cost, error],
#                                         givens={input:valid_input,
#                                                 output:valid_output,},
#                                         allow_input_downcast=True)

#        test_func = None
#        if (test_input):
#            # Full batch evaluation
#            test_func = theano.function(inputs=[],
#                                        outputs=[cost, error],
#                                        givens={input:test_input,
#                                                output:test_output,},
#                                        allow_input_downcast=True)

        ###############
        # TRAIN MODEL #
        ###############
        print('... training')
        
        best_valid_error = 1.
        best_epoch = 0

        last_tr_mean = 0.

        start_time = timeit.default_timer()

        for epoch in range(self.epochs):
            
            # For each batch of the unsupervised data, we show all labeled datapoints
            train_batchinds = np.arange(unlabeled_train_input.shape.eval()[0] // self.batchsize)
            self.rng.shuffle(train_batchinds)
            
            sys.stdout.write('\n')
            
            tr_costs  = []
            tr_errors = []
            tr_us = []
            for bii, bi in enumerate(train_batchinds):
                tr_cost, tr_error, us = train_func(bi)
                tr_us.append(us)

#                print us.shape
#                exit()

                # tr_error might be nan for a batch without labels in semi-supervised learning
                if not np.isnan(tr_error):
                    tr_errors.append(tr_error)

                tr_costs.append(tr_cost)
                if np.isnan(tr_costs[-1]): 
                    return
                if bii % (int(len(train_batchinds) / 1000) + 1) == 0:
                    sys.stdout.write('\r[Epoch %i]  %0.1f%% mean training error: %.5f unsupervised_cost: %.5f' % (epoch, 100 * float(bii)/len(train_batchinds), np.mean(tr_errors) * 100., np.mean(tr_us)))
                    sys.stdout.flush()

            curr_tr_mean = np.mean(tr_errors)
            diff_tr_mean, last_tr_mean = curr_tr_mean-last_tr_mean, curr_tr_mean

#            valid_cost, valid_error = valid_func()
#            valid_diff = valid_error - best_valid_error
            valid_error = 0.
            valid_diff  = 0.

            sys.stdout.write('\r[Epoch %i] 100.0%% mean training error: %.5f training diff: %.5f unsupervised_cost: %.5f validation error: %.5f validation diff: %.5f %s\n' % 
                (epoch, curr_tr_mean * 100., diff_tr_mean * 100., np.mean(tr_us), valid_error * 100., valid_diff * 100., str(datetime.now())[11:19]))
            sys.stdout.flush()

            # if we got the best validation score until now
#            if valid_error < best_valid_error:
#                best_valid_error = valid_error
#                best_epoch = epoch
#
#                # Only save the model if the validation error improved
#                # TODO: Don't add time needed to save model to training time
#                network.save(filename)

        end_time = timeit.default_timer()

        ####################
        # Final Validation #
        ####################

#        test_cost, test_error = test_func()
#
#        sys.stdout.write(('Optimization complete. Best validation score of %f %% '
#                          'obtained at epoch %i, with test performance %f %%\n') %
#                         (best_valid_error * 100., best_epoch + 1, test_error * 100.))
#        sys.stdout.flush()
#
#        sys.stdout.write(('Training took %.2fm\n' % ((end_time - start_time) / 60.)))
#        sys.stdout.flush()
