import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T
import timeit
import sys

from AdamTrainer import AdamTrainer
from datetime import datetime
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from theano.tensor.shared_randomstreams import RandomStreams
from nn.AnimationPlotLines import animation_plot

class AdversarialAdamTrainer(object):
    def __init__(self, rng, batchsize, 
                    gen_cost, disc_cost,
                    epochs=100, 
                    gen_alpha=0.00001, gen_beta1=0.85, gen_beta2=0.90, 
                    disc_alpha=0.00001, disc_beta1=0.85, disc_beta2=0.90, 
                    eps=1e-08, 
                    l1_weight=0.0, l2_weight=0.1, n_hidden_source = 100., mean=0, std=1):

        self.gen_alpha = gen_alpha
        self.gen_beta1 = gen_beta1
        self.gen_beta2 = gen_beta2

        self.disc_alpha = disc_alpha
        self.disc_beta1 = disc_beta1
        self.disc_beta2 = disc_beta2

        self.eps = eps
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.rng = rng
        self.theano_rng = RandomStreams(rng.randint(2 ** 30))
        self.epochs = epochs
        self.batchsize = batchsize
        self.n_hidden_source = n_hidden_source

        self.generator = lambda network, x: network(x)
        self.discriminator = lambda network, x: network(x)

        self.generator_cost = gen_cost
        self.discriminator_cost = disc_cost

    def randomize_uniform_data(self, n_input):
        return self.rng.uniform(size=(n_input, 800), 
                low=-np.sqrt(5, dtype=theano.config.floatX), 
                high=np.sqrt(5, dtype=theano.config.floatX)).astype(theano.config.floatX)

    def l1_regularization(self, network, target=0.0):
        return sum([T.mean(abs(p - target)) for p in network.params])

    def l2_regularization(self, network, target=0.0):
        return sum([T.mean((p - target)**2) for p in network.params])

    def get_cost_updates(self, gen_network, disc_network, input, gen_rand_input):

        gen_result = self.generator(gen_network, gen_rand_input)
        concat_gen_input = T.concatenate([gen_result, input], axis = 0)
        disc_result = self.discriminator(disc_network, concat_gen_input)

        # for mnist
        disc_fake_result = T.nnet.sigmoid(disc_result[:self.batchsize])
        disc_real_result = T.nnet.sigmoid(disc_result[self.batchsize:])   

        # generator update
        gen_cost = self.generator_cost(disc_fake_result)

        gen_param_values = [p.value for p in self.gen_params]
        gen_gparams = T.grad(gen_cost, gen_param_values)

        gen_m0params = [self.gen_beta1 * m0p + (1-self.gen_beta1) *  gp     for m0p, gp in zip(self.gen_m0params, gen_gparams)]
        gen_m1params = [self.gen_beta2 * m1p + (1-self.gen_beta2) * (gp*gp) for m1p, gp in zip(self.gen_m1params, gen_gparams)]

        gen_params = [p - self.gen_alpha * 
                  ((m0p/(1-(self.gen_beta1**self.gen_t[0]))) /
            (T.sqrt(m1p/(1-(self.gen_beta2**self.gen_t[0]))) + self.eps))
            for p, m0p, m1p in zip(gen_param_values, gen_m0params, gen_m1params)]

        updates = ([( p,  pn) for  p,  pn in zip(gen_param_values, gen_params)] +
                   [(m0, m0n) for m0, m0n in zip(self.gen_m0params, gen_m0params)] +
                   [(m1, m1n) for m1, m1n in zip(self.gen_m1params, gen_m1params)] +
                   [(self.gen_t, self.gen_t+1)])

        # discriminator update
        disc_cost = self.discriminator_cost(disc_fake_result, disc_real_result)

        disc_param_values = [p.value for p in self.disc_params]
        disc_gparams = T.grad(disc_cost, disc_param_values)

        disc_m0params = [self.disc_beta1 * m0p + (1-self.disc_beta1) *  gp     for m0p, gp in zip(self.disc_m0params, disc_gparams)]
        disc_m1params = [self.disc_beta2 * m1p + (1-self.disc_beta2) * (gp*gp) for m1p, gp in zip(self.disc_m1params, disc_gparams)]

        disc_params = [p - self.disc_alpha * 
                  ((m0p/(1-(self.disc_beta1**self.disc_t[0]))) /
            (T.sqrt(m1p/(1-(self.disc_beta2**self.disc_t[0]))) + self.eps))
            for p, m0p, m1p in zip(disc_param_values, disc_m0params, disc_m1params)]

        updates += ([( p,  pn) for  p,  pn in zip(disc_param_values, disc_params)] +
                   [(m0, m0n) for m0, m0n in zip(self.disc_m0params, disc_m0params)] +
                   [(m1, m1n) for m1, m1n in zip(self.disc_m1params, disc_m1params)] +
                   [(self.disc_t, self.disc_t+1)])

        return (gen_cost, disc_cost, updates)

    def train(self, gen_network, disc_network, train_input, filename=None):

        """ Conventions: For training examples with labels, pass a one-hot vector, otherwise a numpy array with zero values.
        """
        
        # variables to store parameters
        self.gen_network = gen_network

        input = train_input.type()
        
        # Match batch index
        index = T.lscalar()
        rand_input = T.matrix()
        
        self.gen_params = gen_network.params
        gen_param_values = [p.value for p in self.gen_params]

        self.gen_m0params = [theano.shared(np.zeros(p.shape.eval(), dtype=theano.config.floatX), borrow=True) for p in gen_param_values]
        self.gen_m1params = [theano.shared(np.zeros(p.shape.eval(), dtype=theano.config.floatX), borrow=True) for p in gen_param_values]
        self.gen_t = theano.shared(np.array([1], dtype=theano.config.floatX))

        self.disc_params = disc_network.params
        disc_param_values = [p.value for p in self.disc_params]

        self.disc_m0params = [theano.shared(np.zeros(p.shape.eval(), dtype=theano.config.floatX), borrow=True) for p in disc_param_values]
        self.disc_m1params = [theano.shared(np.zeros(p.shape.eval(), dtype=theano.config.floatX), borrow=True) for p in disc_param_values]
        self.disc_t = theano.shared(np.array([1], dtype=theano.config.floatX))

        gen_cost, disc_cost, updates = self.get_cost_updates(gen_network, disc_network, input, rand_input)

        train_func = theano.function(inputs=[index, rand_input], 
                                     outputs=[gen_cost, disc_cost], 
                                     updates=updates, 
                                     givens={input:train_input[index*self.batchsize:(index+1)*self.batchsize],}, 
                                     allow_input_downcast=True)

        ###############
        # TRAIN MODEL #
        ###############
        print('... training')
        
        best_epoch = 0
        last_tr_mean = 0.

        start_time = timeit.default_timer()

        gen_cost_mean = []
        disc_cost_mean = []

        for epoch in range(self.epochs):
            
            train_batchinds = np.arange(train_input.shape.eval()[0] // self.batchsize)
            self.rng.shuffle(train_batchinds)

            sys.stdout.write('\n')

            tr_gen_costs  = []
            tr_disc_costs = []
            for bii, bi in enumerate(train_batchinds):
                rand_data = self.randomize_uniform_data(self.batchsize)
                
                tr_gen_cost, tr_disc_cost = train_func(bi, rand_data)

                sys.stdout.write('\r[Epoch %i]   generative cost: %.5f   discriminative cost: %.5f' % (epoch, tr_gen_cost, tr_disc_cost))
                sys.stdout.flush()

                tr_gen_costs.append(np.absolute(1-tr_gen_cost))
                if np.isnan(tr_gen_costs[-1]):
                    print "NaN in generator cost."
                    return

                tr_disc_costs.append(np.absolute(0.5-tr_disc_cost))
                if np.isnan(tr_disc_costs[-1]):
                    print "NaN in discriminator cost."
                    return

                tr_gen_costs.append(tr_gen_cost)
                tr_disc_costs.append(tr_disc_cost)

            gen_network.save(filename)

            gen_cost_mean.append(np.mean(tr_gen_costs))
            disc_cost_mean.append(np.mean(tr_disc_costs))

        gen_plot, = plt.plot(gen_cost_mean, label='Gen. error')
        disc_plot, = plt.plot(disc_cost_mean, label='Disc. error')
        plt.legend(handles=[gen_plot, disc_plot,])
        plt.show()

        np.savez_compressed('gan_stats.npz', n_epochs=self.epochs, gen_cost_mean=gen_cost_mean, disc_cost_mean=disc_cost_mean)

        print "Finished training..."
