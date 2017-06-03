import sys
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from datetime import datetime

class AdamTrainer:
    
    def __init__(self, rng, batchsize, misc_cost, dec_cost, disc_cost, epochs=100, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-08, gamma=0.1):

        self.misc_alpha = alpha
        self.misc_beta1 = beta1
        self.misc_beta2 = beta2

        self.dec_alpha = alpha
        self.dec_beta1 = beta1
        self.dec_beta2 = beta2
        
        self.disc_alpha = alpha
        self.disc_beta1 = beta1
        self.disc_beta2 = beta2

        self.eps = eps
        self.gamma = gamma
        self.rng = rng
        self.theano_rng = RandomStreams(rng.randint(2 ** 30))
        self.epochs = epochs
        self.batchsize = batchsize

        self.misc_cost = misc_cost
        self.dec_cost = dec_cost
        self.disc_cost = disc_cost

        self.misc = lambda network, x: network(x)
        self.decoder = lambda network, x: network(x)
        self.discriminator = lambda network, x: network(x)

    def random_samples(self, n_input):
        return self.rng.uniform(size=(n_input, 128, 60), 
                low=-1.7, 
                high=1.7).astype(theano.config.floatX)

    def regularization(self, network, target=0.0):
        return sum([T.mean(abs(p - target)) for p in network.params]) / len(network.params)
        
    def get_cost_updates(self, misc_network, dec_network, disc_network, input_joint, input_control, output, rand_input):
        misc_cost = (self.misc_cost(misc_network, dec_network, input_joint, input_control, output) + self.gamma * self.regularization(misc_network))
        misc_gparams = T.grad(misc_cost, self.misc_params)

        misc_m0params = [self.misc_beta1 * m0p + (1-self.misc_beta1) *  gp     for m0p, gp in zip(self.misc_m0params, misc_gparams)]
        misc_m1params = [self.misc_beta2 * m1p + (1-self.misc_beta2) * (gp*gp) for m1p, gp in zip(self.misc_m1params, misc_gparams)]
        
        misc_params = [p - (self.misc_alpha / self.batchsize) * 
                  ((m0p/(1-(self.misc_beta1**self.misc_t[0]))) /
            (T.sqrt(m1p/(1-(self.misc_beta2**self.misc_t[0]))) + self.eps))
            for p, m0p, m1p in zip(self.misc_params, misc_m0params, misc_m1params)]
        
        updates = ([( p,  pn) for  p,  pn in zip(self.misc_params, misc_params)] +
                   [(m0, m0n) for m0, m0n in zip(self.misc_m0params, misc_m0params)] +
                   [(m1, m1n) for m1, m1n in zip(self.misc_m1params, misc_m1params)] +
                   [(self.misc_t, self.misc_t+1)])
        
        dec_cost = (self.dec_cost(misc_network, dec_network, disc_network, input_joint, input_control, output, rand_input) + self.gamma * self.regularization(dec_network))
        dec_gparams = T.grad(dec_cost, self.dec_params)

        dec_m0params = [self.dec_beta1 * m0p + (1-self.dec_beta1) *  gp     for m0p, gp in zip(self.dec_m0params, dec_gparams)]
        dec_m1params = [self.dec_beta2 * m1p + (1-self.dec_beta2) * (gp*gp) for m1p, gp in zip(self.dec_m1params, dec_gparams)]
        
        dec_params = [p - (self.dec_alpha / self.batchsize) * 
                  ((m0p/(1-(self.dec_beta1**self.dec_t[0]))) /
            (T.sqrt(m1p/(1-(self.dec_beta2**self.dec_t[0]))) + self.eps))
            for p, m0p, m1p in zip(self.dec_params, dec_m0params, dec_m1params)]
        
        updates += ([( p,  pn) for  p,  pn in zip(self.dec_params, dec_params)] +
                   [(m0, m0n) for m0, m0n in zip(self.dec_m0params, dec_m0params)] +
                   [(m1, m1n) for m1, m1n in zip(self.dec_m1params, dec_m1params)] +
                   [(self.dec_t, self.dec_t+1)])

        disc_cost = (self.disc_cost(misc_network, dec_network, disc_network, input_joint, input_control, output, rand_input) + self.gamma * self.regularization(disc_network))
        disc_gparams = T.grad(disc_cost, self.disc_params)

        disc_m0params = [self.disc_beta1 * m0p + (1-self.disc_beta1) *  gp     for m0p, gp in zip(self.disc_m0params, disc_gparams)]
        disc_m1params = [self.disc_beta2 * m1p + (1-self.disc_beta2) * (gp*gp) for m1p, gp in zip(self.disc_m1params, disc_gparams)]
        
        disc_params = [p - (self.disc_alpha / self.batchsize) * 
                  ((m0p/(1-(self.disc_beta1**self.disc_t[0]))) /
            (T.sqrt(m1p/(1-(self.disc_beta2**self.disc_t[0]))) + self.eps))
            for p, m0p, m1p in zip(self.disc_params, disc_m0params, disc_m1params)]
        
        updates += ([( p,  pn) for  p,  pn in zip(self.disc_params, disc_params)] +
                   [(m0, m0n) for m0, m0n in zip(self.disc_m0params, disc_m0params)] +
                   [(m1, m1n) for m1, m1n in zip(self.disc_m1params, disc_m1params)] +
                   [(self.disc_t, self.disc_t+1)])

        return (misc_cost, dec_cost, disc_cost, updates)
        
    def train(self, misc_network, dec_network, disc_network, input_motion_data, input_control_data, output_data, f_misc, f_dec):

        self.misc_network = misc_network
        self.dec_network = dec_network
        self.disc_network = disc_network
        
        input_motion = input_motion_data.type()
        input_control = input_control_data.type()
        output = output_data.type()

        index = T.lscalar()
        rand_input = T.tensor3()

        self.misc_params = misc_network.params
        self.misc_m0params = [theano.shared(np.zeros(p.shape.eval(), dtype=theano.config.floatX), borrow=True) for p in self.misc_params]
        self.misc_m1params = [theano.shared(np.zeros(p.shape.eval(), dtype=theano.config.floatX), borrow=True) for p in self.misc_params]
        self.misc_t = theano.shared(np.array([1], dtype=theano.config.floatX))
        
        self.dec_params = dec_network.params
        self.dec_m0params = [theano.shared(np.zeros(p.shape.eval(), dtype=theano.config.floatX), borrow=True) for p in self.dec_params]
        self.dec_m1params = [theano.shared(np.zeros(p.shape.eval(), dtype=theano.config.floatX), borrow=True) for p in self.dec_params]
        self.dec_t = theano.shared(np.array([1], dtype=theano.config.floatX))

        self.disc_params = disc_network.params
        self.disc_m0params = [theano.shared(np.zeros(p.shape.eval(), dtype=theano.config.floatX), borrow=True) for p in self.disc_params]
        self.disc_m1params = [theano.shared(np.zeros(p.shape.eval(), dtype=theano.config.floatX), borrow=True) for p in self.disc_params]
        self.disc_t = theano.shared(np.array([1], dtype=theano.config.floatX))
        
        misc_cost, dec_cost, disc_cost, updates = self.get_cost_updates(misc_network, dec_network, disc_network, input_motion, input_control, output, rand_input)

        train_func = theano.function(inputs=[index, rand_input], 
            outputs=[misc_cost, dec_cost, disc_cost], 
            updates=updates, 
            givens={
            input_motion:input_motion_data[index*self.batchsize:(index+1)*self.batchsize],
            input_control:input_control_data[index*self.batchsize:(index+1)*self.batchsize],
            output:output_data[index*self.batchsize:(index+1)*self.batchsize],
        }, allow_input_downcast=True)
        
        print('... training')

        misc_cost_mean = []
        dec_cost_mean = []
        disc_cost_mean = []

        for epoch in range(self.epochs):
                     
            batchinds = np.arange(input_motion_data.shape.eval()[0] // self.batchsize)
            self.rng.shuffle(batchinds)
            
            sys.stdout.write('\n')            
            misc_costs = []
            dec_costs = []
            disc_costs = []

            for bii, bi in enumerate(batchinds):

                rand_data = self.random_samples(self.batchsize)

                misc_cost_, dec_cost_, disc_cost_ = train_func(bi, rand_data)

                sys.stdout.write('\r[Epoch %i]   misc cost: %.5f   dec cost: %.5f   disc cost: %.5f' % (epoch, misc_cost_, dec_cost_, disc_cost_))
                sys.stdout.flush()

                misc_costs.append(misc_cost_)
                if np.isnan(misc_costs[-1]):
                    print "NaN in misc cost."
                    return

                dec_costs.append(dec_cost_)
                if np.isnan(dec_costs[-1]):
                    print "NaN in decoder cost."
                    return                    

                disc_costs.append(np.absolute(2-disc_cost_))
                if np.isnan(disc_costs[-1]):
                    print "NaN in discriminator cost."
                    return

            misc_cost_mean.append(np.mean(misc_costs))
            dec_cost_mean.append(np.mean(dec_costs))
            disc_cost_mean.append(np.mean(disc_costs))

            misc_network.save(f_misc)
            dec_network.save(f_dec)
