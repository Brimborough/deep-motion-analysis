import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv, sigmoid
from theano.tensor.shared_randomstreams import RandomStreams

class LSTM1DTestLayer(object):
    
    def __init__(self, encoder, recoder, encode_igate, recode_igate, encode_fgate, recode_fgate, activation, dropout, initial, border_mode='clamp'):
        self.encoder = encoder
        self.recoder = recoder
        self.encode_igate = encode_igate
        self.recode_igate = recode_igate
        self.encode_fgate = encode_fgate
        self.recode_fgate = recode_fgate
        self.activation = activation
        self.dropout = dropout
        self.initial = initial
        self.border_mode = border_mode
        self.params = (
            self.encoder.params + 
            self.recoder.params + 
            self.encode_igate.params +
            self.recode_igate.params +
            self.encode_fgate.params +
            self.recode_fgate.params)
        
    def __call__(self, input):
        
        input = input.dimshuffle(2, 0, 1)
        initial = self.initial.dimshuffle(2, 0, 1)
        
        def step(e, h):
            ig = sigmoid(self.encode_igate(e) + self.recode_igate(h))
            fg = sigmoid(self.encode_fgate(e) + self.recode_fgate(h))
            return self.activation(fg * self.recoder(h) + ig * self.encoder(e))
        
        h = theano.scan(step, sequences=[input, initial], outputs_info=None)[0]
        
        return h.dimshuffle(1, 2, 0)
        
    def save(self, database, prefix=''):
        self.encoder.save(database, '%sRE_' % prefix)
        self.recoder.save(database, '%sRR_' % prefix)
        self.encode_igate.save(database, '%sRGei_' % prefix)
        self.recode_igate.save(database, '%sRGri_' % prefix)
        self.encode_fgate.save(database, '%sRGef_' % prefix)
        self.recode_fgate.save(database, '%sRGrf_' % prefix)
        
    def load(self, database, prefix=''):
        self.encoder.load(database, '%sRE_' % prefix)
        self.recoder.load(database, '%sRR_' % prefix)
        self.encode_igate.load(database, '%sRGei_' % prefix)
        self.recode_igate.load(database, '%sRGri_' % prefix)
        self.encode_fgate.load(database, '%sRGef_' % prefix)
        self.recode_fgate.load(database, '%sRGrf_' % prefix)
        
