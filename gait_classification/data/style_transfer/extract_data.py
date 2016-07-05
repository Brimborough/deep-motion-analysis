#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from tempfile import TemporaryFile

np.set_printoptions(threshold=np.nan)

# ['classes', 'clips']
data = np.load('data_styletransfer.npz')

#(Examples, Time frames, joints)
clips   = data['clips']
#(Motion, Styles)
classes = data['classes']

motions = {0:'fast_punching',
           1:'fast_walking',
           2:'jumping',
           3:'kicking',
           4:'normal_walking',
           5:'punching',
           6:'running',
           7:'transitions'}

styles = {0:'angry',
          1:'childlike',
          2:'depressed',
          3:'neutral',
          4:'old',
          5:'proud',
          6:'sexy',
          7:'strutting'}

#clips = np.swapaxes(clips, 1, 2)
#clips = clips[:,:-4]
#
#nb_datapoints = classes.shape[0]
#nb_attributes = clips.shape[1] * clips.shape[2]
#
### Convert to one-hot representation for classification
## ['fast_punching', 'fast_walking', 'jumping', 'kicking', 'normal_walking', 'punching', 'running', 'transitions']
#one_hot_motions = np.zeros([nb_datapoints, 8])
#one_hot_motions[np.arange(nb_datapoints), classes[:,0]] = 1
#
## ['angry', 'childlike', 'depressed', 'neutral', 'old', 'proud', 'sexy', 'strutting']
#one_hot_styles  = np.zeros([nb_datapoints, 8])
#one_hot_styles[np.arange(nb_datapoints), classes[:,1]] = 1
#
#with open('styletransfer_motions_one_hot.npz', 'w') as smo_f:
#    np.savez(smo_f, one_hot_vectors=one_hot_motions)
#
#with open('styletransfer_styles_one_hot.npz', 'w') as sso_f:
#    np.savez(sso_f, one_hot_vectors=one_hot_styles)

# Exploring labels

# This prints the number of motions for each style
#for style_id in xrange(8):
#    print 'Style %i' % style_id
#    elements = np.array([y for y in Y if y[1] == style_id])
#
#    c = Counter(elements[:,0])
#    print sorted(zip(c.keys(), c.values()))

# Prints the ids of distinct (style, motion) pairs
#y_old = None
#
#print 0
#print '%s %s' % (styles[classes[0][1]], motions[classes[0][1]])
#for y_id, y in enumerate(classes):
#    if (y_old is None):
#        y_old = y
#    elif ((y != y_old).any()):
#        print y_id
#        print '%s %s' % (styles[y[1]], motions[y[0]])
#    else:
#        pass
#
#    y_old = y
