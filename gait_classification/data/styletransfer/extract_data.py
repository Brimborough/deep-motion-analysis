#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

from collections import Counter


np.set_printoptions(threshold=np.nan)

# ['classes', 'clips']
data = np.load('data_styletransfer.npz')

#(Examples, Time frames, joints)
#clips   = data['clips']
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

# Convert to one-hot representation for classification
# 8 motions
# 8 styles
# 62 combinations (no instances of 'neutral fast punching' 
#                  or 'old fast punching') 
one_hot_motions  = np.eye(8)[classes[:,0]]
one_hot_styles   = np.eye(8)[classes[:,1]]

# Created list of unique classes
id_dict = {}
combined_classes = []

i = 0
for y in classes:
    if (not id_dict.has_key(str(y))):
        id_dict[str(y)] = i
        i += 1

    combined_classes.append(id_dict[str(y)])

one_hot_combined = np.eye(62)[combined_classes]

with open('styletransfer_one_hot.npz', 'w') as f:
    np.savez(f, motions=one_hot_motions, styles=one_hot_styles,
                combined=one_hot_combined)

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
