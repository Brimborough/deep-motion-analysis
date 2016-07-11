import numpy as np
import os, sys, inspect
sys.path.append('../../representation_learning/')

# Stick figures
from nn.AnimationPlotLines import animation_plot
from nn.Network import AutoEncodingNetwork
from nn.NoiseLayer import NoiseLayer
from nn.Pool1DLayer import Pool1DLayer

rng = np.random.RandomState(45425)

data = np.load('data_hdm05.npz')

X = data['clips'].swapaxes(1, 2)
X = X[:,:-4]

Y = data['classes']

classes = {-1:'no class',
           0:'cartwheel',
           1:'clap',
           2:'climb',
           3:'elbow_to_knee',
           4:'grab',
           5:'hop',
           6:'jog',
           7:'jump',
           8:'kick',
           9:'lie_down',
           10:'punch',
           11:'rotate_arms',
           12:'shuffle',
           13:'sit_down',
           14:'ski',
           15:'sneak',
           16:'squat',
           17:'stand_up',
           18:'throw',
           19:'turn',
           20:'walk_backward',
           21:'walk_forward',
           22:'walk_inlpace',
           23:'walk_left',
           24:'walk_right'}

#motion_id1 = np.random.randint(0, 3190)
#motion_id2 = motion_id1+1#np.random.randint(0, 3190)

motion_id1 = 1035
#motion_id2 = motion_id1+1#np.random.randint(0, 3190)
motion_id2 = 1025#np.random.randint(0, 3190)
    
X1 = X[motion_id1:motion_id1+1]
X2 = X[motion_id2:motion_id2+5]

print 'Left motion: %s, Id: %i' % (classes[Y[motion_id1]], motion_id1)
print 'Right motion: %s, Id: %i' % (classes[Y[motion_id2]], motion_id2)

animation_plot([X1, X2], filename='./animation.mp4' , interval=15.15)
#animation_plot([X1, X2],  interval=15.15)

