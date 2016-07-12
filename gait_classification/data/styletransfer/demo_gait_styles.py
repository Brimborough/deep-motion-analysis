import numpy as np
import os, sys, inspect
sys.path.append('../../representation_learning/')

# Stick figures
from nn.AnimationPlotLines import animation_plot
from nn.Network import AutoEncodingNetwork
from nn.NoiseLayer import NoiseLayer
from nn.Pool1DLayer import Pool1DLayer


rng = np.random.RandomState(45425)

# 31-38: Angry kicking
# 110-123: Childlike walking
# 451-468: Sexy walk

data = np.load('data_styletransfer.npz')

X = data['clips'].swapaxes(1, 2)
X = X[:,:-4]

Y = data['classes']

#preprocessed = np.load('styletransfer_preprocessed.npz')
#Xmean        = preprocessed['Xmean']
#Xmean        = Xmean.reshape(1,len(Xmean),1)
#Xstd         = preprocessed['Xstd']
#Xstd         = Xstd.reshape(1,len(Xstd),1)
#
#Xstd[np.where(Xstd == 0)] = 1
#
#X = (X - Xmean) / Xstd 

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

motion_id1 = np.random.randint(0, 559)
motion_id2 = np.random.randint(0, 559)
    
X1 = X[motion_id1:motion_id1+1]
X2 = X[motion_id2:motion_id2+1]

print 'Left motion: %s %s' % (styles[Y[motion_id1][1]], motions[Y[motion_id1][0]])
print 'Right motion: %s %s' % (styles[Y[motion_id2][1]], motions[Y[motion_id2][0]])

animation_plot([X1, X2], interval=15.15)

