import numpy as np
import os, sys, inspect
sys.path.append('../../representation_learning/')

# Stick figures
from nn.AnimationPlotLines import animation_plot

rng = np.random.RandomState(45425)

data = np.load('data_cmu_small.npz')

X = data['clips'].swapaxes(1, 2)
X = X[:,:-4]

motion_id1 = np.random.randint(0, 1000)
motion_id2 = np.random.randint(0, 1000)
    
X1 = X[motion_id1:motion_id1+1]
X2 = X[motion_id2:motion_id2+1]

print 'Left motion id: %i' % (motion_id1)
print 'Right motion id: %i' % (motion_id2)

animation_plot([X1, X2], interval=15.15)

