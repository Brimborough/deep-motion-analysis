import matplotlib.pyplot as plt
import numpy as np

from sklearn import mixture
from sklearn import hmm
#from tools.utils import load_cmu

rng = np.random.RandomState(23455)

x = np.arange(10)
y = rng.normal(size=[10])
#
#print x 
#print y

#ax = plt.gca()
#ax.

plt.plot(x,y)

plt.xlabel('Number of components')
plt.ylabel('Data Log-Likelihood')
plt.show()

#train_set_x = load_cmu(rng)[0][0]
#
#max_components = 15
#
#for c in xrange(max_components):
#    gmm = mixture.GMM(n_components=c)
#
#    gmm.fit(train_set_x) 
#    #print gmm.predict([[ 0], [2], [9], [10]])
#    print gmm.predict_proba([[0], [2], [9], [10]])
