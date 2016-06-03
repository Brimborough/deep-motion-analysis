import numpy as np

X = np.load('../representation_learning/HiddenActivations.npz')['Orig']

# 1to1 array for counting
countNumZeros = np.zeros(len(X))
countPercent = np.zeros(len(X))

#Go through only inner matrix
for i in range(len(X)):
    # Go through from 1 to 17924 checking if any arr got a number
    countNumZeros[i] = not X[i].any()
    countPercent[i] = np.count_nonzero(X[i])


countNumZeros = countNumZeros.astype(int)

assert (np.sum(countNumZeros)==0)

print 'Percent not zero' + str(np.sum(countPercent).astype(float)/len(X))