import numpy as np

X = np.load('../data/Joe/HiddenActivations.npz')['Orig']

# 1to1 array for counting
countNumZeros = np.zeros(len(X))
countNumZeros2 = np.zeros((len(X),len(X[0])))
countPercent = np.zeros(len(X))

print len(X)
print len(X[0])
#Go through only inner matrix
for i in range(len(X)):
    # Go through from 1 to 17924 checking if any arr got a number
    countNumZeros[i] = not X[i].any()
    for j in range(len(X[i])):
        countNumZeros2[i, j] = not X[i, j].any()
    countPercent[i] = np.count_nonzero(X[i])

countNumZeros = countNumZeros.astype(int)
countNumZeros2 = countNumZeros2.astype(int)

assert (np.sum(countNumZeros)==0)
print 'Percent not zero: ' + str(np.sum(countPercent).astype(float)/np.sum(X.shape[0]*X.shape[1]*X.shape[2]))

assert (np.sum(countNumZeros2)<10), 'There are ' + str(np.sum(countNumZeros2)) + ' zero arrays'
