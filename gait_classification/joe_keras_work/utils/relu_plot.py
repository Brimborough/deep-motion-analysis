import matplotlib
import os
os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2012/bin/x86_64-darwin'
import matplotlib.pyplot as plt 
import numpy as np

plt.rc('text', usetex=True)
def f(x, alpha):
    return max(0,x) + alpha*(min(0,np.exp(x)-1))
 
def g(x, alpha):
    return max(0,x) + alpha*(min(0,x))

y = np.empty(np.arange(-5,5,0.1).shape[0])
for x in np.arange(-5,5,0.1):
    idx = ((x+5.1)*10)
    y[idx] = f(x,1.)
    
plt.plot(np.arange(-5,5,0.1),y, color='r', label=r'$ELU:\quad\quad \alpha = 1.0$')

y = np.empty(np.arange(-5,5,0.1).shape[0])
for x in np.arange(-5,5,0.1):
    idx = ((x+5.1)*10)
    y[idx] = g(x,.1)
    
plt.plot(np.arange(-5,5,0.1),y, color='b', label=r'$PReLU:\quad \alpha = 0.1$')

y = np.empty(np.arange(-5,5,0.1).shape[0])
for x in np.arange(-5,5,0.1):
    idx = ((x+5.1)*10)
    y[idx] = g(x,.5)
    
plt.plot(np.arange(-5,5,0.1),y, color='g', label=r'$PReLU:\quad \alpha = 0.5$')
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend(bbox_to_anchor=(0.33, 1.1), loc=2, borderaxespad=0.)
plt.savefig('relu_plots.pdf', format='pdf', dpi='1200')
plt.show()