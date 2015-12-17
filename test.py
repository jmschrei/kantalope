# test.py

import time
import numpy
import matplotlib.pyplot as plt
import seaborn as sns

from kantalope import *

n = 50
X = numpy.concatenate(list( numpy.random.randn(10000, 50) * 0.5 + i for i in range(n)))
numpy.random.shuffle(X)

for i in range(1, 9):
	clf = Kantalope(n)
	tic = time.time()
	clf.fit(X, i)
	print time.time() - tic

'''
y = clf.predict(X)

plt.scatter( X[y==0, 0], X[y==0, 1], c='c', alpha=0.5, linewidth=0 )
plt.scatter( X[y==1, 0], X[y==1, 1], c='m', alpha=0.5, linewidth=0 )
plt.show()
'''