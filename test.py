# test.py

import time
import numpy
import matplotlib.pyplot as plt
import seaborn as sns

from kantalope import *


X = numpy.concatenate(( numpy.random.randn(50000, 5000) * 0.5, numpy.random.randn(50000, 5000) * 0.5 + 2 ))
numpy.random.shuffle(X)

for i in range(1, 9):
	clf = Kantalope(2)
	tic = time.time()
	clf.fit(X, i)
	print time.time() - tic

'''
y = clf.predict(X)

plt.scatter( X[y==0, 0], X[y==0, 1], c='c', alpha=0.5, linewidth=0 )
plt.scatter( X[y==1, 0], X[y==1, 1], c='m', alpha=0.5, linewidth=0 )
plt.show()
'''