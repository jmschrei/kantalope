#cython: boundscheck=False
#cython: cdivision=True
# model.pyx
# Author: Jacob Schreiber <jmschr@cs.washington.edu>

'''
Lock-Free Parallel K-means using Stochastic Descent.
'''

from cython.parallel import prange

import numpy
cimport numpy

cdef double inf = float("inf")

cdef class Kantalope( object ):
	"""Kantalope kmeans estimator."""

	cdef int k
	cdef public numpy.ndarray centroids_ndarray
	cdef numpy.ndarray sums_ndarray
	cdef numpy.ndarray weights_ndarray
	cdef double* centroids
	cdef double* sums
	cdef double* weights

	def __init__( self, k ):
		self.k = k

	cpdef fit( self, numpy.ndarray X, int nthreads=1 ):
		"""Fit to the data using random initializations."""

		cdef int n = X.shape[0]
		cdef int d = X.shape[1]
		cdef int k = self.k

		self.centroids_ndarray = numpy.zeros((k, d), dtype=numpy.float64)
		self.centroids = <double*> self.centroids_ndarray.data

		self.sums_ndarray = numpy.zeros((k, d), dtype=numpy.float64)
		self.sums = <double*> self.sums_ndarray.data

		self.weights_ndarray = numpy.zeros((k, d), dtype=numpy.float64)
		self.weights = <double*> self.weights_ndarray.data

		cdef double* X_data = <double*> X.data

		with nogil:
			self._fit( X_data, n, d, nthreads )

	cdef void _fit( Kantalope self, double* X, int n, int d, int nthreads ) nogil:
		"""Cython inner loop."""

		cdef int i, j, k, m, min_centroid
		cdef double distance, min_distance

		for i in range(self.k):
			for j in range(d):
				m = j + i*d
				self.centroids[m] = X[m]
		
		for i in prange(n, num_threads=nthreads, schedule='guided'):
			k = self._predict( X + i*d, d )

			for j in range(d):
				m = k*d + j

				self.sums[m] += X[i*d + j]
				self.weights[m] += 1

				self.centroids[m] = self.sums[m] / self.weights[m]

	cpdef predict( self, numpy.ndarray X, int nthreads=1 ):
		"""Predict nearest centroid, python wrapper."""

		cdef int i
		cdef int n = X.shape[0]
		cdef int d = X.shape[1]

		cdef numpy.ndarray y_ndarray = numpy.zeros(n, dtype=numpy.int32)
		cdef int* y = <int*> y_ndarray.data
		cdef double* X_data = <double*> X.data

		for i in prange(n, nogil=True, num_threads=nthreads):
			y[i] = self._predict( X_data + i*d, d )

		return y_ndarray

	cdef int _predict( Kantalope self, double* X, int d ) nogil:
		"""Predict the nearest centroid."""

		cdef int j, k, min_centroid
		cdef double distance, min_distance

		min_distance = inf
		min_centroid = -1

		for k in range(self.k):
			distance = 0.0

			for j in range(d):
				distance += ( X[j] - self.centroids[k*d + j] ) ** 2.0

			if distance < min_distance:
				min_distance = distance
				min_centroid = k

		return min_centroid