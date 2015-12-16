# model.pyx
# Author: Jacob Schreiber <jmschr@cs.washington.edu>

'''
Lock-Free Parallel K-means using Stochastic Descent.
'''

from cython.parallel import parallel, prange
from libc.math cimport sqrt as csqrt

import numpy
cimport numpy

from joblib import Parallel, delayed

cdef class Centroid( object ):
	"""
	A centroid which stores summary statistics and values.
	"""

	cdef int d

	cdef double summary_weight
	cdef numpy.ndarray summary_ndarray
	cdef double* summary

	cdef numpy.ndarray position_ndarray
	cdef double* position

	def __init__( self, position ):
		self.d = position.shape[0]
		self.position_ndarray = position
		self.position = <double*> self.position_ndarray.data

		self.summary_weight = 0.
		self.summary_ndarray = numpy.zeros(self.d, dtype=numpy.float64)
		self.summary = <double*> self.summary_ndarray.data

	cdef double distance( self, double* X ) nogil:
		"""Calculate the euclidean distance between this centroid and a point."""

		cdef int i
		cdef double distance = 0

		for i in range(self.d):
			distance += ( self.position[i] - X[i] ) ** 2.0

		return csqrt(distance)

	cdef void summarize( self, double* X, double weight ) nogil:
		"""Add this to the growing summary statistics."""

		cdef int i

		self.summary_weight += weight
		for i in range(self.d):
			self.summary[i] += X[i]

	cdef void from_summaries( self ) nogil:
		"""Use the growing summary statistics to update the centroid position."""

		cdef int i

		for i in range(self.d):
			self.position[i] = self.summary[i] / self.summary_weight


cdef class Kantalope( object ):
	"""Kantalope kmeans estimator."""

	cdef int k
	cdef numpy.ndarray centroids

	def __init__( self, k, n_jobs=1 ):
		self.k = k
		self.centroids = numpy.empty(k, dtype=numpy.object_)

	cpdef fit( self, numpy.ndarray X, nthreads=1 ):
		"""Fit to the data using random initializations."""

		cdef int i, y
		cdef int n = X.shape[0]
		cdef int d = X.shape[1]
		cdef void** centroids = <void**> self.centroids.data
		cdef double* X_data = <double*> X.data

		for i in range(self.k):
			self.centroids[i] = Centroid( X[i] )

		with nogil, parallel( num_threads=nthreads ):
			for i in prange(n):
				y = self.__predict_single_point( X_data + i*d, d, centroids )
				(<Centroid> centroids[y]).summarize( X_data + i*d, 1 )
				(<Centroid> centroids[y]).from_summaries()

	def predict( self, X ):
		"""Predict the centroid associated with each point."""

		y_pred = numpy.empty( X.shape[0] )
		for i in xrange( X.shape[0] ):
			y_pred[i] = self._predict_single_point( X[i] )

		return y_pred

	cpdef int _predict_single_point( self, numpy.ndarray X ):
		"""Python wrapper for parallel processing."""

		cdef double* X_data = <double*> X.data
		cdef void** centroids = <void**> self.centroids.data
		cdef int y_pred

		with nogil:
			y_pred = self.__predict_single_point( X_data, X.shape[0], centroids )

		return y_pred

	cdef int __predict_single_point( self, double* X, int n, void** centroids ) nogil:
		"""Predict the label of a single datapoint."""

		cdef int i
		cdef double distance
		cdef double min_distance = 1e8
		cdef int min_centroid = -1

		for i in range(self.k):
			distance = (<Centroid> centroids[i]).distance( X )

			if distance < min_distance:
				min_distance = distance
				min_centroid = i

		return min_centroid
