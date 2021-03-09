
# coding: utf-8

import itertools
import sys, random
import numpy
from scipy import sparse as scipy_sparse

from sklearn.base import BaseEstimator

from csr.Log import start_statusbar
from csr.misc import CACHE

STATUS = start_statusbar(sys.stderr)

class BalancedFixedSizePairSelection:
    def __init__(self, n_pairs_per_combination = 10000):
        self.n_pairs_per_combination = n_pairs_per_combination
    
    def __call__(self, X, y):
        X_new = []
        y_new = []
        y = numpy.asarray(y)
        
        classes = sorted(numpy.unique(y))
        class_comb = itertools.combinations(classes, 2)
        # Note: combinations returns all distinct pairs with preserved order
        for (class_i, class_j) in class_comb:
            # We consider all positive class combination pairs
            assert(class_j > class_i)
            X_i = X[numpy.where(y == class_i)[0], :]
            X_j = X[numpy.where(y == class_j)[0], :]
            n_i = X_i.shape[0]
            n_j = X_j.shape[0]
            n = n_i * n_j
            sample_n = 2*self.n_pairs_per_combination
            # Avoid edge cases:
            if sample_n < 1 or sample_n > n: sample_n = n
            STATUS[10]("Pairwise case in data: (%i, %i), selecting %i / %i examples" % (class_j,
                                                                                        class_i,
                                                                                        sample_n,
                                                                                        n))
            # Enumerate all pairs (i, j) in (X_i x X_j) by giving each
            # a unique index p_ij = i * n_i + j
            # Now the set of all p_ij is in {0, 1, 2, 3, ..., n-1}
            P_ij = random.sample(range(n), sample_n)
            # Translate back to (i, j)
            ii = [p_ij // n_j for p_ij in P_ij]
            jj = [p_ij  % n_j for p_ij in P_ij]
            X_new.append(X_j[jj, :] - X_i[ii, :])

        if scipy_sparse.issparse(X):
            X_new = scipy_sparse.vstack(X_new)
        else:
            X_new = numpy.vstack(X_new)
        # Flip direction of every second pair
        y_new = numpy.resize([1, -1], [X_new.shape[0], 1])
        STATUS[10]("X: (%i x %i), y: %i" % (X_new.shape[0], X_new.shape[1], len(y_new)))
        if scipy_sparse.issparse(X):
            X_new = X_new.multiply(y_new)
        else:
            X_new = y_new * X_new
        
        if scipy_sparse.issparse(X):
            STATUS[10]("Density: %f -> %f" % (1.0*X.getnnz() / numpy.multiply(*X.shape),
                                             1.0*X_new.getnnz() / numpy.multiply(*X_new.shape)))
        # Upstream classifier expects labels in YMN
        y_new = [y_i == 1 and 'Y' or 'N' for y_i in y_new.ravel()]
        c, count = numpy.unique(y_new, return_counts = True)
        STATUS[10]("Pairwise training data created: %s" % (str(dict(zip(c, count)))))
        return X_new, y_new

class Pairwise(BaseEstimator):
    '''
    Learn-to-rank pair-wise metaclassifier
    Adapted from https://gist.github.com/fabianp/2020955
    but uses dependency injection to allow using general
    classifiers adhering to the sklearn API.
    Requires a LINEAR classifier with coefficients stored
    in 'coef_'.
    '''

    def __init__(self, _classifier):
        self._classifier = _classifier
        self._label_to_utility = {
            'Y': 2,
            'M': 1,
            'N': 0
        }
        self.pair_selection_strategy = BalancedFixedSizePairSelection(10000)

    @property
    def max_iter(self):
        return self._classifier.max_iter
    
    def fit(self, X, y, sample_weight = None):
        y_utility = numpy.array(
            [self._label_to_utility[y_i] for y_i in y],
            dtype = numpy.int32
        )
        X_trans, y_trans = CACHE(self.pair_selection_strategy)(X, y_utility)
        self._classifier.fit(X_trans, y_trans, sample_weight = sample_weight)
        return self
    
    def partial_fit(self, X, y, classes = None, sample_weight = None):
        y_utility = numpy.array(
            [self._label_to_utility[y_i] for y_i in y],
            dtype = numpy.int32
        )
        X_trans, y_trans = CACHE(self.pair_selection_strategy)(X, y_utility)
        self._classifier.partial_fit(X_trans, y_trans, classes = ['Y', 'N'], sample_weight = sample_weight)
        return self
    
    def predict(self, X):
        if hasattr(self._classifier, 'coef_'):
            return np.argsort(np.dot(X, self.coef_.ravel()))
        else:
            raise ValueError("Must call fit() prior to predict()")
        
    def predict_proba(self, X):
        confidence = X.dot(self._classifier.coef_.ravel())
        
        # Proba will be two-class since this is a ranking
        proba = numpy.array([1-confidence, confidence]).transpose()
        assert proba.shape == (X.shape[0], 2)
        return proba
