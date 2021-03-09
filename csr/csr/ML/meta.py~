
import argparse
import pickle
import math
import random
import sys
import itertools
import collections
from collections import namedtuple

import numpy
import scipy
from scipy.sparse import issparse, csr_matrix

from sklearn import metrics, preprocessing
from sklearn.linear_model.base import LinearClassifierMixin
from sklearn.utils import check_array

from csr import Log

STATUS = Log.start_statusbar(sys.stderr)

def safe_sparse_multiply(a, b, dense_output = False):
    """ Multiply that handles the sparse matrix case correctly.
    Because it is too much to expect multiplication to behave
    consistently in numpy and scipy...
    """
    if issparse(a) or issparse(b):
        # b is ducktyped so it doesn't matter if it's a numpy array
        ret = csr_matrix(a).multiply(b)
        if dense_output and hasattr(ret, "toarray"):
            ret = ret.toarray()
        return ret
    else:
        return a * b


def LINEAR_get_activation_pattern(self, X):
    if not hasattr(self, 'coef_') or self.coef_ is None:
        raise NotFittedError("This %(name)s instance is not fitted "
                             "yet" % {'name': type(self).__name__})

    X = check_array(X, accept_sparse='csr')

    n_features = self.coef_.shape[1]
    if X.shape[1] != n_features:
        raise ValueError("X has %d features per sample; expecting %d"
                         % (X.shape[1], n_features))
    
#    scores = safe_sparse_multiply(X, self.coef_) + self.intercept_
    # The intercept is a scaler, so it only shifts the the pattern
    # It would however make the results dense
    scores = safe_sparse_multiply(X, self.coef_).tocsr()
    
    assert scores.shape == X.shape # We should have one value per sample/feature pair

    # Note: the activation pattern columne must not change depending
    # on X, which means that we cannot calculate std from the data
    # we receive here
    if hasattr(self, '_train_std_X'):
        feature_importance = abs(self._train_std_X * self.coef_[0])
    else:
        # This should give the same results, assuming the input data
        # from the pipeline was normalized
        feature_importance = abs(self.coef_[0])
    top_n_features = numpy.argsort(feature_importance)[-100:][::-1]    
    scores = scores[:, top_n_features]
    
    return scores
LinearClassifierMixin.get_activation_pattern = LINEAR_get_activation_pattern

# Monkey patch for keeping the std of the training data
# Not currently used
# Note that we might have fit the model using partial fit
#LINEAR_orig_fit = LinearClassifierMixin.fit
#def LINEAR_fit(self, X, *args, **kw_args):
#    self._train_std_X = numpy.std(X, 0)
#    return LINEAR_orig_fit(self, X, *args, **kw_args)
#LinearClassifierMixin.fit = LINEAR_get_activation_pattern

RFData = namedtuple('RFData', 'indices, scores, global_features')

def _safe_sparse_hstack(X):
#    i = 0
#    for block in X:
#        i += 1
#        STATUS[10]('Block %i: [%i x %i]' % (i, block.shape[0], block.shape[1]))
    if any([scipy.sparse.issparse(d) for d in X]):
        X = scipy.sparse.hstack(X).tocsr()
    else:
        X = numpy.hstack(X)
    return X

def _safe_sparse_vstack(X):
    if any([scipy.sparse.issparse(d) for d in X]):
        X = scipy.sparse.vstack(X).tocsr()
    else:
        X = numpy.vstack(X)
    return X

class MetaClassifier:
    def __init__(self, classifier_prot, X, labels, static_classifiers, RF_data_train, RF_data_test):
        STATUS.status('Building metaclassifier...')
        self._classifier_prot = classifier_prot
        if hasattr(self._classifier_prot, 'build'):
            self._classifier = classifier_prot.build()
        else:
            self._classifier = classifier_prot
        self._X = X # Not necessary
        self._Y = labels
        self._static_classifiers = static_classifiers

        self._X_p = []
        for classifier in self._static_classifiers:
            STATUS.status('Extracting conf...')
                #conf = numpy.array([numpy.random.rand(len(X))]).T
            conf = numpy.array([classifier.calculate_score(X)]).T
            self._X_p.append(conf)
            STATUS[10]('Conf: [%i x %i]' % (conf.shape[0], conf.shape[1]))
#                STATUS.status('Extracting activation pattern...')
#                X_p = scipy.sparse.rand(len(X), 100, 0.3)
#                X_p = classifier.get_activation_pattern(X)
#                X_p = preprocessing.scale(X_p, axis = 0, with_mean = False)
#                STATUS[10]('X_p: [%i x %i]' % (X_p.shape[0], X_p.shape[1]))
#                self._X_p.append(X_p)             
        self._X_p = _safe_sparse_hstack(self._X_p)
                
        assert self._X_p.shape[0] == len(X)

        STATUS[10]('Done')
        self._RF_data_train = RF_data_train
        self._RF_data_test = RF_data_test
    
    def rebuild_classifier(self):
        self._classifier = self.classifier_prot.build()
    
    def train(self):
        
        STATUS.status('Training...')
        # We don't need to do this every iteration but we can decrease
        # the GPU memory footprint by using smaller random samples
        # rather than a huge batch
        test_n = min(10000, len(self._RF_data_test))
        test_i   = random.sample(range(len(self._RF_data_test)), test_n)
        test_X_t = []
        test_y   = []
        for i in test_i:
            X_t, y = self.make_meta_trainer_X_t(self._RF_data_test[i])
            test_X_t.append(X_t)
            test_y.append(y)
        #        test_X_t  = _safe_sparse_vstack(test_X_t)
        #        test_y    = list(itertools.chain(*test_y))
        
        last_score = -1
        last_classifier = None
        overfit_attempts_remaining = 5
        for training_iteration in range(1000):
            train_i = random.sample(range(len(self._RF_data_train)), 50)
            train_X_t = []
            train_y   = []
            for i in train_i:
                X_t, y = self.make_meta_trainer_X_t(self._RF_data_train[i],
                                                    undersample = True,
                                                    sampling_method = 'uniform'
                                                    )
                train_X_t.append(X_t)
                train_y.append(y)
            if scipy.sparse.issparse(train_X_t): train_X_t = train_X_t.todense()
            train_X_t  = _safe_sparse_vstack(train_X_t)
            train_y    = list(itertools.chain(*train_y))
            STATUS[10]('Constructed train data: [%i x %i]' % (train_X_t.shape[0],
                                                              train_X_t.shape[1]))
            
            last_classifier = pickle.dumps(self._classifier)
            self._classifier.partial_fit(train_X_t, train_y, classes = ['Y', 'N'])
#            self._classifier.fit(train_X_t, train_y)
            
            ap_values = []
            ap_values_baseline = []
            for X, y in zip(test_X_t, test_y):
                assert X.shape[0] == len(y)
                truth = numpy.array(y) == 'Y'
                if collections.Counter(y)['Y'] == 0:
                    pass
#                    ap_values.append(0)
#                    ap_values_baseline.append(0)
                    # ...or just ignore these?
                else:
                    if scipy.sparse.issparse(X): X = X.todense()
                    conf = self._classifier.predict_proba(X)[:, 1].tolist()
                    ap   = metrics.average_precision_score(truth, conf)
                    # RF score is in first column
#                    if scipy.sparse.issparse(X):
#                        conf_baseline = list(itertools.chain(*X[:, 0].toarray()))
#                    else:
                    conf_baseline = X[:, 0].tolist()
                    ap_baseline   = metrics.average_precision_score(truth, conf_baseline)
                    ap_values_baseline.append(ap_baseline)
                    ap_values.append(ap - ap_baseline)
            score = numpy.mean(ap_values)
            msg = "Epoch %4i: MAP diff %+.3f (baseline: %.3f)" % (training_iteration,
                                                                  numpy.mean(ap_values),
                                                                  numpy.mean(ap_values_baseline))
            STATUS[10](msg)
            if score < last_score:
                # We are starting to overfit
                # Revert last update
                STATUS[10]('Overfitting. Reverting last training attempt...')
                self._classifier = pickle.loads(last_classifier)
                if overfit_attempts_remaining == 0:
                    return
                else:
                    overfit_attempts_remaining -= 1
            else:
                overfit_attempts_remaining = 5
                last_score = score
#            STATUS.status(msg)
#        STATUS.status('')
    
    def calculate_score(self, rf_data):
        X_t, _ = self.make_meta_trainer_X_t(rf_data)
        conf = self._classifier.predict_proba(X_t)[:, 1].tolist()
        return conf
    
    def make_meta_trainer_X_t(self, rf_data, undersample = False, sampling_method = 'uniform'):
        
        labels = [self._Y[i] for i in rf_data.indices]
        counts = collections.Counter(labels)
#        assert counts['Y'] > 0,           "Training data without positive labels"
#        assert counts['Y'] < len(labels), "Training data without negative labels"
#        STATUS[10]("Using RF block of length %i, Y: %i, N: %i" % (len(rf_data.indices),
#                                                                  counts['Y'],
#                                                                  counts['N']))
        
        if undersample:
            # Undersample
            pos_i_local = numpy.where(numpy.array(labels) == 'Y')[0].tolist()
            neg_i_local = numpy.where(numpy.array(labels) != 'Y')[0].tolist()
            opt_n = max(len(pos_i_local), 10)
            neg_n = min(opt_n, len(neg_i_local))
            if sampling_method == 'false positives':
                # Variant of uncertainty sampling intended to return
                # (on average) for every positive sample one negative
                # sample ranked higher
                # Prob is prop to the number of 'Y' after it
                prob = numpy.cumsum(numpy.array(labels[-1::-1]) == 'Y')[-1::-1]
                # ...and on average one from the tail end
                # This also assures some probability > 0
                prob = prob + 1.0 / len(neg_i_local)
                prob = prob[neg_i_local]
                prob = prob / sum(prob)
                neg_i_local = numpy.random.choice(neg_i_local, neg_n,
                                                  p = prob,
                                                  replace = False).tolist()
            elif sampling_method == 'uniform':
                # Uniform sampling
                neg_i_local = random.sample(neg_i_local, neg_n)
            indices_local = pos_i_local + neg_i_local
            random.shuffle(indices_local)
            
            indices = [rf_data.indices[i] for i in indices_local]
            scores  = [rf_data.scores [i] for i in indices_local]
        else:
            indices = rf_data.indices
            scores  = rf_data.scores
        
        n = len(indices)
#        STATUS[10]("         undersampled to %i" % (n))
        X_p = []
        # Convention: first column is RF score
        X_p.append(numpy.array([scores]).T)
        for feature_name in ['n_pos_found', 'n_feedback']:
            feature = rf_data.global_features[feature_name]
            # Features used atm are amount of feedback and positives found
            # This custom scaling is engineered to be indicative for features in
            # the range (0, 256] and to be more discriminative for lower values
            feature = math.copysign(math.log2(1+abs(feature))/8, feature)
            X_p.append(numpy.array([[feature] * n]).T)
#        rf_stage = rf_data.global_features['RF_stage']
#        X_p.append(numpy.array([[rf_stage] * n]).T)
#        perc_feedback = rf_data.global_features['perc_feedback']
#        X_p.append(numpy.array([[perc_feedback] * n]).T)
        
        X_p.append(self._X_p[indices, :])
        X_p = _safe_sparse_hstack(X_p)
        assert X_p.shape[0] == n
#        STATUS[10]('Constructed training block: [%i x %i]' % (X_p.shape[0], X_p.shape[1]))
        
        y = [self._Y[i] for i in indices]
        
        return (X_p, y)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'Construct meta classifier')
    parser.add_argument('--static_models',    nargs = '*', required = True,
                        help = 'Pickled static model files to be used as helpers in meta training')
    parser.add_argument('--classifier',       required = True,
                        help = "Classifier parameters encoded in YAML for meta classification.")
    
    static_models = []
    if args.static_models:
        for model_filename in args.static_models:
            with open(model_filename, 'rb') as model_file:
                loaded_model = pickle.load(model_file)
                assert len(loaded_model) == 2
                STATUS[7]("Loading static model from file: %s" % model_filename)
                static_models.append(StaticModel(pipeline   = loaded_model[0],
                                                 classifier = loaded_model[1]))
                STATUS[7]("Done")
