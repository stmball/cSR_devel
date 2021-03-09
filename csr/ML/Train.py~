
# coding: utf-8

from enum import Enum
import sys, string
from time import time
import collections
from collections import OrderedDict, defaultdict
from future.utils import iteritems

from joblib import Memory

from sklearn import cluster
from sklearn import metrics
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
import json
import random
import math
import numpy, scipy
import re
from copy import deepcopy
from collections import namedtuple, Counter

from csr.Data import DataStream
from csr.Log import start_statusbar
import csr.ML.Evaluation
from csr.ML.meta import RFData, safe_sparse_multiply

# ~~~~~~ Setup ~~~~~~
import warnings
from scipy.sparse import SparseEfficiencyWarning
# This warning happens if sparse matrices inadvertently end up being
# dense, which cause the system to slow down by several orders of
# magnitude -- Treat it as an error
warnings.filterwarnings("error", category = SparseEfficiencyWarning)

STATUS = start_statusbar(sys.stderr)

# ~~~~~~~~ Misc helper functions ~~~~~~~~
# Bash color formatting
def red(s):
    return '\033[0;31m%s\033[0m' % s
def blue(s):
    return '\033[0;34m%s\033[0m' % s
def green(s):
    return '\033[0;32m%s\033[0m' % s

class Split:
    TRAIN, TEST, VALIDATE = range(3)

class TrainingSplit:

    def __init__(self, name, train_split, test_split, labels):
        self.name = name
        self.indices = {
            Split.TRAIN: train_split,
            Split.TEST:  test_split
            }
        assert len(set(self.indices[Split.TRAIN]).intersection(set(self.indices[Split.TEST]))) == 0
        self.labels = labels


def undersample(split, ratio = 1.0, pos_labels = ['Y'], min_threshold = 100, max_threshold = 500):
#    labels = normalize_labels(split.labels, pos_labels)
    labels = split.labels
    # Each split should have all classes present, but let us assume
    # that this is not necessarily the case
    classes = numpy.unique(labels)
    samples_per_class   = {}
    minority_class_size = float('inf')
    minority_class = None
    undersampled_train_split = []
    
    for c in classes:
        # Find the set of instances of class c in the training split
        class_subset_logical = numpy.array(labels) == c
        class_subset = set(numpy.where(class_subset_logical)[0])
        train_subset = set(split.indices[Split.TRAIN])
        subset = list(class_subset.intersection(train_subset))
        if len(subset) < minority_class_size:
            minority_class_size = len(subset)
            minority_class = c
        samples_per_class[c] = subset
    
#    sys.stderr.write("Minority class '%s' has %i instances\n" % (minority_class,
#                                                                 minority_class_size))
    minority_class_size = max(minority_class_size, min_threshold)
    minority_class_size = min(minority_class_size, max_threshold)
    
    for c in classes:
        subset = samples_per_class[c]
        random.shuffle(subset)
        class_n = min(minority_class_size, int(ratio*len(subset)))
#        class_n = minority_class_size
        subset_c = subset[0:class_n]
#        sys.stderr.write("Adding %i samples\n" % (len(subset_c)))
        undersampled_train_split += subset_c
    random.shuffle(undersampled_train_split)
    assert(len(set(undersampled_train_split).intersection(set(split.indices[Split.TEST]))) == 0)
    
#    sys.stderr.write("Undersampled split to %i samples\n" % (len(undersampled_train_split)))
    ret_split = deepcopy(split)
    ret_split.indices[Split.TRAIN] = undersampled_train_split
    return ret_split

def _log_structure(structure, log_level, depth):
    if type(structure) == Pipeline:
        for label, child in structure.steps:
            STATUS[log_level]('%s  %s %s:' % (' ' * depth,
                                              green(child.__class__.__name__),
                                              blue(label)))
            _log_structure(child, log_level, depth + 4)
    elif type(structure) == FeatureUnion:
        # transformer_list is not usually meant to be accessed...
        for label, child in structure.transformer_list:
            STATUS[log_level]('%s  %s %s:' % (' ' * depth,
                                              green(child.__class__.__name__),
                                              blue(label)))
            _log_structure(child, log_level, depth + 4)
    else:
        STATUS[log_level]('%s%s:' % (' ' * depth, structure))
        

class SharedDataView:
    def __init__(self, transformer_prot, X):
        
        # Is there data to process?
        self._has_unprocessed_data = (X is not None)
        # The raw data that will be passed through the pipeline.
        self._X = X
        # The transformed feature representations for each
        # data point in X.
        # The splits are those passed through the pipelines,
        # and the actual splits used for training will depend
        # on e.g. relevance feedback and may span the split
        # boundaries. The data should therefore be only ever
        # be accessed using the accessor functions to ensure
        # the correct and actual splits are used.
        self._X_t = None
        self._transformer = transformer_prot.build()
        
        STATUS[8](green("Pipeline:"))
        _log_structure(self._transformer, 8, 0)
        
    def __len__(self):
        if not self._X: return 0
        else: return len(self._X)
    
    def get_raw_data(self, indices):
        ''' Retrieves the specified subset of X before pipeline transformation.
        Returns a copy.
        '''
        assert len(indices) == 0 or max(indices) < len(self)
        return self._X[indices]
    
    def get_data(self, indices):
        '''
        Retrieves the specified subset of X_T.
        The returned data will be a new instance.
        '''
        assert max(indices) < len(self)
        self._execute_pipeline_if_unprocessed()
        # Duck typing. The matrix type should behave like a numpy matrix
        # or a scipy sparse matrix
        MAT_TYPE = type(self._X_t)

        # The resulting shape should only differ in dimension 1
        shape = (len(indices), *list(self._X_t.shape[1:]))
        
        X_t_i = self._X_t[indices, :]
        
        # Sanity check
        assert MAT_TYPE == type(X_t_i)
        assert X_t_i.shape == shape
        return X_t_i

    def get_feature_names(self):
        self._execute_pipeline_if_unprocessed()
        if not hasattr(self._transformer, 'get_feature_names') and hasattr(self._transformer, 'steps'):
            # Hackity-hack!
            # get_feature_names should really be a method of Pipeline
            # but this is not going to happen because reasons
            ret_val = self._transformer.steps[-1][1].get_feature_names()
        else:
            ret_val = self._transformer.get_feature_names()
        ret_val = [name.split("__")[-1] for name in ret_val]
        return ret_val
    
    def _execute_pipeline_if_unprocessed(self):
        if not self._has_unprocessed_data: return
        self._has_unprocessed_data = False
        STATUS[1]("Extracting features...")
        self._X_t = self._X

        y_t = self._X_t.label
        
        n_samples = self._X_t.shape[0]
        color_k = len(green('')) - len('') # The number of chars to specify a bash color
        
        for step in self._transformer.steps:
            assert self._X_t.shape[0] == n_samples
            msg = "Running step '%s' on data %s" % (blue(step[0]),
                                                    green(type(self._X_t)))
            n_features = self._X_t.shape[1]
            STATUS[2](msg + ('%80s' % ('x %i' % n_features))[len(msg)-2*color_k:])
            assert self._X_t is not None
            self._X_t = step[1].fit_transform(self._X_t, y = y_t)
        assert self._X_t.shape[0] == n_samples
        msg = "             result: data %s" % (green(type(self._X_t[Split.TRAIN])))
        n_features = self._X_t.shape[1]
        STATUS[2](msg + ('%80s' % ('x %i' % n_features))[len(msg)-1*color_k:])
    
    def transform(self, X):
        return self._transformer.transform(X)

class StaticModel:
    def __init__(self,
                 pipeline,
                 classifier):
        self._pipeline   = pipeline
        self._classifier = classifier
        self._model            = Pipeline([
                ('pipeline',   self._pipeline),
                ('classifier', self._classifier)
            ])
    
    def rank(self, X):
        # TODO This should probably return the confidence column
        # and let the caller handle it
        ''' Ranks the rows of X in place. Adds a, or overwrites the
        confidence column in X with relevant confidence scores
        '''
        conf = self.calculate_score(X)
        # Only add the column after making sure the score
        # calculation succeeded
        X.add_column('confidence')
        X.confidence = conf
        X.sort(lambda row: 1-row.confidence)
    
    def calculate_score(self, X):
        ''' Calculates the scores for each row of X.
        '''
        if hasattr(self._model, 'predict_proba'):
            conf = self._model.predict_proba(X)
            return conf[:, 1].tolist()
        else:
            raise ValueError('Loaded model does not support ranking')

    def get_activation_pattern(self, X):
        ''' Extracts the activation pattern for each row of X.
        '''
        if hasattr(self._classifier, 'get_activation_pattern'):
            X_t = self._pipeline.transform(X)
            return self._classifier.get_activation_pattern(X_t)
        else:
            raise ValueError('Loaded model does not implement activation pattern extraction')


class Trainer:
    
    def __init__(self,
                 shared_data_view,
                 labels,
                 classifier_prot,
                 mode = 'Y|MN',
                 undersample_range = None):
        
        assert len(shared_data_view) == len(labels)
        
        self._classifier_prot  = classifier_prot
        self._classifier       = classifier_prot.build()
        self._shared_data_view = shared_data_view
        self._model            = Pipeline([
                ('pipeline',   self._shared_data_view._transformer),
                ('classifier', self._classifier)
            ])
        # The current epoch of training.
        self.epoch = 0
        # Raw Y|M|N labels. These should never be accessed
        # directly to ensure they are correctly normalized
        # according to the training and testing modes. Note
        # that the training and test modes are allowed to
        # change during execution
        self._Y = labels
        # Tables for remporarily reassigning portions of the
        # labels
        self._Y_reassignment = {}
        # Split indices. Should not be accessed directly.
        # The indices are allowed to change during execution.
        self._indices = {
            Split.TRAIN:    [],
            Split.VALIDATE: [],
            Split.TEST:     []
            }
        # Indices into the test split for which we have done
        # relevance feedback and which we thus have to freeze
        # the output order
        self._feedback_provided = set()
        
        if not mode in ['Y|MN', 'YM|N', 'Y|M|N', 'CUSTOM']:
            msg = 'Unrecognized training mode: %s' % mode
            raise ValueError(msg)
        self.mode = mode
        if undersample_range:
            if len(undersample_range) != 2 or undersample_range[0] > undersample_range[1]:
                msg = 'Undersample must be a valid range'
                raise ValueError(msg)
        self.undersample_range = undersample_range
        
        STATUS[8](green("Classifier:"))
        _log_structure(self._classifier, 8, 2)
    
    def rebuild_classifier(self):
        self._classifier = self._classifier_prot.build()
        self._model      = Pipeline([
                ('pipeline',   self._shared_data_view._transformer),
                ('classifier', self._classifier)
            ])
        self.epoch = 0
    
    def _normalize(self, raw_labels, split):
        if self.mode == 'CUSTOM':
            return raw_labels
        pos_labels = ['Y']
        if split == Split.TRAIN:
            if self.mode in ['YM|N', 'Y|M|N']:
                pos_labels.append('M')
#        elif split == Split.VALIDATE or split == Split.TRAIN:
        if self.mode in ['YM|N']:
            pos_labels.append('M')
#        STATUS[8]('Training mode: (%s), Pos labels: %s' % (self.mode, str(pos_labels)))
        return [y in pos_labels and 'Y' or 'N' for y in raw_labels]
    
    def get_raw_data(self, split):
        ''' Retrieves X for the specified split without considering
        permutations or pipeline transformation.
        Returns a copy.
        '''
        indices = self._indices[split]
        return self._shared_data_view.get_raw_data(indices)
#        return self._X[self._indices[split]]
    
    def get_data(self, split):
        '''
        Retrieves X_T^i for the current epoch and the specified split.
        The returned data will be a new instance.
        '''
        indices = self._indices[split]
        return self._shared_data_view.get_data(indices)
    
    def get_all_labels(self, split):
        ''' Retrieves Y_T^i for the current epoch normalized as though it
        belonged to the specified split.
        '''
        raw_labels = self._Y[:]
        for i, reassigned_label in self._Y_reassignment.items():
            raw_labels[i] = reassigned_label
        return self._normalize(raw_labels, split)
    
    def get_labels(self, split):
        ''' Retrieves Y_T^i for the current epoch and the specified split.
        '''
        labels = self.get_all_labels(split)
        indices = self._indices[split]
        return [labels[index] for index in indices]
    
    def is_feedback_provided(self, i):
        return i in set(self._feedback_provided)
    
    def provide_feedback(self, indices):
        new_indices = set(indices) - set(self._feedback_provided)
        if set(indices) != new_indices:
            STATUS[2]('WARN: Relevance feedback provided multiple times for some test data')
#        assert(len(set(self._feedback_provided).intersection(indices)) == 0)
        assert(len(set(self._indices[Split.TRAIN]).intersection(new_indices)) == 0)
        self._feedback_provided.update(new_indices)
        self._indices[Split.TRAIN] += new_indices
        
#        print(self._indices[Split.TRAIN][Split.TEST])
    
    def remove_feedback(self, indices):
        indices = set(indices)
        unknown_indices = indices - set(self._feedback_provided)
        if len(unknown_indices) != 0:
            STATUS[2]('WARN: Removing relevance feedback but feedback was not provided')
        self._feedback_provided -= indices
        old = self._indices[Split.TRAIN]
        self._indices[Split.TRAIN] = [i for i in old if not i in indices]
        
    def get_amount_feedback(self):
        return 1.0*len(self._feedback_provided) / len(self._indices[Split.TEST])
    
    def reassign(self, i, label):
        if label == None:
            del(self._Y_reassignment[i])
        else:
            self._Y_reassignment[i] = label
    
    def _check_can_train(self):
        if self.mode == 'CUSTOM':
            return True
        if not 'Y' in numpy.unique(self.get_labels(Split.TRAIN)):
            msg = 'Trying to build classifier without positive examples'
            raise ValueError(msg)
        if not 'N' in numpy.unique(self.get_labels(Split.TRAIN)):
            msg = 'Trying to build classifier without negative examples'
            raise ValueError(msg)
    
    def can_calculate_ranked_score(self, split = Split.TEST):
        return hasattr(self._classifier, 'predict_proba') and len(self._indices[split]) != 0
    
    def use_split(self, train    = [],
                        validate = [],
                        test     = []):
        # Change the permutation table
        self._indices[Split.TRAIN]    = train[:]
        self._indices[Split.VALIDATE] = validate[:]
        self._indices[Split.TEST]     = test[:]
        
        def colorize(label):
            if label == 'Y': return green(label)
            elif label == 'N': return red(label)
            else: return blue(label)
        def print_dist_stats(Y_part, msg):
            labels, counts = numpy.unique(Y_part, return_counts = True)
            counts = dict(zip(labels, counts))
            STATUS[8]("  %s %i (%s)" % (msg, len(Y_part),
                    ', '.join(['%s: %5i' % (colorize(l), c) for l, c in iteritems(counts)])))
        STATUS[8](green('Data split:'))
        if train    != []: print_dist_stats(numpy.array(self._Y)[train],    'TRAIN:   ')
        if validate != []: print_dist_stats(numpy.array(self._Y)[validate], 'VALIDATE:')
        if test     != []: print_dist_stats(numpy.array(self._Y)[test],     'TEST:    ')
        print_dist_stats(self._Y, 'Total:   ')
        
    def get_all_possible_labels(self):
        assert self.mode in ['Y|MN', 'YM|N', 'Y|M|N', 'CUSTOM']
        if self.mode == 'CUSTOM':
            return numpy.unique(self._Y)
        if self.mode in ['Y|MN', 'YM|N']:
            return ['Y', 'N']
        if self.mode in ['Y|M|N']:
            return ['Y', 'M', 'N']
    
    def calculate_ranking_score(self, split, score_func):
        truth = self.get_labels(split)
        if not truth or sum(numpy.array(truth) == 'Y') == 0:
            return numpy.nan
        
        if hasattr(self._classifier, 'predict_proba'):
            with warnings.catch_warnings():
                # This happens for linear models early in the training, it is
                # harmless apart from yielding terrible ranking results
                warnings.filterwarnings("ignore", message="overflow encountered in exp")
                conf = self._classifier.predict_proba(self.get_data(split))
            conf = conf[:, 1].tolist()
        else:
            raise ValueError('Loaded model does not support ranking')
        return score_func(numpy.array(truth) == 'Y', conf)
    
    def calculate_auc(self, split):
        return self.calculate_ranking_score(split, metrics.roc_auc_score)
    
    def calculate_ap(self, split):
        return self.calculate_ranking_score(split, metrics.average_precision_score)
    
    def train(self):
#        if self.epoch == 0:
#        if self._has_unprocessed_data:
#            self._execute_pipeline()
        
        self._check_can_train()
        
        use_incremental_training = hasattr(self._classifier, 'partial_fit')
        if use_incremental_training:
            STATUS[8]('Using incremental training')
            target_classes = self.get_all_possible_labels()
        else:
            STATUS[5]('Using batch training')
            target_classes = numpy.unique(self.get_labels(Split.TRAIN))
        assert len(target_classes) > 1
        if len(target_classes) > 2:
            STATUS[0](red('WARN: More than two classes in training: [%s]' % ', '.join(target_classes)))
#            STATUS[0]('WARN: Training will use multiclass training which will decrease performance')
        
        STATUS[10]('Training on %i rows' % (self.get_data(Split.TRAIN).shape[0]))
        running = True
        i = 0
        while running:
            i += 1
            X_t = self.get_data(Split.TRAIN)
            Y = self.get_labels(Split.TRAIN)
            assert X_t.shape[0] == len(Y)
            STATUS[8]('Training on %i examples' % len(Y))
            uniq_labels, counts = numpy.unique(Y, return_counts = True)
            STATUS[10]('Training on ' + ' '.join(['%s: %i' % (label, count) for label, count in zip(uniq_labels, counts)]))

#            all_train_labels  = self.get_all_labels(Split.TRAIN)
#            all_train_indices = self._indices[Split.TRAIN]
#            all_train_labels  = [all_train_labels[debug_train_i] for debug_train_i in all_train_indices]
#            STATUS[10]('Label mapping: ' + ' '.join(['%i: %s' % (debug_train_i, debug_train_label) for debug_train_i, debug_train_label in zip(all_train_indices, all_train_labels)]))
            
            if hasattr(self._classifier, 'shuffle') and self._classifier.shuffle:
                if self.undersample_range:
                    STATUS[10]("Shuffling undersample range (%i, %i)" % self.undersample_range)
                    pos_indices = list(numpy.where(numpy.array(Y) == 'Y')[0])
                    neg_indices = list(numpy.where(numpy.array(Y) != 'Y')[0])
                    numpy.random.shuffle(pos_indices)
                    numpy.random.shuffle(neg_indices)
                    min_n = min(len(pos_indices), len(neg_indices))
                    min_n = min(min_n, self.undersample_range[1])
                    min_n = max(min_n, self.undersample_range[0])
                    indices = pos_indices[:min_n] + neg_indices[:min_n]
#                    n_pos = len(pos_indices[:min_n])
#                    n_neg = len(neg_indices[:min_n])
                else:
                    STATUS[10]("Shuffling...")
                    indices = numpy.arange(X_t.shape[0])
#                    n_pos = sum(numpy.array(Y) == 'Y')
#                    n_neg = sum(numpy.array(Y) != 'Y')
#                STATUS[10]("Training on %s vs %s samples" % (green(n_pos), red(n_neg)))
                numpy.random.shuffle(indices)
                X_t = X_t[indices, :]
                Y = numpy.array(Y)[indices]
            if use_incremental_training:
                STATUS[8]("Fitting incrementally...")
                self._classifier.partial_fit(X_t, Y, classes = target_classes)
            else:
                STATUS[8]("Fitting...")
                self._classifier.fit(X_t, Y)
            
            if i >= 100:
                running = False
                STATUS[8]("Max training attempts reached")
            else:
                # Test if the training (iteration) was successful
                if self.can_calculate_ranked_score(Split.VALIDATE):
                    cur_auc = self.calculate_auc(Split.VALIDATE)
                    if cur_auc > 0.5: running = False
                    STATUS[8]("VALIDATE AUC score: %.3f" % cur_auc)
                elif self.can_calculate_ranked_score(Split.TRAIN):
                    cur_auc = self.calculate_auc(Split.TRAIN)
                    if cur_auc > 0.5: running = False
                    STATUS[8]("TRAIN AUC score: %.3f" % cur_auc)
                else: # No way to tell if it failed
                    running = False
            if running:
                STATUS[8]("Training failed with AUC %.3f. Trying again..." % cur_auc)
        
        old_epoch = self.epoch
        if use_incremental_training:
            self.epoch += 1
        else:
            try:
#                STATUS[0]('Using batch training')
                if self._classifier.max_iter < 0:
                    self.epoch += 1
                else:
                    self.epoch += self._classifier.max_iter
                # Note that some classifiers in sklearn pre 0.19
                # used n_iter instead of max_iter
            except AttributeError: # No max_iter
                raise AttributeError("Batch training model does not implement 'max_iter' attribute")
#        STATUS[8]('Training step finished with epoch increment: %i => %i' % (old_epoch, self.epoch))
    
    def finished(self):
        try:
            if self._classifier.max_iter is not None:
                if self._classifier.max_iter < 0: 
                    # Sklearn seems to use neg numbers to signify not applicable
                    return self.epoch > 0
                else:
                    return self.epoch >= self._classifier.max_iter
            else:
                return self.epoch > 0
        except AttributeError: # No max_iter
            return self.epoch > 0
    
    def predict(self, X):
        if self.epoch == 0:
            raise ValueError('Trying to predict using unfitted model')
        return self._model.predict(X)
    
    def rank(self, X):
        # TODO This should probably return the confidence column
        # and let the caller handle it
        ''' Ranks the rows of X in place. Adds a, or overwrites the
        confidence column in X with relevant confidence scores
        '''
        conf = self.calculate_score(X)
        # Only add the column after making sure the score
        # calculation succeeded
        X.add_column('confidence')
        X.confidence = conf
        X.sort(lambda row: 1-row.confidence)
    
    def calculate_score(self, X):
        ''' Calculates the scores for each row of X.
        '''
        if len(X) == 0:
            return []
        if self.epoch == 0:
            raise ValueError('Trying to use unfitted model')
        if hasattr(self._model, 'predict_proba'):
            conf = self._model.predict_proba(X)
            return conf[:, 1].tolist()
        else:
            raise ValueError('Loaded model does not support ranking')
    
    def calculate_feature_activation(self, X):
#        X = self.get_data(split)
        X_t = self._shared_data_view.transform(X)
        assert X_t.shape[0] == X.shape[0]
        assert X_t.shape[1] == len(self._shared_data_view.get_feature_names())
        if hasattr(self._classifier, 'coef_'):
#            activation = safe_sparse_multiply(X_t, self._classifier.coef_)
            activation = X_t.multiply(self._classifier.coef_).toarray()
            STATUS[10]("%s%s X %s%s -> %s%s" % (type(X_t),
                                                str(X_t.shape),
                                                type(self._classifier.coef_),
                                                str(self._classifier.coef_.shape),
                                                type(activation),
                                                str(activation.shape)))
            assert X_t.shape == activation.shape
        else:
            raise ValueError('Feature activation is not supported for non-linear models')
        return activation
    
    def calculate_most_important_features(self, X, k = 10, negative = False):
        activation = self.calculate_feature_activation(X)
        # Quick and dirty in O(m n log n) with sorting
        # TODO: Use numpy.argpartition for O(m n + m k log k)
        order = numpy.argsort(activation, axis=1)
        assert order.shape == activation.shape
        if negative: return order[:, -k:]
        else:        return order[:, :k]

    def print_most_important_features(self, X, k = 10, negative = False):
        ind_mat = self.calculate_most_important_features(X, k, negative)
        feature_names = self._shared_data_view.get_feature_names()
        STATUS[10]("%s%s" % (type(ind_mat), str(ind_mat.shape)))
        
        for ind_row in ind_mat:
            STATUS[10]("[%s]" % ', '.join(["'%s'" % feature_names[i] for i in ind_row]))
    
def create_crossvalidation_splits(X, n_folds = 10):
    splits = []
    fold_names = ["<Fold %i>" % (i + 1) for i in range(n_folds)]
    folds = StratifiedKFold(n_splits = n_folds, shuffle = True).split(X, X.label)
    for fold_name, split in zip(fold_names, folds):
        splits.append(TrainingSplit(fold_name,
                                    split[0], split[1], X.label))
    return splits

TrainingRun = namedtuple('TrainingRun', 'name, trainer, data')

class Runner:
    def __init__(self, trainer_prot,
#                 X,
                 splits,
                 n_repetitions,
                 undersample_range = None):
        self._trainer_prot = trainer_prot
        if undersample_range:
            splits = [undersample(split,
                                  min_threshold = undersample_range[0],
                                  max_threshold = undersample_range[1]) for split in splits]
        self._splits = splits
        self._n_repetitions = n_repetitions
        
        self.runs = OrderedDict()
        for fold_i, split in enumerate(self._splits):
            assert not split.name in self.runs
            STATUS[8]("Processing trainer %i: %s" % (fold_i+1, green(split.name)))
            self.runs[split.name] = []
            for rep_i in range(self._n_repetitions):
                trainer = self._trainer_prot.build()
                if self._n_repetitions > 1:
                    STATUS[8]("Repetition %i:" % (rep_i+1))
                assert len(set(split.indices[Split.TRAIN]).intersection(set(split.indices[Split.TEST]))) == 0
                trainer.use_split(
                    train = split.indices[Split.TRAIN],
                    test  = split.indices[Split.TEST]
                    )
#                trainer.load_data(X, X.label,
#                                  train = split.indices[Split.TRAIN],
#                                  test  = split.indices[Split.TEST]
#                                  )
                self.runs[split.name].append(trainer)
#        i = 0
        for trainer_name in self.runs.keys():
#            i += 1
#            if i == 1:
#                run[0].debug_log_model(8)
#            if len(run) == 1:
            STATUS[8]("Model built for split %s" % (green(trainer_name)))
#            else:
#                STATUS[8]("Model built for split %s x %i" % (green(trainer_name), len(run)))
    
    def get_ranked_test_data(self):
        '''
        Get the order of the testing data for each split and iteration after
        training. Returns a list tuples on the format (run_name, ranked_X)
        '''
        # This is a rather ugly API, but we need to ask the class itself to
        # rank the data for us if we want to use the same API for ordinary
        # runners and active learners, since the data and ranking active
        # learners needs to keep of track of which samples were considered
        # at each time step.
        ret_val = []
        for split_name in self.runs.keys():
            trainers = self.runs[split_name]
            rep_i = 0
            for trainer in trainers:
                rep_i += 1
                
                run_name = "%s, rep %i" % (split_name, rep_i)
                
                X = deepcopy(trainer.get_raw_data(Split.TEST))
                trainer.rank(X)
                ret_val.append((run_name, X))
                
#                X_pos = X[numpy.where(X.label == 'Y')[0]]
#                X_pos = X[list(range(10))]
#                trainer.print_most_important_features(X_pos)
        return ret_val
    
    def run(self, display = None, eval_metrics = ['AUC'], verbosity = 0):
        '''
        Run the metatraining until the stopping criteria for all trainers
        have been reached.
        '''
        avg_score = dict(zip(eval_metrics, [0 for _ in eval_metrics]))
#        eval_metrics = set(eval_metrics)
        from functools import partial
        get_metric = csr.ML.Evaluation.resolve_metric_by_string
        func = Trainer.calculate_ranking_score
        eval_metric_func = dict([(s, partial(func, score_func=get_metric(s))) for s in eval_metrics])
        running = True
        while running:
#            epoch += 1
            
            score = dict(zip(eval_metrics, [[] for _ in eval_metrics]))
            # Update each trainer
            for split_name in self.runs.keys():
                trainers = self.runs[split_name]
                j = 0
                for trainer in trainers:
                    j += 1
                    msg = []
                    if display:
                        msg.append(display)
                    # Some classifiers in sklearn use
                    # max_iter < 0 to signify batch training
                    max_iter = max(1, trainer._classifier.max_iter)
                    msg.append('Epoch %i/%i' % (trainer.epoch, max_iter))
                    if trainer.epoch > 1 and trainer.can_calculate_ranked_score():
                        for eval_metric in eval_metrics:
                            msg.append('%s: %.3f' % (eval_metric,  avg_score[eval_metric]))
                    if len(trainers) > 1:
                        # Two-pass to make j constant width
                        f_s = '%%%ii/%%i' % len(str(len(trainers)))
                        msg.append(f_s % (j, len(trainers)))
                    msg.append('of %s' % split_name)
                    STATUS.status(' '.join(msg))
                    if not trainer.finished():
                        trainer.train()
#                        msg = []
#                        msg.append('Split %s' % split_name)
#                        if len(trainers) > 1:
#                            msg.append('repetition %i/%i' % (j, len(trainers)))
                        if trainer.epoch > 0 and trainer.can_calculate_ranked_score(Split.TEST):
                            for eval_metric in eval_metrics:
                                func = eval_metric_func[eval_metric]
                                score[eval_metric].append(func(trainer, Split.TEST))
#                            msg.append('AUC: %.3f' % (cur_auc))
#                        STATUS[1](', '.join(msg))
                if all([trainer.finished() for trainer in trainers]):
                    running = False
#            STATUS[1]('[%s]' % ', '.join([str(val) for val in auc]))
            for eval_metric in eval_metrics:
                valid_scores = numpy.array(score[eval_metric])
                valid_scores = valid_scores[~numpy.isnan(valid_scores)]
                if len(valid_scores) > 0:
                    avg_score[eval_metric] = numpy.mean(valid_scores)
        STATUS[1]('Finished training')
        
        for eval_metric in eval_metrics:
            STATUS[1]('%s per fold:' % eval_metric)
            split_names = []
            for split_name in self.runs.keys():
                for _ in self.runs[split_name]:
                    split_names.append(split_name)
            split_names_max_len = max(map(len, split_names))
            for score_i, split_name in zip(score[eval_metric], split_names):
                left_col = split_name.rjust(split_names_max_len)
                STATUS[1]('  %s: %.3f' % (left_col, score_i))
            if avg_score[eval_metric] > 0:
                STATUS[1]('  %s: %.3f' % ('mean'.rjust(split_names_max_len), avg_score[eval_metric]))
        STATUS.status('')
    
    def check_amount_feedback(self):
        return numpy.mean([trainer.get_amount_feedback() for trainer in self.trainers()])
    
    '''
    def provide_feedback(self):
        
        STATUS[5]('Performing relevance feedback')
        for trainer in self.trainers():
            result = trainer.get_raw_data(Split.TEST)
            conf = trainer.calculate_score(result)
            indices_sorted = numpy.argsort(conf)[::-1]
            labels = []
            feedback_indices = []
            for i in indices_sorted:
                if trainer.is_feedback_provided(i): continue
                feedback_indices.append(i)
                labels.append(result[i].label)
                if len(feedback_indices) == 10: break
            labels, counts = numpy.unique(labels, return_counts = True)
            STATUS[5](' '.join(['%s: %i' % (label, count) for label, count in zip(labels, counts)]))
            trainer.provide_feedback(feedback_indices)
    '''
    
    def trainers(self):
        for split_name in self.runs.keys():
            for trainer in self.runs[split_name]:
                yield trainer
    
    def output_test_splits_ranked(self, columns):
        results = []
        for trainer in self.trainers():
            result = trainer.get_raw_data(Split.TEST)
            trainer.rank(result)
            reduced = DataStream(*columns)
            for row in result:
                d = [row[c] for c in columns]
                reduced.append(d)
            results.append(reduced)
        return results
    
    def calculate_avg_auc(self):
        auc = []
        for trainer in self.trainers():
            if trainer.can_calculate_ranked_score():
                auc.append(trainer.calculate_auc(Split.TEST))
        if len(auc) > 0:
            return numpy.mean(auc)
        else:
            import warnings
            warnings.warn('Trying to calculate AUC for metatrainer but none of the trainers support AUC', RuntimeWarning)
            return float('NaN')

'''
RFData = namedtuple('RFData', 'indices, scores, global_features')

class MetaTrainer:
    def __init__(self, X, static_classifiers, RF_data):
        self._static_classifiers = static_classifiers
        self._X = X # Not necessary
        
        self._X_p = []
        for classifier in self._static_classifiers:
            data = []
            data.append(static_model.calculate_score(X))
#            X_p = static_model.get_activation_pattern(X)
#            data.append(X_p.T)
        if any([scipy.sparse.issparse(d) for d in data]):
            data = numpy.transpose(scipy.sparse.vstack(data))
        else:
            data = numpy.transpose(numpy.vstack(data))
        
        self._RF_data
    
    def make_meta_trainer_X_t(self, rf_data):
        data = []
        return data
'''

class RF_Stage(Enum):
    SYNTHETIC_SEED = -1
    M_SEED         =  0
    USE_Y_ONLY     =  1

class ActiveLearner:
    
    def __init__(self, runner,
                 RF_mode,
                 static_models = [],
                 meta_classifier = None,
                 ):
        self._runner = runner
        
#        if train_meta_classifier: assert meta_classifier
#        self.train_meta_classifier = train_meta_classifier
        self._meta_classifier     = meta_classifier
        
        for model in static_models:
            STATUS[8](green("Static models:"))
            STATUS[8](green("  Pipeline:"))
            _log_structure(model._pipeline, 8, 2)
            STATUS[8](green("  Classifier:"))
            _log_structure(model._pipeline, 8, 2)
        self._static_models = static_models
        
        self.RF_data = []
        
        self._RF_stage_per_trainer = []
        self._RF_seed_per_trainer = []
        self._unseen_indices_per_trainer = []
        self._seen_indices_per_trainer = []
        for trainer in self._runner.trainers():
#            self._data_per_trainer.append(deepcopy(trainer.get_raw_data(Split.TEST)))
#            self._future_labels_per_trainer.append(deepcopy(trainer.get_labels(Split.TEST)))
#            self._checked_labels_per_trainer.append([])
            X_train = trainer._shared_data_view._X[trainer._indices[Split.TRAIN]]
#            print(X_train.split)
#            print(X_train.label)
#            print()
            if 'seed' in X_train.split:
                self._RF_stage_per_trainer.append([RF_Stage.SYNTHETIC_SEED])
                self._RF_seed_per_trainer.append(trainer._indices[Split.TRAIN])
            else:
                self._RF_stage_per_trainer.append([RF_Stage.USE_Y_ONLY])
                self._RF_seed_per_trainer.append([])
            self._unseen_indices_per_trainer.append(set(trainer._indices[Split.TEST][:]))
            self._seen_indices_per_trainer.append([])
    
    def get_amount_feedback(self):
        feedback = []
        for unseen_i, seen_i in zip(self._unseen_indices_per_trainer,
                                    self._seen_indices_per_trainer):
            unseen_n = len(unseen_i)
            seen_n = len(seen_i)
            feedback.append(1.0*seen_n / (seen_n + unseen_n))
        return feedback
    
    def get_ranked_test_data(self):
        '''
        Get the order of the testing data for each split and iteration after
        training. Returns a list tuples on the format (run_name, ranked_X)
        '''
        ret_val = []
        i = 0
        for trainer, seen_i in zip(self._runner.trainers(),
                                   self._seen_indices_per_trainer):
            i += 1
            run_name = "rep %i" % i
            
            X = trainer._shared_data_view.get_raw_data(seen_i)
            X.add_column('confidence')
            X.confidence = list(range(len(seen_i) + 1, 1, -1))
            ret_val.append((run_name, X))
        return ret_val
    
    def run(self, display = None, verbosity = 0):
        
        chars = string.ascii_uppercase + string.digits
        run_id = ''.join(random.SystemRandom().choice(chars) for _ in range(8))
        
        for trainer in self._runner.trainers():
            trainer.mode = 'YM|N'
        
        B = 1
        C = 100
        
        iteration = 0
        while all([feedback < 1 for feedback in self.get_amount_feedback()]):
            iteration += 1
            
            # Add temporary (negative) indices and keep track of these
            temp_indices_per_trainer = []
            for trainer, unseen_i, seen_i in zip(self._runner.trainers(),
                                                 self._unseen_indices_per_trainer,
                                                 self._seen_indices_per_trainer):
                
                assert len(set(unseen_i).intersection(seen_i)) == 0
                temp_indices = random.sample(unseen_i, max(min(C, len(unseen_i)), 0))
                trainer.provide_feedback(temp_indices)
                for i in temp_indices: trainer.reassign(i, 'N')
                temp_indices_per_trainer.append(temp_indices)
            STATUS[5]('Feedback iter %i: seen %i (%.1f%%)' % (iteration,
                                                min(map(len, self._seen_indices_per_trainer)),
                                                              100*min(self.get_amount_feedback())))
            self._runner.run(display, [], verbosity)
            # Remove temporary indices
            for trainer, temp_indices in zip(self._runner.trainers(), temp_indices_per_trainer):
                trainer.remove_feedback(temp_indices)
                for i in temp_indices: trainer.reassign(i, None)
            
            auc_per_trainer = []
            ap_per_trainer  = []
            for trainer, unseen_i_set, seen_i, seed_i, RF_stage in zip(self._runner.trainers(),
                                                                   self._unseen_indices_per_trainer,
                                                                   self._seen_indices_per_trainer,
                                                                   self._RF_seed_per_trainer,
                                                                   self._RF_stage_per_trainer):
                
                # Fix order
                unseen_i = list(unseen_i_set)
                # This breaks encapsulation, but avoids having to redo the pipeline
#                unseen_X_t = trainer._X_t[unseen_i, :]
                unseen_X   = trainer._shared_data_view.get_raw_data(unseen_i)
                unseen_X_t = trainer._shared_data_view.get_data(unseen_i)
                unseen_conf_RF = trainer._classifier.predict_proba(unseen_X_t)[:, 1].tolist()
                
                labels        = trainer.get_all_labels(Split.TEST)
                test_labels   = trainer.get_labels(Split.TEST)
                unseen_labels = [labels[i] for i in unseen_i]
                seen_labels   = [labels[i] for i in seen_i]
                seen_conf = list(range(len(seen_i) + 1, 1, -1))
                
                n_pos = Counter(seen_labels)['Y']
                n_RF  = len(seen_i)
                n_total = len(seen_i) + len(unseen_i)
                meta_X = RFData(indices = unseen_i,
                                scores  = unseen_conf_RF,
                                global_features = {'n_pos_found':    n_pos,
                                                   'n_feedback':     n_RF,
                                                   'perc_feedback':  1.0*n_RF / (n_total),
                                                   'RF_stage':       RF_stage[0].value
                                                   }
                                )
                self.RF_data.append(meta_X)
                
                if self._meta_classifier:
                    unseen_conf = self._meta_classifier.calculate_score(meta_X)
                else:
                    unseen_conf = unseen_conf_RF
                indices_sorted = [i for c, i in sorted(zip(unseen_conf, unseen_i), reverse = True)]
                
                auc = metrics.roc_auc_score(numpy.array(seen_labels + unseen_labels) == 'Y',
                                            seen_conf + unseen_conf)
                auc_per_trainer.append(auc)
                ap=metrics.average_precision_score(numpy.array(seen_labels + unseen_labels) == 'Y',
                                                      seen_conf + unseen_conf)
                ap_per_trainer.append(ap)
                
                feedback_indices = []
                for i in indices_sorted:
                    data_row = trainer._shared_data_view.get_raw_data([i])[0]
#                    if data_row.abstract == '_': continue
#                    if len(feedback_indices) < 1:
#                        STATUS[10]('Checked index %i with PMID: %s' % (i, data_row.PMID))
                    feedback_indices.append(i)
                    if len(feedback_indices) == B: break
                
                checked_labels = [labels[i] for i in feedback_indices]
                uniq_labels, counts = numpy.unique(checked_labels, return_counts = True)
                n_found = sum(numpy.array(seen_labels + checked_labels) == 'Y')
                n_total = sum(numpy.array(test_labels) == 'Y')
                STATUS[5]('RF found %i/%i (%.1f%%) AP: %.3f' % (n_found, n_total, 100.0*n_found/n_total, ap))
                STATUS[5]('Uncovered ' + ' '.join(['%s: %i' % (label, count) for label, count in zip(uniq_labels, counts)]))

                
                pos_feedback = [i for i in feedback_indices if labels[i] == 'Y']
                trainer.provide_feedback(pos_feedback)
#                trainer.provide_feedback(feedback_indices)
                
                seen_i       += feedback_indices
                unseen_i_set -= set(feedback_indices)
                
                
                raw_labels       = trainer._shared_data_view._X.label
                seen_raw_labels  = [raw_labels[i] for i in seen_i]
                seen_raw_label_counts = collections.Counter(seen_raw_labels)
                
                uniq_labels, counts = numpy.unique(seen_raw_labels, return_counts = True)
#                STATUS[5]('Uncovered (total raw) ' + ' '.join(['%s: %i' % (label, count) for label, count in zip(uniq_labels, counts)]))
                
#                if RF_stage[0] == RF_Stage.SYNTHETIC_SEED:
                train_indices = trainer._indices[Split.TRAIN]
                train_X = trainer._shared_data_view._X[train_indices]
                seed_i = [i for i, s in zip(train_indices, train_X.split) if s == 'seed']
#                print(list(zip(train_indices, train_X.split)))
                if seed_i:
                    if seen_raw_label_counts['Y'] > 0 or seen_raw_label_counts['M'] > 0:
#                    if seen_raw_label_counts['Y'] > 0:
                            STATUS[10]("Removing seed: [%s]" % ",".join(map(str, seed_i)))
                            trainer._indices[Split.TRAIN] = list(set(trainer._indices[Split.TRAIN]) - set(seed_i))
                            STATUS[10]("Training indices are now: [%s]" % ",".join(map(str, trainer._indices[Split.TRAIN])))
                #'''
                if seen_raw_label_counts['Y'] > 0:
                    if RF_stage[0] != RF_Stage.USE_Y_ONLY:
                        STATUS[5]("Switching to training on Y")
                        trainer.mode = 'Y|MN'
                        RF_stage[0] = RF_Stage.USE_Y_ONLY
                elif seen_raw_label_counts['M'] > 0:
                    if RF_stage[0] != RF_Stage.M_SEED:
                        STATUS[5]("Using M as seed")
                        trainer.mode = 'YM|N'
                        RF_stage[0] = RF_Stage.M_SEED
                else:
                    pass
                #'''
                trainer.rebuild_classifier()
#                STATUS[5]('Intertopic training data used: %i' % (len(train_ind)))
            
            STATUS[5]('Mean AUC: %0.3f' % numpy.mean(auc_per_trainer))
            STATUS[5]('Mean AP:  %0.3f' % numpy.mean(ap_per_trainer))
            
            source_name = trainer._shared_data_view._X[seen_i][0].source
            log_output_file = "log/ap_%s_%s" % (run_id, source_name)
            with open(log_output_file, 'a') as log_out:
                if len(seen_i) == 1:
                    log_out.write("%s\n" % (list(self._runner.runs.keys())[0]))
                    if self._meta_classifier:
                        log_out.write("Metaclassifier\n")
                    else:
                        log_out.write("AF only\n")
                log_out.write("%i %.3f\n" % (len(seen_i), numpy.mean(ap_per_trainer)))
            
#            B += math.ceil(B / 10)
            
        return auc_per_trainer
    
