
# coding: utf-8

import sys, string

from joblib import Memory

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn import svm, linear_model, tree, ensemble, naive_bayes, multiclass
from sklearn import feature_selection
from sklearn import preprocessing
from sklearn import decomposition
from sklearn import cluster
from sklearn import metrics
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, StratifiedKFold
#from sklearn.cross_validation import cross_val_score, StratifiedKFold
import nltk
import json
import html
import random
import math
import numpy, scipy
import pickle
import re
from copy import deepcopy

#from Data import DataStream

def add_placeholder_for_missing_data(data):
    for row in data:
        for col in data.header:
            entry = row[col]
            # Testing 'not entry' would inadvertantly trigger on numerical
            # zero, and boolean False, which could be reasonable entries
#            if entry == '': row[col] = 'MISSING_DATA'
            if entry == '': row[col] = '_'
    return data

def strip_numerals(data):
    def substitute_numerals(text):
        # Replace numeral spans with zero
        return re.sub('\d+', '0', text)
    fields = [field for field in ['title',
                                  'abstract',
                                  'abstract_BACKGROUND',
                                  'abstract_METHODS',
                                  'abstract_RESULTS',
                                  'abstract_CONCLUSIONS'] if field in data.header]
    for row in data:
        for field in fields:
            row[field] = substitute_numerals(row[field])
    return data

def clean_text(data):
    def clean_text_inner(text):
        
        # Convert HTML entities to unicode
        while True:
            old_text = text
            text = html.unescape(text)
            if text == old_text: break

        # Remove HTML formatting
        text = re.sub('<[^<]+?>', '', text)
        
        return text
    fields = [field for field in ['title',
                                  'abstract',
                                  'abstract_BACKGROUND',
                                  'abstract_METHODS',
                                  'abstract_RESULTS',
                                  'abstract_CONCLUSIONS'] if field in data.header]
    for row in data:
        for field in fields:
            row[field] = clean_text_inner(row[field])
    return data

class StructuredAbstractPreprocessor:
    def __init__(self):
        
        features = []
        features.append(('BOW', Pipeline([
                ('selector', _Classifier_ItemSelector(key = 'sentence')),
                ('vectorizer', CountVectorizer(ngram_range = (1, 3),
                                               binary = True,
                                               lowercase = True)),
                ('f_selector', feature_selection.VarianceThreshold(threshold = 0.001)),
                ('densify', _Classifier_DenseTransformer()),
                ('typecast<float>', _Classifier_Typecast('float_')),
                ('scaling', preprocessing.StandardScaler())
                ])))
        features.append(('BOW_prev', Pipeline([
                ('selector', _Classifier_ItemSelector(key = 's_prev')),
                ('vectorizer', CountVectorizer(ngram_range = (1, 3),
                                               binary = True,
                                               lowercase = True)),
                ('f_selector', feature_selection.VarianceThreshold(threshold = 0.001)),
                ('densify', _Classifier_DenseTransformer()),
                ('typecast<float>', _Classifier_Typecast('float_')),
                ('scaling', preprocessing.StandardScaler())
                ])))
        features.append(('BOW_next', Pipeline([
                ('selector', _Classifier_ItemSelector(key = 's_next')),
                ('vectorizer', CountVectorizer(ngram_range = (1, 3),
                                               binary = True,
                                               lowercase = True)),
                ('f_selector', feature_selection.VarianceThreshold(threshold = 0.001)),
                ('densify', _Classifier_DenseTransformer()),
                ('typecast<float>', _Classifier_Typecast('float_')),
                ('scaling', preprocessing.StandardScaler())
                ])))
        features.append(('pos', Pipeline([
                ('selector', _Classifier_ItemSelector2(key = 'pos')),
#                ('vectorizer', CountVectorizer(binary = True)),
#                ('densify', _Classifier_DenseTransformer()),
                ('typecast<float>', _Classifier_Typecast('float_')),
                ('scaling', preprocessing.StandardScaler())
                ])))
        
#        dimension_reducer = decomposition.PCA(n_components = 4000)
        dimension_reducer = decomposition.PCA()
        
        classifier_model = linear_model.LogisticRegression(penalty = 'l2',
                                                           multi_class = 'multinomial',
                                                           solver = 'sag',
                                                           max_iter = 400,
                                                           verbose = 1
                                                           )
        '''
        classifier_model = multiclass.OneVsRestClassifier(
            svm.SVC(kernel = 'linear',
                    probability = True,
                    class_weight = 'balanced',
                    verbose = True
                    )
            )
        '''
        
        self.pipeline = Pipeline([
                ('features',            FeatureUnion(features)),
                ('dimension_reduction', dimension_reducer),
                ('classifier',          classifier_model)
                ])
        
    def train(self, data):
        
        X_train = DataStream('sentence', 's_prev', 's_next', 'pos', 'label')
        
        from nltk import sent_tokenize
        sentence_tokenizer = sent_tokenize
        # Pretrained tokenizer, might not be optimal on the domain...
        # It appears to work well regardless (abbreviations seem OK)
        try: nltk.data.find('tokenizers/punkt')
        except LookupError: nltk.download('punkt')
        
        sys.stderr.write('Processing data...\n')
        i = 0
        for d in data:
            # Let us assume that this order is always followed
            # The alternative is to look up the order in 'abstract'
            abs_pos = 0
            for label in ['BACKGROUND', 'METHODS', 'RESULTS', 'CONCLUSIONS']:
                block = d['abstract_' + label]
                sentences = sentence_tokenizer(block)
                sentences = [re.sub('\d+', 'NUM', s) for s in sentences]
                sys.stderr.write('%s\r' % (' ' * 40))
                sys.stderr.write('Progress: %i / %i' % (i+1, len(data)))
                for s_prev, sentence, s_next in zip(['ABSTRACT_START'] + sentences[:-1],
                                                    sentences,
                                                    sentences[1:] + ['ABSTRACT_END']
                                                    ):
                    abs_pos += 1
                    pos = 1.0*abs_pos / len(sentences)
                    X_train.append([sentence, s_prev, s_next, pos, label])
#                    sys.stderr.write('%s\n' % sentence)
            i += 1
        sys.stderr.write('\nDone.\n')
        
        target_size = 20000
        cur_size = len(X_train)
        n_splits = cur_size / target_size
        if n_splits < 2: n_splits = 2

        import itertools
        folds = StratifiedKFold(n_splits = n_splits).split(X_train, X_train.label)
        folds = list(itertools.islice(folds, 2))
        X_eval  = X_train[folds[1][1]]
        X_train = X_train[folds[0][1]]
        
        sys.stderr.write('Training on %i sentences\n' % len(X_train))
        
        sys.stderr.write('Training...\n')
        self.pipeline.fit(X_train, X_train.label)
        Y_eval_pred = self.pipeline.predict(X_eval)
        sys.stderr.write('%s\n' % metrics.classification_report(X_eval.label, Y_eval_pred))

    def reconstruct(self, data):
                
        from nltk import sent_tokenize
        sentence_tokenizer = sent_tokenize
        # Pretrained tokenizer, might not be optimal on the domain...
        # It appears to well regardless (abbreviations seem OK)
        try: nltk.data.find('tokenizers/punkt')
        except LookupError: nltk.download('punkt')
        
        sys.stderr.write('Reconstructing...\n')
        i = 0
        for d in data:
            block = d.abstract
            
            X = DataStream('sentence', 'orig_sentence', 's_prev', 's_next', 'pos')
            abs_pos = 0
            orig_sentences = sentence_tokenizer(block)
            sentences = [re.sub('\d+', 'NUM', s) for s in orig_sentences]
            sys.stderr.write('%s\r' % (' ' * 40))
            sys.stderr.write('Progress: %i / %i' % (i+1, len(data)))
            for s_prev, sentence, s_next, orig_sentence in zip(['ABSTRACT_START'] + sentences[:-1],
                                                               sentences,
                                                               sentences[1:] + ['ABSTRACT_END'],
                                                               orig_sentences
                                                               ):
                abs_pos += 1
                pos = 1.0*abs_pos / len(sentences)
                X.append([sentence, orig_sentence, s_prev, s_next, pos])
            
            Y_pred = self.pipeline.predict(X)
            for label in ['BACKGROUND', 'METHODS', 'RESULTS', 'CONCLUSIONS']:
                ii = numpy.where(numpy.array(Y_pred) == label)[0]
                block = ' '.join(X[ii].orig_sentence)
                d['abstract_' + label] = block
            
            i += 1
        sys.stderr.write('\nDone.\n')

if __name__ == "__main__":
    import sys, traceback
    import argparse
    
    parser = argparse.ArgumentParser(description = 'Preprocess data')
    parser.add_argument('--data',            nargs  = '+')
    parser.add_argument('--method', nargs = '*', default = [])

    args = parser.parse_args()
    
    records = []
    for filename in args.data:
        records.append(DataStream.parse(open(filename).read()))
    if type(records) == DataStream:
        records = [records]
    data = DataStream(*records[0].header)
    for d in records:
        data.merge(d)
    
    if 'structure' in args.method:
        structurer = StructuredAbstractPreprocessor()
        structurer.train(data)
        structurer.reconstruct(data)
    
    print(data.marshall())
