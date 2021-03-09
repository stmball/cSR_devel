
# ~~~~~~~~~~~~ Imports ~~~~~~~~~~~~

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.class_weight import compute_class_weight

from gensim.models.keyedvectors import KeyedVectors

# Shut up keras 'using XX backend'
import sys, os
#stderr = sys.stderr
#sys.stderr = open(os.devnull, 'w')
import keras.backend
import keras.layers
import keras.models
#sys.stderr = stderr

import numpy

import types
import tempfile

# ~~~~~~~~~~~~ Setup ~~~~~~~~~~~~

def make_keras_picklable():
    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            keras.models.save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        d = { 'model_str': model_str }
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            model = keras.models.load_model(fd.name)
        self.__dict__ = model.__dict__


    cls = keras.models.Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__

make_keras_picklable()

# Shut up tensorflow
# It would be better to redirect output to the logger
# but this should do for now
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ~~~~~~~~~~~~ Definitions ~~~~~~~~~~~~

class DeepNetwork(BaseEstimator, TransformerMixin):
    '''
    Wrapper for keras models to simplify construction and to
    use the same API as sklearn.
    '''
    def __init__(self,
                 layers,
                 loss         = 'binary_crossentropy',
                 optimizer    = 'rmsprop',
                 input_type   = 'text',
                 shuffle      = True,
                 batch_size   = 32,
                 class_weight = None,
                 seq_length   = 100,
                 embeddings   = None,
                 emb_size     = None,
                 emb_train    = False,
                 vocabulary   = None,
                 epochs       = 100):

        self.layers       = '...' # For pretty printing
        self.loss         = loss
        self.optimizer    = optimizer
        self.shuffle      = shuffle
        self.batch_size   = batch_size
        self.class_weight = class_weight
        self.seq_length   = seq_length
        self.embeddings   = embeddings
        self.emb_size     = emb_size
        self.vocabulary   = vocabulary
        self.epochs       = epochs
        
        assert input_type in ['text', 'float32']
        
        if input_type == 'text':
            sequence_input = keras.layers.Input(shape = (seq_length,), dtype = 'int32')
            if embeddings is None:
                embedding_layer = keras.layers.Embedding(4000,
                                                         128,
                                                         input_length = seq_length,
                                                         trainable = emb_train)
            else:
                # For now we assume binary format
                embedding_vectors = KeyedVectors.load_word2vec_format(embeddings,
#                                                                      binary = True,
#                                                                      unicode_errors='ignore',
                                                                      limit = emb_size)
                embedding_layer = embedding_vectors.get_keras_embedding(train_embeddings = emb_train)
            x = embedding_layer(sequence_input)
        else: # Input is numerical - no processing required
            sequence_input = keras.layers.Input(shape = (seq_length,), dtype = 'float32')
            x = sequence_input
        for layer in layers:
            x = layer(x)

        self._model = keras.models.Model(sequence_input, x)
        
        self._model.compile(loss = loss,
                            optimizer = optimizer)
        self.y_labels = None
        self.max_iter = epochs
        self._epoch = 0

    #'''
    def partial_fit(self, x, y, classes):
        if not self.y_labels:
            self.y_labels = classes
        assert set(self.y_labels) == set(classes)
        
        y_index = [self.y_labels.index(y_i) for y_i in y]
        index_to_one_hot = numpy.eye(len(self.y_labels))
        y_one_hot = index_to_one_hot[numpy.array(y_index)]
        
        if self.class_weight is None:
            sample_weight = None
        else:
            if self.class_weight == "balanced":
                class_weight = dict(zip(self.y_labels, compute_class_weight('balanced', self.y_labels, y)))
            else:
                class_weight = self.class_weight
            sample_weight = numpy.array([class_weight[y_i] for y_i in y])
        
        self._model.fit(x, y_one_hot,
                        epochs = self._epoch + 1,
                        batch_size = self.batch_size,
                        shuffle = True,
                        sample_weight = sample_weight,
                        initial_epoch = self._epoch,
                        verbose = 0)
        self._epoch += 1
    #'''
    
    def fit(self, x, y):
        if not self.y_labels:
            self.y_labels = list(numpy.unique(y))
        assert set(self.y_labels) == set(numpy.unique(y))
        
        y_index = [self.y_labels.index(y_i) for y_i in y]
        index_to_one_hot = numpy.eye(len(self.y_labels))
        y_one_hot = index_to_one_hot[numpy.array(y_index)]
    
        if self.class_weight is None:
            sample_weight = None
        else:
            if self.class_weight == "balanced":
                class_weight = dict(zip(self.y_labels, compute_class_weight('balanced', self.y_labels, y)))
            else:
                class_weight = self.class_weight
            sample_weight = numpy.array([class_weight[y_i] for y_i in y])
        
        self._model.fit(x, y_one_hot,
                        epochs = self.max_iter,
                        batch_size = self.batch_size,
                        shuffle = True,
                        sample_weight = sample_weight,
                        verbose = 0)
        self._epoch += self.max_iter

    def predict(self, x):

        assert self.y_labels is not None
        
        y_one_hot = self._model.predict(x)
        # Assume the final activation is softmax
        y = numpy.where(y_one_hot == 1)[1]
        return [self.y_labels[y_i] for y_i in y]

    def predict_proba(self, x):
        
        assert self.y_labels is not None
        assert set(self.y_labels) == set(['Y', 'N'])
        
        pred = self._model.predict(x)
        # Assume the final activation is softmax (logistic regression)
        prob = pred[:, self.y_labels.index('Y')]
        return numpy.vstack([1-prob, prob]).transpose()
    
#    def get_activation_pattern_per_layer(self, X):
#        inp = self._model.input
#        outputs = [layer.output for layer in self._model.layers]
#        functor = keras.backend.function([inp] + [keras.backend.learning_phase()], outputs)
#        
#        layer_outs = functor([X, 1.])
#        print layer_outs
    
    def get_activation_pattern(self, X):
#        return self.get_activation_pattern_per_layer(X)[-1]
        n_samples = X.shape[0]
        activation_model = Model(inputs  = self._model.input,
                                 outputs = self._model.layers[-2].output)
        scores = activation_model.predict(X)
        return scores.shape[0] == n_samples
