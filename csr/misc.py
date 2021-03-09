
from future.utils import iteritems

__cache_store = {}
def CACHE(func):
    if not func in __cache_store:
        __cache_store[func] = {}
    def __inner(*args, **kwargs):
        args_str = [hasattr(arg, 'tostring') and arg.tostring() or str(arg) for arg in args]
        key = str((args_str, kwargs))
        if not key in __cache_store:
            __cache_store[key] = func(*args, **kwargs)
        return __cache_store[key]
    return __inner

class ObjectBuilder:
    '''
    Takes a structure of dicts, list, and objects and constructs
    an object based on the type definitions given at construction.
    By convention (or by definition), capitalized dict entries are
    treated as classes, and non-capitalized dict entries are treated
    normally.
    Thus, {Foo: [1, 2, 3]} would construct a Foo object with 1, 2, 3
    as arguments, provided a Foo class is defined in the 'type_def'
    argument to the constructor. Named arguments can be passed only
    in lower case, e.g. {SVC: {class_weight: 'balanced', tol: 0.01}}.
    I.e. {SVC: {C: 1}} is not supported at present.
    This class is mainly intended to provide a base for constructing
    domain languages for pipeline construction. Examples can be found
    in the 'data/param' directory (in YAML).
    '''
    def __init__(self, type_defs):
        '''
        Construct a builder with the given type translations.
        '''
        self.type_defs = type_defs
    
    def parse(self, param):
        '''
        Construct an object from the given structure.
        '''
        # We get this from YAML, so we should only get primitives
        if type(param) == list:
            # Recurse on each item
            return [self.parse(p) for p in param]
            ret_val = []
            for p in param:
                assert len(p) == 1
        elif type(param) == dict:
            if all([klass in self.type_defs for klass in param.keys()]):
                # Treat as type definitions
                for k in param.keys():
                    v = param[k]
                    class_name = k
                    if not class_name in self.type_defs:
                        raise TypeError("Unknown type name: '%s'" % class_name)
                    args = self.parse(v)
                    klass = self.type_defs[class_name]
                    if type(args) == dict:
                        try:
                            return klass(**args)
                        except TypeError as e:
                            raise TypeError('Failed to instantiate %s(%s): %s' % (
                                    klass.__name__,
                       ', '.join(['%s=%s'%(x, y.__class__.__name__) for x, y in iteritems(args)]),
                                    e))
                    elif type(args) == list:
                        try:
                            return klass(*args)
                        except TypeError as e:
                            raise TypeError('Failed to instantiate %s(%s): %s' % (
                                    klass.__name__,
                                    ', '.join([x.__class__.__name__ for x in args]),
                                    e))
                    else: # Primitive type
                        try:
                            return klass(args)
                        except TypeError as e:
                            raise TypeError('Failed to instantiate %s(%s): %s' % (
                                    klass.__name__,
                                    args.__class__.__name__,
                                    e))
            else:
                # Treat as a dict
                return dict([(k, self.parse(v)) for k, v in iteritems(param)])
        else:
            # Base case
            return param
    #        raise ArgumentError("Trying to parse '%s' (list or dict expected)" % type(param))

from sklearn.base import BaseEstimator, TransformerMixin

class DenseTransformer(BaseEstimator, TransformerMixin):
    def transform(self, X, y = None, **fit_params):
        return X.todense()
    def fit_transform(self, X, y = None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)
    def fit(self, X, y, **fit_params):
        return self

class Prototype:
    '''
    Implementation of the Prototype patterns without
    resorting to deepcopying. This should therefore be
    safe to use even for classes that cannot be cloned
    such as files, sockets, or tensorflow models (which
    depend on gpu state)
    '''
    class PrototypeObj:
        def __init__(self, klass, *args, **kv_args):
            self._klass = klass
            self._args = args
            self._kv_args = kv_args
            self.__name__ = 'Prototype<%s>' % klass.__name__
            
        def build(self):
            return self._klass(*self._args, **self._kv_args)
    
    def __init__(self, klass):
        self._klass = klass
        self.__name__ = 'Prototype<%s>' % klass.__name__

    def __call__(self, *args, **kv_args):
        return Prototype.PrototypeObj(self._klass, *args, **kv_args)
                                                        
