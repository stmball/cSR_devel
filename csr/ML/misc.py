
from sklearn.base import BaseEstimator, TransformerMixin

# Selection transformer
# Takes dict-like objects and extracts the value for the specified
# key
# TODO make this work for list-like objects too (using indices)
# TODO change the constructor to (optionally) take lists of keys
class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.keys = key.split(' OR ')
    def fit(self, x, y = None): return self
    def transform(self, data):
        # Given data of length n, construct keys x n size table
        blocks = [data[key] for key in self.keys]
        # Merge column by column, resulting in a 1 x n size vector
        return ['\n'.join(block) for block in zip(*blocks)]

