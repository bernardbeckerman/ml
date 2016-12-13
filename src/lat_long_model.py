#import ipdb
import numpy as np
import sklearn.base as base
from sklearn.neighbors import KNeighborsRegressor
import json
import pandas as pd
import dill

#load file and convert to pandas df
with open('../data/yelp_train_academic_dataset_business.json', 'r') as f:
    df = map(json.loads,f)
df = pd.DataFrame(df)

class ColumnSelectTransformer(base.BaseEstimator, base.TransformerMixin):
#    def __init__(self, ...):
#        # initialization code
        
    def fit(self, X, y=None):
        # fit the transformation
        return self

    def transform(self, X):
#        if len(X['latitude']) == 1:
#            return (X['latitude'],X['longitude'])
        return zip(X['latitude'],X['longitude'])

class ShellEstimator(base.BaseEstimator, base.RegressorMixin):
    """
    A shell estimator that combines a transformer and regressor into a single object.
    """
    def __init__(self):
        self.transformer = ColumnSelectTransformer()
        self.model = KNeighborsRegressor(n_neighbors=10)

    def fit(self, X, y):
        X_trans = self.transformer.fit(X, y).transform(X)
        self.model.fit(X_trans, y)
        return self
    
    def score(self, X, y):
        X_test = self.transformer.transform(X)
        self.mean_all = y.mean()
        return self.model.score(X_test, y)
    
    def predict(self, X):
        if ('latitude' in X and 'longitude' in X):
            X_test = self.transformer.transform(X)
            return self.model.predict(X_test)
        return self.mean_all

she  = ShellEstimator()

X, y = df.drop('stars', 1), df['stars']

she.fit(X, y)

test = df.sample()

print (she.predict(test))
print (test['stars'])

#ipdb.set_trace()

with open('../pickle/she2.dill', "wb") as f:
    dill.dump(she,f)
