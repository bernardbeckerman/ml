import pdb
import numpy as np
import sklearn.base as base
from sklearn import linear_model
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import json
import pandas as pd
import dill

class AttributeFlattener(base.BaseEstimator, base.TransformerMixin):
    def fit(self, X, y=None):
        # fit the transformation
        return self

    def transform(self, X):
        X = X.attributes.tolist()
        Xtrans = []
        for item in X:
            tdict = {}
            for key,value in item.iteritems():
                if type(value) == type({}):
                    for key1,value1 in value.iteritems():
                        tdict[key + key1 + str(value1)] = 1
                else:
                    tdict[key + str(value)] = 1
            #print tdict
            Xtrans.append(tdict)
        return Xtrans

#load file and convert to pandas df
with open('../data/yelp_train_academic_dataset_business.json', 'r') as f:
    df = map(json.loads,f)
df = pd.DataFrame(df)

#pdb.set_trace()

from sklearn import pipeline

att_pipe = pipeline.Pipeline([
        ('at_flat', AttributeFlattener()),
        ('att_vec', DictVectorizer()),
        ('c_tfidf', TfidfTransformer()),
        ('linreg', linear_model.LassoCV())
        ])

X, y = df,df['stars'].tolist()

print ('fitting')
att_pipe.fit(X,y)
print('fitting complete')

with open('../pickle/att.dill', "wb") as f:
    dill.dump(att_pipe,f)

test = df.sample()

test_dict = [test.attributes]

pdb.set_trace()

#print (att_pipe.predict(test_dict))
#print (test['stars'])
