import sklearn.base as base

typed = type({})

class AttributeFlattener(base.BaseEstimator, base.TransformerMixin):
#    def __init__(self):
#        self.vec = CountVectorizer(token_pattern=r'[^\|]+')
#        True

    def fit(self, X, y=None):
        # fit the transformation
        return self

    def transform(self, X):
        Xtrans = []
        for item in X:
            tdict = {}
            for key,value in item.iteritems():
                if type(value) == typed:
                    for key1,value1 in value.iteritems():
                        tdict[key + key1 + str(value1)] = 1
                else:
                    tdict[key + str(value)] = 1
            #print tdict
            Xtrans.append(tdict)
        return Xtrans
