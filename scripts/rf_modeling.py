import rpy2
from rpy2.robjects import pandas2ri, Formula
from rpy2.robjects.packages import importr
import rpy2.rinterface as ri
from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
import pandas as pd
from helper_function import *

pandas2ri.activate()
rf = importr('randomForest')
stats = importr('stats')

class R_randomForest(BaseEstimator, TransformerMixin):
    '''
    Random forest regression model ported from R using rpy2
    Interface followed sklearn:
    implemented functions:
    1. fit
    2. predict
    3. fit_transform
    3. score
    '''
    def __init__(self):
        self.formula = Formula('Y~.')
        self.fitted_rf = None


    def fit(self, X, Y):
        X['Y'] = Y
        X = pandas2ri.DataFrame(X) 
        self.fitted_rf = rf.randomForest(formula = self.formula, data = X)

    def predict(self, X):
        pred = stats.predict(self.fitted_rf, pandas2ri.DataFrame(X))
        return pandas2ri.ri2py_vector(pred)
    
    def fit_transform(X, Y):
        self.fit(X, Y)
        return self.predict(X)

    def score(self, X, y):
        pred_Y = self.predict(pandas2ri.DataFrame(X))
        pred_Y = pandas2ri.ri2py_vector(pred_Y)
        return r2_score(pred_Y, y)



def rename_col(x):
    '''
    rename column names such that:
    head0 -->> 1
    tail0 --> -3
    '''
    out = ''
    if 'head' in x:
        out = '+' + str(int(x.replace('head','')) + 1)
    elif 'tail' in x:
        pos = re.search('[0-9]+$', x).group(0)
        pos = int(pos) - 3
        out = str(pos)
    return out

def train_to_cat(d):
    '''
    categorize all nucleotide columns
    '''
    for col in d.columns:
        if col != 'Y':
            d[col] = d[col].astype('category')
    return d

def test_nucleotides(nucleotides=[0,1,2,-3,-2,-1], return_model = False, validation = False):
    '''
    extract nucleotides from miRNAs according to positions (input: nucleotides)
    train model and test model
    return the test data frame
    '''

    rf = R_randomForest()
    label = 'End nucleotides' if set(nucleotides) - set([0,1,2,-3,-2,-1]) == set() else 'Random predictors'
    df = pd.read_feather('../data/miR_count.feather') \
        .groupby(["prep","seq_id"], as_index=False) \
        .agg({'seq_count':'sum'})\
        .merge(get_seq_base(shuffle = nucleotides))\
        .assign(cpm = lambda d: d.groupby('prep').seq_count.transform(count_to_cpm))\
        .assign(expected_cpm = lambda d: 1e6 / 962) \
        .assign(Y = lambda d: np.log10(d['cpm']) - np.log10(d['expected_cpm']))  \
        .query('prep == "NTT"')\
        .reset_index() 

    model_df = df.filter(regex = 'head|tail|Y')\
        .pipe(train_to_cat)
    if validation:
        # set random_state to preserve same test set between two models
        train_df, test_df = train_test_split(model_df, test_size=0.2, random_state=4)     
    else:
        train_df, test_df = model_df, model_df
    
    rf.fit(train_df.drop('Y', axis=1), train_df['Y'])
    test_df['predict'] = rf.predict(test_df)
    if return_model:
        return test_df.assign(label = label), rf
    
    else:
        return test_df.assign(label = label) 

