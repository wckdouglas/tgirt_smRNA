import rpy2
from rpy2.robjects import pandas2ri, Formula
from rpy2.robjects.packages import importr
import rpy2.rinterface as ri
from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, KFold
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
    
    def score(self, X, y):
        pred_Y = self.predict(pandas2ri.DataFrame(X))
        pred_Y = pandas2ri.ri2py_vector(pred_Y)
        return r2_score(pred_Y, y)


class h2o_randomForest(BaseEstimator, TransformerMixin):
    def __init__(self):
        import h2o
        from h2o.estimators.random_forest import H2ORandomForestEstimator
        h2o.init()
        self.rf = H2ORandomForestEstimator()

    def fit(self, X, y):
        '''
        X: pandas dataframe (n x m)
        y: numpy array (n)
        '''
        X_row, X_col  = X.shape
        assert(X_row > 0 and X_col > 0)
        x_colnames = X.columns.tolist()

        assert(y.ndim == 1 and X_row == len(y))
        X.reset_index(inplace=True)
        X['y'] = y.tolist()
        train_df = h2o.H2OFrame.from_python(X)
        self.rf.train(x_colnames, 'y', training_frame=train_df)


    def coeficients(self):
        '''
        return variable importance
        '''
        return self.rf._model_json['output']['variable_importances']\
            .as_data_frame()

    def predict(self, X):
        '''
        X: dataframe containing training columns
        return: predicted values (list)
        '''
        X = h2o.H2OFrame.from_python(X) 
        y = self.rf.predict(X)
        return y.as_data_frame()['predict'].tolist()
    
    def score(self, X, y):
        pred_Y = self.predict(X)
        return r2_score(pred_Y, y)

    def save_model(self, model_file):
        model_path = h2o.save_model(model=self.rf,  path = model_file)
        return model_path

    def load_model(self, model_path):
        self.rf = h2o.load_model(model_path)



def rename_col(x, inner=True):
    '''
    rename column names such that:
    head0 -->> 1
    tail0 --> -3
    '''
    out = ''
    if 'head' in x:
        offset = 1 if not inner else 4
        out = '+' + str(int(x.replace('head','')) + offset)
    elif 'tail' in x:
        offset = 3 if not inner else 6
        pos = re.search('[0-9]+$', x).group(0)
        pos = int(pos) - offset
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


def k_fold_cv(train_df, test_df):
    '''
    Do a 8-fold cv
    and return prediction results on test data frame
    and variance importance
    '''
    rf = R_randomForest()
    kf = KFold(n_splits = 8, random_state=123)
    res_df = []
    var_df = []
    for i, (train_idx, test_idx) in enumerate(kf.split(train_df)):
        sub_train_df = train_df.iloc[train_idx,:]
        sub_test_df = train_df.iloc[test_idx]  
        
        X = sub_train_df.drop('Y', axis=1)
        Y = sub_train_df['Y'].values
        rf.fit(X, Y)
        pred = rf.predict(sub_test_df)
        
        var = pandas2ri.ri2py_vector(rf.fitted_rf.rx2('importance')).reshape(-1)
        var_df.append(pd.DataFrame({'imp_score': var,
                            'k' : i,
                            'variable': sub_train_df.drop('Y', axis=1).columns}))
        res_df.append(pd.DataFrame({'pred': pred, 'Y': sub_test_df['Y'], 'k': i}))
        #print(sub_test_df.shape)
    res_df = pd.concat(res_df)
    var_df = pd.concat(var_df)
    rf.fit(train_df.drop('Y', axis=1), train_df['Y'].values)
    test_df['pred'] = rf.predict(test_df)
    return res_df, var_df, test_df


def plot_kfold_reg(res_df, ax, ce):
    '''
    plot predictions vs observations on kfold results
    '''
    scale = np.arange(-3,3)
    ax.set_xticks(scale)
    ax.set_yticks(scale)
    ax.set_xlim(scale[0], scale[-1])
    ax.set_ylim(scale[0], scale[-1])
    for k, k_df in res_df.groupby('k'):
        sns.regplot(k_df.Y, k_df.pred, 
                    label = k + 1, 
                    color = ce.encoder[k+1],
                    ci = None, 
                    truncate=False,
                    line_kws = {'alpha': 0.5, 'linestyle':'-'},
                    scatter_kws = {'alpha' : 0.5})
    ax.plot([-5,3], [-5,3], color='red')
    ax.set_xlabel('Observed $\Delta(log_{10} CPM)$')
    ax.set_ylabel('Predicted $\Delta(log_{10} CPM)$')

    ax.legend().set_visible(False)

def plot_var(var_df, ax, ce, inner=False):
    var_plot_df = var_df\
        .sort_values('imp_score', ascending=False)\
        .assign(variable = lambda d: d.variable.map(lambda x: rename_col(x, inner=inner)))\
        .assign(imp_score = lambda d: d.groupby('k').imp_score.transform(lambda x: x/x.sum()))
    sns.stripplot(data = var_plot_df,
                order = var_plot_df.variable.unique(),
                jitter=0.3,
                y = 'imp_score', 
                x=  'variable', 
                hue = 'k', palette = list(ce.encoder.values()),
                color = 'steelblue', ax = ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=70, 
                    rotation_mode =  'anchor', ha = 'right')
    ax.set_xlabel('Position')
    ax.set_ylabel('Relative importance of the position')
    lgd = ce.show_legend(ax, title = 'Subgroup', fontsize=15, ncol=2)
    lgd.get_title().set_fontsize('15')


def plot_R2(res_df, ax, ce):
    r2_df = res_df\
        .groupby('k')\
        .apply(lambda d: r2_score(d.Y,d.pred))\
        .transform(pd.DataFrame)\
        .reset_index()\
        .rename(columns={0:'r2'}) \
        .assign(colors = lambda d: d['k'].map(lambda k: ce.encoder[k+1]))
    ax.scatter(y = r2_df['r2'], x = r2_df.shape[0] * [1],
            color = r2_df.colors.tolist())
    ax.xaxis.set_visible(False)
    ax.set_ylabel('$R^2$')
    return r2_df


def plot_test(test_df, control_test_df, ax):
    prep_ce = color_encoder()
    scale = np.arange(-3,3)
    ax.set_xticks(scale)
    ax.set_yticks(scale)
    ax.set_xlim(scale[0], scale[-1])
    ax.set_ylim(scale[0], scale[-1])

    random_df = pd.concat([control_test_df.assign(label = 'Random nucleotides'),
                      test_df.assign(label = 'End nucletides')])
    plot_random_df = random_df \
        .groupby('label', as_index=False)\
        .apply(lambda D: pd.DataFrame({'r2':[r2_score(D.Y, D.pred)], 
                                    'label':[D.label.values[0]]})) \
        .merge(random_df) \
        .assign(label = lambda d: d.label + ' ($R^2: ' + d.r2.round(3).astype(str) + '$)')\
        .assign(colors = lambda d: prep_ce.fit_transform(d.label, simpsons_palette()[1:]))

    for i, ((label, color), lab_df) in enumerate(plot_random_df.groupby(['label', 'colors'])):
        sns.regplot(data = lab_df, 
                ax = ax,
                color = color,
                x = 'Y', 
                y = 'pred', 
                ci = None,
                scatter_kws={'alpha':0.5},
                line_kws = {'linestyle':':', 'alpha':0.7})
        
        ax.text(-1.7, -2.5 - i/2.5, label, fontsize=15, color = color)
    ax.plot([-5,3], [-5,3], color='red')
    ax.set_xlabel('Observed $\Delta(log_{10} CPM)$ on test set')
    ax.set_ylabel('Predicted $\Delta(log_{10} CPM)$ on test set')
