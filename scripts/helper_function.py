#!/usr/bin/env python

import pandas as pd
import numpy as np
import re
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold, LeaveOneOut, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from sequencing_tools.viz_tools import color_encoder, okabeito_palette, simpsons_palette
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from operator import itemgetter
import os
import pysam
from collections import Counter
end_ce = color_encoder()
end_ce.fit(["3' end", "5' end"],['darkgoldenrod','purple'])

def positioning(x):
    return x[-1]

def count_to_cpm(count_array):
    count_array = np.true_divide(count_array,count_array.sum()) * 1e6 
    return count_array

def get_end(x):
    if 'head' in x:
        return "5' N+"
    elif 'tail' in x:
        return "3' N-"

def make_column_name(colnames):
    colnames = pd.Series(colnames)
    col_d = pd.DataFrame({'nucleotide':colnames.str.slice(-1),
                 'position':colnames.str.slice(4,5),
                 'end':colnames.map(get_end),
                 'colname': colnames}) \
        .assign(offset = lambda d: np.where(d.colname.str.contains('5|head'),-1, 3)) \
        .assign(adjusted_position = lambda d: np.abs(d.position.astype(int) - d.offset))\
        .assign(colnames = colnames) \
        .assign(outnames = lambda d: d.end + d.adjusted_position.astype(str) +':'+ d.nucleotide)
    return col_d.outnames


def preprocess_rf_dataframe(df):
    '''
    labelEncoder for categorical training in RF
    '''
    le = LabelEncoder()
    le.fit(list('ACTG'))
    nucleotides = df.columns[df.columns.str.contains('^head|^tail')]
    poses = map(lambda x: int(x[-1]),nucleotides)
    offset  = max(poses)
    for col in nucleotides:
        pos_end = "5'" if 'head' in col else "3'"
        pos = int(col[-1])
        pos = pos + 1 if pos_end == "5'" else pos - offset -1
        new_col = pos_end +':'+ str(pos)
        df[new_col] = le.transform(df[col])
    
    return df

def extract_training(df):
    X = df.filter(regex="^[35]':")
    Y = df.Y
    return X, Y



def preprocess_dataframe(df):
    nucleotides = df.columns[df.columns.str.contains('^head|^tail')]
    dummies = pd.get_dummies(df[nucleotides]) 
    dummies.columns = make_column_name(dummies.columns)
    df = pd.concat([df.drop(nucleotides, axis=1),
                dummies],axis=1) 
    return df

def extract_train_cols(d):
    return d.loc[:,d.columns[d.columns.str.contains('^head|^tail|^5|^3')]]



def get_label(x):
    lab = ''
    if 'NTM' in x:
        lab = 'Diaminopurine'
    if 'noTA' in x:
        lab = 'NTT-noTA'
    elif 'NTTR' in x:
        lab = 'NTTR'
    elif 'NTT' in x:
        lab = 'NTT'
    elif 'NTC' in x or re.search('R[0-9]', x):
        lab = 'NTC'
    elif 'UMI' in x:
        lab = 'UMI'
    elif 'circ' in x:
        lab = 'CircLigase'
    return lab

def correct_prep(x):
    prep = ''
    if 'CleanTag' in x:
        prep = 'CleanTag'
    elif 'NEXTflex' in x:
        prep = 'NEXTflex'
    elif '4N' in x:
        prep = '4N'
    elif 'NEBNext' in x:
        prep = 'NEBNext'
    elif 'TruSeq' in x:
        prep = 'TruSeq'
    elif 'NTC' in x or re.search('R[0-9]', x):
        prep = 'NTC'
    elif 'NTTR' in x:
        prep = 'NTTR'
    elif 'noTA' in x:
        prep = 'NTT-noTA'
    elif 'NTT' in x:
        prep = 'NTT'
    elif 'MTT' in x or 'NTM' in x or 'Diamin' in x:
        prep = 'MTT'
    elif 'UMI' in x:
        prep = '6N-NTTR'
    elif 'NTTR' in x:
        prep = 'NTTR'
    elif 'circ' in x or 'Circ' in x:
        prep = 'CircLigase'
    else:
        prep = x

    if re.search('[Cc]orrec', x):
        prep += ' (Corrected)'
    return prep


prep_encoder = {'4N': '#D55E00',
                 'CleanTag': '#009E73',
                 'NEBNext': '#56B4E9',
                 'NEXTflex': '#F0E442',
                 'TruSeq': '#CC79A7',
                 'NTC': '#999999',
                 'NTT': '#0072B2',
                 'MTT': '#E69F00',
                 '6N-NTTR': 'black',
                 'NTTR': '#96b6ea',
                 'TruSeq': '#CC79A7',
                 'NTC (Corrected)': 'red',
                 '6N-NTTR (Corrected)': 'red',
                 'NTTR (Corrected)': 'red',
                 'MTT (Corrected)': 'red',
                 'NTT (Corrected)': 'red',
                 'NTT-noTA (Corrected)': 'red',
                 'NTC': 'red',
                 'NTT': 'red',
                 'NTT-noTA': 'red',
                 'MTT': 'red',
                 '6N-NTTR': 'red',
                 'NTTR': 'red',
                 '4N': '#D55E00',
                 'NEXTflex': '#D55E00',
                 'CleanTag': 'gray',
                 'NEBNext': '#56B4E9',
                 'TruSeq': 'gray',
                'CircLigase':'#5e0700'}
figure_path = '/stor/work/Lambowitz/cdw2854/miRNA/new_NTT'


def get_published(return_count = False):
    #published dataframe
    id_table = pd.read_table('/stor/work/Lambowitz/ref/Mir9_2/MiRxplorer.fa.fai',
                            names = ['id','0','1','2','3'])
    sample_table = pd.read_table('/stor/home/cdw2854/miRNA_seq/download_data/sra.tsv', names=['GSM','prep_name'])   
    pdf=pd.read_table('/stor/work/Lambowitz/cdw2854/miRNA/published_data/mirExplore_count.tsv') \
                .pipe(pd.melt,id_vars='id', 
                              var_name='samplename', 
                              value_name='count') \
                .assign(GSM = lambda d: d['samplename'].map(lambda x: x.split('_')[0]))\
                .merge(sample_table) \
                .drop(['GSM'],axis=1) \
                .pipe(lambda d: d[d['id'].isin(id_table.id.tolist())]) \
                .assign(cpm = lambda d: d.groupby(['samplename','prep_name'])['count'].transform(count_to_cpm))   \
                .assign(prep_name = lambda d: d.prep_name.str.replace('.SynthEQ',''))
    if return_count:
        return pdf
    else:
        return pdf.drop('count',axis=1)


def get_seq_base(shuffle=None):
    fa = '/stor/work/Lambowitz/ref/Mir9_2/MiRxplorer.fa'
    bases = []
    if not shuffle:
        indices = [0,1,2,-3,-2,-1]
    else:
        indices = shuffle

    ref_fa = pysam.Fastafile(fa)
    for ref in ref_fa.references:
        b = np.array(list(ref_fa[ref]))
        b = b[indices]
        b = list(b)
        b.append(ref)
        bases.append(b)
    
    if len(indices) == 6:
        headers = ['head0','head1','head2','tail0','tail1','tail2']

    else:
        headers = []
        col_count = Counter()
        for j in indices:
            if j < 0:
                a = 'tail'
            else:
                a = 'head'
            i = col_count[a]
            col_count[a] += 1
            headers.append(a + str(int(i)))
    headers.append('seq_id')
    return pd.DataFrame(bases, columns = headers)
        
    



#### plotting ####
def train_lm(d, ax, red_line=True, fitted_line=False):
    X = extract_train_cols(d).values
    Y = d['Y']
    lm = Ridge(fit_intercept=False)    
    lm.fit(X, Y)
    pred_Y = lm.predict(X)
    rsqrd = r2_score(Y, pred_Y)
    rho, pval = pearsonr(pred_Y, Y)
    ax.scatter(Y, pred_Y, alpha=0.7, color = 'steelblue')
    ax.text(-2.5, 1, '$R^2$ = %.3f' %(rsqrd), fontsize=13)
    ax.text(-2.5, 0.8, r'$\rho$ = %.3f' %rho, fontsize=13)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel(r'Observed $\Delta$($log_{10}$ CPM)', fontsize=15)
    ax.set_ylabel(r'Predicted $\Delta$($log_{10}$ CPM)', fontsize=15)
    perfect = np.arange(np.min(Y), np.max(Y), 0.01)
    if red_line:
        ax.plot(perfect, perfect, color= 'red',linewidth=4)

    if fitted_line:
        lm_fit = LinearRegression()
        lm_fit.fit(Y.values.reshape(-1,1), pred_Y)
        ax.plot(perfect, lm_fit.predict(perfect.reshape(-1,1)), 
                color = 'gold', linestyle='--', linewidth = 4)

    ax.set_xlim(-3.5,2)
    ax.set_ylim(-3.5,2)
    return lm


def plot_coef(lm, sample_df, ax=None):
    coef_df = pd.DataFrame({'label':extract_train_cols(sample_df).columns,
                                'coef':lm.coef_}) \
        .sort_values('coef')
    
    if ax:
        sns.barplot(data=coef_df, x='label',
            y='coef',color='steelblue', 
            ax = ax)

        for xt in ax.get_xticklabels():
            if "3'" in xt.get_text():
                col = end_ce.encoder["3' end"]
                xt.set_color(col)
            else:
                col = end_ce.encoder["5' end"]
                xt.set_color(col)


        ax.hlines(y = 0, color='black', xmin=-1, xmax=100)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=70, 
                            rotation_mode='anchor', ha = 'right')
        ax.set_xlabel('')
        ax.set_ylabel('Coefficient', fontsize=15)
        sns.despine()
        #end_ce.show_legend(ax, bbox_to_anchor = (0.7,0.3), fontsize=13)
    else:
        return coef_df

def cpm_rmse_function(validation_df, predicted, observed, colors = ["#E69F00","#56B4E9"], seq_id=None):
    validation_df = pd.DataFrame({'Corrected': np.log10(observed) - predicted,
                              'Uncorrected': np.log10(observed)}) 
    if seq_id is not None:
        validation_df = validation_df \
            .assign(seq_id = seq_id) \
            .pipe(pd.melt, id_vars = 'seq_id', 
                var_name = 'correction', 
                value_name = 'log10_cpm')
    else:
        validation_df = validation_df \
            .pipe(pd.melt, 
                var_name = 'correction', 
                value_name = 'log10_cpm')
    validation_df = validation_df \
        .assign(pseudo_count = lambda d: d['log10_cpm'].rpow(10))\
        .assign(cpm = lambda d: d.groupby('correction')['pseudo_count'].transform(count_to_cpm)) \
        .assign(new_log10_cpm = lambda d: np.log10(d['cpm']))\
        .assign(predicted = 1e6/962) \
        .assign(err = lambda d: d.cpm - d.predicted)

    
    
    rmse_df = validation_df\
        .groupby('correction', as_index=False) \
        .agg({'err':lambda x: np.sqrt((x**2).mean())}) \
        .assign(color = colors) \
        .assign(label = lambda d: d.correction + ' (RMSE: ' + d.err.map(lambda x: '%3.f' %x) + ')')
    return validation_df, rmse_df

def plot_rmse(lm, main_df, ax=None):
    colors = ["#E69F00","#56B4E9"]
    
    validation_df, rmse_df = cpm_rmse_function(main_df, lm.predict(extract_train_cols(main_df).values), 
                main_df['cpm'], 
                colors=colors, 
                seq_id = main_df.seq_id) 
    
    if ax:
        sns.stripplot(data=validation_df, 
              x = 'correction', 
              y = 'log10_cpm',
             jitter = 0.2,
             palette = colors,
             alpha=0.4, 
             ax = ax)
        ax.hlines(y = np.log10(1e6/962), xmin=0,xmax=4, color = 'red')
        ax.xaxis.set_visible(False)
        sns.despine()
        ax.set_ylabel(r'$log_{10}$ CPM', fontsize=15)
        ax.set_xlabel(' ')

        pat = [mpatches.Patch(color=row['color'], label=row['label']) for i, row in rmse_df.iterrows()]
        ax.legend(handles=pat, loc='best', fontsize=13)
    return validation_df



def cross_validation(X_train, X_test, Y_train, Y_test, i, obs_cpm):
    lm = Ridge(fit_intercept=False)
    lm.fit(X_train, Y_train)
    
    predicted = lm.predict(X_test)
    validation_df, rmse_df = cpm_rmse_function(validation_df, predicted, obs_cpm)
    return validation_df.assign(train_idx = i), rmse_df.assign(train_idx = i)

def plot_cv(ax, train_df, k):
    X = extract_train_cols(train_df).values
    Y = train_df['Y']
    kf = KFold(n_splits=k, random_state=0)
    valid_dfs = []
    rmse_dfs = []
    train_idx, test_idx = [],[]
    variations = []
    coef_dfs = []
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index,:], X[test_index,:] 
        Y_train, Y_test = Y[train_index], Y[test_index]
        validation_df, rmse_df = cross_validation(X_train, X_test, Y_train, Y_test, i, train_df.cpm[test_index]) 
        valid_dfs.append(validation_df)
        rmse_dfs.append(rmse_df)
        train_idx.append(train_index)
        test_idx.append(test_index)
        variations.append(np.max(Y_test) - np.min(Y_test))
        coef_dfs.append(plot_coef(lm, train_df))
    error_df = pd.concat(rmse_dfs)
    validation_df = pd.concat(valid_dfs)
    coef_df = pd.concat(coef_dfs)
    
    
    sns.stripplot(data=error_df, x = 'correction', color = 'steelblue', 
                y = 'err', order=['Uncorrected','Corrected'])
    ax.set_xlabel(' ')
    ax.set_ylabel(r'Root-mean-square error ($\Delta$ CPM)', fontsize=15)
    par_error_df = error_df.pipe(pd.pivot_table, columns='correction', index=['train_idx'],values='err')
    for i, row in par_error_df.iterrows():
        ax.plot([0,1], [row['Uncorrected'],row['Corrected']], color='grey', alpha=0.5)
    return validation_df, error_df, train_idx, test_idx, variations, coef_df

def plot_comparison(ax, rmse_df):
    rmse_df = rmse_df\
        .groupby(['prep','prep_name'], as_index=False)\
        .agg({'error':lambda x: np.sqrt(np.mean(x.pow(2)))}) \
        .sort_values('error') 
    
    colors = ['#444a60',
            '#605e5d', 
            '#999961',
             '#ff0000', 
            '#20b57e',
            '#3d8268']
    labs = ['Bioo NEXTflex','4N','NEBNext','TGIRT','CleanTag', 'TruSeq']
    ce_encoder = {lab:col for lab, col in zip(labs,colors)}
    rmse_df['colors'] = rmse_df.prep.map(ce_encoder)
    print(rmse_df.pipe(lambda d: d[pd.isnull(d.colors)]))
    sns.barplot(data=rmse_df, 
            x = 'prep_name', 
            y = 'error', 
           palette = rmse_df['colors'],
           ax = ax)
    sns.despine()
    xt = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    pat = [mpatches.Patch(color=col, label=lab) for lab, col in ce_encoder.items()]

    ax.legend(handles=pat, 
              bbox_to_anchor = (0.48,0.46),
              fontsize=13)
    for xt in ax.get_xticklabels():
        if 'Corrected' in xt.get_text():
            xt.set_color('red')
    ax.set_xlabel('Library')
    ax.set_ylabel('RMSE (CPM)')
    
def validation_to_rmse(validation_df):
    return validation_df \
        .assign(prep = lambda d: d.prep + '.' + d.correction) \
        .assign(cpm = lambda d: np.power(10, d.new_log10_cpm))\
        .filter(regex = 'prep|^cpm$|seq_id') \
        .assign(samplename = lambda d: d.prep )\
        .rename(columns = {'prep':'prep_name',
            'seq_id':'id'})  \
        .pipe(lambda d: pd.concat([d, get_published()], sort=True, axis=0)) \
        .assign(model = 1e6/962.0) \
        .assign(error = lambda d: d.cpm-d.model) \
        .assign(log_cpm = lambda d: np.log10(d.cpm+1)) \
        .assign(prep = lambda d: d.prep_name.map(lambda x: x.split('.')[0]))\
        .assign(prep = lambda d: d.prep.map(lambda x: x.split('_')[0]))\
        .assign(prep = lambda d: list(map(lambda x,y: 'Bioo NEXTflex' if 'NEXTflex' in y else x, d.prep, d.prep_name))) \
        .assign(prep = lambda d: np.where(d.prep.str.contains('NTC|Diamino|NTT|UMI|ratio'), 'TGIRT', d.prep)) \
        .assign(prep_name = lambda d: np.where(d.prep.str.contains('NTC|Diamino|NTT|UMI|ratio'), 'TGIRT-' + d.prep_name , d.prep_name))
    
    
def plot_figure(fig, main_df, prep, 
                return_model=False,
               no_compare = False,
               lm = False):
    lm_ax = fig.add_axes([0,0.5,0.4,0.5])
    #lm_ax = fig.add_subplot(2,2,1)
    if not lm:
        lm = train_lm(main_df, lm_ax)
    coef_ax = fig.add_axes([0.5,0.5,0.5,0.5])
    #coef_ax = fig.add_subplot(2,2,2)
    plot_coef(lm, main_df, coef_ax)
    rmse_ax = fig.add_axes([0,0,0.4,0.4])
    #rmse_ax =fig.add_subplot(2,2,3)
    validation_df = plot_rmse(lm, main_df, rmse_ax) \
        .assign(prep = prep)
    
    #all_compare_ax = fig.add_subplot(2,2,4)
    rmse_df = validation_to_rmse(validation_df)
    if not no_compare:
        all_compare_ax = fig.add_axes([0.5,0,0.5,0.38])
        plot_comparison(all_compare_ax, rmse_df)
    if return_model:
        return validation_df, lm
    else:
        return validation_df



def plot_rmse_strip(ax, published, corrected, original):
    prep_rmse_df = pd.concat([published, corrected, original]) \
        .assign(model = 1e6/962) \
        .assign(error = lambda d: d.cpm - d.model) \
        .groupby(['samplename','prep'], as_index=False)\
        .agg({'error': lambda x:  np.sqrt(np.mean(x.pow(2)))}) 

    prep_order = prep_rmse_df\
        .groupby('prep',as_index=False)\
        .agg({'error':'mean'})\
        .sort_values('error').prep
    colors = np.where(prep_order.str.contains('Correct'),
                      'red', 
                      prep_order.map(prep_encoder))
    
    sns.stripplot(data = prep_rmse_df, jitter=0.2,
            order=prep_order,
           x = 'prep',
           y = 'error',
           palette = colors,
           ax = ax, s = 10, alpha=0.7)
    xts = ax.set_xticklabels(ax.get_xticklabels(), 
                         rotation=70, ha = 'right',
                        rotation_mode = 'anchor')
    ax.set_xlabel('')
    ax.set_ylabel('RMSE (CPM)')
    for xt in ax.get_xticklabels():
        if 'Corrected' in xt.get_text():
            xt.set_color('red')
    sns.despine()
