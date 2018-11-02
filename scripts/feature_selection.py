import pandas as pd
import pysam
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sequencing_tools.viz_tools import color_encoder, okabeito_palette
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize, OneHotEncoder, LabelBinarizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import PCA, FactorAnalysis
import matplotlib.patches as mpatches
from sklearn.model_selection import LeaveOneOut, KFold, train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from helper_function import *
import glob
import os

pca_ce = color_encoder()


def pca_color(labs, colors):
    return {lab:col for lab, col in zip(labs, colors)}
    
def pca_biplot(train_df, ax):
    pca = PCA(n_components=3)
    train_df = train_df \
        .query('label !="Zero" & label != ""') 
    tdf = extract_train_cols(train_df)
    d = pca.fit_transform(StandardScaler().fit_transform(tdf))
    pca_df = pd.DataFrame(d)
    pca_df.columns = ['PC%i' %(int(col) +1) for col in pca_df.columns]
    pca_df['label'] = train_df.label.values
    color_dict = pca_color(pca_df.label.unique(), ['red','purple','green'])
    ax.scatter(pca_df.PC1,pca_df.PC2, 
            alpha=0.4, 
           # c= pca_df.row_sum,
            c= pca_ce.fit_transform(pca_df.label))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    pca_ce.show_legend(ax, title = '', 
                bbox_to_anchor = (0.7,-0.15), fontsize=13)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    return pca, pca_df

def loading_plot(train_df, pca, ax):
    loading_df = pd.DataFrame(pca.components_)
    loading_df.columns = extract_train_cols(train_df).columns
    loading_df = loading_df \
        .reset_index() \
        .rename(columns={'index':'PC'}) \
        .pipe(pd.melt, id_vars=['PC'], var_name='feature', value_name='loading') \
        .assign(feature = lambda d: d.feature.str.replace(':-[ACTG]',''))  \
        .groupby(['PC','feature'],as_index=False) \
        .agg({'loading':lambda x: np.sum(x)})  \
        .assign(color = lambda d: np.where(d.feature.str.contains('^3'), 
                                           end_ce.encoder["3' end"], 
                                           end_ce.encoder["5' end"]))\
        .query('PC == 0')\
        .sort_values('loading')
    sns.barplot(data=loading_df,
               x = 'feature', palette = loading_df['color'].tolist(),
               y='loading', ax = ax)
    ax.hlines(y=0,xmin=-1,xmax=25)
    ax.legend(bbox_to_anchor= (1,1))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for xt in ax.get_xticklabels():
        if "3'" in xt.get_text():
            col = end_ce.encoder["3' end"]
            xt.set_color(col)
        else:
            col = end_ce.encoder["5' end"]
            xt.set_color(col)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=70, 
                    rotation_mode = 'anchor',
                    ha = 'right')
    ax.set_xlabel(' ')
    ax.set_ylabel('Feature loading on PC1')
    end_ce.show_legend(ax,
              bbox_to_anchor = (0.7,0.3), 
              fontsize=13,title='')
    
    
def variance_plot(pca, ax):
    ax.plot(pca.explained_variance_ratio_*100)
    ax.xaxis.set_ticks(np.arange(pca.n_components_))
    xtick = [ 'PC%i' %(i+1) for i in range(pca.n_components_)]
    x=ax.set_xticklabels(xtick, rotation = 90)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel('% Variance explained ')
    ax.set_xlabel(' ')

def plot_outliers(ax1, ax2, train_df):
    #ax1: all disttibutions
    #ax2: outlier distributions
    sns.distplot(train_df.log_expression, ax = ax1)
    ax1.set_title('All miRNA in miRXplore\n(n=%i)' %(train_df.shape[0]))
    ax1.vlines(x = np.log10(1e6/962), ymin = 0, ymax=1, color='red')
    ax1.set_ylabel('Density')
    train_df = train_df.query('label != "" & label != "Zero"')

    for (lab, label_d), col in zip(train_df.groupby('label'),['purple','red','green']):
        sns.distplot(label_d.log_expression, 
                    label=lab,
                    color = col,
                    ax = ax2)
#        ax2.set_ylim(0,1)
    ax2.set_ylabel('Density')
    ax2.legend(title = '', bbox_to_anchor = (1,0.8))
    [ax.spines[spine].set_visible(False) for ax in [ax1, ax2] for spine in ['top','right']]
    [ax.set_xlabel('Count-per-million ($log10$)') for ax in [ax1, ax2]]
    ax2.set_title('Only outliers > 1 $\sigma$ from $\mu$\n(n=%i)' %(train_df.shape[0]))
    plt.savefig(figure_path + '/expression_cut.eps', bbox_inches='tight', transparent = True)
    #train_df.drop(['log_expression'], axis=1, inplace=True)


def read_count_table(tablename):
    samplename = os.path.basename(tablename).split('.')[0]
    return pd.read_table(tablename, names = ['seq_id','seq_count'])\
        .assign(samplename = samplename)


def labeling_expression(expression):
    exp_vector = expression[expression!=0]
    mean = exp_vector.mean()
    sd = np.std(exp_vector)
    lower_1sd = mean - sd
    high_1sd = mean + sd
    
    
    label_classes = []
    for i, ex in enumerate(expression):
        if ex == 0:
            label_class = 'Zero'
        elif ex < lower_1sd:
            label_class = 'Under-represented'
        elif ex > high_1sd:
            label_class = 'Over-represented'
        else:
            label_class = ''
        
        label_classes.append(label_class)
    return label_classes
        
def mean_cpm(count_mat):
    count_mat = count_mat/count_mat.sum(axis=0) * 1e6 
    return np.log2(count_mat.mean(axis=1)+1)


def make_pca_df():
    tables = glob.glob('/stor/work/Lambowitz/cdw2854/miRNA/new_NTT/*counts')
    tables = glob.glob('../data/*counts')
    tables = filter(lambda x: re.search('NTT[0-9]+', x), tables)
    df = pd.concat(map(read_count_table, tables))\
        .groupby('seq_id', as_index=False)\
        .agg({'seq_count':'sum'})\
        .assign(cpm = lambda d: count_to_cpm(d.seq_count)) 

    data_df = seq_to_base_table('../data/MiRxplorer.fa') \
        .merge(df, on = 'seq_id', how='right') 

    transform_df = data_df.pipe(preprocess_dataframe)

    train_df = transform_df \
        .assign(log_expression = lambda d: d.cpm.transform(lambda x: np.log2(x+1)))\
        .fillna(0) \
        .assign(label = lambda d: labeling_expression(d['log_expression'])) 
    
    return train_df


def seq_to_base_table(fa):
    infa = pysam.Fastafile(fa)
    rows = []
    for ref in infa.references:
        seq = infa.fetch(ref)
        row = (ref, seq[0], seq[1], seq[2], seq[-3], seq[-2], seq[-1])
        rows.append(row)
    return pd.DataFrame(rows, columns = ['seq_id', 'head0', 'head1', 'head2','tail0','tail1', 'tail2'])

    
