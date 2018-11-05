#!/usr/bin/env python

import pandas as pd
import glob
import gzip
import os

def read_file(filename):
    mir_id= []
    counts = []
    samplename = os.path.basename(filename).replace('.clipped.trimmed.filtered.calibratormapped.counts.txt.gz','')
    for line in gzip.open(filename,'rt'):
        line = line.lstrip().rstrip('\n')
        fields = line.split(' ')
        mir_id.append(fields[1])
        counts.append(fields[0])
    return pd.DataFrame({'id':mir_id,
                         samplename:counts})

        
def merge_table(x, y):
    return x.merge(y, how='outer', on='id')
        
datapath = '/stor/work/Lambowitz/cdw2854/miRNA/published_data'
files = glob.glob(datapath + '/*gz')
dfs = map(read_file, files)
df = reduce(merge_table, dfs) \
        .fillna(0) \
        .assign(id = lambda d: map(lambda x: x.split(';')[0], d['id']))
out_table = datapath + '/mirExplore_count.tsv'
df.to_csv(out_table, index=False, sep='\t')
print 'Written ', out_table

