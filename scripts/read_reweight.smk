import pysam
import pandas as pd
from collections import defaultdict


wildcard_constraints:
    PROCESS = '[A-Za-z]+'

WORKING_PATH = '/stor/work/Lambowitz/cdw2854/miRNA'
ERCC_PATH = WORKING_PATH + '/ercc'
MIR_PATH = WORKING_PATH + '/new_NTT/MiR'
WEIGHT_TABLE = 'model/weights.pkl'
MIR_BAM_TEMPLATE = MIR_PATH + '/{SAMPLENAME}/bowtie2.bam'
MIR_FILTER_BAM_TEMPLATE = MIR_BAM_TEMPLATE.replace('.bam','.filter.bam')
CORRECTED_BAM_TEMPLATE = MIR_BAM_TEMPLATE.replace('.bam','.corrected.bam')
COUNT_BAM_TEMPLATE = MIR_BAM_TEMPLATE.replace('.bam', '.{PROCESS}.bam')
COUNT_TABLE = MIR_PATH + '/{SAMPLENAME}/count.{PROCESS}.csv'
END_BASE_TABLE = COUNT_BAM_TEMPLATE.replace('.bam','.csv')
OUT_COUNT_TABLE = MIR_PATH + '/mir_count.feather'
NUC_COUNT_TABLE = MIR_PATH + '/nuc_count.feather'
SAMPLENAMES, = glob_wildcards(MIR_BAM_TEMPLATE)


rule all:
    input:
        OUT_COUNT_TABLE,
        NUC_COUNT_TABLE


rule merge_nuc:
    input:        
        TABS = expand(END_BASE_TABLE, 
            SAMPLENAME = SAMPLENAMES,
            PROCESS = ['filter','corrected'])

    output:
        TAB = NUC_COUNT_TABLE

    run:
        def read_nuc(c):
            tab_name = os.path.basename(c)
            pro = tab_name.split('.')[1]
            samplename = os.path.basename(os.path.dirname(c))
            return pd.read_csv(c) \
                .assign(samplename = samplename + '_' + pro)

        pd.concat(map(read_nuc, input.TABS)) \
            .reset_index(drop=True)\
            .to_feather(output.TAB)



rule merge_table:
    input:
        TABS = expand(COUNT_TABLE, 
            SAMPLENAME = SAMPLENAMES,
            PROCESS = ['filter','corrected'])

    output:
        TAB = OUT_COUNT_TABLE
    
    run:
        def read_counts(c):
            tab_name = os.path.basename(c)
            pro = tab_name.split('.')[1]
            samplename = os.path.basename(os.path.dirname(c))
            return pd.read_csv(c, names = ['mir', 'mir_count']) \
                .assign(samplename = samplename + '_' + pro)

        pd.concat(map(read_counts, input.TABS)) \
            .pipe(pd.pivot_table, 
                  columns = 'samplename', 
                  values = 'mir_count',
                  index = 'mir')\
            .reset_index()\
            .to_feather(output.TAB)


rule count_corrected:
    input:
        BAM = COUNT_BAM_TEMPLATE
    
    output:
        TAB = COUNT_TABLE
    
    run:
        def make_table(inbam):
            mir_count = defaultdict(int)
            with pysam.Samfile(inbam) as bam:
                for aln in bam:
                    if aln.is_read1:
                        count = aln.get_tag('ZW') if aln.has_tag('ZW') else 1
                        mir_count[aln.reference_name] += count
            return mir_count
        
        pd\
            .DataFrame()\
            .from_dict(make_table(input.BAM), orient='index', columns = ['mir_count'])  \
            .reset_index() \
            .to_csv(output.TAB, index=False, header=False)


rule end_nuc:
    input:
        BAM = COUNT_BAM_TEMPLATE

    output:
        TAB = END_BASE_TABLE

    run:
        base_count = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        with pysam.Samfile(input.BAM) as bam:
            for read_count, aln in enumerate(bam):
                read = 'read1' if aln.is_read1 else 'read2'

                count = aln.get_tag('ZW') if aln.has_tag('ZW') else 1
                for i, b in enumerate(aln.get_forward_sequence()[:15]):
                    base_count[read][i][b] += count
                if read_count == 100000:
                    break
        
        dfs = []
        for end, end_dict in base_count.items():
            for pos, pos_dict in end_dict.items():
                dfs.append(pd.DataFrame({'base':list(pos_dict.keys()),
                            'base_count': list(pos_dict.values())})\
                    .assign(end = end)\
                    .assign(pos = pos))
        pd.concat(dfs)\
            .reset_index(drop=True)\
            .to_csv(output.TAB, index=False)
            



rule correction:
    '''
    https://github.com/wckdouglas/tgirt_smRNA_correction
    '''
    input:
        IDX = WEIGHT_TABLE,
        BAM = MIR_FILTER_BAM_TEMPLATE,

    output:
        CORRECTED_BAM_TEMPLATE

    shell:
        'tgirt_correction.py correct '\
        '-i {input.BAM} -t "ZW" '\
        '-x {input.IDX} -o {output} '
        

rule filter_bam:
    input:
        MIR_BAM_TEMPLATE

    params:
        TMP = MIR_FILTER_BAM_TEMPLATE + '_TMP'
    output:
        MIR_FILTER_BAM_TEMPLATE

    shell:
        "mkdir -p {params.TMP} "\
        '; samtools view -hF4 -F256 {input} '\
        "| awk '{{if ($6~/^[1-5]S[1-3][0-9]M[1-5]S$|^[1-3][0-9]M$|^[1-5]S[1-3][0-9]M$|^[1-3][0-9]M[1-5]S$/ || $1~\"^@\") print $0}}' "\
        "| samtools fixmate - - "\
        "| samtools view -h "\
        "| awk '$2~/^99$|^147$/ || $1 ~ /^@/ '"\
        "| samtools view -b "\
        "> {output}"\
        "; rm -rf {params.TMP}"


rule train_table:
    input:
        ERCC_PATH + '/NTTF1.bam'

    params:
        ITER = 500000

    output:
        WEIGHT_TABLE

    shell:
        'tgirt_correction.py train -i {input} -x {output} --iter {params.ITER} -n 4 '

