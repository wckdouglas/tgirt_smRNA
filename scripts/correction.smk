import pysam
from collections import defaultdict


wildcard_constraints:
    PROCESS = '[A-Za-z]+'

WORKING_PATH = '/stor/work/Lambowitz/cdw2854/miRNA'
ERCC_PATH = WORKING_PATH + '/ercc'
MIR_PATH = WORKING_PATH + '/new_NTT/MiR'
WEIGHT_TABLE = ERCC_PATH + '/weights.pkl'
MIR_BAM_TEMPLATE = MIR_PATH + '/{SAMPLENAME}/bowtie2.bam'
MIR_FILTER_BAM_TEMPLATE = MIR_BAM_TEMPLATE.replace('.bam','.filter.bam')
CORRECTED_BAM_TEMPLATE = MIR_BAM_TEMPLATE.replace('.bam','.corrected.bam')
COUNT_BAM_TEMPLATE = MIR_BAM_TEMPLATE.replace('.bam', '.{PROCESS}.bam')
COUNT_TABLE = MIR_PATH + '/{SAMPLENAME}/count.{PROCESS}.tsv'
SAMPLENAMES, = glob_wildcards(MIR_BAM_TEMPLATE)
print(MIR_BAM_TEMPLATE,SAMPLENAMES)

rule all:
    input:
        expand(COUNT_TABLE, 
            SAMPLENAME = SAMPLENAMES,
            PROCESS = ['filter','corrected'])

rule count_corrected:
    input:
        BAM = COUNT_BAM_TEMPLATE
    
    output:
        TAB = COUNT_TABLE
    
    run:
        mir_count = defaultdict()
        with pysam.Samfile(input) as bam:
            for aln in bam:
                try:
                    mir_count[aln.reference_name] += aln.get_tag('AS')
                except ValueError:
                    mir_count[aln.reference_name] += 1
        
        pd\
            .DataFrame()\
            .from_dict(mir_count, orient='index', columns = ['mir_count'])  \
            .reset_index() \
            .to_csv(output.TAB, index=False, header=False)


rule correction:
    input:
        IDX = WEIGHT_TABLE,
        BAM = MIR_FILTER_BAM_TEMPLATE,

    output:
        CORRECTED_BAM_TEMPLATE

    shell:
        'tgirt_correction.py correct '\
        '-i {input.BAM}  '\
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
        'tgirt_correction.py train -i {input} -x {output} --iter {params.ITER} '

