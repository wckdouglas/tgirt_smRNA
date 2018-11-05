

BASE_PATH=/scratch/02190/yaojun/Work/JA18029

for TREATMENT in MiR_NTM MiR_NTT MiR_R
do
    for SAMPLE_NUM in 1 2 3
    do
        echo rsync $BASE_PATH/${TREATMENT}${SAMPLE_NUM}/bowtie2.list \
            cdw2854@lambcomp01.ccbb.utexas.edu:/stor/work/Lambowitz/cdw2854/miRNA/new_NTT/${TREATMENT}${SAMPLE_NUM}.counts
    done
done
