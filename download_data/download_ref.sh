
REF_LINK=http://homes.gersteinlab.org/people/rrk24/exceRpt/exceRptDB_v4_hg38_lowmem.tgz
REF_PATH=${REF}/GRCh38/exceRpt_reference

FILENAME=$(basename $REF_LINK)
curl -o $REF_PATH/$FILENAME $REF_LINK

MIRNA_REF=http://biorxiv.org/content/biorxiv/suppl/2017/05/17/113050.DC1/113050-8.txt
curl -o $REF_PATH/miRNA.fa $MIRNA_REF
