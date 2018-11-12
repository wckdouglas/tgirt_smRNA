#!/usr/bin/env Rscript

library(tidyverse)
library(DESeq2)

ercc_df <- read_tsv('../data/ercc.tsv') %>%
    select(-Len) %>%
    rename(refA1 = 'NTCF1') %>%
    rename(refA2 = 'NTCF2') %>%
    rename(refA3 = 'NTCF3') %>%
    select(grep('TPM',names(.),invert=T)) %>%
    mutate(Type = 'ERCC') %>%
    mutate(Name = ID) %>%
    tbl_df
gene_df <- read_tsv('../data/all.tsv') %>%
    select(grep('ID|Name|Type|NT[CT][F][0-9]+',names(.))) %>%
    tbl_df
abrf <- read_tsv('/stor/work/Lambowitz/Data/archived_work/2016/TGIRT_ERCC_project/result/countTables/countsData.tsv') %>%
    select(grep('ABRF|id', names(.))) %>%
    rename(id = 'ID') %>%
    tbl_df
ntt_df <-  rbind(gene_df, ercc_df) %>%
    inner_join(abrf)  %>%
    filter(!grepl('rRNA', Type))


count_mat <- ntt_df %>% select(-ID, -Name, -Type) %>% data.frame()
row.names(ntt_df$ID)
col_data <- data.frame(samplename = names(count_mat)) %>%
    mutate(treatment = samplename) %>%
    data.frame()
row.names(col_data) <- col_data$samplename

dds <- DESeqDataSetFromMatrix(countData = count_mat,
                        colData = col_data,
                        design = ~treatment)
dds <- estimateSizeFactors(dds) 
count_data <- counts(dds, normalized=T) %>%
    data.frame() %>%
    rownames_to_column('ID') %>%
    cbind(ntt_df %>% select(ID:Type)) %>%
    tbl_df %>%
    write_tsv('../data/normalized_counts.tsv')
