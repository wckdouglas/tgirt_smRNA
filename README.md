# TGIRT smRNA #

This repository stores code for analyzing bias from [mirXplore miRNA reference data](https://www.miltenyibiotec.com/US-en/products/macsmolecular/reagents/nucleic-acid-research/microrna-research/mirxplore-universal-reference.html) sequenced by [TGIRT-seq](http://www.ingex.com/tgirt-enzyme/) using a set of new NTT primer.

We compared our data to small RNA data generated by [several different small RNA methods](https://www.nature.com/articles/nbt.4183).


## Bias modeling ##

We have seen non-random base distributions within the three nuclotides from each end (5' and 3' ends) from TGIRT [RNA-seq](http://wckdouglas.github.io/assets/article_images/tgirt-seq/seqEnds.png) and [ssDNA-seq (Fig. S5)](https://static-content.springer.com/esm/art%3A10.1038%2Fs41598-017-09064-w/MediaObjects/41598_2017_9064_MOESM1_ESM.pdf). We hypothesized these 6 positions contributes to the sampling bias in small RNA TGIRT-seq. In a TGIRT-seq dataset from miRXplore sample, we demonstrated that the variations from the nucleotides at these positions can explain the over- and under-represented miRNAs (https://github.com/wckdouglas/tgirt_smRNA/blob/master/scripts/RF_correction.ipynb).

Using a random forest regression model, we further modeled how these positions on a miRNA contributed to the sample bias.

<img src="/tex/68f7a7363e19746f92c6045e0ee22f43.svg?invert_in_darkmode&sanitize=true" align=middle width=327.93247575pt height=24.65753399999998pt/>

where <img src="/tex/61f26f6f091ade6e6189bd6ac6995773.svg?invert_in_darkmode&sanitize=true" align=middle width=21.208250249999992pt height=22.465723500000017pt/> indicates the difference between observed <img src="/tex/aa551a9dc8603eac977e0d199b1a1992.svg?invert_in_darkmode&sanitize=true" align=middle width=34.14207224999999pt height=22.831056599999986pt/>CPM and expected <img src="/tex/aa551a9dc8603eac977e0d199b1a1992.svg?invert_in_darkmode&sanitize=true" align=middle width=34.14207224999999pt height=22.831056599999986pt/>CPM for miRNA <img src="/tex/0e51a2dede42189d77627c4d742822c3.svg?invert_in_darkmode&sanitize=true" align=middle width=14.433101099999991pt height=14.15524440000002pt/>. <img src="/tex/190083ef7a1625fbc75f243cffb9c96d.svg?invert_in_darkmode&sanitize=true" align=middle width=9.81741584999999pt height=22.831056599999986pt/> indicates the random forest regression function, and <img src="/tex/41f3fb6539b5c51fd4237a8f78c697e1.svg?invert_in_darkmode&sanitize=true" align=middle width=29.61486989999999pt height=14.15524440000002pt/>  indicates the nucleotide of miRNA <img src="/tex/0e51a2dede42189d77627c4d742822c3.svg?invert_in_darkmode&sanitize=true" align=middle width=14.433101099999991pt height=14.15524440000002pt/> at position <img src="/tex/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode&sanitize=true" align=middle width=5.663225699999989pt height=21.68300969999999pt/>. Only the first 3 bases (<img src="/tex/0ac75c805f5e7bf3181cb114d8ac5ae4.svg?invert_in_darkmode&sanitize=true" align=middle width=35.80006649999999pt height=21.68300969999999pt/> to 3) and the last 3 bases (<img src="/tex/8479bc40956c8a127f0327f9ee934f16.svg?invert_in_darkmode&sanitize=true" align=middle width=48.5855007pt height=21.68300969999999pt/> to -1) of each miRNA were considered. Correction of miRNA abundances was done by subtracting <img src="/tex/61f26f6f091ade6e6189bd6ac6995773.svg?invert_in_darkmode&sanitize=true" align=middle width=21.208250249999992pt height=22.465723500000017pt/> from the experimental <img src="/tex/aa551a9dc8603eac977e0d199b1a1992.svg?invert_in_darkmode&sanitize=true" align=middle width=34.14207224999999pt height=22.831056599999986pt/>CPM for each miRNA. 