## SWOT
SWOT: A spatially weighted optimal transport method for cell-type deconvolution in spatial transcriptomics data


## Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Compared methods](#compared-methods)
- [Datasets availability](#datasets-availability)



## Overview
This project is a python implementation of SWOT, which is introducted in the paper "SWOT: A spatially weighted optimal transport method for cell-type deconvolution in spatial transcriptomics data"

Cell-type deconvolution is a crucial step for the description of cell-type spatial patterns. SWOT is a spatially weighted optimal transport method that integrates single-cell RNA sequencing data with spatial transcriptomics data for cell-type deconvolution. It contains two principal components: an optimal transport module for computing transport plan and a cell-type mapping module for inferring cell-type composition. It applies a spatially weighted strategy, which incorporating gene expression and spatial location, to an optimal transport framework to learn a mapping from cells to spots, thereby estimating cell-type composition. Furthermore, SWOT conducts a cell mapping to infer cell numbers and cell coordinates within each spot.



## Installation
We suggest using a separate conda environment for installing SWOT. SWOT can be run in Windows Powershell or Linux Bash shell.

1. Create conda environment and install `SWOT` package

```bash
conda create -y -n SWOT_env python=3.9

conda activate SWOT_env
```

2. Clone this repository and cd into it as below.

```bash
git clone https://github.com/LanyingWang25/SWOT.git
```

3. Install requirements directly

```bash
cd SWOT

pip install -r requirements.txt
```



## Usage

The SWOT algorithm used in spot-resolution ST data for cell-type deconvolution. SWOT inputs a gene expression matrix with cell type labels of scRNA-seq data and a gene expression matrix with spatial coordinates of ST data. 


### Run the demo
An example can be found under the directory "test/", and the example data are under the directory "Data/".
The example is a simuluated dataset, the scRNA-seq data consisted of 523 cells expressing across 10,000 genes and the ST data has 85 spots in 10,000 genes.

```bash
cd test

python demo.py
```
Related results of SWOT are saved in "SWOT_files/".
* "Distance/" saves four distance matrices, "D_cell_spot.csv", "D_cell.csv", "D_pos.csv", "D_spot.csv" represent, the gene expression distance matrix between cells and spots, the gene expression distance matrix among cells in scRNA-seq data, the spatial coordinates distance matrix among spots in ST data, and the gene expression distance matrix among spots in ST data, respectively.
* "SpatialWeight/" saves "spa_cost.csv" and "spa_weight.csv", represent the spatially weighted distance matrix among spots and the the spatial weights matrix, respectively.
* "OptimalTransport/" saves "T_mapping.csv", represents the transport plan with rows being cells ang columns being spots.
* "CTmapping/" saves "CT_mapping.csv", represents the estimated cell-type proportions matrix with rows being spots and columns being cell types.
* "Cellmapping/" saves "Cell_mapping_xy_new.csv" and "Cell_mapping_exp.csv", represent the estimated cell coordinates informations within each spot, and the obtained cell expression profile with rows being genes and columns being cells, respectively.

### Run your own data
When using your own data for deconvolution, you should provide as:
* The gene expression matrix of scRNA-seq and ST data, rows represent genes and columns represent cells, and saved as .csv format.
* The cell type labels matrix of scRNA-seq data, rows represent cells and columns represent cell type informations having 'celltype' for labels, and saved as .csv format.
* The spatial coordinates matrix of ST data, rows represent spots and columns represent coordinates informations having 'X' and 'Y', and saved as .csv format. 



## Compared methods
* SPOTlight (v1.6.7, https://github.com/MarcElosua/SPOTlight) is executed in an R environment (v4.3.0), we set “mean_AUC=0.5” and selected the top 3,000 highly variable genes. 
* RCTD (v2.2.1, https://github.com/dmcable/spacexr) is executed in an R environment (v4.3.0).
* CARD (v1.1, https://github.com/YingMa0107/CARD) is executed in an R environment (v4.3.0).  
* STRIDE (v0.0.2a, https://github.com/wanglabtongji/STRIDE) is run in a Python environment (v3.8.18).
* Stereoscope (v0.2.0, https://github.com/almaan/stereoscope) is conducted in a Python environment (v3.8.18), the parameters "sc epochs" and "st epochs" are set to 50,000. 
* Uniport (v1.2.2, https://github.com/caokai1073/uniPort) is first implemented in a Python environment (v3.9.18) and then executed in an R environment (v4.3.0) . We adjusted the maximum iteration number to 1,000 during training phase. 
* SONAR (https://github.com/lzygenomics/SONAR) was executed in both R (v4.3.0) and MATLAB (R2021b) environments.
* CytoCommunity (v1.1.0, https://github.com/huBioinfo/CytoCommunity) relied on Python (v3.10.6) and R (v4.3.0) environments. 
* STAGATE (v1.0.1, https://github.com/QIFEIDKN/STAGATE) is conducted within a Python environment (v3.7.16). 



## Datasets availability
* SeqFISH+ data of mouse somatosensory cortex and mouse olfactory bulb came from http://linnarssonlab.org/cortex and https://github.com/CaiGroup/seqFISH-PLUS, respectively. 
* Stereoseq data is available at http://116.6.21.110:8090/share/dd965cba-7c1f-40b2-a275-0150890e005f and the scRNA-seq data came from http://mousebrain.org/adolescent/downloads.html. 
* Mouse olfactory bulb dataset is available at  https://www.spatialresearch.org/resources-published-datasets/doi-10-1126science-aaf2403/ and GSE121891 for ST and scRNA-seq data. 
* Mouse cerebellum dataset is publicly available at https://singlecell.broadinstitute.org/single_cell/study/SCP948/robust-decomposition-of-cell-type-mixtures-in-spatial-transcriptomics#study-download and https://singlecell.broadinstitute.org/single_cell/study/SCP948/robust-decomposition-of-cell-type-mixtures-in-spatial-transcriptomics for ST and scRNA-seq data. 
* Pancreatic ductal adenocarcinoma dataset were downloaded from GSM3036911 and GSE111672 for ST and scRNA-seq data.