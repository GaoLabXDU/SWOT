## SWOT
Reconstructing single-cell spatial maps from spatial transcriptomics data with SWOT


## Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Compared methods](#compared-methods)
- [Datasets availability](#datasets-availability)



## Overview
This project is a python implementation of SWOT, which is introducted in the paper "Reconstructing single-cell spatial maps from spatial transcriptomics data with SWOT"

Although cell-type deconvolution methods revealed spatial patterns, they are predominantly limited to estimating cell-type compositions without considering cell-to-spot mapping for reconstructing single-cell spatial maps.
SWOT is a spatially weighted optimal transport method that integrates single-cell RNA sequencing data with spatial transcriptomics data for cell-type deconvolution and further single-cell spatial maps reconstruction. 
It contains two principal components: an optimal transport module for computing transport plan and a cell mapping module for estimating cell-type compositions, cell numbers and cell coordinates per spot. 
It applies a spatially weighted strategy, which incorporating gene expression and spatial location, to an optimal transport framework to learn a mapping from cells to spots, thereby achieving cell-type deconvolution and further single-cell spatial maps reconstruction.



## Installation
We suggest using a separate conda environment for installing SWOT. SWOT can be run in Windows Powershell or Linux Bash shell.

1. Create conda environment and install `SWOT` package

```bash
conda create -y -n SWOT_env python=3.9

conda activate SWOT_env
```

2. Clone this repository and cd into it as below.

```bash
git clone https://github.com//GaoLabXDU/SWOT.git
```

3. Install requirements directly

```bash
cd SWOT

pip install -r requirements.txt
```



## Usage

The SWOT algorithm used in spot-resolution ST data for cell-type deconvolution and further single-cell spatial maps reconstruction. SWOT inputs a gene expression matrix with cell type labels of scRNA-seq data and a gene expression matrix with spatial coordinates of ST data. 


### Run the demo
An example can be found under the directory "test/", and the example data are under the directory "Data/".
The example is a simuluated dataset, the scRNA-seq data consisted of 523 cells expressing across 10,000 genes and the ST data has 85 spots in 10,000 genes.

```bash
unzip Data.zip

cd test

python demo.py
```
The results of SWOT are saved in "SWOT_files/".
* "OptimalTransport/": 
  * "D_cell_spot.csv", "D_cell.csv", "D_pos.csv", "D_spot.csv" represent, the gene expression distance matrix between cells and spots, the gene expression distance matrix among cells in scRNA-seq data, the spatial coordinates distance matrix among spots in ST data, and the gene expression distance matrix among spots in ST data, respectively.
  * "spa_weight.csv", represents the spatially weighted distance matrix among spots.
  * "TransportPlan.csv", represents the transport plan with rows being cells ang columns being spots.
* "CellMapping/": 
  * "Celltype_proportions.csv", represents the estimated cell-type proportions matrix with rows being spots and columns being cell types. 
  * "Cell_maps_xy.csv", represents the estimated spatial coordinates with rows being reconstructed cells and columns being meta information.
  * "Cell_maps_exp.csv", represents the gene expression profiles of single-cell spatial maps with rows being genes and columns being cells.

### Run your own data
When using your own data for deconvolution, you should provide as:
* The gene expression matrix of scRNA-seq and ST data, rows represent genes and columns represent cells, and saved as .csv format.
* The cell type labels matrix of scRNA-seq data, rows represent cells and columns represent cell type information having 'celltype' for labels, and saved as .csv format.
* The spatial coordinates matrix of ST data, rows represent spots and columns represent coordinates information having 'X' and 'Y', and saved as .csv format. 



## Compared methods
* SPOTlight (v1.6.7, https://github.com/MarcElosua/SPOTlight) is executed in an R environment (v4.3.0), we set “mean_AUC=0.5” and selected the top 3,000 highly variable genes. 
* RCTD (v2.2.1, https://github.com/dmcable/spacexr) is executed in an R environment (v4.3.0).
* CARD (v1.1, https://github.com/YingMa0107/CARD) is executed in an R environment (v4.3.0).  
* STRIDE (v0.0.2a, https://github.com/wanglabtongji/STRIDE) is run in a Python environment (v3.8.18).
* Stereoscope (v0.2.0, https://github.com/almaan/stereoscope) is conducted in a Python environment (v3.8.18), the parameters "sc epochs" and "st epochs" are set to 50,000. 
* Uniport (v1.2.2, https://github.com/caokai1073/uniPort) is first implemented in a Python environment (v3.9.18) and then executed in an R environment (v4.3.0) . We adjusted the maximum iteration number to 1,000 during training phase. 
* SONAR (https://github.com/lzygenomics/SONAR) is executed in both R (v4.3.0) and MATLAB (R2021b) environments.
* CellTrek (v1.1.0, v0.0.94, https://github.com/navinlabcode/CellTrek) is executed in an R environment (v4.3.0).
* CytoSPACE (https://cytospace.stanford.edu/) is executed in a web interface with default parameter settings. For Simulated_Cortex, Simulated_MB, MOB, and PDAC, we set the parameters spot_n and top_spot as 5, 10, 5, and 10, respectively.
* CytoCommunity (v1.1.0, https://github.com/huBioinfo/CytoCommunity) relied on Python (v3.10.6) and R (v4.3.0) environments. 
* STAGATE (v1.0.1, https://github.com/QIFEIDKN/STAGATE) is conducted within a Python environment (v3.7.16). 



## Datasets availability
* SeqFISH+ data of mouse somatosensory cortex comes from http://linnarssonlab.org/cortex. 
* Stereoseq data is available at http://116.6.21.110:8090/share/dd965cba-7c1f-40b2-a275-0150890e005f and the scRNA-seq data comes from http://mousebrain.org/adolescent/downloads.html. 
* Mouse olfactory bulb dataset is available at  https://www.spatialresearch.org/resources-published-datasets/doi-10-1126science-aaf2403/ and GSE121891 for ST and scRNA-seq data. 
* Mouse cerebellum dataset is publicly available at https://singlecell.broadinstitute.org/single_cell/study/SCP948/robust-decomposition-of-cell-type-mixtures-in-spatial-transcriptomics#study-download and https://singlecell.broadinstitute.org/single_cell/study/SCP948/robust-decomposition-of-cell-type-mixtures-in-spatial-transcriptomics for ST and scRNA-seq data. 
* Pancreatic ductal adenocarcinoma dataset is downloaded from GSM3036911 and GSE111672 for ST and scRNA-seq data.