## SWOT
Reconstructing single-cell spatial maps from spatial transcriptomics data with SWOT


## Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)



## Overview
This project is a python implementation of SWOT, which is introducted in the paper "Reconstructing single-cell spatial maps from spatial transcriptomics data with SWOT"

Although cell-type deconvolution methods revealed spatial patterns, they are predominantly limited to estimating cell-type compositions without considering cell-to-spot mapping for reconstructing single-cell spatial maps.
SWOT is a spatially weighted optimal transport method that integrates single-cell RNA sequencing data with spatial transcriptomics data for cell-type deconvolution and further single-cell spatial maps reconstruction. 
It contains two principal components: an optimal transport module for computing transport plan and a cell mapping module for estimating cell-type compositions, cell numbers and cell coordinates per spot. 
It applies a spatially weighted strategy, which incorporating gene expression and spatial location, to an optimal transport framework to learn a mapping from cells to spots, thereby achieving cell-type deconvolution and further single-cell spatial maps reconstruction.



## Installation
We suggest using a separate conda environment for installing SWOT. SWOT can be run in Windows Powershell or Linux Bash shell.

1. Create conda environment and install `SWOT` package.

```bash
conda create -y -n SWOT_env python=3.9

conda activate SWOT_env
```

2. Clone this repository and cd into it as below.

```bash
git clone https://github.com//GaoLabXDU/SWOT.git
```

3. Install requirements directly.

```bash
cd SWOT

pip install -r requirements.txt
```



## Usage

The SWOT algorithm used in spot-resolution ST data for cell-type deconvolution and further single-cell spatial maps reconstruction. SWOT inputs a gene expression matrix with cell type labels of scRNA-seq data and a gene expression matrix with spatial coordinates of ST data. 


### Run Example
An example for pancreatic ductal adenocarcinoma (PDAC) dataset can be found under the directory "Example/", and the example data are under the directory "Data/".

```bash

cd Example

unzip Data.zip

python PDAC.py
```
The results of SWOT are saved in "SWOT_files/".
* "CellMapping/": 
  * "Celltype_proportions.csv", represents the estimated cell-type proportions matrix with rows being spots and columns being cell types. 
  * "Cell_maps_xy.csv", represents the estimated spatial coordinates with rows being reconstructed cells and columns being meta information.
  * "Cell_maps_exp.csv", represents the gene expression profiles of single-cell spatial maps with rows being genes and columns being cells.

### Run your own data
When using your own data to run SWOT, you should provide as:
* The gene expression matrix of scRNA-seq and ST data, rows represent genes and columns represent cells or spots, and saved as .csv format.
* The cell type labels matrix of scRNA-seq data, rows represent cells and columns represent cell type information having 'celltype' for labels, and saved as .csv format.
* The spatial coordinates matrix of ST data, rows represent spots and columns represent coordinates information having 'X' and 'Y', and saved as .csv format. 

