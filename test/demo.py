
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd().replace('\\', '/'))))

import numpy as np
import pandas as pd
from SWOT import swot


if __name__ == '__main__':

    #### 1. reading dataset
    SC_exp = pd.read_csv('../Data/sc_exp.csv', sep=',', index_col=0)
    SC_meta = pd.read_csv( '../Data/sc_meta.csv', sep=',', index_col=0)
    ST_exp = pd.read_csv('../Data/st_exp.csv', sep=',', index_col=0)
    ST_xy = pd.read_csv('../Data/st_xy.csv', sep=',', index_col=0)

    result_path = '../test/'

    if 'SWOT_files' not in os.listdir(result_path):  # Save all results in 'SWOT_files' folder
        os.mkdir(os.path.join(result_path, 'SWOT_files'))
    swot_path = result_path + 'SWOT_files/'

    swotclass = swot.SWOTdecon(sc_exp=SC_exp, sc_meta=SC_meta,
                               st_exp=ST_exp, st_xy=ST_xy,
                               file_path=swot_path)

    #### 2. An optimal transport module for computing transport plan
    D_cell, D_spot, D_pos, D_cell_spot = swotclass.compute_costs(knn_scale_method='MinMaxScaler',
                                                                 save_dis=True,
                                                                 knn_metric_d12='correlation')

    _, Spa_cost = swotclass.compute_sw(d_spot=D_spot, d_pos=D_pos,
                                       cluster='celltype',cluster_method='Louvain',
                                       resolu_cluster=0.2, plot_umap=False, n_neighbors=20,
                                       verbose=False, ps_bandwidth=0.1, sp_bandwidth=0.1,
                                       save_sw=True)

    T_mapping = swotclass.compute_swusot(cost12=D_cell_spot, cost1=D_cell,
                                         spa_cost=Spa_cost, cost2=None,
                                         alpha=0.1, lamda=10.0, ent_reg=0.1,
                                         initdis_method='minus',
                                         save_swusot=True)

    #### 3. A cell-type mapping module for inferring cell-type composition
    CT_mapping = swotclass.compute_ctmapping(t_mapping=T_mapping,
                                             mapping_method='non_zero_sum',
                                             cluster = 'celltype',
                                             minto0=0.05, mincut=0.1,
                                             save_ctmapping=True)

    np.random.seed(1234)
    Cell_mapping = swotclass.compute_cellmapping(ct_mapping=CT_mapping, t_mapping=T_mapping,
                                                 number_method='auto_tech',
                                                 cells_eachspot=0, tech='10XVisium',
                                                 save_cellmapping=True)
