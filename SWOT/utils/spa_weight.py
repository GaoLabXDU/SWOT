
import os
import random
import anndata
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix

import warnings
warnings.filterwarnings("ignore")

def create_anndata_st(st_exp, st_xy):
    """
    Create AnnData object for ST data.
    """
    if 'X' not in st_xy.columns or 'Y' not in st_xy.columns:
        print("Please check the column names of st_xy data, using 'X' and 'Y' to represent the position of spots.")
        return None

    st_exp_counts = csr_matrix(st_exp.T)
    st_exp_adata = anndata.AnnData(st_exp_counts)
    st_exp_adata.obs_names = st_exp.columns
    st_exp_adata.var_names = st_exp.index
    st_exp_adata.obs['X'] = pd.Categorical(st_xy['X'])
    st_exp_adata.obs['Y'] = pd.Categorical(st_xy['Y'])

    return st_exp_adata




def pre_clustering(st_exp_adata,
                   sc_meta,
                   file_path, save=False,
                   cluster='celltype', cluster_method='Leiden',
                   resolu_cluster=1.5, n_neighbors=20,
                   plot_umap=False):
    """
    Pre_clustering for ST gene expression data.
    """
    if not isinstance(st_exp_adata, anndata._core.anndata.AnnData):
        print("Please enter st_exp_adata data of AnnData type or "
              "using 'create_anndata_st' function to create Anndata.")
        return None
    if not isinstance(cluster, str):
        print('Please enter cluster of string type, which represent the column name '
              'of cell type information in sc_meta data.')
        return None
    assert (cluster_method in ["Leiden", "Louvain"]), \
        "cluster_method argument has to be either one of 'Leiden' or 'Louvain'. "
    if not (isinstance(resolu_cluster, float) or isinstance(resolu_cluster, int)):
        print('Please enter resolu_cluster of float or int type.')
        return None
    if not isinstance(n_neighbors, int):
        print('Please enter n_neighbors of int type.')
        return None
    if save != True and save != False:
        print("Please select 'True' or 'False' for save argument.")
        return None
    if plot_umap != True and plot_umap != False:
        print("Please select 'True' or 'False' for plot_umap argument.")
        return None

    sc.pp.neighbors(st_exp_adata, n_neighbors=2, use_rep='X')
    ct_num = len(sc_meta[cluster].unique())

    # Leiden clustering in Scanpy package
    if cluster_method == 'Leiden':
        sc.tl.leiden(st_exp_adata, resolution=resolu_cluster)
        st_clustering = st_exp_adata.obs['leiden']
        cluster_num = len(st_clustering.unique())

        if cluster_num < ct_num:
            print('Leiden clusters obtained ' + str(cluster_num) +
                  ' clusters. The scRNA-seq data has ' + str(ct_num) + ' cell types, ' +
                  'and we suggest increasing the resolu_cluster so that the number of clusters '
                  'in ST data is the same as the number of cell types in scRNA-seq data!')
        elif cluster_num > ct_num:
            print('Leiden clusters obtained ' + str(cluster_num) +
                  ' clusters. The scRNA-seq data has ' + str(ct_num) + ' cell types, ' +
                  'and we suggest decreasing the resolu_cluster so that the number of clusters '
                  'in ST data is the same as the number of cell types in scRNA-seq data!')
        else:
            print('The number of Leiden clusters is: ' + str(cluster_num) +
                  ', it is equals to the number of cell types in scRNA-seq data!')

        if plot_umap:
            sc.tl.umap(st_exp_adata)
            sc.pl.umap(st_exp_adata, color=['leiden'], save='_st_Leiden.eps')

    # Louvain clustering in Scanpy package
    if cluster_method == 'Louvain':
        sc.tl.louvain(st_exp_adata, resolution=resolu_cluster)
        st_clustering = st_exp_adata.obs['louvain']
        cluster_num = len(st_clustering.unique())

        if cluster_num < ct_num:
            print('Louvain clusters obtained ' + str(cluster_num) +
                  ' clusters. The scRNA-seq data has ' + str(ct_num) + ' cell types, ' +
                  'and we suggest increasing the resolu_cluster so that the number of clusters '
                  'in ST data is the same as the number of cell types in scRNA-seq data!')
        elif cluster_num > ct_num:
            print('Louvain clusters obtained ' + str(cluster_num) +
                  ' clusters. The scRNA-seq data has ' + str(ct_num) + ' cell types, ' +
                  'and we suggest decreasing the resolu_cluster so that the number of clusters '
                  'in ST data is the same as the number of cell types in scRNA-seq data!')
        else:
            print('The number of Louvain clusters is: ' + str(cluster_num) +
                  ', it is equals to the number of cell types in scRNA-seq data!')

        if plot_umap:
            sc.tl.umap(st_exp_adata)
            sc.pl.umap(st_exp_adata, color=['louvain'], save='_st_Louvain.eps')

    st_clustering = pd.DataFrame(st_clustering)

    if save:
        files = os.listdir(file_path)
        if 'SpatialWeight' not in files:
            os.mkdir(os.path.join(file_path, 'SpatialWeight'))

        st_clustering.to_csv(file_path +'SpatialWeight/st_clustering.csv', sep=',', index=True, header=True)
        print('The pre-clustering result of ST data is saved in ' + file_path + 'SpatialWeight/st_clustering.csv.')

    return st_clustering




def spatial_weight_cost(d_pos, d_spot,
                        st_clustering,
                        file_path, save=False,
                        verbose=True,
                        ps_bandwidth=0.1, sp_bandwidth=0.1):
    """
    Compute spatial weight and spatial distance of each spot with respect to
    the other spots in neighbor with bandwidth.
    """
    if not isinstance(d_pos, pd.DataFrame):
        print('Please enter d_pos data of pandas DataFrame type represents the '
              'scaled distance of spatial position among spots in ST data.')
        return None
    if not isinstance(d_spot, pd.DataFrame):
        print('Please enter d_spot data of pandas DataFrame type represents the '
              'scaled distance of gene expression among spots in ST data.')
        return None
    if not isinstance(st_clustering, pd.DataFrame):
        print('Please enter st_clustering data of pandas DataFrame type represents clustering results of ST data.')
        return None
    if save != True and save != False:
        print("Please select 'True' or 'False' for save.")
        return None
    if verbose != True and verbose != False:
        print("Please select 'True' or 'False' for verbose.")
        return None
    if not isinstance(ps_bandwidth, float):
        print('Please enter ps_bandwidth of float type.')
        return None
    if not isinstance(sp_bandwidth, float):
        print('Please enter sp_bandwidth of float type.')
        return None

    spot_name = d_pos.index
    spot_num = d_pos.shape[0]
    spa_weight = np.diag(np.ones(spot_num))
    spa_cost = np.diag(np.zeros(spot_num))

    # four categories:
    # 1. Inside the neighborhood and of the same cluster.
    # 2. Inside the neighborhood but of a different cluster.
    # 3. Outside the neighborhood but of the same cluster.
    # 4. Outside the neighborhood and of a different cluster.

    count_inter_samtype = 0
    count_inter_diftype = 0
    count_intra_samtype = 0
    count_intra_diftype = 0

    for i in range(spot_num):
        for j in range(i + 1, spot_num):
            if verbose:
                print('Spatial distance between spot ' + str(i) + ' and ' + str(j)
                      + ' is: %.4f' % d_pos.iat[i, j])
                print('Expression distance between spot ' + str(i) + ' and ' + str(j)
                      + ' is: %.4f' % d_spot.iat[i, j])

            spot_name_i = d_pos.index[i]
            spot_name_j = d_pos.index[j]

            ps_dist = d_pos.iloc[i, j]
            sp_dist = d_spot.iloc[i, j]

            inter_neigh_weight = (1 - ((ps_dist / ps_bandwidth) ** 2)) ** 2
            intra_neigh_weight = (1 - ((sp_dist / (sp_dist + sp_bandwidth)) ** 2)) ** 2

            # Compute the spatial weight and spatial cost.
            if ps_dist <= ps_bandwidth and st_clustering.iloc[i, 0] == st_clustering.iloc[j, 0]:
                # 1. Inside the neighborhood and of the same cluster.
                spa_w_tmp = inter_neigh_weight
                spa_c_tmp = ps_dist * (1 - inter_neigh_weight)
                count_inter_samtype = count_inter_samtype + 1
                if verbose:
                    print(spot_name_i + ' and ' + spot_name_j +
                          'are inside neighborhood, and have same cluster. Spatial weight is: %.4f'
                          % spa_w_tmp + '. Spatial cost is: %.4f' % spa_c_tmp)

            elif ps_dist <= ps_bandwidth and st_clustering.iloc[i, 0] != st_clustering.iloc[j, 0]:
                # 2. Inside the neighborhood but of a different cluster.
                spa_w_tmp = inter_neigh_weight ** (10 * sp_dist)
                spa_c_tmp = 1 / (1 + sp_dist ** (-(inter_neigh_weight ** sp_dist)))
                count_inter_diftype = count_inter_diftype + 1
                if verbose:
                    print(spot_name_i + ' and ' + spot_name_j +
                          'are inside neighborhood, and have different cluster. Spatial weight is: %.4f'
                          % spa_w_tmp + '.  Spatial cost is: %.4f' % spa_c_tmp)

            elif ps_dist > ps_bandwidth and st_clustering.iloc[i, 0] == st_clustering.iloc[j, 0]:
                # 3. Outside the neighborhood but of the same cluster.
                spa_w_tmp = intra_neigh_weight
                spa_c_tmp = ps_dist ** (10 * intra_neigh_weight)
                count_intra_samtype = count_intra_samtype + 1
                if verbose:
                    print(spot_name_i + ' and ' + spot_name_j +
                          'are outside neighborhood, and have same cluster. Spatial weight is: %.4f'
                          % spa_w_tmp + '.   Spatial cost is: %.4f' % spa_c_tmp)

            elif ps_dist > ps_bandwidth and st_clustering.iloc[i, 0] != st_clustering.iloc[j, 0]:
                # 4. Outside the neighborhood and of a different cluster.
                spa_w_tmp = 0
                spa_c_tmp = random.uniform(0.5, 0.7)
                count_intra_diftype = count_intra_diftype + 1
                if verbose:
                    print(spot_name_i + ' and ' + spot_name_j +
                          'are outside neighborhood, and have different cluster. Spatial weight is: %.4f'
                          % spa_w_tmp + '. Spatial cost is: %.4f' % spa_c_tmp)

            spa_weight[i, j] = spa_w_tmp
            spa_weight[j, i] = spa_w_tmp
            spa_cost[i, j] = spa_c_tmp
            spa_cost[j, i] = spa_c_tmp

    if verbose:
        print('The number of inside the neighborhood and of the same cluster:' + str(count_inter_samtype))
        print('The number of inside the neighborhood but of a different cluster:' + str(count_inter_diftype))
        print('The number of outside the neighborhood but of the same cluster:' + str(count_intra_samtype))
        print('The number of outside the neighborhood and of a different cluster:' + str(count_intra_diftype))

    spa_weight = pd.DataFrame(spa_weight, index=spot_name, columns=spot_name)
    spa_cost = pd.DataFrame(spa_cost, index=spot_name, columns=spot_name)

    spa_weight = spa_weight.round(3)
    spa_cost = spa_cost.round(3)

    if save:
        files = os.listdir(file_path)
        if 'SpatialWeight' not in files:
            os.mkdir(os.path.join(file_path, 'SpatialWeight'))

        spa_weight.to_csv(file_path + 'SpatialWeight/spa_weight.csv', sep=',', index=True, header=True)
        spa_cost.to_csv(file_path + 'SpatialWeight/spa_cost.csv', sep=',', index=True, header=True)
        print('The spatial weight is saved in ' +  file_path + 'SpatialWeight/spa_weight.csv.')
        print('The spatially weighted distance is saved in ' + file_path + 'SpatialWeight/spa_cost.csv.')

    return spa_weight, spa_cost