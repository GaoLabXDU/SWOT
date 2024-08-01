
import os
import scipy
import scipy.stats
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn.neighbors import kneighbors_graph
from scipy.spatial.distance import jensenshannon
from sklearn.preprocessing import MinMaxScaler, normalize

import warnings
warnings.filterwarnings("ignore")

class NonConvergenceError(Exception):
    pass


def data_scaling(data, scale_method):
    if not isinstance(data, np.ndarray):
        print("Please enter data of numpy ndarray type with rows being genes and columns being cells.")
        return None
    assert (scale_method in ['Max', 'MinMaxScaler', 'L2_Normalization']), \
        "scale_method argument has to be either one of 'Max', 'MinMaxScaler', 'L2_Normalization'."

    if scale_method == 'Max':
        data_scaled = data / data.max()

    if scale_method == 'MinMaxScaler':
        minmaxScaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = minmaxScaler.fit_transform(data)

    if scale_method == 'L2_Normalization':
        data_scaled = normalize(data, norm="l2", axis=1)
        print('Distance matrices were Scaled by L2 Normalization.')

    return data_scaled



def construct_KNNgraph(data, n_neighbors, mode="connectivity"):
    if not isinstance(data, pd.DataFrame):
        print('Please enter data of pandas DataFrame type with rows being '
              'samples(cells or spots) and columns being features(genes or x/y).')
        return None
    if not isinstance(n_neighbors, int):
        print('Please enter n_neighbors of int type.')
        return None
    assert (mode in ["connectivity", "distance"]), \
        "mode argument has to be either one of 'connectivity', or 'distance'. "

    data = data.to_numpy()

    if mode == "connectivity":
        include_self = True
    else:
        include_self = False

    data_graph = kneighbors_graph(X=data, n_neighbors=n_neighbors, mode=mode,
                                  metric="minkowski", include_self=include_self)
    return data_graph


def compute_cost_knn(sc_exp, st_exp, st_xy, file_path,
                     scaling= True, scale_method='MinMaxScaler',
                     save=True,
                     n_neighbors_cell=5, n_neighbors_spot=5, n_neighbors_pos=5):
    """
    Compute the distance between samples in KNN graph using the shortest path distance.
    # For the gene expression among cells in scRNA-seq data,
          the gene expression among spots in ST data,
          the spatial coordinates among spots in ST data.
    """
    if scaling != True and scaling != False:
        print("Please select 'True' or 'False' for scaling.")
        return None
    if save != True and save != False:
        print("Please select 'True' or 'False' for save.")
        return None
    if not isinstance(n_neighbors_cell, int):
        print('Please enter n_neighbors_cell of int type.')
        return None
    if not isinstance(n_neighbors_spot, int):
        print('Please enter n_neighbors_spot of int type.')
        return None
    if not isinstance(n_neighbors_pos, int):
        print('Please enter n_neighbors_pos of int type.')
        return None

    sc_exp_graph = construct_KNNgraph(sc_exp.T, n_neighbors=n_neighbors_cell)
    st_exp_graph = construct_KNNgraph(st_exp.T, n_neighbors=n_neighbors_spot)
    st_xy_graph = construct_KNNgraph(st_xy, n_neighbors=n_neighbors_pos)

    scexp_shortestPath = dijkstra(csgraph=csr_matrix(sc_exp_graph), directed=False, return_predecessors=False)
    stexp_shortestPath = dijkstra(csgraph=csr_matrix(st_exp_graph), directed=False, return_predecessors=False)
    stxy_shortestPath = dijkstra(csgraph=csr_matrix(st_xy_graph), directed=False, return_predecessors=False)

    scexp_shortestPath_max = np.nanmax(scexp_shortestPath[scexp_shortestPath != np.inf])
    stexp_shortestPath_max = np.nanmax(stexp_shortestPath[stexp_shortestPath != np.inf])
    stxy_shortestPath_max = np.nanmax(stxy_shortestPath[stxy_shortestPath != np.inf])

    scexp_shortestPath[scexp_shortestPath > scexp_shortestPath_max] = scexp_shortestPath_max
    stexp_shortestPath[stexp_shortestPath > stexp_shortestPath_max] = stexp_shortestPath_max
    stxy_shortestPath[stxy_shortestPath > stxy_shortestPath_max] = stxy_shortestPath_max

    # Normalize the distance matrix based on shortest paths matrices
    if scaling:
        assert (scale_method in ['Max', 'MinMaxScaler', 'L2_Normalization']), \
            "scale_method argument has to be either one of 'Max', 'MinMaxScaler', 'L2_Normalization'."
        d_cell = data_scaling(scexp_shortestPath, scale_method)
        d_spot = data_scaling(stexp_shortestPath, scale_method)
        d_pos = data_scaling(stxy_shortestPath, scale_method)

    cell_name = sc_exp.columns
    spot_name = st_xy.index

    d_cell = pd.DataFrame(d_cell, index=cell_name, columns=cell_name)
    d_spot = pd.DataFrame(d_spot, index=spot_name, columns=spot_name)
    d_pos = pd.DataFrame(d_pos, index=spot_name, columns=spot_name)
    d_cell = d_cell.round(2)
    d_spot = d_spot.round(3)
    d_pos = d_pos.round(3)

    if save:
        files = os.listdir(file_path)
        if 'Distance' not in files:
            os.mkdir(os.path.join(file_path, 'Distance'))
        d_cell.to_csv(file_path + 'Distance/D_cell.csv', sep=',', index=True, header=True)
        d_spot.to_csv(file_path + 'Distance/D_spot.csv', sep=',', index=True, header=True)
        d_pos.to_csv(file_path +  'Distance/D_pos.csv', sep=',', index=True, header=True)
        print('Three distance matrices (Dse, Dte, Dtc) are saved in ' + file_path + 'Distance/.')
    return d_cell, d_spot, d_pos


def compute_cost_c12(sc_exp, st_exp,
                     file_path, save=True,
                     metric='correlation',
                     scaling=True, scale_method='MinMaxScaler'):

    if not isinstance(metric, str):
        print('Please enter metric of string type.')
        return None
    if save != True and save != False:
        print("Please select 'True' or 'False' for save.")
        return None
    if scaling != True and scaling != False:
        print("Please select 'True' or 'False' for scaling.")
        return None

    dist_12_tmp = scipy.spatial.distance.cdist(sc_exp.T, st_exp.T, metric=metric)

    if scaling:
        assert (scale_method in ['Max', 'MinMaxScaler', 'L2_Normalization']), \
            "scale_method argument has to be either one of 'Max', 'MinMaxScaler', 'L2_Normalization'."
        dist_12_scaled = data_scaling(dist_12_tmp, scale_method=scale_method)
        dist_12 = dist_12_scaled
    else:
        dist_12 = dist_12_tmp

    d_cell_spot = pd.DataFrame(dist_12, index=sc_exp.columns, columns=st_exp.columns)
    d_cell_spot = d_cell_spot.round(3)

    if save:
        files = os.listdir(file_path)
        if 'Distance' not in files:
            os.mkdir(os.path.join(file_path, 'Distance'))

        d_cell_spot.to_csv(file_path + 'Distance/D_cell_spot.csv', sep=',', index=True, header=True)
        print('The distance matrices (D) is saved in ' + file_path + 'Distance/D.csv.')
    return d_cell_spot