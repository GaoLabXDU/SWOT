
import pandas as pd
from .utils import dis_matrix, spa_weight, sw_usot, celltype_mapping

import warnings
warnings.filterwarnings("ignore")

class NonConvergenceError(Exception):
    pass


class SWOTdecon(object):
    """
        This is the SWOTdecon class of SWOT, it products an object for the spatially weighted optimal transport model
        for cell-type deconvolution in ST data. It contains two principal components: an optimal transport module for
        computing transport plan and a cell-type mapping module for inferring cell-type composition.

        SWOT inputs gene expression profile and cell type labels in scRNA-seq data, as well as gene expression profile
        and spatial coordinates in ST data. The core output of SWOT is a cell-type proportion matrix, and moreover,
        through cell mapping, it provides an estimation of cell numbers and cell coordinates within each spot.

        See more details in our paper.
    """

    def __init__(self, sc_exp, sc_meta, st_exp, st_xy, file_path):
        """
        :param sc_exp: pandas.DataFrame, expression profile of scRNA-seq data with rows being genes and columns being cells.
        :param sc_meta: pandas.DataFrame, cell type information of scRNA-seq data with rows being cells and columns having 'celltype' for labels.
        :param st_exp: pandas.DataFrame, expression profile of ST data with rows being genes and columns being spots.
        :param st_xy: pandas.DataFrame, spatial coordinates information of ST data with rows being spots and columns being 'X' and 'Y'.
        :param file_path: string, file path for saving SWOT results.
        """

        self.sc_exp = sc_exp
        self.sc_meta = sc_meta
        self.st_exp = st_exp
        self.st_xy = st_xy
        self.file_path = file_path

        if not isinstance(sc_exp, pd.DataFrame):
            self.sc_exp = pd.DataFrame(sc_exp)
        if not isinstance(sc_meta, pd.DataFrame):
            self.sc_meta = pd.DataFrame(sc_meta)
        if not isinstance(st_exp, pd.DataFrame):
            self.st_exp = pd.DataFrame(st_exp)
        if not isinstance(st_xy, pd.DataFrame):
            self.st_xy = pd.DataFrame(st_xy)

    def compute_costs(self,
                      knn_neighbors_d1=5,
                      knn_neighbors_d2s=5,
                      knn_neighbors_d2=5,
                      knn_scale=True,
                      knn_scale_method='MinMaxScaler',
                      knn_metric_d12='correlation',
                      save_dis=False):
        """
        Calculate four costs/distances matrices.
        :param knn_neighbors_d1: number of neighbors to use for k-neighbors queries of gene expression in scRNA-seq data.
        :param knn_neighbors_d2s: number of neighbors to use for k-neighbors queries of gene expression in ST data.
        :param knn_neighbors_d2: number of neighbors to use for k-neighbors queries of spatial position in ST data.
        :param knn_scale: whether the cost matrices by KNN need to scaling?
        :param knn_scale_method: scaling method, the string name can be: 'Max', 'MinMaxScaler', 'L2_Normalization'.
        :param knn_metric_d12: metric to be computed in scipy for D in KNN method.
        :param save_dis: whether the computed distance matrices need to save in file_path?
        :returns:
            four distances matrices：
            # Dse: distance metric of gene expression among cells in scRNA-seq data.
            # Dte: distance metric of gene expression among spots in ST data.
            # Dtc: distance metric of coordinates among spots in ST data.
            # D:   distance metric of gene expression between cells and spots in scRNA-seq and ST data.
        """

        print('Computing distance matrices ......')
        d_cell, d_spot, d_pos = dis_matrix.compute_cost_knn(sc_exp=self.sc_exp,
                                                            st_exp=self.st_exp,
                                                            st_xy=self.st_xy,
                                                            file_path=self.file_path,
                                                            save=save_dis,
                                                            scaling=knn_scale,
                                                            scale_method=knn_scale_method,
                                                            n_neighbors_cell=knn_neighbors_d1,
                                                            n_neighbors_spot=knn_neighbors_d2s,
                                                            n_neighbors_pos=knn_neighbors_d2)
        d_cellspot = dis_matrix.compute_cost_c12(sc_exp=self.sc_exp,
                                                 st_exp=self.st_exp,
                                                 file_path=self.file_path,
                                                 save=save_dis,
                                                 metric=knn_metric_d12,
                                                 scaling=knn_scale,
                                                 scale_method=knn_scale_method)

        return d_cell, d_spot, d_pos, d_cellspot


    def compute_sw(self,
                   d_pos,
                   d_spot,
                   cluster='celltype',
                   cluster_method='Leiden',
                   resolu_cluster=1.0,
                   n_neighbors=20,
                   verbose=False,
                   plot_umap=False,
                   ps_bandwidth=0.1,
                   sp_bandwidth=0.1,
                   save_sw=False):
        """
        Compute the spatial weights and spatially weighted distance among spots.
        The spatially weighted strategy incorporates gene expression, derived from pre-clustering of spots,
        with spatial location, derived from spatial neighborhood of coordinates
        :param d_pos: distance metric of coordinates among spots in ST data.
        :param d_spot: distance metric of gene expression among spots in ST data.
        :param cluster: the column name of cell type information in sc_meta data.
        :param cluster_method:  clustering method, the string name can be: 'Leiden' or 'Louvain'.
        :param resolu_cluster: controlling the coarseness of the clustering. Higher values lead to more clusters.
        :param n_neighbors: number of neighbors for Leiden/Louvain clustering.
        :param verbose: whether show the neighborhoods and cell type relationship between spots?
        :param plot_umap: whether to draw or save the clustered UMAP picture results?
        :param ps_bandwidth: bandwidth of spatial coordinates determining the maximum neighbors radius.
        :param sp_bandwidth: bandwidth of gene expression determining the maximum neighbors radius.
        :param save_sw: whether the spatial weight and spatially weighted distance results need to save in file_path?
        :returns:
            Spa_weight: spatial weights distance matrix among spots.
            Spa_cost: spatially weighted distance matrix among spots.
        """

        print('Computing spatial weight ......')

        st_exp_adata = spa_weight.create_anndata_st(st_exp=self.st_exp,
                                                    st_xy=self.st_xy)

        st_clustering = spa_weight.pre_clustering(st_exp_adata=st_exp_adata,
                                                  sc_meta=self.sc_meta,
                                                  file_path=self.file_path,
                                                  cluster=cluster,
                                                  cluster_method=cluster_method,
                                                  resolu_cluster=resolu_cluster,
                                                  n_neighbors=n_neighbors,
                                                  plot_umap=plot_umap,
                                                  save=False)

        Spa_weight, Spa_cost = spa_weight.spatial_weight_cost(d_pos=d_pos,
                                                              d_spot=d_spot,
                                                              st_clustering=st_clustering,
                                                              file_path=self.file_path,
                                                              save=save_sw,
                                                              verbose=verbose,
                                                              ps_bandwidth=ps_bandwidth,
                                                              sp_bandwidth=sp_bandwidth)

        return Spa_weight, Spa_cost


    def compute_swusot(self,
                       cost12,
                       cost1,
                       cost2,
                       spa_cost,
                       alpha=0.1,
                       lamda=10.0,
                       ent_reg=0.1,
                       initdis_method='minus',
                       numItermax=5,
                       save_swusot=False):
        """
        A spatially weighted unbalanced and structured optimal transport module for computing transport plan.
        :param cost12: cost matrix of gene expression between cells and spots, with rows being cells and columns being spots.
        :param cost1: cost matrix of cells in scRNA-seq data.
        :param cost2: cost matrix of coordinates in ST data.
        :param spa_cost: spatially weighted distance matrix among spots.
        :param alpha: weight for structure term.
        :param lamda: weight for KL divergence penalizing unbalanced transport.
        :param ent_reg: weight for entropy regularization term.
        :param initdis_method: initialization method, the string name can be: 'minus', 'minus_exp', 'uniform_distribution'.
        :param numItermax: max number of iterations.
        :param save_swusot: whether the transport plan result need to save in file_path?
        :return: a transport plan of cell-to-spot between cells in scRNA-seq data and spots in ST data.
        """

        print('Computing transport plan ......')

        T_mapping = sw_usot.transport_plan(cost12=cost12,
                                           spa_cost=spa_cost,
                                           cost1=cost1,
                                           alpha=alpha,
                                           lamda=lamda,
                                           ent_reg=ent_reg,
                                           numItermax=numItermax,
                                           initdis_method=initdis_method,
                                           file_path=self.file_path,
                                           save=save_swusot)
        cell_name = self.sc_exp.columns
        spot_name = self.st_exp.columns

        T_mapping = pd.DataFrame(T_mapping, index=cell_name, columns=spot_name)

        return T_mapping


    def compute_ctmapping(self,
                          t_mapping,
                          mapping_method='rawmean',
                          cluster = 'celltype',
                          minto0=0.05,
                          mincut=0.1,
                          save_ctmapping=True):
        """
        A cell-type mapping module for inferring cell-type composition of SWOT.
        :param t_mapping: a transport plan of cell-to-spot, with rows being cells and columns being spots.
        :param mapping_method: cell type mapping method.
        :param cluster: the column name of cell type information in sc_meta data.
        :param minto0: the threshold of setting 0.
        :param mincut: the minimum threshold.
        :param save_ctmapping: whether the cell-type mapping result need to save in file_path?
        :return: A cell-type proportion matrix.
        """
        print('Cell type mapping ......')
        ct_order = pd.Index(self.sc_meta.celltype.unique(), dtype=object)
        CT_mapping = celltype_mapping.ct_mapping(t_mapping,
                                                 mapping_method=mapping_method,
                                                 st_xy=self.st_xy,
                                                 sc_meta=self.sc_meta,
                                                 ct_order=ct_order,
                                                 file_path=self.file_path,
                                                 cluster=cluster,
                                                 save=save_ctmapping,
                                                 minto0=minto0,
                                                 mincut=mincut)

        return CT_mapping


    def compute_cellmapping(self,
                            ct_mapping,
                            t_mapping,
                            number_method,
                            cells_eachspot=0,
                            tech='10XVisium',
                            save_cellmapping=False):
        """
        Estimation of cell numbers and cell coordinates within each spot by cell mapping of SWOT.
        :param ct_mapping: a transport plan of cell-to-spot, with rows being cells and columns being spots.
        :param t_mapping: a cell-type proportion matrix, with rows being spots and columns being cell types.
        :param number_method: the method for compute or allocate the number of cells in each spot.
        :param cells_eachspot: the number of initialization.
        :param tech: the sequencing technology of ST data.
        :param save_cellmapping: whether the cell mapping result need to save in file_path?
        :return: A cell mapping results.
        """
        print('Cell mapping ......')
        cellnum_spot = celltype_mapping.compu_cells_eachspot(number_method=number_method,
                                                             st_exp=self.st_exp,
                                                             cells_eachspot=cells_eachspot,
                                                             tech=tech)

        Cell_mapping_xy, cs_ctnum = celltype_mapping.cell_mapping_xy(ct_mapping=ct_mapping,
                                                                     st_xy=self.st_xy,
                                                                     cellnum_spot=cellnum_spot)

        cell_mapping_exp, cell_mapping_xy_new = celltype_mapping.cell_mapping_expression(t_mapping=t_mapping,
                                                                                         cs_xy=Cell_mapping_xy,
                                                                                         cellnum_spot=cellnum_spot,
                                                                                         sc_exp=self.sc_exp,
                                                                                         sc_meta=self.sc_meta,
                                                                                         cs_ctnum=cs_ctnum,
                                                                                         file_path=self.file_path,
                                                                                         save=save_cellmapping)
        Cell_mapping = [Cell_mapping_xy, cell_mapping_exp, cell_mapping_xy_new]
        return Cell_mapping
