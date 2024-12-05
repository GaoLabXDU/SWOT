
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

import warnings
warnings.filterwarnings("ignore")
class NonConvergenceError(Exception):
    pass



def sum_to_1(ct_spot):
    """
    Summing the cell-type proportions in each spot equals 1.
    """
    OT_spotmap_sum = ct_spot.copy()
    for st in ct_spot.index:
        spotmap_old = OT_spotmap_sum.loc[st]
        st_sum = ct_spot.apply(lambda x: sum(x), axis=1)[st]
        spotmap_new = spotmap_old / st_sum
        OT_spotmap_sum.loc[st] = spotmap_new
    return OT_spotmap_sum

def min_cut(ct_spot, mincut):
    """
    Pruning the cell-type proportion matrix.
    """
    OT_spotmap_col = ct_spot.columns
    OT_spotmap_ind = ct_spot.index
    ct_spot = np.apply_along_axis(lambda x: np.where(x < np.max(x) * mincut, 0, x),
                                  axis=1, arr=ct_spot)
    OT_spotmap_mincut = pd.DataFrame(ct_spot, columns=OT_spotmap_col, index=OT_spotmap_ind)
    return OT_spotmap_mincut


def ct_mapping_non_zero_sum(T_mapping,
                            st_xy, sc_meta,
                            minto0, cluster='celltype', mincut=0.1):
    """
    Mapping transport plan into cell-type compositions.
    """
    celltypes = sc_meta[cluster].unique()
    ct_spot = pd.DataFrame(columns=celltypes, index=st_xy.index)

    for ct in celltypes:
        ct_cells = sc_meta.loc[sc_meta[cluster] == ct, 'cellname']
        ot_ctcells = T_mapping.loc[ct_cells,]
        ot_ctcells_new = ot_ctcells.copy()
        col_mean = ot_ctcells.mean(axis=0)

        ot_ctcells_new[ot_ctcells > col_mean] = ot_ctcells
        ot_ctcells_new[ot_ctcells <= col_mean] = 0
        ct_spot[ct] = ot_ctcells_new.apply(lambda x: x.sum())

    OT_spotmap_mincut = min_cut(ct_spot, mincut)
    OT_spotmap_sum1 = sum_to_1(OT_spotmap_mincut)

    OT_spotmap_cut = OT_spotmap_sum1.copy()
    OT_spotmap_cut[OT_spotmap_cut < minto0] = 0
    OT_spotmap_cut_sum1 = sum_to_1(OT_spotmap_cut)
    CT_mapping = OT_spotmap_cut_sum1
    return CT_mapping


def ct_mapping_rawmean(T_mapping,
                       st_xy, sc_meta,
                       minto0, cluster='celltype', mincut=0.1):
    """
    Mapping transport plan into cell-type compositions.
    """
    celltypes = sc_meta[cluster].unique()
    OT_spotmap = pd.DataFrame(columns=celltypes, index=st_xy.index)

    for ct in celltypes:
        cells = sc_meta.loc[sc_meta[cluster] == ct, 'cellname']
        OT_ct = T_mapping.loc[cells,]
        OT_spotmap[ct] = OT_ct.apply(lambda x: x.mean())

    OT_spotmap_mincut = min_cut(OT_spotmap, mincut)
    OT_spotmap_sum1 = sum_to_1(OT_spotmap_mincut)
    OT_spotmap_cut = OT_spotmap_sum1.copy()
    OT_spotmap_cut[OT_spotmap_cut < minto0] = 0
    OT_spotmap_cut_sum1 = sum_to_1(OT_spotmap_cut)
    CT_mapping = OT_spotmap_cut_sum1
    return CT_mapping



def ct_mapping(t_mapping,
               mapping_method,
               st_xy, sc_meta,
               ct_order, file_path,
               cluster='celltype', save=True,
               minto0=0.05, mincut=0.1):
    """
    Estimation of cell-type compositions for cell-type deconvolution.
    """
    if not isinstance(t_mapping, pd.DataFrame):
        print('Please enter T_mapping data of pandas DataFrame type represents a transport plan.')
        return None
    assert (mapping_method in ['non_zero_sum', 'rawmean']), \
        "mapping_method argument has to be either one of 'non_zero_sum', 'rawmean'."
    if not isinstance(st_xy, pd.DataFrame):
        print('Please enter st_xy data of pandas DataFrame type with rows '
              'being spots and columns being X and Y.')
        return None
    if not isinstance(sc_meta, pd.DataFrame):
        print("Please enter sc_meta data of pandas DataFrame type with rows "
              "being cells and columns having 'celltype'.")
        return None
    if not isinstance(ct_order, pd.Index):
        print("Please enter ct_order of Index object, e.g. Index(['a', 'b', 'c'], "
              "dtype='object', name='name1').")
        return None
    if save != True and save != False:
        print("Please select 'True' or 'False' for save.")
        return None
    if not isinstance(cluster, str):
        print('Please enter cluster of string type represents the column name of '
              'cell type information in sc_meta data.')
        return None
    if cluster not in sc_meta.columns:
        print('Please confirm whether cluster name is a column of sc_meta?')
        return None
    if not isinstance(minto0, float):
        print('Please enter minto0 of float type.')
        return None
    if not isinstance(mincut, float):
        print('Please enter mincut of float type.')
        return None

    if set(t_mapping.index) == set(sc_meta.index):
        t_mapping = t_mapping.apply(lambda x: pd.to_numeric(x), axis=0)
    else:
        print("Please make the cell names of T_mapping and sc_meta correspond.")
        return None
    if 'cellname' not in sc_meta.columns:
        sc_meta['cellname'] = sc_meta.index


    if mapping_method == 'non_zero_sum':
        ct_mapping = ct_mapping_non_zero_sum(T_mapping=t_mapping,
                                             st_xy=st_xy, sc_meta=sc_meta,
                                             cluster='celltype',
                                             minto0=minto0, mincut=mincut)
    if mapping_method == 'rawmean':
        ct_mapping = ct_mapping_rawmean(T_mapping=t_mapping,
                                        st_xy=st_xy, sc_meta=sc_meta,
                                        cluster='celltype',
                                        minto0=minto0, mincut=mincut)

    if not ct_mapping.columns.equals(ct_order):
        ct_mapping = ct_mapping[ct_order]

    file_name = 'Celltype_proportions.csv'
    if save:
        files = os.listdir(file_path)
        if 'CellMapping' not in files:
            os.mkdir(os.path.join(file_path, 'CellMapping'))

        ct_mapping.to_csv(file_path + 'CellMapping/' + file_name, sep=',', index=True, header=True)
        print('The cell-type deconvolution results are saved in ' + file_path + 'CellMapping/' + file_name + '.')

    return ct_mapping




def compu_cells_eachspot(number_method, st_exp,
                         cells_eachspot=0, tech=None):
    """
    Estimation of cell numbers per spot for reconstructing single-cell spatial maps.
    """

    assert (number_method in ['allocate_dire', 'allocate_tech', 'auto_tech']), \
        "number_method argument has to be either one of 'allocate_dire', 'allocate_tech', 'auto_tech'."

    cellnum_spot = pd.DataFrame(columns=(['cell_num']), index=st_exp.columns)

    if number_method == 'allocate_dire':
        # directly assigns a fixed number of cells to all spot
        print('Directly allocated by the user.')
        cellnum_spot['cell_num'] = cells_eachspot

    if number_method == 'allocate_tech':
        # obtains cell numbers randomly within the minimum and maximum resolution ranges
        # based on the sequencing technology used,
        # with support provided for 10x Genomics Visium, Spatial Transcriptomics and Slide-seq technologies.
        assert (tech in ['10XVisium', 'SpatialTranscriptomics', 'Slide-seq']), \
            "tech argument has to be either one of '10XVisium', 'SpatialTranscriptomics', 'Slide-seq'."
        cells_num_min = 0
        cells_num_max = 0

        if tech == '10XVisium':
            print('Based on 10X Visium technology, '
                  'randomly assigned in the largest and smallest range.')
            cells_num_min = 1
            cells_num_max = 10
        elif tech == 'SpatialTranscriptomics':
            print('Based on Spatial Transcriptomics technology, '
                  'randomly assigned in the largest and smallest range.')
            cells_num_min = 10
            cells_num_max = 40
        elif tech == 'Slide-seq':
            print('Based on Slide-seq technology, '
                  'randomly assigned in the largest and smallest range.')
            cells_num_min = 1
            cells_num_max = 3
        cells_num = np.random.randint(cells_num_min, cells_num_max, size=cellnum_spot.shape[0])
        cellnum_spot['cell_num'] = cells_num

    if number_method == 'auto_tech':
        # automatically assigns a varying number of cells to different spots based on
        # the proportion of zero values in gene expression profiles.
        assert (tech in ['10XVisium', 'SpatialTranscriptomics', 'Slide-seq']), \
            "tech argument has to be either one of '10XVisium', 'SpatialTranscriptomics', 'Slide-seq'."
        cells_num_min = 0
        cells_num_max = 0
        zero_ratio = st_exp.apply(lambda x: (x == 0).sum() / len(x))
        zero_ratio_mean = zero_ratio.mean()

        if tech == '10XVisium':
            print('Based on 10X Visium technology and zero proportions, '
                  'automatically randomly allocated within the range.')
            cells_num_min = 1
            cells_num_max = 10
        elif tech == 'SpatialTranscriptomics':
            print('Based on Spatial Transcriptomics technology and zero proportions, '
                  'automatically randomly allocated within the range.')
            cells_num_min = 10
            cells_num_max = 40
        elif tech == 'Slide-seq':
            print('Based on Slide-seq technology and zero proportions, '
                  'automatically randomly allocated within the range.')
            cells_num_min = 1
            cells_num_max = 3

        cells_num_avg = (cells_num_min + cells_num_max) / 2
        zero_ratio = pd.DataFrame(zero_ratio, columns=['zero_ratio'])
        cellnum_spot_zero = cellnum_spot.copy()
        cellnum_spot_zero['zero_ratio'] = zero_ratio

        for st in cellnum_spot_zero.index:
            if cellnum_spot_zero.at[st, 'zero_ratio'] > zero_ratio_mean:
                cellnum_spot_zero.at[st, 'cell_num'] = np.random.randint(1, cells_num_avg)
            else:
                cellnum_spot_zero.at[st, 'cell_num'] = np.random.randint(cells_num_avg, cells_num_max)
        cellnum_spot = cellnum_spot_zero.drop(['zero_ratio'], axis=1)

    cells_sum = sum(cellnum_spot.cell_num)
    print('The number of cells in all spots is: ', str(cells_sum))

    return cellnum_spot



def cell_mapping_xy(ct_mapping, st_xy, cellnum_spot):
    """
    Estimation of spatial coordinates for reconstructing single-cell spatial maps.
    """
    if not isinstance(ct_mapping, pd.DataFrame):
        print('Please enter CT_mapping data of pandas DataFrame type represents cell-type proportions.')
        return None

    cell_total = sum(cellnum_spot.cell_num)
    cs_columns = np.array(['cs_name', 'cs_type', 'cs_x', 'cs_y', 'spot_name', 'spot_x', 'spot_y'])
    cs_index = np.array(['cell_' + str(i) for i in np.arange(1, cell_total + 1, step=1)])
    cs_xy = pd.DataFrame(index=cs_index, columns=cs_columns)
    cs_xy['cs_name'] = cs_index

    spot_name_list = []
    spot_x_list = []
    spot_y_list = []
    for st in cellnum_spot.index:
        spot_name_list_tmp = np.repeat(st, cellnum_spot.at[st, 'cell_num'])
        spot_x_list_tmp = np.repeat(st_xy.at[st, 'X'], cellnum_spot.at[st, 'cell_num'])
        spot_y_list_tmp = np.repeat(st_xy.at[st, 'Y'], cellnum_spot.at[st, 'cell_num'])
        spot_name_list.append(spot_name_list_tmp)
        spot_x_list.append(spot_x_list_tmp)
        spot_y_list.append(spot_y_list_tmp)
    spot_name = [item for list in spot_name_list for item in list]
    spot_x = [item for list in spot_x_list for item in list]
    spot_y = [item for list in spot_y_list for item in list]
    cs_xy['spot_name'] = spot_name
    cs_xy['spot_x'] = spot_x
    cs_xy['spot_y'] = spot_y

    coord = st_xy[['X', 'Y']].to_numpy()
    nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(coord)
    distances, indices = nbrs.kneighbors(coord)
    for i in range(0, len(distances)):
        if distances[i, 1] == 0:
            dist_tmp = distances[i,]
            dist_non0 = next((num for num in dist_tmp if num != 0), None)
            distances[i, 1] = dist_non0

    radius = distances[:, 1] / 2
    radius = radius.tolist()
    all_coord = cs_xy[['spot_x', 'spot_y']].to_numpy()

    spot_len = cellnum_spot.values.tolist()
    all_radius = list()
    for i in range(len(spot_len)):
        all_radius_tmp = [radius[i]] * spot_len[i][0]
        all_radius.extend(all_radius_tmp)

    length = np.random.uniform(0, all_radius)
    angle = np.pi * np.random.uniform(0, 2, all_coord.shape[0])
    cs_xy['cs_x'] = all_coord[:, 0] + length * np.cos(angle)
    cs_xy['cs_y'] = all_coord[:, 1] + length * np.sin(angle)

    cs_ctnum = pd.DataFrame(index=ct_mapping.index, columns=ct_mapping.columns)  #
    for st in ct_mapping.index:
        st_cenum = cellnum_spot.at[st, 'cell_num']
        ct_prop = ct_mapping.loc[st]
        cs_ctnum.loc[st] = round(ct_prop * st_cenum)

        rowsum = cs_ctnum.loc[st].sum()
        if rowsum > st_cenum:
            duo = int(rowsum - st_cenum)
            ctprop_csctnum = pd.DataFrame(ct_prop)
            ctprop_csctnum['csctnum'] = cs_ctnum.loc[st]
            ctprop_csctnum_order = ctprop_csctnum.sort_values(by=st, axis=0, ascending=True)
            ct_non0 = []
            for i in ctprop_csctnum_order.index:
                if (ctprop_csctnum_order.at[i, st] != 0) and (ctprop_csctnum_order.at[i, 'csctnum'] != 0):
                    ct_non0.append(i)
            ct_non0_jian = ct_non0[0:duo]
            cs_ctnum.loc[st, ct_non0_jian] = cs_ctnum.loc[st, ct_non0_jian] - 1

        elif rowsum < st_cenum:
            ctp_min = min([i for i in ct_prop if i != 0])
            ctp_min_index = ct_prop.loc[ct_prop == ctp_min].index[0]
            cs_ctnum.loc[st, ctp_min_index] = cs_ctnum.loc[st, ctp_min_index] + (st_cenum - rowsum)

        cs_st_ctnum = cs_ctnum.loc[st]
        cs_st_name = cs_xy.loc[cs_xy['spot_name'] == st].index
        cs_st_name_ct = []
        for ct in cs_st_ctnum.index:
            cs_st_name_ct_tmp = [ct] * int(cs_st_ctnum[ct])
            cs_st_name_ct.append(cs_st_name_ct_tmp)
        cs_ct = [item for list in cs_st_name_ct for item in list]
        cs_xy.loc[cs_st_name, 'cs_type'] = cs_ct

    return cs_xy, cs_ctnum



def cell_mapping_expression(t_mapping, cs_xy, cellnum_spot, sc_exp,
                            sc_meta, cs_ctnum, file_path, save=False):
    """
    Obtain the single-cell spatial gene expression profiles for reconstructing single-cell spatial maps,
    with rows being genes and columns being cells.
    :return:
    cs_expression： gene expression profile of reconstructed cells.
    cs_xy_new: spatial coordinates of reconstructed single-cell spatial maps
               with rows being cells and columns being:
                'cs_name': new cell name;
                'cs_type': new cell type;
                'cs_x': new cell coordinate of X;
                'cs_y': new cell coordinate of Y;
                'spot_name': name of the spot to which the new cell belongs;
                'spot_x': X coordinate of the spot to which the new cell belongs;
                'spot_y': Y coordinate of the spot to which the new cell belongs;
                'cell_name_old': old cell name in scRNA-seq data.
    """

    if not isinstance(t_mapping, pd.DataFrame):
        print('Please enter T_mapping data of pandas DataFrame type represents a transport plan.')
        return None
    if not isinstance(cs_xy, pd.DataFrame):
        print('Please enter st_xy data of pandas DataFrame type with rows being spots and columns being X and Y.')
        return None
    if save != True and save != False:
        print("Please select 'True' or 'False' for save argument.")
        return None

    cs_ce_st = pd.concat([cs_xy['cs_name'], cs_xy['cs_type'], cs_xy['spot_name']], axis=1)
    cs_ce_st['cell_name'] = cs_xy['cs_name']

    for st in t_mapping.columns:
        st_cells_ct = pd.concat([t_mapping[st], sc_meta['celltype']], axis=1)
        st_cells_ct_order = st_cells_ct.sort_values(by=st, ascending=False)
        st_ct = cs_xy['cs_type'][cs_xy['spot_name'] == st]
        st_ct_num = cellnum_spot.loc[st]
        st_ct_list = []
        st_ct = pd.DataFrame(st_ct)
        for ct in st_ct['cs_type'].unique().tolist():
            if (sc_meta['celltype'] == ct).sum() < (st_ct['cs_type'] == ct).sum():
                #print('The number of cells in SC is less than estimated numbers for ', ct)
                shangeshu = -((st_ct['cs_type'] == ct).sum() - (sc_meta['celltype'] == ct).sum())
                st_ct_drop = st_ct.drop(st_ct[st_ct['cs_type'] == ct].index[shangeshu:])
                st_ct = st_ct_drop
                st_ct_list_tmp11 = st_cells_ct_order.index[st_cells_ct_order['celltype'] == ct]
                st_ct_list_tmp21 = st_ct_list_tmp11[:(st_ct['cs_type'] == ct).sum()].values.tolist()
                st_ct_list.append(st_ct_list_tmp21)
                continue

            st_ct_list_tmp1 = st_cells_ct_order.index[st_cells_ct_order['celltype'] == ct]
            st_ct_list_tmp2 = st_ct_list_tmp1[:(st_ct['cs_type'] == ct).sum()].values.tolist()
            st_ct_list.append(st_ct_list_tmp2)

        st_ct_list1 = [item for list in st_ct_list for item in list]
        if len(st_ct_list1) != len(cs_ce_st['cell_name'][cs_ce_st['spot_name'] == st]):
            cs_ce_st_tmp1 = cs_ce_st.loc[st_ct_drop.index, 'cell_name']
            cs_ce_st['cell_name'][cs_ce_st.loc[cs_ce_st_tmp1, 'cs_name']] = st_ct_list1
        else:
            cs_ce_st.loc[cs_ce_st['spot_name'] == st, 'cell_name'] = st_ct_list1

    cs_cells_fei = cs_ce_st['cell_name'][cs_ce_st['cell_name'] == cs_ce_st['cs_name']]
    cs_ce_st_fin = cs_ce_st.drop(cs_cells_fei)
    cs_cells = cs_ce_st_fin['cell_name'].tolist()

    cs_expression = sc_exp[cs_cells]
    cs_expression.columns = cs_ce_st_fin['cs_name']

    cs_xy_new = cs_xy.drop(cs_cells_fei)
    cs_ctnum_new = cs_ctnum.copy()
    for st in cs_ctnum.index:
        for ct in cs_ctnum.columns:
            ctnumt_SC = sc_meta['celltype'].value_counts()[ct]
            if cs_ctnum.at[st, ct] > ctnumt_SC:
                #print('For spot: ', st, 'the cell type: ', ct)
                cs_ctnum_new.at[st, ct] = ctnumt_SC
            else:
                cs_ctnum_new.at[st, ct] = cs_ctnum.at[st, ct]

    cs_xy_new['cell_name_old'] = cs_cells

    file_name = 'Cell_maps_exp.csv'
    #file_name_cs_cells = 'cs_cells.csv'
    file_name_cs_xy = 'Cell_maps_xy.csv'
    if save:
        files = os.listdir(file_path)
        if 'CellMapping' not in files:
            os.mkdir(os.path.join(file_path, 'CellMapping'))

        cs_expression.to_csv(file_path + 'CellMapping/' + file_name, sep=',', index=True, header=True)
        #cs_cells.to_csv(file_path + 'CellMapping/' + file_name_cs_cells, sep=',', index=True, header=True)
        cs_xy_new.to_csv(file_path + 'CellMapping/' + file_name_cs_xy, sep=',', index=True, header=True)
        #cs_ctnum_new.to_csv(file_path + 'Cellmapping/'  + file_name_cs_ctnum_new, sep=',', index=True, header=True)
        print('The gene expression and single cell information after cell mapping is saved in ' +
              file_path + 'CellMapping/Cell_maps_xy.csv, Cell_maps_exp.csv.')

    return cs_expression, cs_xy_new
