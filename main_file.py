import csv
import glob
import itertools
import os
import pickle
import pprint
import random
import sys
import time
import zipfile

import joblib
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from scipy.linalg import kron
from sklearn.cluster import KMeans
from sklearn.svm import SVC

import metric
from kfold_dataset_generator import generate_ds
from metric import *

random.seed(1234567890)
np.random.seed(1234567890)

dat_ix = 2
FOLD = 5
test_index = 1
copy_no = 0

N_min = 5
N_max = 5
K_FINAL = 4

generate_ds(dat_ix, FOLD, test_index, copy_no)

dat_suffix = ['Osak', 'Glas', 'Edin', 'Toro', 'Melb', 'caliAdv', 'disHolly', 'disland', 'epcot', 'MagicK']
embedding_name = dat_suffix[dat_ix]
poi_name = "poi-" + dat_suffix[dat_ix] + ".csv"  # Edin
tra_name = "traj-" + dat_suffix[dat_ix] + ".csv"

if not os.path.exists('model_files'):
    os.mkdir('model_files')
else:
    files = glob.glob('model_files/*')
    for f in files:
        os.remove(f)

if not os.path.exists('recset_markov'):
    os.mkdir('recset_markov')
else:
    files = glob.glob('recset_markov/*')
    for f in files:
        os.remove(f)

# if not os.path.exists(os.path.join('model_files_ds_' + str(dat_ix) + '_index_' + str(test_index))):
#     os.mkdir(os.path.join('model_files_ds_' + str(dat_ix) + '_index_' + str(test_index)))

model_zip_name = 'model_files_ds_' + str(dat_ix) + '_index_' \
                 + str(test_index) + '.zip'

with zipfile.ZipFile(os.path.join('model_repository', model_zip_name), 'r') as zip_ref:
    zip_ref.extractall('./model_files')

model_directory = os.path.join('model_files')

LMDA = 0.5

pp = pprint.PrettyPrinter(indent=4)

LOG_SMALL = -10
LOG_ZERO = -1000

data_dir = 'origin_data'

ALPHA_SET = [0.1, 0.3, 0.5, 0.7, 0.9]  # trade-off parameters

BIN_CLUSTER = 5  # discritization parameter

RANKSVM_COST = 10  # RankSVM regularisation constant
N_JOBS = 4  # number of parallel jobs
USE_GUROBI = False  # whether to use GUROBI as ILP solver

run_rank = True
run_tran = True
run_comb = True
run_rand = True

##################################################################################################

fpoi = os.path.join(data_dir, 'poi-' + dat_suffix[dat_ix] + '.csv')

poi_all = pd.read_csv(fpoi)
poi_all_np = np.array(poi_all)
poi_all.set_index('poiID', inplace=True)

ftraj = os.path.join(data_dir, 'traj-' + dat_suffix[dat_ix] + '.csv')

traj_all = pd.read_csv(ftraj)


def extract_traj(tid, traj_all):
    traj = traj_all[traj_all['trajID'] == tid].copy()
    traj.sort_values(by=['startTime'], ascending=True, inplace=True)
    return traj['poiID'].tolist()


num_user = traj_all['userID'].unique().shape[0]
num_poi = traj_all['poiID'].unique().shape[0]
num_traj = traj_all['trajID'].unique().shape[0]


def calc_dist_vec(longitudes1, latitudes1, longitudes2, latitudes2):
    """Calculate the distance (unit: km) between two places on earth, vectorised"""
    # convert degrees to radians
    lng1 = np.radians(longitudes1)
    lat1 = np.radians(latitudes1)
    lng2 = np.radians(longitudes2)
    lat2 = np.radians(latitudes2)
    radius = 6371.0088  # mean earth radius, en.wikipedia.org/wiki/Earth_radius#Mean_radius

    # The haversine formula, en.wikipedia.org/wiki/Great-circle_distance
    dlng = np.fabs(lng1 - lng2)
    dlat = np.fabs(lat1 - lat2)
    dist = 2 * radius * np.arcsin(np.sqrt(
        (np.sin(0.5 * dlat)) ** 2 + np.cos(lat1) * np.cos(lat2) * (np.sin(0.5 * dlng)) ** 2))
    return dist


##################################################################################################


query_dict_trajectory_train = dict()
query_dict_users_train = dict()
query_dict_traj_ids_train = dict()
query_dict_traj_time_train = dict()
query_dict_freq_train = dict()


def isSame(traj_a, traj_b):
    if (len(traj_a) != len(traj_b)):
        return False
    for i in range(len(traj_a)):
        if (traj_a[i] != traj_b[i]):
            return False

    return True


# KFOLD = 5
# kfold_dataset_generator.generate_ds_kfold_parts(KFOLD=KFOLD)
# kfold_dataset_generator.generate_train_test_data(3, KFOLD)

with open("processed_data/" + embedding_name + '_train_set.csv', mode='r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    line_no = 0

    row0 = []
    row1 = []
    row2 = []

    for row in csv_reader:

        if (line_no == 0):
            row0 = row.copy()
        elif (line_no == 1):
            row1 = row.copy()
        elif (line_no == 2):
            row2 = row.copy()

            curr_traj = [int(poi) for poi in row0]
            st_poi = curr_traj[0]
            ed_poi = curr_traj[-1]
            qu = str(st_poi) + "-" + str(ed_poi)

            curr_traj_time = [int(poi_time) for poi_time in row1]
            curr_user = row2[0]
            curr_traj_id = int(row2[1])

            gotBefore = False
            all_traj_pos = -1

            if qu in query_dict_trajectory_train.keys():
                all_traj = query_dict_trajectory_train[qu]
                for prev_traj_itr in range(len(all_traj)):
                    if isSame(all_traj[prev_traj_itr], curr_traj):
                        gotBefore = True
                        all_traj_pos = prev_traj_itr
                        break

                if not gotBefore:

                    all_traj.append(curr_traj)
                    query_dict_trajectory_train[qu] = all_traj

                    all_u = query_dict_users_train[qu]
                    all_u.append([curr_user])
                    query_dict_users_train[qu] = all_u

                    all_traj_id = query_dict_traj_ids_train[qu]
                    all_traj_id.append([curr_traj_id])
                    query_dict_traj_ids_train[qu] = all_traj_id

                    all_traj_time = query_dict_traj_time_train[qu]
                    all_traj_time.append([curr_traj_time])
                    query_dict_traj_time_train[qu] = all_traj_time

                    all_freq = query_dict_freq_train[qu]
                    all_freq.append(1)
                    query_dict_freq_train[qu] = all_freq

                else:

                    all_u = query_dict_users_train[qu]
                    all_u[all_traj_pos].append(curr_user)
                    query_dict_users_train[qu] = all_u

                    all_traj_id = query_dict_traj_ids_train[qu]
                    all_traj_id[all_traj_pos].append(curr_traj_id)
                    query_dict_traj_ids_train[qu] = all_traj_id

                    all_traj_time = query_dict_traj_time_train[qu]
                    all_traj_time[all_traj_pos].append(curr_traj_time)
                    query_dict_traj_time_train[qu] = all_traj_time

                    all_freq = query_dict_freq_train[qu]
                    all_freq[all_traj_pos] += 1
                    query_dict_freq_train[qu] = all_freq

            else:

                query_dict_trajectory_train.setdefault(qu, []).append(curr_traj)
                query_dict_users_train.setdefault(qu, []).append([curr_user])
                query_dict_traj_ids_train.setdefault(qu, []).append([curr_traj_id])
                query_dict_traj_time_train.setdefault(qu, []).append([curr_traj_time])
                query_dict_freq_train.setdefault(qu, []).append(1)

        line_no = (line_no + 1) % 3

query_dict_trajectory_test = dict()
query_dict_users_test = dict()
query_dict_traj_ids_test = dict()
query_dict_traj_time_test = dict()
query_dict_freq_test = dict()

with open("processed_data/" + embedding_name + '_test_set.csv', mode='r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    line_no = 0

    row0 = []
    row1 = []
    row2 = []

    for row in csv_reader:

        if (line_no == 0):
            row0 = row.copy()
        elif (line_no == 1):
            row1 = row.copy()
        elif (line_no == 2):
            row2 = row.copy()

            curr_traj = [int(poi) for poi in row0]
            st_poi = curr_traj[0]
            ed_poi = curr_traj[-1]
            qu = str(st_poi) + "-" + str(ed_poi)

            curr_traj_time = [int(poi_time) for poi_time in row1]
            curr_user = row2[0]
            curr_traj_id = int(row2[1])

            gotBefore = False
            all_traj_pos = -1

            if (qu in query_dict_trajectory_test.keys()):
                all_traj = query_dict_trajectory_test[qu]
                for prev_traj_itr in range(len(all_traj)):
                    if (isSame(all_traj[prev_traj_itr], curr_traj)):
                        gotBefore = True
                        all_traj_pos = prev_traj_itr
                        break

                if not gotBefore:

                    all_traj.append(curr_traj)
                    query_dict_trajectory_test[qu] = all_traj

                    all_u = query_dict_users_test[qu]
                    all_u.append([curr_user])
                    query_dict_users_test[qu] = all_u

                    all_traj_id = query_dict_traj_ids_test[qu]
                    all_traj_id.append([curr_traj_id])
                    query_dict_traj_ids_test[qu] = all_traj_id

                    all_traj_time = query_dict_traj_time_test[qu]
                    all_traj_time.append([curr_traj_time])
                    query_dict_traj_time_test[qu] = all_traj_time

                    all_freq = query_dict_freq_test[qu]
                    all_freq.append(1)
                    query_dict_freq_test[qu] = all_freq

                else:

                    all_u = query_dict_users_test[qu]
                    all_u[all_traj_pos].append(curr_user)
                    query_dict_users_test[qu] = all_u

                    all_traj_id = query_dict_traj_ids_test[qu]
                    all_traj_id[all_traj_pos].append(curr_traj_id)
                    query_dict_traj_ids_test[qu] = all_traj_id

                    all_traj_time = query_dict_traj_time_test[qu]
                    all_traj_time[all_traj_pos].append(curr_traj_time)
                    query_dict_traj_time_test[qu] = all_traj_time

                    all_freq = query_dict_freq_test[qu]
                    all_freq[all_traj_pos] += 1
                    query_dict_freq_test[qu] = all_freq

            else:

                query_dict_trajectory_test.setdefault(qu, []).append(curr_traj)
                query_dict_users_test.setdefault(qu, []).append([curr_user])
                query_dict_traj_ids_test.setdefault(qu, []).append([curr_traj_id])
                query_dict_traj_time_test.setdefault(qu, []).append([curr_traj_time])
                query_dict_freq_test.setdefault(qu, []).append(1)

        line_no = (line_no + 1) % 3


def get_test_raw_dict():
    q_u = dict()
    for k, v in query_dict_users_test.items():
        v_new = []
        for i in range(len(v)):
            v_new.append(v[i][0])
        q_u[k] = v_new

    q_tid = dict()
    for k, v in query_dict_traj_ids_test.items():
        v_new = []
        for i in range(len(v)):
            v_new.append(v[i][0])
        q_tid[k] = v_new

    q_tt = dict()
    for k, v in query_dict_traj_time_test.items():
        v_new = []
        for i in range(len(v)):
            v_new.append(v[i][0])
        q_tt[k] = v_new

    test_data_dicts = query_dict_trajectory_test, q_u, q_tt, q_tid

    return test_data_dicts


def get_test_trajid():
    test_trajid = []

    for k, v in query_dict_traj_ids_test.items():

        for i in range(len(v)):
            for j in range(len(v[i])):
                test_trajid.append(v[i][j])

    return test_trajid


test_data_dicts = get_test_raw_dict()
trajid_test = get_test_trajid()


##################################################################################################

def print_progress(cnt, total):
    """Display a progress bar"""
    assert (cnt > 0 and total > 0 and cnt <= total)
    length = 80
    ratio = cnt / total
    n = int(length * ratio)
    sys.stdout.write('\r[%-80s] %d%%' % ('-' * n, int(ratio * 100)))
    sys.stdout.flush()


def calc_poi_info(trajid_list, traj_all, poi_all):
    '''

    :param trajid_list: :param traj_all: :param poi_all: :return: poi-info dataframe , attributes : poiID(index), nVisit, poiCat, poiLon, poiLat, popularity[no of unique users]
    '''

    assert (len(trajid_list) > 0)
    poi_info = traj_all[traj_all['trajID'] == trajid_list[0]][['poiID']].copy()
    for i in range(1, len(trajid_list)):
        traj = traj_all[traj_all['trajID'] == trajid_list[i]][['poiID', 'poiDuration']]
        poi_info = poi_info.append(traj, ignore_index=True)

    poi_info = poi_info.groupby('poiID').agg([np.mean, np.size])
    poi_info.columns = poi_info.columns.droplevel()
    poi_info.reset_index(inplace=True)
    poi_info.rename(columns={'size': 'nVisit'}, inplace=True)
    poi_info.set_index('poiID', inplace=True)
    poi_info['poiCat'] = poi_all.loc[poi_info.index, 'poiCat']
    poi_info['poiLon'] = poi_all.loc[poi_info.index, 'poiLon']
    poi_info['poiLat'] = poi_all.loc[poi_info.index, 'poiLat']

    # POI popularity: the number of distinct users that visited the POI
    pop_df = traj_all[traj_all['trajID'].isin(trajid_list)][['poiID', 'userID']].copy()
    pop_df = pop_df.groupby('poiID').agg(pd.Series.nunique)
    pop_df.rename(columns={'userID': 'nunique'}, inplace=True)
    poi_info['popularity'] = pop_df.loc[poi_info.index, 'nunique']

    return poi_info.copy()


POI_DISTMAT = pd.DataFrame(data=np.zeros((poi_all.shape[0], poi_all.shape[0]), dtype=np.float),
                           index=poi_all.index, columns=poi_all.index)

ALL_POIs = list(poi_all.index)
poi_poi_distance_matrix = np.zeros((max(ALL_POIs) + 1, max(ALL_POIs) + 1))

for ix in poi_all.index:
    POI_DISTMAT.loc[ix] = calc_dist_vec(poi_all.loc[ix, 'poiLon'], poi_all.loc[ix, 'poiLat'], poi_all['poiLon'],
                                        poi_all['poiLat'])

for i in ALL_POIs:
    for j in ALL_POIs:
        poi_poi_distance_matrix[i][j] = np.round(POI_DISTMAT.loc[i, j], 3)

trajid_set_all = sorted(traj_all['trajID'].unique().tolist())
trajid_train = list(set(trajid_set_all) - set(trajid_test))

poi_info_all = calc_poi_info(trajid_train, traj_all, poi_all)

traj_dict = dict()

for trajid in trajid_set_all:
    traj = extract_traj(trajid, traj_all)
    assert (trajid not in traj_dict)
    traj_dict[trajid] = traj

QUERY_ID_DICT = dict()  # (start, end, length) --> qid

cnt = 0
for k, v in query_dict_trajectory_train.items():
    str_k = k.split("-")
    poi_start = int(str_k[0])
    poi_end = int(str_k[1])
    key = (poi_start, poi_end)
    QUERY_ID_DICT[key] = cnt
    cnt += 1

for k, v in query_dict_trajectory_test.items():
    str_k = k.split("-")
    poi_start = int(str_k[0])
    poi_end = int(str_k[1])
    key = (poi_start, poi_end)
    QUERY_ID_DICT[key] = cnt
    cnt += 1

DF_COLUMNS = ['poiID', 'label', 'queryID', 'category', 'neighbourhood', 'popularity', 'nVisit',
              'sameCatStart', 'sameCatEnd', 'distStart', 'distEnd', 'diffPopStart', 'diffPopEnd',
              'diffNVisitStart', 'diffNVisitEnd',
              'sameNeighbourhoodStart', 'sameNeighbourhoodEnd']


def gen_train_subdf(poi_id, query_id_set, poi_info, poi_clusters, cats, clusters, query_id_rdict):
    assert (isinstance(cats, list))
    assert (isinstance(clusters, list))

    columns = DF_COLUMNS
    poi_distmat = POI_DISTMAT
    df_ = pd.DataFrame(index=np.arange(len(query_id_set)), columns=columns)

    pop, nvisit = poi_info.loc[poi_id, 'popularity'], poi_info.loc[poi_id, 'nVisit']
    cat, cluster = poi_info.loc[poi_id, 'poiCat'], poi_clusters.loc[poi_id, 'clusterID']

    for j in range(len(query_id_set)):
        qid = query_id_set[j]
        assert (qid in query_id_rdict)  # qid --> (start, end, length)
        (p0, pN) = query_id_rdict[qid]
        idx = df_.index[j]
        df_.loc[idx, 'poiID'] = poi_id
        df_.loc[idx, 'queryID'] = qid
        df_.at[idx, 'category'] = tuple((cat == np.array(cats)).astype(np.int) * 2 - 1)
        df_.at[idx, 'neighbourhood'] = tuple((cluster == np.array(clusters)).astype(np.int) * 2 - 1)
        df_.loc[idx, 'popularity'] = LOG_SMALL if pop < 1 else np.log10(pop)
        df_.loc[idx, 'nVisit'] = LOG_SMALL if nvisit < 1 else np.log10(nvisit)
        df_.loc[idx, 'sameCatStart'] = 1 if cat == poi_info.loc[p0, 'poiCat'] else -1
        df_.loc[idx, 'sameCatEnd'] = 1 if cat == poi_info.loc[pN, 'poiCat'] else -1
        df_.loc[idx, 'distStart'] = poi_distmat.loc[poi_id, p0]
        df_.loc[idx, 'distEnd'] = poi_distmat.loc[poi_id, pN]
        df_.loc[idx, 'diffPopStart'] = pop - poi_info.loc[p0, 'popularity']
        df_.loc[idx, 'diffPopEnd'] = pop - poi_info.loc[pN, 'popularity']
        df_.loc[idx, 'diffNVisitStart'] = nvisit - poi_info.loc[p0, 'nVisit']
        df_.loc[idx, 'diffNVisitEnd'] = nvisit - poi_info.loc[pN, 'nVisit']
        df_.loc[idx, 'sameNeighbourhoodStart'] = 1 if cluster == poi_clusters.loc[p0, 'clusterID'] else -1
        df_.loc[idx, 'sameNeighbourhoodEnd'] = 1 if cluster == poi_clusters.loc[pN, 'clusterID'] else -1

    return df_


def gen_train_df(trajid_list, traj_dict, poi_info, poi_clusters, cats, clusters, n_jobs=-1):
    columns = DF_COLUMNS
    poi_distmat = POI_DISTMAT
    query_id_dict = QUERY_ID_DICT
    train_trajs = [traj_dict[x] for x in trajid_list if len(traj_dict[x]) > 2]

    # pp.pprint(traj_dict)
    # for t in train_trajs:
    #     if t[0] == 14 and t[-1] == 25:
    #         print(t)

    qid_set = sorted(set([query_id_dict[(t[0], t[-1])] for t in train_trajs]))
    poi_set = set()
    for tr in train_trajs:
        poi_set = poi_set | set(tr)

    query_id_rdict = dict()
    for k, v in query_id_dict.items():
        query_id_rdict[v] = k  # qid --> (start, end)

    train_df_list = Parallel(n_jobs=n_jobs) \
        (delayed(gen_train_subdf)(poi, qid_set, poi_info, poi_clusters, cats, clusters, query_id_rdict)
         for poi in poi_set)

    assert (len(train_df_list) > 0)
    df_ = train_df_list[0]
    for j in range(1, len(train_df_list)):
        df_ = df_.append(train_df_list[j], ignore_index=True)

        # set label
    df_.set_index(['queryID', 'poiID'], inplace=True)
    df_['label'] = 0
    for t in train_trajs:
        qid = query_id_dict[(t[0], t[-1])]
        for poi in t[1:-1]:  # do NOT count if the POI is startPOI/endPOI
            df_.loc[(qid, poi), 'label'] += 1

    df_.reset_index(inplace=True)
    return df_


def gen_test_df(startPOI, endPOI, poi_info, poi_clusters, cats, clusters):
    assert (isinstance(cats, list))
    assert (isinstance(clusters, list))

    columns = DF_COLUMNS
    poi_distmat = POI_DISTMAT
    query_id_dict = QUERY_ID_DICT
    key = (p0, pN) = (startPOI, endPOI)
    assert (key in query_id_dict)
    assert (p0 in poi_info.index)
    assert (pN in poi_info.index)

    df_ = pd.DataFrame(index=np.arange(poi_info.shape[0]), columns=columns)
    poi_list = sorted(poi_info.index)

    qid = query_id_dict[key]
    df_['queryID'] = qid
    df_['label'] = np.random.rand(df_.shape[0])  # label for test origin_data is arbitrary according to libsvm FAQ

    for i in range(df_.index.shape[0]):
        poi = poi_list[i]
        lon, lat = poi_info.loc[poi, 'poiLon'], poi_info.loc[poi, 'poiLat']
        pop, nvisit = poi_info.loc[poi, 'popularity'], poi_info.loc[poi, 'nVisit']
        cat, cluster = poi_info.loc[poi, 'poiCat'], poi_clusters.loc[poi, 'clusterID']
        idx = df_.index[i]
        df_.loc[idx, 'poiID'] = poi
        df_.at[idx, 'category'] = tuple((cat == np.array(cats)).astype(np.int) * 2 - 1)
        df_.at[idx, 'neighbourhood'] = tuple((cluster == np.array(clusters)).astype(np.int) * 2 - 1)
        df_.loc[idx, 'popularity'] = LOG_SMALL if pop < 1 else np.log10(pop)
        df_.loc[idx, 'nVisit'] = LOG_SMALL if nvisit < 1 else np.log10(nvisit)
        df_.loc[idx, 'sameCatStart'] = 1 if cat == poi_all.loc[p0, 'poiCat'] else -1
        df_.loc[idx, 'sameCatEnd'] = 1 if cat == poi_all.loc[pN, 'poiCat'] else -1
        df_.loc[idx, 'distStart'] = poi_distmat.loc[poi, p0]
        df_.loc[idx, 'distEnd'] = poi_distmat.loc[poi, pN]
        df_.loc[idx, 'diffPopStart'] = pop - poi_info.loc[p0, 'popularity']
        df_.loc[idx, 'diffPopEnd'] = pop - poi_info.loc[pN, 'popularity']
        df_.loc[idx, 'diffNVisitStart'] = nvisit - poi_info.loc[p0, 'nVisit']
        df_.loc[idx, 'diffNVisitEnd'] = nvisit - poi_info.loc[pN, 'nVisit']
        df_.loc[idx, 'sameNeighbourhoodStart'] = 1 if cluster == poi_clusters.loc[p0, 'clusterID'] else -1
        df_.loc[idx, 'sameNeighbourhoodEnd'] = 1 if cluster == poi_clusters.loc[pN, 'clusterID'] else -1

    return df_


def gen_data_svm(df_, df_columns=DF_COLUMNS):
    for col in df_columns:
        assert (col in df_.columns or col == 'label')

    lines = []
    for idx in df_.index:
        line = []
        for j in range(2, len(df_columns)):
            values_ = df_.at[idx, df_columns[j]]
            values_ = values_ if isinstance(values_, tuple) else [values_]
            for v in values_:
                line.append(v)
        lines.append(line)
    return lines


def softmax(x):
    x1 = x.copy()
    x1 -= np.max(x1)  # numerically more stable, REF: http://cs231n.github.io/linear-classify/#softmax
    expx = np.exp(x1)
    return expx / np.sum(expx, axis=0)  # column-wise sum


def normalise_transmat(transmat_cnt):
    transmat = transmat_cnt.copy()
    assert (isinstance(transmat, pd.DataFrame))
    for row in range(transmat.index.shape[0]):
        rowsum = np.sum(transmat.iloc[row] + 1)
        assert (rowsum > 0)
        transmat.iloc[row] = (transmat.iloc[row] + 1) / rowsum
    return transmat


poi_train = sorted(poi_info_all.index)  # list of all POIs in the DATASET

poi_cats = poi_all.loc[poi_train, 'poiCat'].unique().tolist()
poi_cats.sort()
POI_CAT_LIST = poi_cats


def gen_transmat_cat(trajid_list, traj_dict, poi_info, poi_cats=POI_CAT_LIST):
    transmat_cat_cnt = pd.DataFrame(data=np.zeros((len(poi_cats), len(poi_cats)), dtype=np.float),
                                    columns=poi_cats, index=poi_cats)
    for tid in trajid_list:
        t = traj_dict[tid]
        if len(t) > 1:
            for pi in range(len(t) - 1):
                p1 = t[pi]
                p2 = t[pi + 1]
                assert (p1 in poi_info.index and p2 in poi_info.index)
                cat1 = poi_info.loc[p1, 'poiCat']
                cat2 = poi_info.loc[p2, 'poiCat']
                transmat_cat_cnt.loc[cat1, cat2] += 1
    return normalise_transmat(transmat_cat_cnt)


poi_pops = poi_info_all.loc[poi_train, 'popularity']
# print(np.array(poi_pops.reset_index(inplace=False)))


expo_pop1 = np.log10(max(1, min(poi_pops)))
expo_pop2 = np.log10(max(poi_pops))
# print(expo_pop1, expo_pop2)

nbins_pop = BIN_CLUSTER
logbins_pop = np.logspace(np.floor(expo_pop1), np.ceil(expo_pop2), nbins_pop + 1)
logbins_pop[0] = 0  # deal with underflow
if logbins_pop[-1] < poi_info_all['popularity'].max():
    logbins_pop[-1] = poi_info_all['popularity'].max() + 1


def gen_transmat_pop(trajid_list, traj_dict, poi_info, logbins_pop=logbins_pop):
    nbins = len(logbins_pop) - 1
    transmat_pop_cnt = pd.DataFrame(data=np.zeros((nbins, nbins), dtype=np.float),
                                    columns=np.arange(1, nbins + 1), index=np.arange(1, nbins + 1))
    for tid in trajid_list:
        t = traj_dict[tid]
        if len(t) > 1:
            for pi in range(len(t) - 1):
                p1 = t[pi]
                p2 = t[pi + 1]
                assert (p1 in poi_info.index and p2 in poi_info.index)
                pop1 = poi_info.loc[p1, 'popularity']
                pop2 = poi_info.loc[p2, 'popularity']
                pc1, pc2 = np.digitize([pop1, pop2], logbins_pop)
                transmat_pop_cnt.loc[pc1, pc2] += 1
    return normalise_transmat(transmat_pop_cnt), logbins_pop


poi_visits = poi_info_all.loc[poi_train, 'nVisit']

expo_visit1 = np.log10(max(1, min(poi_visits)))
expo_visit2 = np.log10(max(poi_visits))
# print(expo_visit1, expo_visit2)

nbins_visit = BIN_CLUSTER
logbins_visit = np.logspace(np.floor(expo_visit1), np.ceil(expo_visit2), nbins_visit + 1)
logbins_visit[0] = 0  # deal with underflow
if logbins_visit[-1] < poi_info_all['nVisit'].max():
    logbins_visit[-1] = poi_info_all['nVisit'].max() + 1


def gen_transmat_visit(trajid_list, traj_dict, poi_info, logbins_visit=logbins_visit):
    nbins = len(logbins_visit) - 1
    transmat_visit_cnt = pd.DataFrame(data=np.zeros((nbins, nbins), dtype=np.float), \
                                      columns=np.arange(1, nbins + 1), index=np.arange(1, nbins + 1))
    for tid in trajid_list:
        t = traj_dict[tid]
        if len(t) > 1:
            for pi in range(len(t) - 1):
                p1 = t[pi]
                p2 = t[pi + 1]
                assert (p1 in poi_info.index and p2 in poi_info.index)
                visit1 = poi_info.loc[p1, 'nVisit']
                visit2 = poi_info.loc[p2, 'nVisit']
                vc1, vc2 = np.digitize([visit1, visit2], logbins_visit)
                transmat_visit_cnt.loc[vc1, vc2] += 1
    return normalise_transmat(transmat_visit_cnt), logbins_visit


X = poi_all.loc[poi_train, ['poiLon', 'poiLat']]
nclusters = BIN_CLUSTER

kmeans = KMeans(n_clusters=nclusters, random_state=987654321)
kmeans.fit(X)

clusters = kmeans.predict(X)
POI_CLUSTER_LIST = sorted(np.unique(clusters))
POI_CLUSTERS = pd.DataFrame(data=clusters, index=poi_train)
POI_CLUSTERS.index.name = 'poiID'
POI_CLUSTERS.rename(columns={0: 'clusterID'}, inplace=True)
POI_CLUSTERS['clusterID'] = POI_CLUSTERS['clusterID'].astype(np.int)


def gen_transmat_neighbor(trajid_list, traj_dict, poi_info, poi_clusters=POI_CLUSTERS):
    nclusters = len(poi_clusters['clusterID'].unique())
    transmat_neighbor_cnt = pd.DataFrame(data=np.zeros((nclusters, nclusters), dtype=np.float), \
                                         columns=np.arange(nclusters), index=np.arange(nclusters))
    for tid in trajid_list:
        t = traj_dict[tid]
        if len(t) > 1:
            for pi in range(len(t) - 1):
                p1 = t[pi]
                p2 = t[pi + 1]
                assert (p1 in poi_info.index and p2 in poi_info.index)
                c1 = poi_clusters.loc[p1, 'clusterID']
                c2 = poi_clusters.loc[p2, 'clusterID']
                transmat_neighbor_cnt.loc[c1, c2] += 1
    return normalise_transmat(transmat_neighbor_cnt), poi_clusters


def gen_poi_logtransmat(trajid_list, poi_set, traj_dict, poi_info, debug=False):
    transmat_cat = gen_transmat_cat(trajid_list, traj_dict, poi_info)
    transmat_pop, logbins_pop = gen_transmat_pop(trajid_list, traj_dict, poi_info)
    transmat_visit, logbins_visit = gen_transmat_visit(trajid_list, traj_dict, poi_info)
    transmat_neighbor, poi_clusters = gen_transmat_neighbor(trajid_list, traj_dict, poi_info)

    # Kronecker product
    transmat_ix = list(
        itertools.product(transmat_cat.index, transmat_pop.index, transmat_visit.index, transmat_neighbor.index))
    transmat_value = transmat_cat.values
    for transmat in [transmat_pop, transmat_visit, transmat_neighbor]:
        transmat_value = kron(transmat_value, transmat.values)
    transmat_feature = pd.DataFrame(data=transmat_value, index=transmat_ix, columns=transmat_ix)

    poi_train = sorted(poi_set)
    feature_names = ['poiCat', 'popularity', 'nVisit', 'clusterID']
    poi_features = pd.DataFrame(data=np.zeros((len(poi_train), len(feature_names))),
                                columns=feature_names, index=poi_train)
    poi_features.index.name = 'poiID'
    poi_features['poiCat'] = poi_info.loc[poi_train, 'poiCat']
    poi_features['popularity'] = np.digitize(poi_info.loc[poi_train, 'popularity'], logbins_pop)
    poi_features['nVisit'] = np.digitize(poi_info.loc[poi_train, 'nVisit'], logbins_visit)
    poi_features['clusterID'] = poi_clusters.loc[poi_train, 'clusterID']

    # shrink the result of Kronecker product and deal with POIs with the same features
    poi_logtransmat = pd.DataFrame(data=np.zeros((len(poi_train), len(poi_train)), dtype=np.float),
                                   columns=poi_train, index=poi_train)
    for p1 in poi_logtransmat.index:
        rix = tuple(poi_features.loc[p1])
        for p2 in poi_logtransmat.columns:
            cix = tuple(poi_features.loc[p2])
            value_ = transmat_feature.loc[(rix,), (cix,)]
            poi_logtransmat.loc[p1, p2] = value_.values[0, 0]

    # group POIs with the same features
    features_dup = dict()
    for poi in poi_features.index:
        key = tuple(poi_features.loc[poi])
        if key in features_dup:
            features_dup[key].append(poi)
        else:
            features_dup[key] = [poi]
    if debug == True:
        for key in sorted(features_dup.keys()):
            print(key, '->', features_dup[key])

    # deal with POIs with the same features
    for feature in sorted(features_dup.keys()):
        n = len(features_dup[feature])
        if n > 1:
            group = features_dup[feature]
            v1 = poi_logtransmat.loc[group[0], group[0]]  # transition value of self-loop of POI group

            # divide incoming transition value (i.e. unnormalised transition probability) uniformly among group members
            for poi in group:
                poi_logtransmat[poi] /= n

            # outgoing transition value has already been duplicated (value copied above)

            # duplicate & divide transition value of self-loop of POI group uniformly among all outgoing transitions,
            # from a POI to all other POIs in the same group (excluding POI self-loop)
            v2 = v1 / (n - 1)
            for pair in itertools.permutations(group, 2):
                poi_logtransmat.loc[pair[0], pair[1]] = v2

    # normalise each row
    for p1 in poi_logtransmat.index:
        poi_logtransmat.loc[p1, p1] = 0
        rowsum = poi_logtransmat.loc[p1].sum()
        assert (rowsum > 0)
        logrowsum = np.log10(rowsum)
        for p2 in poi_logtransmat.columns:
            if p1 == p2:
                poi_logtransmat.loc[p1, p2] = LOG_ZERO  # deal with log(0) explicitly
            else:
                poi_logtransmat.loc[p1, p2] = np.log10(poi_logtransmat.loc[p1, p2]) - logrowsum

    return poi_logtransmat


QUERY_ID_RDICT = dict()
for k, v in QUERY_ID_DICT.items():
    QUERY_ID_RDICT[v] = k


def find_viterbi(V, E, ps, pe, L, withNodeWeight=False, alpha=0.5, withStartEndIntermediate=False):
    assert (isinstance(V, pd.DataFrame))
    assert (isinstance(E, pd.DataFrame))
    assert (ps in V.index)
    assert (pe in V.index)
    if L > V.index.shape[0]:
        L = V.index.shape[0]

    if withNodeWeight == True:
        assert (0 < alpha < 1)
        beta = 1 - alpha
    else:
        alpha = 0
        beta = 1
        weightkey = 'weight'
        if weightkey not in V.columns:
            V['weight'] = 1  # dummy weights, will not be used as alpha=0
    if withStartEndIntermediate == True:
        excludes = [ps]
    else:
        excludes = [ps, pe]

    A = pd.DataFrame(data=np.zeros((L - 1, V.shape[0]), dtype=np.float), columns=V.index, index=np.arange(2, L + 1))
    B = pd.DataFrame(data=np.zeros((L - 1, V.shape[0]), dtype=np.int), columns=V.index, index=np.arange(2, L + 1))
    A += np.inf
    for v in V.index:
        if v not in excludes:
            A.loc[2, v] = alpha * (V.loc[ps, 'weight'] + V.loc[v, 'weight']) + beta * E.loc[ps, v]  # ps--v
            B.loc[2, v] = ps

    for l in range(3, L + 1):
        for v in V.index:
            if withStartEndIntermediate == True:  # ps-~-v1---v
                values = [A.loc[l - 1, v1] + alpha * V.loc[v, 'weight'] + beta * E.loc[v1, v] for v1 in V.index]
            else:  # ps-~-v1---v
                values = [A.loc[l - 1, v1] + alpha * V.loc[v, 'weight'] + beta * E.loc[v1, v] \
                              if v1 not in [ps, pe] else -np.inf for v1 in V.index]  # exclude ps and pe

            maxix = np.argmax(values)
            A.loc[l, v] = values[maxix]
            B.loc[l, v] = V.index[maxix]

    path = [pe]
    v = path[-1]
    l = L
    while l >= 2:
        path.append(B.loc[l, v])
        v = path[-1]
        l -= 1
    path.reverse()
    return path


def find_beam(V, E, ps, pe, withNodeWeight=True, alpha=0.5,
              max_traj_length=10, B1=20, B2=10, lmda=0.5, NO_OF_DIVERSE_TRAJECTORIES=3, eligibility_div=0.2,
              min_traj_length=3):
    assert (isinstance(V, pd.DataFrame))
    assert (isinstance(E, pd.DataFrame))
    assert (ps in V.index)
    assert (pe in V.index)
    # assert (2 < max_traj_length <= V.index.shape[0])

    # if max_traj_length > V.index.shape[0]:
    #     max_traj_length = V.index.shape[0]

    if withNodeWeight:
        assert (0 < alpha < 1)
        beta = 1 - alpha
    else:
        alpha = 0
        beta = 1
        weightkey = 'weight'
        if weightkey not in V.columns:
            V['weight'] = 1  # dummy weights, will not be used as alpha=0

    A = np.ones(((max_traj_length + 1), max(poi_train) + 1, max(poi_train) + 1)) * LOG_ZERO

    node_scores = np.ones((max(poi_train) + 1), dtype=float) * LOG_ZERO
    for i in range(1, max(poi_train) + 1):
        if (i in V.index):
            node_scores[i] = V.loc[i, 'weight']

    trans_prob = np.ones((max(poi_train) + 1, max(poi_train) + 1), dtype=float) * LOG_ZERO
    for i in range(1, len(trans_prob)):
        for j in range(1, len(trans_prob[i])):
            if (i in V.index and j in V.index):
                trans_prob[i, j] = E.loc[i, j]

    for i in range(1, max(poi_train) + 1):
        if (i != ps and i != pe):
            A[2, i, ps] = alpha * node_scores[i] + beta * trans_prob[ps, i]
    for l in range(3, (max_traj_length + 1)):
        for pp in range(1, max(poi_train) + 1):
            if (pp == ps or pp == pe):
                continue
            for p in range(1, max(poi_train) + 1):
                if (pp == ps):
                    continue
                A[l, p, pp] = max([A[l - 1, pp, pdp] for pdp in range(1, max(poi_train) + 1)]) + alpha * node_scores[
                    p] + \
                              beta * trans_prob[pp, p]

    def find_fixed_length_beam(ps, pe, L, B1=20, B2=10, lmda=0.5):

        def get_B_argmax(scores_list, B):

            '''
            :param scores_list: list of scores
            :param B: no of POIs to return
            :return: pois sorted in order
            '''

            poi_list = list(poi_info_all.index)
            ans_set = []
            pois = np.argsort(-np.array(scores_list))
            no_pois = min(B, len(pois))
            for i in range(no_pois):
                if pois[i] in poi_list:
                    ans_set.append(pois[i])
            return ans_set

        candidate_poi_set_pe = get_B_argmax(A[L, pe, :], B1)

        B1_candidate_paths = []
        B1_candidate_paths_scores = []
        B1_selection_scores = []

        for i in range(len(candidate_poi_set_pe)):
            candidate_poi = candidate_poi_set_pe[i]
            candidate_path = [pe, candidate_poi]

            B1_candidate_paths.append(candidate_path)
            B1_candidate_paths_scores.append(beta * trans_prob[int(candidate_poi), int(pe)] +
                                             alpha * node_scores[int(pe)] - lmda * i)
            B1_selection_scores.append(A[L, pe, candidate_poi] + B1_candidate_paths_scores[-1])

        def get_next_iteration_beam(previous_itr_candidate_paths, previous_itr_candidate_paths_scores, node_no):

            all_candidate_paths = []
            all_candidate_paths_scores = []
            all_selection_scores = []

            for i in range(len(previous_itr_candidate_paths)):

                curr_itr_candidate_path = list(previous_itr_candidate_paths[i].copy())
                next_candidate_poi_set_for_path = get_B_argmax(A[node_no, int(previous_itr_candidate_paths[i][-1]), :],
                                                               B2)
                for j in range(len(next_candidate_poi_set_for_path)):
                    curr_itr_candidate_poi = next_candidate_poi_set_for_path[j]
                    if curr_itr_candidate_poi in curr_itr_candidate_path:
                        continue
                    candidate_path_ = curr_itr_candidate_path + [curr_itr_candidate_poi]

                    all_candidate_paths.append(candidate_path_)

                    partial_path_score = previous_itr_candidate_paths_scores[i] + \
                                         beta * trans_prob[
                                             int(curr_itr_candidate_poi), int(previous_itr_candidate_paths[i][-1])] + \
                                         alpha * node_scores[int(previous_itr_candidate_paths[i][-1])] - lmda * j

                    selection_score = A[node_no, int(previous_itr_candidate_paths[i][-1]), int(
                        curr_itr_candidate_poi)] + \
                                      partial_path_score

                    all_candidate_paths_scores.append(partial_path_score)
                    all_selection_scores.append(selection_score)
            # print("995")
            # print(all_candidate_paths)
            # print(all_candidate_paths_scores)
            # print(all_selection_scores)
            # print("\n")

            zp = zip(all_candidate_paths, all_candidate_paths_scores, all_selection_scores)
            l_zp = list(zp)
            l_zp = sorted(l_zp, key=lambda tup: tup[2], reverse=True)

            all_candidate_paths, all_candidate_paths_scores, all_selection_scores = zip(*l_zp)
            all_candidate_paths = list(all_candidate_paths)
            all_candidate_paths_scores = list(all_candidate_paths_scores)
            all_selection_scores = list(all_selection_scores)

            if B1 > len(all_candidate_paths):
                return all_candidate_paths, all_candidate_paths_scores

            return all_candidate_paths[:B1], all_candidate_paths_scores[:B1]

        for i in range(L - 1, 2, -1):
            B1_candidate_paths, B1_candidate_paths_scores = get_next_iteration_beam(B1_candidate_paths,
                                                                                    B1_candidate_paths_scores, i)

        for i in range(len(B1_candidate_paths)):
            B1_candidate_paths[i].append(ps)
            B1_candidate_paths[i].reverse()

        return B1_candidate_paths

    def path_prob(path):
        score = trans_prob[path[0], path[1]]
        for i in range(1, len(path) - 1):
            score += (alpha * node_scores[path[i]] + beta * trans_prob[path[i], path[i + 1]])
        return (1 / (len(path) ** 0.75)) * score

    def trajecory_diversity(traj1, traj2):
        return 1 - calc_F1(traj1, traj2)

    all_candidate_paths = []
    for length in range(3, (max_traj_length + 1)):
        L_cd_paths = find_fixed_length_beam(ps, pe, length, B1=B1, B2=B2, lmda=lmda)
        all_candidate_paths = all_candidate_paths + L_cd_paths

    # (all_candidate_paths)
    def refine_path(path):
        refined_inner = []
        for i in range(1, len(path) - 1):
            if path[i] not in refined_inner and path[i] != path[0] and path[i] != path[-1]:
                refined_inner.append(path[i])

        return [path[0]] + refined_inner + [path[-1]]

    all_candidate_paths_s = []
    for path in all_candidate_paths:
        rp = refine_path(path)
        if min_traj_length <= len(rp) <= max_traj_length:
            all_candidate_paths_s.append(rp)

    all_candidate_paths = sorted(all_candidate_paths_s, key=lambda path: path_prob(path), reverse=True)
    # print(all_candidate_paths)

    waiting_trajectories = []
    answer_trajectories = []
    for i in range(len(all_candidate_paths)):
        current_traj = all_candidate_paths[i]
        eligible = True
        for j in range(len(answer_trajectories)):
            if trajecory_diversity(current_traj, answer_trajectories[j]) < eligibility_div:
                eligible = False
        if eligible:
            answer_trajectories.append(current_traj)
            if len(answer_trajectories) == NO_OF_DIVERSE_TRAJECTORIES:
                break
        else:
            waiting_trajectories.append(current_traj)

    itr = 0
    while len(answer_trajectories) < NO_OF_DIVERSE_TRAJECTORIES:
        answer_trajectories.append(waiting_trajectories[itr])
        itr += 1

    return answer_trajectories


def cv_choose_alpha(alpha_set):
    print("a")

    short_traj_set = list(traj_all[traj_all['trajLen'] <= 2]['trajID'].unique())
    long_traj_set_train = list(set(trajid_train) - set(short_traj_set))

    best_score = 0
    best_alpha = 0
    trainset_ratio = 0.8

    alpha_training_set = short_traj_set + long_traj_set_train[:int(trainset_ratio * len(long_traj_set_train))]
    alpha_validation_set = long_traj_set_train[int(trainset_ratio * len(long_traj_set_train)):]

    print("b")

    trajid_list_train = alpha_training_set
    poi_info = calc_poi_info(trajid_list_train, traj_all, poi_all)

    print("c")

    train_df = gen_train_df(trajid_list_train, traj_dict, poi_info, poi_clusters=POI_CLUSTERS,
                            cats=POI_CAT_LIST, clusters=POI_CLUSTER_LIST, n_jobs=N_JOBS)

    print("d")

    def convert_query(query_tuple):
        poi1 = query_tuple[0]
        poi2 = query_tuple[1]
        final_q = list()
        for poi1_i in range(1, max(poi_train) + 1):
            if (poi1_i == poi1):
                final_q.append(1)
            else:
                final_q.append(0)

        for poi2_i in range(1, max(poi_train) + 1):
            if (poi2_i == poi2):
                final_q.append(1)
            else:
                final_q.append(0)

        final_q = tuple(final_q)

        return final_q

    query_list = []
    for idx in train_df.index:
        query_list.append(convert_query(QUERY_ID_RDICT[train_df.at[idx, 'queryID']]))

    train_df = train_df.drop(['queryID'], axis=1)
    train_df.insert(2, 'queryID', query_list, True)
    final_train_Y = np.array(train_df['label'])
    train_df_without_label = train_df.drop(['label'], axis=1)
    final_train_X = gen_data_svm(train_df_without_label)
    final_train_X = np.array(final_train_X)
    final_train_Y = np.array(final_train_Y)
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(final_train_X)
    final_train_X_scaled = scaler.transform(final_train_X)

    print("Training Start")
    svm_model_linear = SVC(kernel='linear', C=RANKSVM_COST, class_weight='balanced', random_state=5).fit(
        final_train_X_scaled,
        final_train_Y)

    poi_logtransmat = gen_poi_logtransmat(trajid_list_train, set(poi_info.index), traj_dict, poi_info)
    print("Training Done")

    for alpha_i in alpha_set:

        scores = []
        for i in range(len(alpha_validation_set)):
            tid = alpha_validation_set[i]
            te = traj_dict[tid]
            assert (len(te) > 2)
            # start/end is not in training set
            if not (te[0] in poi_info.index and te[-1] in poi_info.index):
                print('Failed cross-validation instance:', te)
                continue
            test_df = gen_test_df(te[0], te[-1], poi_info, poi_clusters=POI_CLUSTERS, cats=POI_CAT_LIST,
                                  clusters=POI_CLUSTER_LIST)
            query_list_test = []
            for idx in test_df.index:
                query_list_test.append(convert_query(QUERY_ID_RDICT[test_df.at[idx, 'queryID']]))
            test_df = test_df.drop(['queryID'], axis=1)
            test_df.insert(2, 'queryID', query_list_test, True)
            test_df_without_label = test_df.drop(['label'], axis=1)
            final_test_X = gen_data_svm(test_df_without_label)
            final_test_X = np.array(final_test_X)
            final_test_X_scaled = scaler.transform(final_test_X)
            rank_df_t = svm_model_linear.predict(final_test_X_scaled)
            poi_rank_df = pd.DataFrame(rank_df_t)
            poi_rank_df.rename(columns={0: 'rank'}, inplace=True)
            poi_rank_df['poiID'] = test_df['poiID'].astype(np.int)
            poi_rank_df.set_index('poiID', inplace=True)
            poi_rank_df['probability'] = softmax(poi_rank_df['rank'])

            # print(poi_rank_df)
            # print(rank_df)
            # print(te)
            # print("\n\n")

            rank_df = poi_rank_df
            edges = poi_logtransmat.copy()
            nodes = rank_df.copy()
            nodes['weight'] = np.log10(nodes['probability'])
            nodes.drop('probability', axis=1, inplace=True)
            comb = find_viterbi(nodes, edges, te[0], te[-1], len(te), withNodeWeight=True, alpha=alpha_i)
            scores.append(calc_pairsF1(te, comb))

        mean_score = np.mean(scores)
        print('alpha:', alpha_i, ' mean pairs-F1:', mean_score)
        if best_score > mean_score: continue
        best_score = mean_score
        best_alpha = alpha_i
    return best_alpha


def write_to_file(dictionary, directory, N_min, N_max, isFreq=False):
    if not isFreq:
        file_path = os.path.join(directory, str(dat_suffix[dat_ix])
                                 + "_index_" + str(test_index)
                                 + "_min_" + str(N_min)
                                 + "_max_" + str(N_max)
                                 + "_copy_" + str(copy_no)) + '.csv'
    else:
        file_path = os.path.join(directory, str(dat_suffix[dat_ix]) + "_" +
                                 str(test_index)) + '_freq.csv'

    write_lines = []

    for k, v in dictionary.items():
        for i in range(len(v)):
            write_lines.append(v[i])
        write_lines.append([-1])

    with open(file_path, mode='w+', newline="") as to_csv_file:
        csv_file_writer = csv.writer(to_csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in write_lines:
            csv_file_writer.writerow(row)

    return


def get_results(K_start, K_end, N_min_, N_max_, load_from_file=False):
    def convert_query(query_tuple):
        poi1 = query_tuple[0]
        poi2 = query_tuple[1]
        final_q = list()
        for poi1_i in range(1, max(poi_train) + 1):
            if poi1_i == poi1:
                final_q.append(1)
            else:
                final_q.append(0)

        for poi2_i in range(1, max(poi_train) + 1):
            if poi2_i == poi2:
                final_q.append(1)
            else:
                final_q.append(0)

        final_q = tuple(final_q)

        return final_q

    if run_comb:
        if not load_from_file:
            alpha_cv = cv_choose_alpha(ALPHA_SET)
            print('alpha:', alpha_cv)

            with open(os.path.join(model_directory, 'alpha.txt'), mode='w') as file:
                file.write(str(alpha_cv))

            trajid_list_train = trajid_train
            poi_info = calc_poi_info(trajid_list_train, traj_all, poi_all)

            train_df = gen_train_df(trajid_list_train, traj_dict, poi_info, poi_clusters=POI_CLUSTERS,
                                    cats=POI_CAT_LIST, clusters=POI_CLUSTER_LIST, n_jobs=N_JOBS)

            query_list = []
            for idx in train_df.index:
                query_list.append(convert_query(QUERY_ID_RDICT[train_df.at[idx, 'queryID']]))

            train_df = train_df.drop(['queryID'], axis=1)
            train_df.insert(2, 'queryID', query_list, True)
            final_train_Y = np.array(train_df['label'])
            train_df_without_label = train_df.drop(['label'], axis=1)
            final_train_X = gen_data_svm(train_df_without_label)
            final_train_X = np.array(final_train_X)
            final_train_Y = np.array(final_train_Y)
            scaler = sklearn.preprocessing.StandardScaler()
            scaler.fit(final_train_X)
            final_train_X_scaled = scaler.transform(final_train_X)

            svm_model_linear = SVC(kernel='linear', C=RANKSVM_COST, class_weight='balanced', random_state=5).fit(
                final_train_X_scaled,
                final_train_Y)

            with open(os.path.join(model_directory, "svm_model"), mode='wb') as file:
                pickle.dump(svm_model_linear, file)

            with open(os.path.join(model_directory, "scaler"), mode='wb') as file:
                pickle.dump(scaler, file)

            # svm_predictions = svm_model_linear.predict(final_train_X_scaled)
            # cm = confusion_matrix(final_train_Y, svm_predictions)
            # print(cm)

            poi_logtransmat = gen_poi_logtransmat(trajid_list_train, set(poi_info.index), traj_dict, poi_info)
            max_traj_length = np.min(
                [np.max([len(traj_dict[tid]) for tid in trajid_train]), len(list(poi_info_all.index))])

        else:

            trajid_list_train = trajid_train
            poi_info = calc_poi_info(trajid_list_train, traj_all, poi_all)
            poi_logtransmat = gen_poi_logtransmat(trajid_list_train, set(poi_info.index), traj_dict, poi_info)

            with open(os.path.join(model_directory, 'alpha.txt'), mode='r') as file:
                alpha_cv = float(file.read())

            with open(os.path.join(model_directory, "scaler"), mode='rb') as file:
                scaler = pickle.load(file)

            with open(os.path.join(model_directory, "svm_model"), mode='rb') as file:
                svm_model_linear = pickle.load(file)

        def isSame_(traj_a, traj_b):
            if len(traj_a) != len(traj_b):
                return False
            for i in range(len(traj_a)):
                if traj_a[i] != traj_b[i]:
                    return False

            return True

        for K in range(K_start, K_end):
            NO_OF_DIVERSE_TRAJECTORIES = K

            total_score_curr_f1 = 0
            total_score_curr_pf1 = 0
            total_score_likability = 0
            total_score_intra_div_f1 = 0

            total_traj_curr = 0
            count = 1

            all_gtset = dict()
            all_recset = dict()
            all_gtfreqset = dict()

            st = time.time()
            for k, v in test_data_dicts[0].items():
                str_k = str(k).split("-")
                start_poi = int(str_k[0])
                end_poi = int(str_k[1])

                query = [start_poi, end_poi]

                GT_set = []
                for i_GT in range(len(v)):
                    gotBefore = False
                    for j in range(len(GT_set)):
                        if isSame_(v[i_GT], GT_set[j]):
                            gotBefore = True
                    if not gotBefore:
                        GT_set.append(v[i_GT])

                GT_freq = query_dict_freq_test[k]

                if not (start_poi in poi_info.index and end_poi in poi_info.index):
                    print('Failed cross-validation instance:' + str(k))
                    continue

                test_df = gen_test_df(start_poi, end_poi, poi_info_all, poi_clusters=POI_CLUSTERS, cats=POI_CAT_LIST,
                                      clusters=POI_CLUSTER_LIST)

                query_list_test = []
                for idx in test_df.index:
                    query_list_test.append(convert_query(QUERY_ID_RDICT[test_df.at[idx, 'queryID']]))

                test_df = test_df.drop(['queryID'], axis=1)
                test_df.insert(2, 'queryID', query_list_test, True)

                test_df_without_label = test_df.drop(['label'], axis=1)
                final_test_X = gen_data_svm(test_df_without_label)

                final_test_X = np.array(final_test_X)

                final_test_X_scaled = scaler.transform(final_test_X)
                rank_df_t = svm_model_linear.predict(final_test_X_scaled)

                poi_rank_df = pd.DataFrame(rank_df_t)
                poi_rank_df.rename(columns={0: 'rank'}, inplace=True)
                poi_rank_df['poiID'] = test_df['poiID'].astype(np.int)
                poi_rank_df.set_index('poiID', inplace=True)
                poi_rank_df['probability'] = softmax(poi_rank_df['rank'])

                rank_df = poi_rank_df

                edges = poi_logtransmat.copy()

                nodes = rank_df.copy()
                nodes['weight'] = np.log10(nodes['probability'])
                nodes.drop('probability', axis=1, inplace=True)

                rec_set = find_beam(nodes, edges, start_poi, end_poi, alpha=alpha_cv, min_traj_length=N_min_,
                                    max_traj_length=N_max_, lmda=LMDA,
                                    NO_OF_DIVERSE_TRAJECTORIES=NO_OF_DIVERSE_TRAJECTORIES,
                                    eligibility_div=0.2)

                '''
                find_beam(V, E, ps, pe, withNodeWeight=True, alpha=0.5,
                      max_traj_length=10,B1 = 10,B2 = 3,lmda=0.5,NO_OF_DIVERSE_TRAJECTORIES=3,eligibility_div=0.2)
                '''

                all_gtset[k] = GT_set
                all_recset[k] = rec_set
                all_gtfreqset[k] = [GT_freq]

                total_score_curr_f1 += metric.tot_f1_evaluation(GT_set, GT_freq, rec_set)
                total_score_curr_pf1 += metric.tot_pf1_evaluation(GT_set, GT_freq, rec_set)
                total_score_likability += metric.likability_score_3(GT_set, GT_freq, rec_set)
                total_score_intra_div_f1 += metric.intra_div_F1(rec_set)

                total_traj_curr += np.sum(GT_freq) * len(rec_set)

                avg_likability = total_score_likability / count
                avg_div = total_score_intra_div_f1 / count
                avg_f1 = total_score_curr_f1 / total_traj_curr
                avg_pf1 = total_score_curr_pf1 / total_traj_curr

                print("Avg. upto now: Likability: " + str(avg_likability) + " F1: " + str(avg_f1) + " PF1: " + str(
                    avg_pf1)
                      + " Div: " + str(avg_div))

                count += 1

            en = time.time()
            print(N_max_)
            print(count)
            print("Time: {}".format((en - st) / (count - 1)))
            print("\n")
            print("Final Score - With K = {}".format(K))
            avg_likability = total_score_likability / (count - 1)
            avg_div = total_score_intra_div_f1 / (count - 1)
            avg_f1 = total_score_curr_f1 / total_traj_curr
            avg_pf1 = total_score_curr_pf1 / total_traj_curr

            print("Likability: " + str(avg_likability) + " F1: " + str(avg_f1) + " PF1: " + str(avg_pf1)
                  + " Div: " + str(avg_div))

            write_to_file(all_recset, 'recset_markov', N_min=N_min, N_max=N_max)


N_s = [(5, 5)]
for Nmn, Nmx in N_s:
    get_results(K_start=K_FINAL, K_end=K_FINAL+1, N_min_=Nmn, N_max_=Nmx, load_from_file=True)
