import config as cfg
import codecs
import json
from os.path import join
import pickle
import os
import re
import scipy.sparse as sp
import scipy
import numpy as np
# import torch

# -------------load and save data--------------


def load_json(rfpath):
    with codecs.open(rfpath, 'r', encoding='utf-8') as rf:
        return json.load(rf)


def dump_json(obj, wfname, indent=None):
    with codecs.open(wfname, 'w', encoding='utf-8') as wf:
        json.dump(obj, wf, ensure_ascii=False, indent=indent)


def dump_data(obj, wfname):
    with open(wfname, 'wb') as wf:
        pickle.dump(obj, wf)


def load_data(rfpath):
    with open(rfpath, 'rb') as rf:
        return pickle.load(rf)


def load_stopWords(rfpath=cfg.STOP_WORDS_PATH):
    with open(rfpath, 'r') as rf:
        stop_words = rf.readlines()
        for i, word in enumerate(stop_words):
            stop_words[i] = word.strip()
        return stop_words


def load_pub_features(rfpath):

    print("Loading from " + rfpath + "......")

    features = {}
    with open(rfpath, 'r') as rf:
        count = 0
        raw_features = rf.readlines()
        for line in raw_features:
            line = line.strip().split(' ')
            paperId = line[0]
            paperFeature = line[1:]
            for i in range(len(paperFeature)):
                paperFeature[i] = float(paperFeature[i])
            paperFeature = np.array(paperFeature)
            features[paperId] = paperFeature
            count += 1
            if count % 10000 == 0:
                print(str(count) + ' Done')
    return features


def save_pub_features(features, rfpath):

    with open(rfpath, 'w') as rf:
        count = 0
        for key, value in features.items():
            value = value.tolist()
            rf.write(key + ' ')
            for num in value[:-1]:
                rf.write(str(num) + ' ')
            rf.write(str(value[-1]) + '\n')
            count += 1
            if count % 10000 == 0:
                print(str(count) + ' Done')

 # Using this path: VAL_SEMATIC_FEATURES_PATH to save features by name


def save_sematic_features_byAuthor(features, rfpath=cfg.VAL_SEMATIC_FEATURES_PATH):

    # Assume we have 4 papers named paper1 - paper4, so features will be like below.
    # features = {
    #     'papaer1': 'feature1'(np.array)
    #     'papaer2': 'feature2'(np.array)
    #     'papaer3': 'feature3'(np.array)
    #     'papaer4': 'feature4'(np.array)
    # }

    pass


# -------------evaluate---------------
def pairwise_evaluate(correct_labels, pred_labels):
    TP = 0.0  # Pairs Correctly Predicted To SameAuthor
    TP_FP = 0.0  # Total Pairs Predicted To SameAuthor
    TP_FN = 0.0  # Total Pairs To SameAuthor

    for i in range(len(correct_labels)):
        for j in range(i + 1, len(correct_labels)):
            if correct_labels[i] == correct_labels[j]:
                TP_FN += 1
            if pred_labels[i] == pred_labels[j]:
                TP_FP += 1
            if (correct_labels[i] == correct_labels[j]) and (pred_labels[i] == pred_labels[j]):
                TP += 1

    if TP == 0:
        pairwise_precision = 0
        pairwise_recall = 0
        pairwise_f1 = 0
    else:
        pairwise_precision = TP / TP_FP
        pairwise_recall = TP / TP_FN
        pairwise_f1 = (2 * pairwise_precision * pairwise_recall) / \
            (pairwise_precision + pairwise_recall)
    return pairwise_precision, pairwise_recall, pairwise_f1


# --------------get author and org features----------------
def format_name(names):
    x = [k.strip() for k in names.lower().strip()]
    return x


'''
# -------------- graph preprocess ---------------
def preprocess_graph(coo_node_list, n_node):
    coo_numpy = np.array(coo_node_list, dtype=np.int32)
    shape = (n_node, n_node)
    rows = coo_numpy[:, 0]
    cols = coo_numpy[:, 1]
    data = coo_numpy[:, 2]
    adj = sp.coo_matrix((data, (rows, cols)), shape)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    # print(np.max(rowsum), np.min(rowsum))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(
        degree_mat_inv_sqrt).tocoo()

    # print(np.max(adj_normalized), np.min(adj_normalized))

    # return sparse_to_tuple(adj_normalized)
    return sparse_mx_to_torch_sparse_tensor(adj_normalized), adj
'''

'''
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
'''


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


# ---------------- cos angle ------------------
def cosangle(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ------------- mapping paper id to idx ----------
def graphMapping(graph, rfpath=cfg.VAL_AUTHOR_PATH):
    pid2idx_dict_by_name, _ = pid2idxMapping(rfpath=rfpath)
    for idx, name in enumerate(graph):
        pid2idx_dict = pid2idx_dict_by_name[name]
        for i, triplets in enumerate(graph[name]):
            graph[name][i][0] = pid2idx_dict[triplets[0]]
            graph[name][i][1] = pid2idx_dict[triplets[1]]
    return graph


def pid2idxMapping(rfpath=cfg.VAL_AUTHOR_PATH):
    author_pubs_raw = load_json(rfpath=rfpath)
    pid2idx_dict_by_name = {}
    names = []
    for i, name in enumerate(author_pubs_raw):
        pid2idx_dict = {}
        names.append(name)
        for j, pid in enumerate(author_pubs_raw[name]):
            pid2idx_dict[pid] = j
        pid2idx_dict_by_name[name] = pid2idx_dict
    return pid2idx_dict_by_name, names


def read_embeddings(name, n_node, filename=cfg.VAL_PUB_FEATURES_PATH, n_embed=100):

    embedding_matrix = np.random.rand(0, n_embed)

    feats = load_pub_features(rfpath=filename)
    author_pubs_raw = load_json(rfpath=cfg.VAL_AUTHOR_PATH)

    for i, pid in enumerate(author_pubs_raw[name]):
        embedding_matrix = np.append(embedding_matrix, np.reshape(
            feats[pid], [1, n_embed]), axis=0)
        # print(embedding_matrix.shape)

    return embedding_matrix


if __name__ == "__main__":
    pass
    # --------load json test-----------
    # pubs_train = load_json(cfg.TRAIN_PUB_PATH)
    # pubs_val = load_json(cfg.VAL_PUB_PATH)
    # print(pubs_train)
    # print(pubs_val)

    # -------- test ----------
    # data = np.load('../results/gen/weiping_liu_gen.npy')
    # print(data.shape)

    # -------- test read embedding
    # m = read_embeddings('weiping_liu', 684)
    # print(m.shape)
