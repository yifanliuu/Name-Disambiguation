import config as cfg
import codecs
import json
from os.path import join
import pickle
import os
import re
import numpy as np
import torch

# -------------load and save data--------------


def load_json(rfpath):
    with codecs.open(rfpath, 'r', encoding='utf-8') as rf:
        return json.load(rf)


def dump_json(obj, wfpath, wfname, indent=None):
    with codecs.open(join(wfpath, wfname), 'w', encoding='utf-8') as wf:
        json.dump(obj, wf, ensure_ascii=False, indent=indent)


def dump_data(obj, wfpath, wfname):
    with open(os.path.join(wfpath, wfname), 'wb') as wf:
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


def load_pub_features(rfpath=cfg.TRAIN_PUB_FEATURES_PATH):
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


def save_pub_features(features, rfpath=cfg.TRAIN_PUB_FEATURES_PATH):
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

# TODO: inverted document frequency

# --------------get author and org features----------------


def format_name(names):
    x = [k.strip() for k in names.lower().strip()]
    return x


def get_author_and_org_features(item, order=None):
    author_features = []
    for i, author in enumerate(item["authors"]):
        if order is not None and i != order:
            continue
        name_features = []
        org_features = []
        names = author.get("org", "")
        names = format_name(names)
    # TODO:


def cosangle(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# -------------- graph preprocess ---------------

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(
        degree_mat_inv_sqrt).tocoo()
    # return sparse_to_tuple(adj_normalized)
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def generate_embeded_features(raw_feat):
    pass


def generate_graph():
    pass


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.FloatTensor(indices, values, shape)
    # TODO: 生成torch向量


if __name__ == "__main__":
    pass
    # --------load json test-----------
    # pubs_train = load_json(cfg.TRAIN_PUB_PATH)
    # pubs_val = load_json(cfg.VAL_PUB_PATH)
    # print(pubs_train)
    # print(pubs_val)
