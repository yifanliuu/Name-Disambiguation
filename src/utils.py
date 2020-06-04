import config as cfg
import codecs
import json
from os.path import join
import pickle
import os
import re

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
        print(stop_words)

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


if __name__ == "__main__":
    pass
    # --------load json test-----------
    # pubs_train = load_json(cfg.TRAIN_PUB_PATH)
    # pubs_val = load_json(cfg.VAL_PUB_PATH)
    # print(pubs_train)
    # print(pubs_val)
