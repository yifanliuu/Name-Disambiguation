from gensim.models.keyedvectors import KeyedVectors
import config as cfg
import numpy as np
import string
import re
import logging
from models import Word2Vec
import multiprocessing
import time
from sklearn.metrics import pairwise_distances
from utils import *


EMBEDDING_SIZE = 100
del_str = string.punctuation
replace_str = ' '*len(del_str)
transTab = str.maketrans(del_str, replace_str)


# ---------------generate all text using for train word embedding--------------
"""
using " organization，title， abstract， venue, keyworlds, year" as corpos to  train WORD EMBEDDING
"""


def generateCorpus():

    corpusFile = open(cfg.ALL_TEXT_PATH, 'w')

    r = r'[!“”"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~—～’]+'
    del_str = string.punctuation
    replace_str = ' '*len(del_str)
    transTab = str.maketrans(del_str, replace_str)

    pubs_raw = load_json(cfg.TRAIN_PUB_PATH)
    for paperId, paperDetail in pubs_raw.items():
        title = paperDetail['title']
        abstract = paperDetail['abstract']
        key_words = paperDetail.get("keywords")
        keyword = ' '.join(key_words)
        venue_name = paperDetail.get('venue')
        year = paperDetail.get('year')
        authors = paperDetail.get('authors')
        orgset = set()
        for author in authors:
            orgs = author['org'].split(';')
            for org in orgs:
                if org not in orgset:
                    orgset.add(org)
        line = title + ' ' + abstract + ' ' + keyword + \
            ' ' + venue_name + ' ' + str(year) + ' '
        for org in orgset:
            line = line + org + ' '
        line = line.translate(transTab)
        line = re.sub(r'\s{2,}', ' ', re.sub(r, ' ', line)).strip()
        line = line + '\n'
        corpusFile.write(line.lower())

    val_pubs_raw = load_json(cfg.VAL_PUB_PATH)
    for paperId, paperDetail in val_pubs_raw.items():
        title = paperDetail['title']
        abstract = paperDetail['abstract']
        key_words = paperDetail.get("keywords")
        keyword = ' '.join(key_words)
        venue_name = paperDetail.get('venue')
        year = paperDetail.get('year')
        authors = paperDetail.get('authors')
        orgset = set()
        for author in authors:
            orgs = author['org'].split(';')
            for org in orgs:
                if org not in orgset:
                    orgset.add(org)
        line = title + ' ' + abstract + ' ' + keyword + \
            ' ' + venue_name + ' ' + str(year) + ' '
        for org in orgset:
            line = line + org + ' '
        line = line.translate(transTab)
        line = re.sub(r'\s{2,}', ' ', re.sub(r, ' ', line)).strip()
        line = line + '\n'
        corpusFile.write(line.lower())

    return

# ---------------load authors.json to generate ground truth and all papaers--------------


def generateCandidateSets(mode='train'):
    '''
    mode: 'train' or 'val'
    '''
    authors = None
    if mode == 'train':
        authors = load_json(cfg.TRAIN_AUTHOR_PATH)
    elif mode == 'val':
        authors = load_json(cfg.VAL_AUTHOR_PATH)
    else:
        raise Exception("mode should be 'train' or 'val'\n")

    pubs_by_name = {}
    labels_by_name = {}
    # {name: pubs}

    for n, name in enumerate(authors):
        i_label = 0
        pubs = []  # all papers
        labels = []  # ground truth

    # TODO: take off stop words and words appear less then 3 times
        for identity in authors[name]:
            identity_pubs = authors[name][identity]
            for pub in identity_pubs:
                pubs.append(pub)
                labels.append(i_label)
            i_label += 1

        # print(n, name, len(pubs))
        pubs_by_name[name] = pubs
        labels_by_name[name] = labels

    return labels_by_name, pubs_by_name

# --------------load authors.json to generate all papaers(for test bt now for val) ----------------


def generateCandidateSetsTest():
    authors = load_json(cfg.VAL_AUTHOR_PATH)

    pubs_by_name = {}
    # {name: pubs}

    for n, name in enumerate(authors):
        pubs = []  # all papers

        for pub in authors[name]:
            pubs.append(pub)
        pubs_by_name[name] = pubs

    return pubs_by_name

# --------------load pubs.json and generate raw features---------------


def generateRawFeatrues(mode='train'):
    '''
    mode: 'train' or 'val' or 'test'
    '''
    stopWords = load_stopWords()

    pubs_raw = None
    if mode == 'train':
        pubs_raw = load_json(cfg.TRAIN_PUB_PATH)
    elif mode == 'val':
        pubs_raw = load_json(cfg.VAL_PUB_PATH)
    elif mode == 'test':
        exit(0)
    else:
        raise Exception("mode should be 'train' or 'val' or 'test'\n")

    r = r'[!“”"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~—～’]+'

    # embeddings, each one 100 dimension
    word_embedding = Word2Vec()
    word_embedding.load()
    # print(word_embedding['optimization'])
    # print(type(word_embedding['optimization']))
    # print(word_embedding['optimization'].size)

    sementic_features = {}

    count = 0
    for paperId, paperDetail in pubs_raw.items():
        title = paperDetail['title']
        abstract = paperDetail['abstract']
        key_words = paperDetail.get("keywords")
        keyword = ' '.join(key_words)
        venue_name = paperDetail.get('venue')
        year = paperDetail.get('year')
        authors = paperDetail.get('authors')

        # relation_features[paperId] = authors

        # NOTE: A paper have many authors with some organizations, but we calculate them several times, may be will be modified here.
        orglist = ''

        for author in authors:
            orgs = author['org'].split(';')
            for org in orgs:
                if org not in orglist:
                    orglist = orglist + org

        paper_words = title + ' ' + abstract + ' ' + keyword + \
            ' ' + venue_name + ' ' + str(year) + ' ' + orglist
        # print(paper_words)
        paper_words = paper_words.translate(transTab)
        paper_words = re.sub(r'\s{2,}', ' ', re.sub(
            r, ' ', paper_words)).strip()
        # print(paper_words)

        paper_words = paper_words.lower().split(' ')
        paper_embedding_words = []

    # delete STOP words and words too short
        for word in paper_words:
            if word in stopWords or len(word) < 3:
                continue
            else:
                paper_embedding_words.append(word)

        paper_embedding = np.zeros(EMBEDDING_SIZE)
        l = len(paper_embedding_words)

        for i, word in enumerate(paper_embedding_words):
            try:
                paper_embedding = paper_embedding + word_embedding.model[word]
            except:
                l = l - 1

        paper_embedding = paper_embedding / l
        sementic_features[paperId] = paper_embedding.copy()
        count += 1
        if count % 1000 == 0:
            print(str(count) + ' Done')

    # print(sementic_features['cFtStBA6'])
    # print(relation_features['cFtStBA6'])

    print('Wrting features to file......')
    if mode == "train":
        rfpath = cfg.TRAIN_PUB_FEATURES_PATH
    elif mode == "val":
        rfpath = cfg.VAL_PUB_FEATURES_PATH
    save_pub_features(sementic_features, rfpath)

    return sementic_features


def cal_simi_thread(param):
    return cosangle(param[0], param[1])


def Cal_Simalarity_byAuthor_labeled(features_path, author_path, save_folder):
    gtime_st = time.time()
    features = load_pub_features(features_path)
    paper_by_author_labeled = load_json(author_path)
    paper_by_author = dict()
    for author, labeled_paper in paper_by_author_labeled.items():
        paper_by_author[author] = []
        for label, papers in labeled_paper.items():
            paper_by_author[author] += papers

    simi_matrix = []
    p = multiprocessing.Pool(4)
    # print(len(paper_by_author))
    for author, papers in paper_by_author.items():
        # print(len(papers))
        time_start = time.time()
        l = len(papers)

        x = []
        for paper in papers:
            x.append(features[paper])
        x = np.array(x)
        res = pairwise_distances(x, metric='cosine', n_jobs=-1)

        # size = l * l
        # simi_matrix = np.zeros(size)
        # jobs_param = []
        # for index in range(size):
        #     i = int(index / l)
        #     j = index % l
        #     jobs_param.append([features[papers[i]], features[papers[j]]])
        # # print("start calculate")
        # res = p.map(cal_simi_thread, jobs_param)
    # write similarity matrices to file

        np.save(save_folder + author + '.npy', np.array(res))
        time_end = time.time()
        print("calculate " + author + " done, using time(s): " +
              str(time_end-time_start))
    gtime_end = time.time()
    print()
    print("TOTAL USING TIME(s): " + str(gtime_st-gtime_end))


def Cal_Simalarity_byAuthor_unlabeled(features_path, author_path, save_folder):
    gtime_st = time.time()
    features = load_pub_features(features_path)
    paper_by_author = load_json(author_path)

    simi_matrix = []
    p = multiprocessing.Pool(4)
    # print(len(paper_by_author))
    for author, papers in paper_by_author.items():
        # print(len(papers))
        time_start = time.time()
        l = len(papers)

        x = []
        for paper in papers:
            x.append(features[paper])
        x = np.array(x)
        res = pairwise_distances(x, metric='cosine', n_jobs=-1)
        # size = l * l
        # simi_matrix = np.zeros(size)
        # jobs_param = []
        # for index in range(size):
        #     i = int(index / l)
        #     j = index % l
        #     jobs_param.append([features[papers[i]], features[papers[j]]])
        # # print("start calculate")
        # res = p.map(cal_simi_thread, jobs_param)
    # write similarity matrices to file
        np.save(save_folder + author + '.npy', np.array(res))
        print(res.shape)
        time_end = time.time()
        print("calculate " + author + " done, using time(s): " +
              str(time_end-time_start))
    gtime_end = time.time()
    print("TOTAL USING TIME(s): " + str(gtime_st-gtime_end))


def generate_wordembedding():
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    word_embedding = Word2Vec()
    word_embedding.train()
    word_embedding.save()

# ------------------- graph generation --------------------


def generateGraph(mode='train'):
    """
    generate graph use co-authors and organization
    """
    pubs_raw = None
    name_to_pubs = None
    if mode == 'train':
        pubs_raw = load_json(cfg.TRAIN_PUB_PATH)
        _, name_to_pubs = generateCandidateSets(mode=mode)
    elif mode == 'val':
        pubs_raw = load_json(cfg.VAL_PUB_PATH)
        name_to_pubs = generateCandidateSetsTest()
    elif mode == 'test':
        exit(0)
    else:
        raise Exception("mode should be 'train' or 'val' or 'test'\n")

    graph = {}
    for i, name in enumerate(name_to_pubs):
        print(i, name)
        paper_list = name_to_pubs[name]
        coo_node_list = []

        for j, pid1 in enumerate(paper_list):
            for k, pid2 in enumerate(paper_list):
                if(j >= k):
                    continue
                name_count, org_count = coAuthorOrg_num(
                    pubs_raw[pid1]['authors'],
                    pubs_raw[pid2]['authors']
                )
                count = name_count + org_count
                if count != 0:
                    coo_node_list.append([pid1, pid2, count])
        graph[name] = coo_node_list
        # generate content
        # wf_content = open(join(graph_dir, '{}_pubs_content.txt'.format(name)), 'w')
    if mode == 'train':
        dump_json(graph, cfg.TRAIN_GRAPH_PATH)
    elif mode == 'val':
        dump_json(graph, cfg.VAL_GRAPH_PATH)
    return graph


def generateRelationFeatures(mode='train'):
    """
    generate graph use co-authors and organization
    """
    pubs_raw = None
    name_to_pubs = None
    if mode == 'train':
        pubs_raw = load_json(cfg.TRAIN_PUB_PATH)
        _, name_to_pubs = generateCandidateSets(mode=mode)
    elif mode == 'val':
        pubs_raw = load_json(cfg.VAL_PUB_PATH)
        name_to_pubs = generateCandidateSetsTest()
    elif mode == 'test':
        exit(0)
    else:
        raise Exception("mode should be 'train' or 'val' or 'test'\n")

    for i, name in enumerate(name_to_pubs):
        print(i, name)
        paper_list = name_to_pubs[name]
        relation_matrix = np.zeros([len(paper_list), len(paper_list)])

        for j, pid1 in enumerate(paper_list):
            for k, pid2 in enumerate(paper_list):
                if(j >= k):
                    continue
                name_count, org_count = coAuthorOrg_num(
                    pubs_raw[pid1]['authors'],
                    pubs_raw[pid2]['authors']
                )
                count = name_count + org_count
                if count != 0:
                    relation_matrix[j, k] = count
                    relation_matrix[k, j] = count

        np.save(cfg.VAL_SIMI_RELATION_FOLDER + name +
                '.npy', np.array(relation_matrix))
        print(relation_matrix.shape)
        print("calculate " + name + " done")
        # generate content
        # wf_content = open(join(graph_dir, '{}_pubs_content.txt'.format(name)), 'w')
    return


if __name__ == "__main__":
    pass
    # ---------generateGraph test -------------------
    # generateGraph(mode='val')
    # ---------generateCandidateSets test------------
    # labels_by_name, pubs_by_name = generateCandidateSets()
    # print(labels_by_name)
    # print(type(pubs_by_name))

    # ---------generateCandidateSets test------------
    # pubs_by_name = generateCandidateSetsTest()
    # print(pubs_by_name)

    # ---------generateRawFeatrues test------------

    # generateRawFeatrues(mode='val')

    # ---------generateCorpus test------------
    # generateCorpus()

    # ---------Cal_Simalarity_byAuthor test------------

    # Cal_Simalarity_byAuthor_labeled(cfg.TRAIN_PUB_FEATURES_PATH, cfg.TRAIN_AUTHOR_PATH, cfg.SIMI_SENMATIC_FOLDER)
    # Cal_Simalarity_byAuthor_unlabeled(cfg.VAL_PUB_FEATURES_PATH, cfg.VAL_AUTHOR_PATH, cfg.VAL_SIMI_SENMATIC_FOLDER)

    # ---------generate_wordembedding test------------
    # generate_wordembedding()

    # ---------Anything else test------------
    # features = load_pub_features(rfpath=cfg.VAL_PUB_FEATURES_PATH)
    # print(features['srDOamxh'])

    # ---------------- generateRelationFeatures ----------
    generateRelationFeatures(mode='val')
