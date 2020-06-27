# from gensim.models.keyedvectors import KeyedVectors
import config as cfg
import numpy as np
import string
import re
import logging
# from models import Word2Vec
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


def save_relation(paper_list, pubs_raw, name):
    # trained by all text in the datasets.

    r = '[!“”"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~—～’]+'
    stopword = ['at', 'based', 'in', 'of', 'for', 'on', 'and', 'to',
                'an', 'using', 'with', 'the', 'by', 'we', 'be', 'is', 'are', 'can']
    stopword1 = ['university', 'univ', 'china', 'department', 'dept', 'laboratory', 'lab', 'school', 'al', 'et',
                 'institute', 'inst', 'college', 'chinese', 'beijing', 'journal', 'science', 'international']

    f1 = open('../dataset/features/'+name +
              '_paper_author.txt', 'w', encoding='utf-8')
    f2 = open('../dataset/features/'+name +
              '_paper_conf.txt', 'w', encoding='utf-8')
    f3 = open('../dataset/features/'+name +
              '_paper_word.txt', 'w', encoding='utf-8')
    f4 = open('../dataset/features/'+name +
              '_paper_org.txt', 'w', encoding='utf-8')

    taken = name.split("_")
    name = taken[0] + taken[1]
    name_reverse = taken[1] + taken[0]
    if len(taken) > 2:
        name = taken[0] + taken[1] + taken[2]
        name_reverse = taken[2] + taken[0] + taken[1]

    authorname_dict = {}

    tcp = set()
    for i, pid in enumerate(paper_list):
        pub = pubs_raw[pid]

        # save authors
        org = ""
        for author in pub["authors"]:
            authorname = re.sub(r, '', author["name"]).lower()
            taken = authorname.split(" ")
            if len(taken) == 2:  # 检测目前作者名是否在作者词典中
                authorname = taken[0] + taken[1]
                authorname_reverse = taken[1] + taken[0]

                if authorname not in authorname_dict:
                    if authorname_reverse not in authorname_dict:
                        authorname_dict[authorname] = 1
                    else:
                        authorname = authorname_reverse
            else:
                authorname = authorname.replace(" ", "")

            if authorname != name and authorname != name_reverse:
                f1.write(pid + '\t' + authorname + '\n')

            else:
                if "org" in author:
                    org = author["org"]

        # save org 待消歧作者的机构名
        pstr = org.strip()
        pstr = pstr.lower()  # 小写
        pstr = re.sub(r, ' ', pstr)  # 去除符号
        pstr = re.sub(r'\s{2,}', ' ', pstr).strip()  # 去除多余空格
        pstr = pstr.split(' ')
        pstr = [word for word in pstr if len(word) > 1]
        pstr = [word for word in pstr if word not in stopword1]
        pstr = [word for word in pstr if word not in stopword]
        pstr = set(pstr)
        for word in pstr:
            f4.write(pid + '\t' + word + '\n')

        # save venue
        pstr = pub["venue"].strip()
        pstr = pstr.lower()
        pstr = re.sub(r, ' ', pstr)
        pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
        pstr = pstr.split(' ')
        pstr = [word for word in pstr if len(word) > 1]
        pstr = [word for word in pstr if word not in stopword1]
        pstr = [word for word in pstr if word not in stopword]
        for word in pstr:
            f2.write(pid + '\t' + word + '\n')
        if len(pstr) == 0:
            f2.write(pid + '\t' + 'null' + '\n')

        # save text
        pstr = ""
        keyword = ""
        if "keywords" in pub:
            for word in pub["keywords"]:
                keyword = keyword+word+" "
        pstr = pstr + pub["title"]
        pstr = pstr.strip()
        pstr = pstr.lower()
        pstr = re.sub(r, ' ', pstr)
        pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
        pstr = pstr.split(' ')
        pstr = [word for word in pstr if len(word) > 1]
        pstr = [word for word in pstr if word not in stopword]
        for word in pstr:
            f3.write(pid + '\t' + word + '\n')

        # save all words' embedding
        pstr = keyword + " " + pub["title"] + " " + pub["venue"] + " " + org
        if "year" in pub:
            pstr = pstr + " " + str(pub["year"])
        pstr = pstr.strip()
        pstr = pstr.lower()
        pstr = re.sub(r, ' ', pstr)
        pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
        pstr = pstr.split(' ')
        pstr = [word for word in pstr if len(word) > 2]
        pstr = [word for word in pstr if word not in stopword]
        pstr = [word for word in pstr if word not in stopword1]

    # the paper index that lack text information
    dump_data(tcp, '../dataset/features/'+name+'_tcp.pkl')
    f1.close()
    f2.close()
    f3.close()
    f4.close()


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

def cut_semetic_features_by_author(features_path=cfg.VAL_PUB_FEATURES_PATH, author_path=cfg.VAL_AUTHOR_PATH, save_folder=cfg.VAL_SEMATIC_FEATURES_PATH):
    features = load_pub_features(features_path)
    paper_by_author = load_json(author_path)
    # paper_by_author_labeled = load_json(author_path)
    # paper_by_author = dict()
    # for author, labeled_paper in paper_by_author_labeled.items():
    #     paper_by_author[author] = []
    #     for label, papers in labeled_paper.items():
    #         paper_by_author[author] += papers
    for author, papers in paper_by_author.items():
        feature_byAuthor = np.zeros([len(papers), 100])
        for i in range(len(papers)):
            feature_byAuthor[i] = features[papers[i]]
        np.save(save_folder+author + '.npy', feature_byAuthor)


def Cal_Simalarity_byAuthor_unlabeled(features_path, author_path, save_folder):
    gtime_st = time.time()
    # features = load_pub_features(features_path)
    paper_by_author = load_json(author_path)

    simi_matrix = []
    # p = multiprocessing.Pool(4)
    # print(len(paper_by_author))
    for author, papers in paper_by_author.items():
        # print(len(papers))
        time_start = time.time()
        l = len(papers)
        features = np.load(features_path + author + '_gen.npy')
        res = pairwise_distances(features, metric='cosine', n_jobs=-1)
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
    print("TOTAL USING TIME(s): " + str(gtime_end-gtime_st))


def generate_wordembedding():
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    word_embedding = Word2Vec()
    word_embedding.train()
    word_embedding.save()

# ------------------- graph generation --------------------


def generateValGraph(name, pid2idx_by_name):
    """
    generate graph use co-authors and organization
    """
    n_relation = 3

    # 0: author 1: org 2: conf
    graph = {}
    for i in range(n_relation):
        graph[i] = {}

    # dict
    paper_author = dict()
    author_paper = dict()
    paper_org = dict()
    org_paper = dict()
    paper_conf = dict()
    conf_paper = dict()
    temp = set()

    with open('../dataset/features/'+name +
              '_paper_author.txt', 'r', encoding='utf-8') as f1:
        for line in f1:
            temp.add(line)
        for line in temp:
            toks = line.strip().split("\t")
            if len(toks) == 2:
                p, a = toks[0], toks[1]
                p = pid2idx_by_name[p]
                if p not in graph:
                    graph[p] = {}
                if p not in paper_author:
                    paper_author[p] = []
                    graph[p][0] = []
                paper_author[p].append(a)
                if a not in author_paper:
                    author_paper[a] = []
                author_paper[a].append(p)
        temp.clear()

    for i, p in enumerate(paper_author):
        for a in paper_author[p]:
            graph[p][0] += author_paper[a]

    with open('../dataset/features/'+name +
              '_paper_org.txt', 'r', encoding='utf-8') as f2:
        for line in f2:
            temp.add(line)
        for line in temp:
            toks = line.strip().split("\t")
            if len(toks) == 2:
                p, a = toks[0], toks[1]
                p = pid2idx_by_name[p]
                if p not in graph:
                    graph[p] = {}
                if p not in paper_org:
                    paper_org[p] = []
                    graph[p][1] = []
                paper_org[p].append(a)
                if a not in org_paper:
                    org_paper[a] = []
                org_paper[a].append(p)
        temp.clear()

    for i, p in enumerate(paper_org):
        for a in paper_org[p]:
            graph[p][1] += org_paper[a]

    with open('../dataset/features/'+name +
              '_paper_conf.txt', 'r', encoding='utf-8') as f3:
        for line in f3:
            temp.add(line)
        for line in temp:
            toks = line.strip().split("\t")
            if len(toks) == 2:
                p, a = toks[0], toks[1]
                p = pid2idx_by_name[p]
                if p not in graph:
                    graph[p] = {}
                if p not in paper_conf:
                    paper_conf[p] = []
                    graph[p][2] = []
                paper_conf[p].append(a)
                if a not in conf_paper:
                    conf_paper[a] = []
                conf_paper[a].append(p)
        temp.clear()

    for i, p in enumerate(paper_conf):
        for a in paper_conf[p]:
            graph[p][2] += conf_paper[a]

    dump_json(graph, cfg.VAL_GRAPH_PATH+name+'.json')
    return len(pid2idx_by_name), n_relation, graph


def generateRelationFeatures():
    """
    generate graph use co-authors and organization
    """
    return


if __name__ == "__main__":
    pass
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
    Cal_Simalarity_byAuthor_unlabeled(
       cfg.VAL_RELATION_FEATURES_PATH_GEN, cfg.VAL_AUTHOR_PATH, cfg.VAL_SIMI_RELATION_FOLDER)

    # ---------generate_wordembedding test------------
    # generate_wordembedding()

    # ---------Anything else test------------
    # features = load_pub_features(rfpath=cfg.VAL_PUB_FEATURES_PATH)
    # print(features['srDOamxh'])

    # ---------------- generateRelationFeatures ----------
    # generateRelationFeatures(mode='val')

    # ---------------- save relation test ----------------
    # name_pubs = load_json(cfg.VAL_AUTHOR_PATH)
    # pubs_raw = load_json(cfg.VAL_PUB_PATH)
    # for i, name in enumerate(name_pubs):
    #    save_relation(name_pubs[name], pubs_raw=pubs_raw, name=name)

    # ---------generateGraph test -------------------
    #
    # pid2idx_by_name, names = pid2idxMapping()
    # # generate graph by name
    # for i, name in enumerate(names):
    #     n_node, n_relation, graph = generateValGraph(
    #         name,
    #         pid2idx_by_name[name]
    #     )
    #     print(name, n_node, n_relation)

    # cut_semetic_features_by_author()