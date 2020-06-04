from utils import *
from gensim.models.keyedvectors import KeyedVectors
import config as cfg
import numpy as np
import string
import re
import logging
from models import Word2Vec


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
        line = title + ' ' + abstract + ' ' + keyword + ' ' + venue_name + ' ' + str(year) + ' '
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
        line = title + ' ' + abstract + ' ' + keyword + ' ' + venue_name + ' ' + str(year) + ' '
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

        print(n, name, len(pubs))
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
        pass
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

        paper_words = title + ' ' + abstract + ' ' + keyword + ' ' + venue_name + ' ' + str(year) + ' ' + orglist
        # print(paper_words)
        paper_words = paper_words.translate(transTab)
        paper_words = re.sub(r'\s{2,}', ' ', re.sub(r, ' ', paper_words)).strip()
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
    # TODO: title/key_words/venue_name embedding input form? -> sematic features
    # TODO: authors: name/org graph construct input form? -> relation features
    
    print('Wrting features to file......')
    save_pub_features(sementic_features)

    return sementic_features


def generate_wordembedding():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    word_embedding = Word2Vec()
    word_embedding.train()
    word_embedding.save()

if __name__ == "__main__":
    pass
    # ---------generateCandidateSets test------------
    # labels_by_name, pubs_by_name = generateCandidateSets()
    # print(labels_by_name)
    # print(pubs_by_name)

    # ---------generateCandidateSets test------------
    # pubs_by_name = generateCandidateSetsTest()
    # print(pubs_by_name)

    # ---------generateRawFeatrues test------------
    generateRawFeatrues()

    # ---------generateCorpus test------------
    # generateCorpus()

    # ---------generate_wordembedding test------------
    # generate_wordembedding()

    # ---------Anything else test------------
    # features = load_pub_features()
    # print(features['cFtStBA6'])
