from utils import load_json
import config as cfg
import re

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

        print(n, name, len(pubs))
        pubs_by_name[name] = pubs

    return pubs_by_name

# --------------load pubs.json and generate raw features---------------


def generateRawFeatrues(mode='train'):
    '''
    mode: 'train' or 'val' or 'test'
    '''
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

    sementic_features = {}
    relation_features = {}
    for i, paper in enumerate(pubs_raw):
        title = paper['title']
        title = re.sub(r'\s{2,}', ' ', re.sub(r, ' ', title)).strip()
        abstract = paper['abstract']
        abstract = re.sub(r'\s{2,}', ' ', re.sub(r, ' ', abstract)).strip()
        key_words = paper.get("keywords")
        keyword = ','.join(key_words)
        venue_name = paper.get('venue')
        authors = paper.get('authors')

    # TODO: title/key_words/venue_name embedding input form? -> sematic features
    # TODO: authors: name/org graph construct input form? -> relation features
    return sementic_features, relation_features


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
