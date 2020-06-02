#
# TODO: load the file and convert to torch tensors

# ---------------load authors.json to generate ground truth and all papaers--------------
from utils import load_json
import config as cfg


def generateCandidateSets(mode):
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


if __name__ == "__main__":
    pass
    # ---------generateCandidateSets test------------
    # labels_by_name, pubs_by_name,  = generateCandidateSets('train')
    # print(labels_by_name)
    # print(pubs_by_name)
