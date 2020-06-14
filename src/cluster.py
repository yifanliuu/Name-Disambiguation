from sklearn.cluster import DBSCAN
from utils import *
import config as cfg
import numpy as np

def dbSCAN_by_Author(author_path, simi_folder, eps=0.4, minsamples=4):
    paper_by_author_labeled = load_json(author_path)
    paper_by_author = dict()
    for author, labeled_paper in paper_by_author_labeled.items():
        paper_by_author[author] = []
        for label, papers in labeled_paper.items():
            paper_by_author[author] += papers
    for author, papers in paper_by_author.items():
        l = len(papers)
        simi = np.load(simi_folder + author + '.npy').reshape(l,l)
        pre = DBSCAN(eps=eps, min_samples=minsamples, metric="precomputed").fit_predict(simi)
        print(pre)
        exit(0)


if __name__ == "__main__":
    dbSCAN_by_Author(cfg.TRAIN_AUTHOR_PATH, cfg.SIMI_SENMATIC_FOLDER)