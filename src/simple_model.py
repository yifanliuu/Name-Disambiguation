
# TODO: implement a simple clustering model

from utils import *
import config as cfg
from sklearn.cluster import DBSCAN
import numpy as np

class PaperCluster():
    
    def __init__(self):
        self.res = []
        self.dbscan_model = None

    def dbscan(self, author_path, simi_folder, eps, min_samples):
        self.dbscan_model = DBSCAN(eps=eps, min_samples = min_samples,metric ="precomputed")
        paper_by_author_labeled = load_json(author_path)
        paper_by_author = dict()
        for author, labeled_paper in paper_by_author_labeled.items():
            paper_by_author[author] = []
            for label, papers in labeled_paper.items():
                paper_by_author[author] += papers
        for author, papers in paper_by_author.items():
            l = len(papers)
            simi = np.load(simi_folder + author + '.npy').reshape(l, l)
            self.res.append(self.dbscan_model.fit_predict(simi))


if __name__ == "__main__":
    papercluster = PaperCluster()
    papercluster.dbscan(cfg.TRAIN_AUTHOR_PATH, cfg.SIMI_SENMATIC_FOLDER, eps=0.4, min_samples=4)