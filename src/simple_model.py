
# TODO: implement a simple clustering model

from utils import *
import config as cfg
from sklearn.cluster import DBSCAN
import numpy as np

class PaperCluster():
    
    def __init__(self):
        self.res = {}
        self.dbscan_model = None

    def dbscan(self, author_path, simi_folder, eps, min_samples):
        self.dbscan_model = DBSCAN(eps=eps, min_samples = min_samples,metric ="precomputed")
        paper_by_author_labeled = load_json(author_path)
        self.paper_by_author = dict()
        for author, labeled_paper in paper_by_author_labeled.items():
            self.paper_by_author[author] = []
            for label, papers in labeled_paper.items():
                self.paper_by_author[author] += papers
        for author, papers in self.paper_by_author.items():
            l = len(papers)
            simi = np.load(simi_folder + author + '.npy').reshape(l, l)
            simi = normalization(-1*simi)
            self.res[author] = self.dbscan_model.fit_predict(simi)
            # print(self.res[author].tolist())
            return

    def tranfer2file(self, savepath):
        resdict = {}
        for author, papers in self.paper_by_author.items():
            l = np.max((self.res[author])) + 1
            resdict[author] = []
            for i in range(l):
                resdict[author].append([])
            for i in range(len(papers)):
                resdict[author][self.res[author][i]].append(papers[i])
            break
        dump_json(resdict, "../dataset", "result_test.json",indent =4)

if __name__ == "__main__":
    papercluster = PaperCluster()
    papercluster.dbscan(cfg.TRAIN_AUTHOR_PATH, cfg.SIMI_SENMATIC_FOLDER, eps=0.15, min_samples=4)
    papercluster.tranfer2file(cfg.TRAIN_RESULT_PATH)