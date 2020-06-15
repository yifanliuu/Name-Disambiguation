
# TODO: implement a simple clustering model

from utils import *
import config as cfg
from sklearn.cluster import DBSCAN

class PaperCluster():
    
    def __init__(self):
        self.res = {}
        self.dbscan_model = None

    def dbscan(self, author_path, simi_folder, eps, min_samples,data_labeled):
        self.dbscan_model = DBSCAN(eps=eps, min_samples = min_samples,metric ="precomputed")
        self.paper_by_author = dict()

        if data_labeled:
            paper_by_author_labeled = load_json(author_path)
            for author, labeled_paper in paper_by_author_labeled.items():
                self.paper_by_author[author] = []
                for label, papers in labeled_paper.items():
                    self.paper_by_author[author] += papers
        else:
            self.paper_by_author = load_json(author_path)
        
        print("Auhtor PaperNum LonelyMountain ClusterNum")
        for author, papers in self.paper_by_author.items():
            l = len(papers)
            simi = np.load(simi_folder + author + '.npy').reshape(l, l)
            self.res[author] = self.dbscan_model.fit_predict(simi)
            print(author,  l, np.sum(self.res[author] == -1), np.max(self.res[author]))

    def tranfer2file(self, savepath):
        resdict = {}
        for author, papers in self.paper_by_author.items():
            # print(set(self.res[author].tolist()))
            # exit(0)

            # TODO: l is temporary +2, we simply classify outliers into one group
            l = np.max((self.res[author])) + 2
            resdict[author] = []
            for i in range(l):
                resdict[author].append([])
            for i in range(len(papers)):
                resdict[author][self.res[author][i]].append(papers[i])
        dump_json(resdict, savepath, indent =4)

if __name__ == "__main__":
    papercluster = PaperCluster()
    papercluster.dbscan(cfg.VAL_AUTHOR_PATH, cfg.VAL_SIMI_SENMATIC_FOLDER, eps=0.15, min_samples=4, data_labeled=0)
    papercluster.tranfer2file(cfg.VAL_RESULT_PATH)