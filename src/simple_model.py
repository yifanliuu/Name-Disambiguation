
# TODO: implement a simple clustering model

from utils import *
import config as cfg
from sklearn.cluster import DBSCAN
import string

del_str = string.punctuation
replace_str = ' '*len(del_str)
transTab = str.maketrans(del_str, replace_str)

class PaperCluster():

    def __init__(self, paperPath):
        self.res = {}
        self.dbscan_model = None
        papers_raw = load_json(paperPath)
        self.paperInfo = {}
        for key, values in papers_raw.items():
            self.paperInfo[key] = {}
            self.paperInfo[key]['title'] = values['title'].lower().strip().split(' ')
            self.paperInfo[key]['venue'] = values['venue'].lower().strip().split(' ')
            authors = []
            orgs = []
            authors_info = values['authors']
            for author_info in authors_info:
                authors.append(author_info['name'].lower())
                orgs = orgs + author_info['org'].split(';')
            self.paperInfo[key]['authors'] = authors
            self.paperInfo[key]['orgs'] = orgs

    def dbscan(self, author_path, simi_sema_folder, simi_rela_folder, eps, min_samples,data_labeled):
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
            sema_simi = np.load(simi_sema_folder + author + '.npy').reshape(l, l)

            rela_simi = np.load(simi_rela_folder + author + '.npy').reshape(l, l)
            rela_simi[rela_simi==0] = 1 
            rela_simi = 1 / rela_simi
            # print(rela_simi)
            # print(sema_simi)
            # exit(0)
            self.res[author] = self.dbscan_model.fit_predict(sema_simi + rela_simi)
            print(author,  l, np.sum(self.res[author] == -1), np.max(self.res[author]))

    def simple_relasimi(self, author_path, savepath):
        self.paper_by_author = load_json(author_path)

        for author, papers in self.paper_by_author.items():
            l = len(papers)
            simi = np.zeros(l * l).reshape(l, l)
            for i, paperi in enumerate(papers):
                for j, paperj in enumerate(papers):
                    simi[i, j] = self.cal_relasimi(paperi, paperj)
            np.save(savepath + author + '.npy', simi)

    def tranfer2file(self, savepath):
        resdict = {}
        
        print('------------ Recovering LonelyMountain -----------------')
        print('author new_cluster_num average_papers')
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
            resdict[author] = self.recover_lonelymountain(resdict[author], 1.5)
            print(author, len(resdict[author]), len(self.paper_by_author[author])/len(resdict[author]))

        dump_json(resdict, savepath, indent =4)

    def same_count(self, a ,b):
        return len([c for c in a if c in b])

    def tanimoto(self, p,q):
        c = [v for v in p if v in q]
        return float(len(c) / (len(p) + len(q) - len(c)))

    def same_org_count(self, orga, orgb):
        c = 0
        orgb_ = orgb.copy()
        for org in orga:
            if org in orgb_:
                orgb_.remove(org)
                c += 1
        return c

    def cal_relasimi(self, paperA, paperB):
        paperAInfo = self.paperInfo[paperA]
        paperBInfo = self.paperInfo[paperB]
        
        # authros
        a = self.same_count(paperAInfo['authors'], paperBInfo['authors'])

        # orgs
        o = self.same_org_count(paperAInfo['orgs'], paperBInfo['orgs'])

        # print(t, a, v, o)
        # if a == 0:
        #     print(paperAInfo['authors'], paperBInfo['authors'])
        #     exit(0)

        return a + o

    def calsimi_lonelymountain(self, paperA, paperB):
        paperAInfo = self.paperInfo[paperA]
        paperBInfo = self.paperInfo[paperB]
        
        # title
        t = self.same_count(paperAInfo['title'], paperBInfo['title'])

        # authros
        a = self.same_count(paperAInfo['authors'], paperBInfo['authors'])

        # venue
        v = self.tanimoto(paperAInfo['venue'], paperBInfo['venue'])

        # orgs
        o = self.same_org_count(paperAInfo['orgs'], paperBInfo['orgs'])

        # print(t, a, v, o)
        # if a == 0:
        #     print(paperAInfo['authors'], paperBInfo['authors'])
        #     exit(0)

        return t / 3.0 + a * 1.5 + v + o

    def recover_lonelymountain(self, paperclasses, eps):
        c = 0
        if len(paperclasses) == 0:
            paperclusters = []
        else:
            paperclusters = paperclasses[:-1]
        lonelymountains = paperclasses[-1]
        for lonelymountain in lonelymountains:
            dis = np.zeros(len(paperclusters))
            if len(dis) == 0:
                paperclusters.append([lonelymountain])
                continue
            for i, papercluster in enumerate(paperclusters):
                sim = 0
                l = len(papercluster)
                for paper in papercluster:
                    sim += self.calsimi_lonelymountain(lonelymountain, paper)
                dis[i] = sim / l
            # print(dis)
            # exit(0)
            maxdis = np.max(dis)
            if maxdis >= eps:
                idx = np.argwhere(dis == maxdis)
                # print(int(idx))
                # exit(0)
                # try:
                if (len(idx) == 1):
                    paperclusters[int(idx[0])].append(lonelymountain)
                else:
                    paperclusters[int(idx[c%len(idx)])].append(lonelymountain)
                    c += 1
            else:
                paperclusters.append([lonelymountain])
        return paperclusters


if __name__ == "__main__":
    papercluster = PaperCluster(cfg.VAL_PUB_PATH)
    # papercluster.simple_relasimi(cfg.VAL_AUTHOR_PATH,cfg.VAL_SIMI_RELATION_FOLDER)
    papercluster.dbscan(cfg.VAL_AUTHOR_PATH, cfg.VAL_SIMI_SENMATIC_FOLDER,cfg.VAL_SIMI_RELATION_FOLDER, eps=0.4, min_samples=4, data_labeled=0)
    papercluster.tranfer2file(cfg.VAL_RESULT_PATH)