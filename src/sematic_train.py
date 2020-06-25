import math
import config as cfg
import torch
import torch.nn.functional as F
import torch.nn as nn
import time
import scipy.sparse as sp
from utils import *
import random
# from preprocess import *

class SematicNN(nn.Module):

    def __init__(self):
        super(SematicNN, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(100, 128), nn.BatchNorm1d(128), nn.ReLU(True))
        self.dropout = nn.Dropout(0.5)
        self.layer2 = nn.Linear(128, 64) 
    def forward(self, x):
        x = self.layer1(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x

class SematicTrainer():
    def __init__(self, features, USE_CUDA=True, max_iter=200, lr=1e-3):
        self.model = SematicNN()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.iterations = max_iter
        self.USE_CUDA = USE_CUDA
        self.pid = features.keys()
        print('load pids done')
        features_list = list(features.values())
        shape = [len(features_list), 100]
        self.features = np.zeros(shape=shape)
        for i in range(len(features_list)):
            self.features[i, :] = features_list[i]
        self.features = torch.from_numpy(self.features)
        print('load features done')

    def loss_function(self, result, triplets):
        triplet_loss = torch.zeros(1)
        for triplet in triplets:
            triplet_loss += 0
        return triplet_loss

    def train(self):
        pass


    def gen_neg_pid(self, not_in_pids, all_pids):
        while True:
            idx = random.randint(0, len(all_pids)-1)
            pid = all_pids[idx]
            if pid not in not_in_pids:
                return pid

    def generateTriplets(self, author_path=cfg.TRAIN_AUTHOR_PATH):
        n_sample_triplets = 0
        triplets = []
        paper_by_author_labeled = load_json(author_path)
        paper_by_author = dict()
        for author, labeled_paper in paper_by_author_labeled.items():
            paper_by_author[author] = []
            for label, papers in labeled_paper.items():
                paper_by_author[author] += papers
        for author, labeled_paper in paper_by_author_labeled.items():
            for aid, papers in labeled_paper.items():
                if(len(papers) == 1):
                    continue
                pids = papers
                cur_n = len(pids)
                random.shuffle(pids)
                for i in range(cur_n):
                    pid  = pids[i]
                    n_samples = min(2, cur_n)
                    idx_pos = random.sample(range(cur_n), n_samples)
                    for i_pos in idx_pos:
                        if i_pos == i:
                            continue
                        if n_sample_triplets % 10000 == 0:
                            print('sampled triplet ids', n_sample_triplets)
                        pid_pos = pids[i_pos]
                        pid_neg = self.gen_neg_pid(pids, paper_by_author[author])
                        triplets.append([pid, pid_pos, pid_neg])
                        n_sample_triplets += 1
                        if n_sample_triplets >= 250000:
                            save_triplets(triplets, cfg.TRIPLETS_PATH)
                            return
        save_triplets(triplets, cfg.TRIPLETS_PATH)

    def load_triplets(self):
        self.triplets = load_triplets(cfg.TRIPLETS_PATH)

if __name__ == "__main__":
    trainer = SematicTrainer(load_pub_features(cfg.TRAIN_PUB_FEATURES_PATH))
    # trainer.generateTriplets()
    trainer.load_triplets()
    print(trainer.triplets[1])