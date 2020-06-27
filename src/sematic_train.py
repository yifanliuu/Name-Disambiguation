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
        self.layer1 = nn.Sequential(nn.Linear(100, 128), nn.ReLU(True))
        self.dropout = nn.Dropout(0.5)
        self.layer2 = nn.Linear(128, 64) 
    def forward(self, x):
        x = self.layer1(x)
        # x = self.dropout(x)
        x = self.layer2(x)
        return x

class SematicTrainer():
    def __init__(self, USE_CUDA=True, max_iter=200, lr=1e-3):
        self.model = SematicNN()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.iterations = max_iter
        self.USE_CUDA = USE_CUDA
        # self.pid = list(features.keys())
        # print('load pids done')
        # features_list = list(features.values())
        # shape = [len(features_list), 100]
        # self.features = np.zeros(shape=shape)
        # for i in range(len(features_list)):
        #     self.features[i, :] = features_list[i]
        self.features = {}
        files = os.listdir(cfg.TRAIN_SEMATIC_FEATURES_PATH)
        for f in files:
            self.features[f[:-4]] = torch.from_numpy(np.load(cfg.TRAIN_SEMATIC_FEATURES_PATH + f))
        # self.features = torch.from_numpy(self.features).double()
        print('load features done')

    def loss_function(self, result, author):
        pid = torch.index_select(result, 0, self.pid[author])
        pid_pos = torch.index_select(result, 0, self.pid_pos[author])
        pid_neg = torch.index_select(result, 0, self.pid_neg[author])
        triplet_loss = F.triplet_margin_loss(pid, pid_pos, pid_neg)
        return triplet_loss

    def gen_neg_pid(self, not_in_pids, all_pids):
        while True:
            idx = random.randint(0, len(all_pids)-1)
            pid = all_pids[idx]
            if pid not in not_in_pids:
                return pid

    def generateTriplets(self, author_path=cfg.TRAIN_AUTHOR_PATH):
        n_sample_triplets = 0
        # triplets = np.zeros([240000, 3])
        paper_by_author_labeled = load_json(author_path)
        paper_by_author = dict()
        for author, labeled_paper in paper_by_author_labeled.items():
            paper_by_author[author] = []
            for label, papers in labeled_paper.items():
                paper_by_author[author] += papers
        for author, labeled_paper in paper_by_author_labeled.items():
            all_paper = paper_by_author[author]
            triplets = np.zeros([0, 3])
            for aid, papers in labeled_paper.items():
                if(len(papers) == 1):
                    continue
                pids = papers
                cur_n = len(pids)
                random.shuffle(pids)
                for i in range(cur_n):
                    pid  = pids[i]
                    n_samples = min(6, cur_n)
                    idx_pos = random.sample(range(cur_n), n_samples)
                    for i_pos in idx_pos:
                        if i_pos == i:
                            continue
                        if n_sample_triplets % 10000 == 0:
                            print('sampled triplet ids', n_sample_triplets)
                        pid_pos = pids[i_pos]
                        pid_neg = self.gen_neg_pid(pids, paper_by_author[author])
                        # print(pid)
                        # print(np.array([all_paper.index(pid), all_paper.index(pid_pos), all_paper.index(pid_neg)]))
                        triplets = np.concatenate((triplets, np.array([[all_paper.index(pid), all_paper.index(pid_pos), all_paper.index(pid_neg)]])))
                        n_sample_triplets += 1
            np.save(cfg.TRIPLETS_PATH + author + '.npy', triplets)


    def load_triplets(self):
        triplets = {}
        files = os.listdir(cfg.TRIPLETS_PATH)
        for f in files:
            triplets[f[:-4]] = torch.from_numpy(np.load(cfg.TRIPLETS_PATH + f))
        self.pid = {}
        self.pid_pos = {}
        self.pid_neg = {}
        for key, value in triplets.items():
            self.pid[key] = value[:,0].long()
            self.pid_pos[key] = value[:, 1].long()
            self.pid_neg[key] = value[:, 2].long()

        

    def save_model(self):
        torch.save(self.model.state_dict(), cfg.SEMATIC_MODEL_PATH)

    def load_model(self):
        self.model.load_state_dict(torch.load(cfg.SEMATIC_MODEL_PATH))

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.model = self.model.double()
        t = time.time()
        for epoch in range(200):
            total_loss = 0
            for author, features in self.features.items():
                self.model.train()
                output = self.model(features)
                loss = self.loss_function(output, author)
                total_loss += loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print("Epoch: {}, train loss={:.5f}, time cost: {:.5f}".format(epoch, loss, time.time() - t))
                
    def generate(self, datapath=cfg.VAL_SEMATIC_FEATURES_PATH, savepath=cfg.VAL_SEMATIC_FEATURES_PATH_64):
        files = os.listdir(datapath)
        for f in files:
            arr = torch.from_numpy(np.load(datapath + f)).float()
            arr = self.model(arr)
            np.save(savepath + f, arr.detach().numpy())

if __name__ == "__main__":
    trainer = SematicTrainer()

    # --- GET TRIPLETS ---
    # trainer.generateTriplets()
    
    # --- TRAIN ---
    # trainer.load_triplets()
    # trainer.train()
    # trainer.save_model()

    # --- TEST ---
    trainer.load_model()
    trainer.generate()