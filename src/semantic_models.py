import math
import config as cfg
import torch
import torch.nn.functional as F
import torch.nn as nn
import time
import scipy.sparse as sp
from utils import *
import random
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE

# FCN consists of two fully connected layers and a dropout layer
class SemanticNN(nn.Module):

    def __init__(self):
        super(SemanticNN, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(100, 128), nn.ReLU(True))
        self.dropout = nn.Dropout(0.5)
        self.layer2 = nn.Linear(128, 64)
    def forward(self, x):
        x = self.layer1(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x

class SemanticTrainer():
    def __init__(self, USE_CUDA=True, max_iter=200, lr=1e-3):
        self.model = SemanticNN()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.iterations = max_iter
        self.USE_CUDA = USE_CUDA
        self.features = {}
        files = os.listdir(cfg.TRAIN_SEMANTIC_FEATURES_PATH)
        for f in files:
            self.features[f[:-4]] = torch.from_numpy(np.load(cfg.TRAIN_SEMANTIC_FEATURES_PATH + f))
        print('load features done')

# Using triplet loss to optimize the parameter
    def loss_function(self, result, author):
        pid = torch.index_select(result, 0, self.pid[author])
        pid_pos = torch.index_select(result, 0, self.pid_pos[author])
        pid_neg = torch.index_select(result, 0, self.pid_neg[author])
        triplet_loss = F.triplet_margin_loss(pid, pid_pos, pid_neg)
        return triplet_loss

# Sample a negative example in triples
    def gen_neg_pid(self, not_in_pids, all_pids):
        while True:
            idx = random.randint(0, len(all_pids)-1)
            pid = all_pids[idx]
            if pid not in not_in_pids:
                return pid

# Sample to generate 6 triples for each paper in the dataset, and save them in file by author name.
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

# load all triplets from files
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
        
# save model parameters
    def save_model(self):
        torch.save(self.model.state_dict(), cfg.SEMANTIC_MODEL_PATH)

# load model parameters
    def load_model(self):
        self.model.load_state_dict(torch.load(cfg.SEMANTIC_MODEL_PATH))

# training function
    def train(self, exp=0):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.model = self.model.double()
        t = time.time()
        MAXEPOCH = 200
        loss_all = np.zeros(MAXEPOCH)
        for epoch in range(MAXEPOCH):
            total_loss = 0
            for author, features in self.features.items():
                self.model.train()
                output = self.model(features)
                loss = self.loss_function(output, author)
                total_loss += loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_all[epoch] = total_loss
            print("Epoch: {}, train loss={:.5f}, time cost: {:.5f}".format(epoch, loss, time.time() - t))
        if exp == 1:
            x = np.arange(0, MAXEPOCH)    
            l3=plt.plot(x,loss_all,'b--',label='loss')
            plt.plot(x,loss_all,'b^-')
            plt.title('Triplet loss change curve in FCN')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()

# Use trained network to transform initial embedding to the final 64-dimensional embedding
    def generate(self, datapath=cfg.TRAIN_SEMANTIC_FEATURES_PATH, savepath=cfg.TRAIN_SEMANTIC_FEATURES_PATH_64):
        files = os.listdir(datapath)
        for f in files:
            arr = torch.from_numpy(np.load(datapath + f)).float()
            arr = self.model(arr)
            np.save(savepath + f, arr.detach().numpy())

# Visualization function for our experiment
    def experiment(self, epoch, datapath=cfg.TRAIN_SEMANTIC_FEATURES_PATH, exp_author='xu_shen'):
        self.model.load_state_dict(torch.load('../model/semantic_model' + str(epoch) + '.pkl'))
        arr = torch.from_numpy(np.load(datapath + exp_author + '.npy')).float()
        arr = self.model(arr).detach().numpy()
        tsne = TSNE(n_components=2, init='pca', random_state=0)
        arr_tsne = tsne.fit_transform(arr)
        # print(arr_tsne)
        np.save('../dataset/exp.npy', arr_tsne)
        authors = load_json(cfg.TRAIN_AUTHOR_PATH)
        exp_author = authors[exp_author]
        target = np.zeros(len(arr), dtype=int)
        i = 0
        j = 0
        for aid, papers in exp_author.items():
            for paper in papers:
                target[i] = j
                i += 1
            j += 1
        if str(epoch) == '1':
            self.plot_embedding_2d(arr_tsne,target, "t-SNE 2D Initial Embedding")
        self.plot_embedding_2d(arr_tsne,target, "t-SNE 2D Trained " + str(epoch) + ' Epoch Embedding')

    def plot_embedding_3d(self, X, target, title=None):
        #坐标缩放到[0,1]区间
        x_min, x_max = np.min(X,axis=0), np.max(X,axis=0)
        X = (X - x_min) / (x_max - x_min)
        # x = X[:, 0]
        # y = X[:, 1]
        # z = X[:, 2]
        #降维后的坐标为（X[i, 0], X[i, 1],X[i,2]），在该位置画出对应的digits
        ax = plt.subplot(111, projection='3d')
        for i in range(X.shape[0]):
            ax.text(X[i, 0], X[i, 1], X[i,2],str(target[i]),
            color=plt.cm.Set1(target[i] / 10.),
            fontdict={'weight': 'bold', 'size': 9})
        # ax.scatter(x, y, z)
        if title is not None:
            plt.title(title)
        plt.show()

    def plot_embedding_2d(self, X, target, title=None):
        #坐标缩放到[0,1]区间
        x_min, x_max = np.min(X,axis=0), np.max(X,axis=0)
        X = (X - x_min) / (x_max - x_min)
        # x = X[:, 0]
        # y = X[:, 1]
        # z = X[:, 2]
        #降维后的坐标为（X[i, 0], X[i, 1],X[i,2]），在该位置画出对应的digits
        ax = plt.subplot()
        for i in range(X.shape[0]):
            ax.text(X[i, 0], X[i, 1], str(target[i]),
            color=plt.cm.Set1(target[i] / 10.),
            fontdict={'weight': 'bold', 'size': 9})
        # ax.scatter(x, y, z)
        if title is not None:
            plt.title(title)
        plt.show()

if __name__ == "__main__":
    trainer = SemanticTrainer()
    pass
    # --- GET TRIPLETS ---
    # trainer.generateTriplets()
    
    # --- TRAIN ---
    # trainer.load_triplets()
    # trainer.train()
    # trainer.save_model()

    # --- TEST ---
    # trainer.load_model()
    # trainer.generate()

    # --- EXPERIMENT_TRAIN ---
    # trainer.load_triplets()
    # trainer.train(exp=1)
    # trainer.save_model()

    # --- EXPERIMENT_TEST ---
    # trainer.experiment('200')
