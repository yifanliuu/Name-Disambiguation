from models import Word2Vec, AutoEncoder
import torch
import torch.nn.functional as F
import time
import scipy.sparse as sp
from utils import *


class AutoEncoderTrainer():
    def __init__(self, USE_CUDA=True):
        self.model = AutoEncoder()
        self.optimizer = torch.optim.Adagrad
        self.iterations = 0
        self.USE_CUDA = USE_CUDA

    def run(self, Y, candidate_set_graph, epochs=1):
        for i in range(1, epochs + 1):
            self.train(Y, candidate_set_graph)

    def loss_function(self, preds, labels, mu, logvar, n_nodes, norm, pos_weight):
        cost = norm * \
            F.binary_cross_entropy_with_logits(
                preds, labels, pos_weight=pos_weight)

        # see VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 / n_nodes * torch.mean(torch.sum(
            1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
        return cost + KLD

    def train(self, Y, normed_A):

        for i in range(1, self.iterations + 1):
            pos_weight = float(
                adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
            norm = adj.shape[0] * adj.shape[0] / \
                float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

            t = time.time()
            self.model.train()
            self.optimizer.zero_grad()
            recovered, mu, logvar = self.model(Y, normed_A)
            loss = self.loss_function(
                preds=recovered,
                labels=adj_label,
                mu=mu,
                logvar=logvar,
                n_nodes=n_nodes,
                norm=norm,
                pos_weight=pos_weight
            )
            loss.backward()
            self.optimizer.zero_grad()
            self.optimizer.step(loss)
        self.iterations += i


if __name__ == "__main__":
    word2vec = Word2Vec()
    # word2vec.train()
    # word2vec.save()
    # how to get Y
    adj, features = load_data(args.dataset_str)
    raw_feat = word2vec.load()
    feat_dict = generate_embeded_features(raw_feat)

    adj_dict = generate_graph()

    adj_norm = preprocess_graph(adj)
    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = torch.FloatTensor(adj_label.toarray())

    trainer = AutoEncoderTrainer()
