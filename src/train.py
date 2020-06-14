from models import Word2Vec, AutoEncoder
import torch
import torch.nn.functional as F
import time
import scipy.sparse as sp
from utils import *
from preprocess import *


class AutoEncoderTrainer():
    def __init__(self, USE_CUDA=True, max_iter=100):
        self.model = AutoEncoder()
        self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr=1e-3)
        self.iterations = max_iter
        self.USE_CUDA = USE_CUDA

    def run(self, Y, candidate_set_graph, n_nodes, epochs=1):
        for i in range(1, epochs + 1):
            self.train(Y, candidate_set_graph, n_nodes)

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

    def train(self, Y, normed_A, labels_A, n_node, norm):
        for i in range(1, self.iterations + 1):
            t = time.time()
            self.model.train()
            self.optimizer.zero_grad()
            recovered, mu, logvar = self.model(Y, normed_A)
            loss = self.loss_function(
                preds=recovered,
                labels=labels_A,
                mu=mu,
                logvar=logvar,
                n_nodes=n_node,
                norm=norm,
                pos_weight=pos_weight
            )
            loss.backward()
            self.optimizer.step()
            print("Epoch: {}, train loss={:.5f}, time cost: {:.5f}".format(
                i, loss, time.time() - t))

    def generate_embedding(self, Y, normed_A):
        mu, _ = self.model.encoder(Y, norm_A)
        return mu


if __name__ == "__main__":
    # word2vec = Word2Vec()
    # word2vec.train()
    # word2vec.save()

    graph = load_json(rfpath=cfg.VAL_GRAPH_PATH)
    feats = load_pub_features(rfpath=cfg.VAL_PUB_FEATURES_PATH)
    author_pubs_raw = load_json(rfpath=cfg.VAL_AUTHOR_PATH)
    relation_features = {}

    for idx, name in enumerate(graph):
        coo_node_list = graph[name]
        n_node = len(author_pubs_raw[name])
        Y = np.empty([n_node, EMBEDDING_SIZE])
        for i, pid in enumerate(author_pubs_raw[name]):
            Y = np.append(Y, np.reshape(
                feats[pid], [1, EMBEDDING_SIZE]), axis=0)

        adj_norm, adj = preprocess_graph(coo_node_list, n_node)
        adj_label = adj + sp.eye(n_node)
        adj_label = torch.FloatTensor(adj_label.toarray())

        pos_weight = float(n_node * n_node - adj.sum()) / adj.sum()
        norm = n_node * n_node / \
            float((n_node * n_node - adj.sum()) * 2)

        trainer = AutoEncoderTrainer()
        trainer.train(
            Y=Y,
            normed_A=adj_norm,
            labels_A=adj_label,
            n_node=n_node,
            norm=norm
        )

        relation_features[name] = trainer.generate_embedding(Y, adj_norm)

    dump_data(relation_features, wfname=cfg.VAL_RELATION_FEATURES_PATH)
