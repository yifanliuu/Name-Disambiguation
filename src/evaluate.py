import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, normalized_mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from utils import *
from sklearn import manifold
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN


class Aminer_evaluation():
    def __init__(self):

        # load author label
        #id - label
        self.paper_label = {}
        self.sample_num = 0
        with open('../dataset/labels/evaluate/paper_label.dat') as infile:
            for line in infile.readlines():
                paper, label = line.strip().split('\t')
                paper = int(paper)
                label = int(label)

                self.paper_label[paper] = label
                self.sample_num += 1

    def evaluate_paper_cluster(self, embedding_matrix):
        embedding_list = embedding_matrix.tolist()
        X = []
        Y = []
        for paper in self.paper_label:
            X.append(embedding_list[paper])
            Y.append(self.paper_label[paper])

        pred_Y = KMeans(20).fit(np.array(X)).predict(X)
        print(pred_Y)
        print(Y)
        #score = normalized_mutual_info_score(np.array(Y), pred_Y)
        precision, recall, f1_score = pairwise_evaluate(np.array(Y), pred_Y)
        '''
        precision = 0
        recall = 0
        f1_score = score
        '''
        return precision, recall, f1_score

    # to visualize each epoch's embedding
    def evaluate_paper_cluster_using_Tsne(self, embedding_matrix):

        embedding_list = embedding_matrix.tolist()
        X = []
        Y = []
        for paper in self.paper_label:
            X.append(embedding_list[paper])
            Y.append(self.paper_label[paper])

        tsne = manifold.TSNE(n_components=2, init='pca',
                             random_state=1, perplexity=5, n_iter=1500)
        X_tsne = tsne.fit_transform(X)

        x_min, x_max = X_tsne.min(0), X_tsne.max(0)
        X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化

        # pred_Y = KMeans(35).fit(np.array(X_tsne)).predict(X_tsne)
        pred_Y = DBSCAN(eps=0.04, min_samples=2).fit_predict(np.array(X_norm))

        #score = normalized_mutual_info_score(np.array(Y), pred_Y)
        precision, recall, f1_score = pairwise_evaluate(np.array(Y), pred_Y)

        plt.figure(figsize=(8, 8))
        for i in range(X_norm.shape[0]):
            plt.text(X_norm[i, 0], X_norm[i, 1], str(Y[i]), color=plt.cm.Set1(Y[i]),
                     fontdict={'weight': 'bold', 'size': 9})
        plt.xticks([])
        plt.yticks([])
        plt.show()

        plt.figure(figsize=(8, 8))
        for i in range(X_norm.shape[0]):
            plt.text(X_norm[i, 0], X_norm[i, 1], str(pred_Y[i]), color=plt.cm.Set1(pred_Y[i]),
                     fontdict={'weight': 'bold', 'size': 9})
        plt.xticks([])
        plt.yticks([])
        plt.show()
        '''
        precision = 0
        recall = 0
        f1_score = score
        '''
        return precision, recall, f1_score


def str_list_to_float(self, str_list):
    return [float(item) for item in str_list]
