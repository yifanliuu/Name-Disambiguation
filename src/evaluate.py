import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, normalized_mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


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
        score = normalized_mutual_info_score(np.array(Y), pred_Y)

        return score


def str_list_to_float(self, str_list):
    return [float(item) for item in str_list]
