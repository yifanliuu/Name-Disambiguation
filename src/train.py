from models import Word2Vec, Discriminator, Generator
from evaluate import Aminer_evaluation
'''
import torch
import torch.nn.functional as F
import torch.nn as nn
'''
import time
import scipy.sparse as sp
from utils import *
from preprocess import *

import os
import tensorflow as tf
import config as cfg
import numpy as np

# NOTE: deprecated Graph AutoEncoder
'''
class AutoEncoderTrainer():
    def __init__(self, USE_CUDA=True, max_iter=200, lr=1e-3):
        self.model = AutoEncoder()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.iterations = max_iter
        self.USE_CUDA = USE_CUDA

    def loss_function(self, preds, labels, mu, logvar, n_nodes, norm, pos_weight):
        # print(preds, labels)
        func = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        cost = norm * func(preds, labels)

        # see VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

        KLD = -0.5 / n_nodes * torch.mean(torch.sum(
            1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
        # print(KLD)
        return cost + KLD

    def train(self, Y, normed_A, labels_A, n_node, norm, pos_weight):
        for i in range(1, self.iterations + 1):
            t = time.time()
            self.model.train()
            self.optimizer.zero_grad()
            recovered, mu, logvar = self.model(Y, normed_A)
            # print(recovered, mu, logvar)
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
        mu, _ = self.model.encoder(Y, normed_A)
        return mu
'''


class Trainer():
    def __init__(self, name, n_node, author_file=cfg.VAL_AUTHOR_PATH, feature_file=cfg.VAL_PUB_FEATURES_PATH):
        t = time.time()
        print('reading graph...')
        self.name = name
        self.n_relation = 3
        # n_node is #papers of one name
        self.n_node = n_node
        self.graph = load_json(cfg.VAL_GRAPH_PATH+name+'.json')
        self.node_list = list(self.graph.keys())  # range(0, self.n_node)
        print('[%.2f] reading graph finished. #node = %d #relation = %d' %
              (time.time() - t, self.n_node, self.n_relation))

        t = time.time()
        print('read initial embeddings...')
        self.node_embed_init_d = read_embeddings(name=self.name,
                                                 n_node=self.n_node,
                                                 rfpath=author_file,
                                                 filename=feature_file,
                                                 n_embed=cfg.n_emb)
        self.node_embed_init_g = self.node_embed_init_d
        print('[%.2f] read initial embeddings finished.' % (time.time() - t))

        print('build GAN model...')
        self.discriminator = None
        self.generator = None
        self.build_generator()
        self.build_discriminator()

        self.latest_checkpoint = tf.train.latest_checkpoint(cfg.model_log)
        self.saver = tf.train.Saver()

        self.config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(allow_growth=True))

        self.aminer_evaluation = Aminer_evaluation()
        self.init_op = tf.group(
            tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess = tf.Session(config=self.config)
        self.sess.run(self.init_op)
        self.writer = tf.summary.FileWriter(cfg.model_log, self.sess.graph)
        self.summary_op = tf.summary.merge_all()

    def save_model(self):
        self.saver.save(self.sess, cfg.model_log+self.name+'.ckpt')

    def restore_model(self):
        self.saver.restore(self.sess, self.latest_checkpoint)

    def build_generator(self):
        # with tf.variable_scope("generator"):
        self.generator = Generator(n_node=self.n_node,
                                   n_relation=self.n_relation,
                                   node_emd_init=self.node_embed_init_g,
                                   relation_emd_init=None)

    def build_discriminator(self):
        # with tf.variable_scope("discriminator"):
        self.discriminator = Discriminator(n_node=self.n_node,
                                           n_relation=self.n_relation,
                                           node_emd_init=self.node_embed_init_d,
                                           relation_emd_init=None)

    def train(self, n_epoch=cfg.n_epoch):
        print('start traning...')
        for epoch in range(1, n_epoch + 1):
            print('epoch %d' % epoch)
            t = time.time()

            one_epoch_gen_loss = 0.0
            one_epoch_dis_loss = 0.0
            one_epoch_batch_num = 0.0

            # D-step
            # t1 = time.time()
            for d_epoch in range(cfg.d_epoch):
                np.random.shuffle(self.node_list)
                one_epoch_dis_loss = 0.0
                one_epoch_pos_loss = 0.0
                one_epoch_neg_loss_1 = 0.0
                one_epoch_neg_loss_2 = 0.0

                range_n = int(np.floor(len(self.node_list) / cfg.batch_size))
                for index in range(range_n):
                    # t1 = time.time()
                    pos_node_ids, pos_relation_ids, pos_node_neighbor_ids, neg_node_ids_1, neg_relation_ids_1, neg_node_neighbor_ids_1, neg_node_ids_2, neg_relation_ids_2, node_fake_neighbor_embedding = self.prepare_data_for_d(
                        index)
                    # t2 = time.time()
                    _, dis_loss, pos_loss, neg_loss_1, neg_loss_2 = self.sess.run([self.discriminator.d_updates, self.discriminator.loss, self.discriminator.pos_loss, self.discriminator.neg_loss_1, self.discriminator.neg_loss_2],
                                                                                  feed_dict={self.discriminator.pos_node_id: np.array(pos_node_ids),
                                                                                             self.discriminator.pos_relation_id: np.array(pos_relation_ids),
                                                                                             self.discriminator.pos_node_neighbor_id: np.array(pos_node_neighbor_ids),
                                                                                             self.discriminator.neg_node_id_1: np.array(neg_node_ids_1),
                                                                                             self.discriminator.neg_relation_id_1: np.array(neg_relation_ids_1),
                                                                                             self.discriminator.neg_node_neighbor_id_1: np.array(neg_node_neighbor_ids_1),
                                                                                             self.discriminator.neg_node_id_2: np.array(neg_node_ids_2),
                                                                                             self.discriminator.neg_relation_id_2: np.array(neg_relation_ids_2),
                                                                                             self.discriminator.node_fake_neighbor_embedding: np.array(node_fake_neighbor_embedding)})

                    one_epoch_dis_loss += dis_loss
                    one_epoch_pos_loss += pos_loss
                    one_epoch_neg_loss_1 += neg_loss_1
                    one_epoch_neg_loss_2 += neg_loss_2

            # G-step

            for g_epoch in range(cfg.g_epoch):
                np.random.shuffle(self.node_list)
                one_epoch_gen_loss = 0.0

                range_n = int(np.floor(len(self.node_list) / cfg.batch_size))
                for index in range(range_n):

                    gen_node_ids, gen_relation_ids, gen_noise_embedding, gen_dis_node_embedding, gen_dis_relation_embedding = self.prepare_data_for_g(
                        index)
                    t2 = time.time()

                    _, gen_loss = self.sess.run([self.generator.g_updates, self.generator.loss],
                                                feed_dict={self.generator.node_id:  np.array(gen_node_ids),
                                                           self.generator.relation_id:  np.array(gen_relation_ids),
                                                           self.generator.noise_embedding: np.array(gen_noise_embedding),
                                                           self.generator.dis_node_embedding: np.array(gen_dis_node_embedding),
                                                           self.generator.dis_relation_embedding: np.array(gen_dis_relation_embedding)})

                    one_epoch_gen_loss += gen_loss

            one_epoch_batch_num = range_n

            # print t2 - t1
            # exit()
            print('[%.2f] gen loss = %.4f, dis loss = %.4f pos loss = %.4f neg loss-1 = %.4f neg loss-2 = %.4f' %
                  (time.time() - t, one_epoch_gen_loss / one_epoch_batch_num, one_epoch_dis_loss / one_epoch_batch_num,
                   one_epoch_pos_loss / one_epoch_batch_num, one_epoch_neg_loss_1 / one_epoch_batch_num, one_epoch_neg_loss_2 / one_epoch_batch_num))

            gen_score, dis_score = self.evaluate_paper_cluster()
            print('Generator: Precision = %.4f, Recall = %.4f, F1_score=%.4f' %
                  (gen_score[0], gen_score[1], gen_score[2]))
            print('Generator: Precision = %.4f, Recall = %.4f, F1_score=%.4f' %
                  (dis_score[0], dis_score[1], dis_score[2]))

            # save model
            if epoch % 4 == 0:
                self.save_model()

        self.write_embeddings_to_file()
        print("training completes")

    def prepare_data_for_d(self, index):

        pos_node_ids = []
        pos_relation_ids = []
        pos_node_neighbor_ids = []

        # real node and wrong relation
        neg_node_ids_1 = []
        neg_relation_ids_1 = []
        neg_node_neighbor_ids_1 = []

        # fake node and true relation
        neg_node_ids_2 = []
        neg_relation_ids_2 = []
        node_fake_neighbor_embedding = None

        for node_id in self.node_list[index * cfg.batch_size: (index + 1) * cfg.batch_size]:
            for i in range(cfg.n_sample):

                # sample real node and true relation
                relations = list(self.graph[node_id].keys())
                relation_id = relations[np.random.randint(0, len(relations))]
                neighbors = self.graph[node_id][relation_id]
                node_neighbor_id = neighbors[np.random.randint(
                    0, len(neighbors))]

                pos_node_ids.append(node_id)
                pos_relation_ids.append(relation_id)
                pos_node_neighbor_ids.append(node_neighbor_id)

                # sample real node and wrong relation
                neg_node_ids_1.append(node_id)
                neg_node_neighbor_ids_1.append(node_neighbor_id)
                neg_relation_id_1 = np.random.randint(0, self.n_relation)
                while neg_relation_id_1 == relation_id:
                    neg_relation_id_1 = np.random.randint(0, self.n_relation)
                neg_relation_ids_1.append(neg_relation_id_1)

                # sample fake node and true relation
                neg_node_ids_2.append(node_id)
                neg_relation_ids_2.append(relation_id)

        # generate fake node
        noise_embedding = np.random.normal(
            0.0, cfg.sig, (len(neg_node_ids_2), cfg.n_emb))

        node_fake_neighbor_embedding = self.sess.run(self.generator.node_neighbor_embedding,
                                                     feed_dict={self.generator.node_id: np.array(neg_node_ids_2),
                                                                self.generator.relation_id: np.array(neg_relation_ids_2),
                                                                self.generator.noise_embedding: np.array(noise_embedding)})

        return pos_node_ids, pos_relation_ids, pos_node_neighbor_ids, neg_node_ids_1, neg_relation_ids_1, \
            neg_node_neighbor_ids_1, neg_node_ids_2, neg_relation_ids_2, node_fake_neighbor_embedding

    def prepare_data_for_g(self, index):
        node_ids = []
        relation_ids = []

        for node_id in self.node_list[index * cfg.batch_size: (index + 1) * cfg.batch_size]:
            for i in range(cfg.n_sample):
                relations = list(self.graph[node_id].keys())
                relation_id = relations[np.random.randint(0, len(relations))]

                node_ids.append(node_id)
                relation_ids.append(relation_id)

        noise_embedding = np.random.normal(
            0.0, cfg.sig, (len(node_ids), cfg.n_emb))

        dis_node_embedding, dis_relation_embedding = self.sess.run([self.discriminator.pos_node_embedding, self.discriminator.pos_relation_embedding],
                                                                   feed_dict={self.discriminator.pos_node_id: np.array(node_ids),
                                                                              self.discriminator.pos_relation_id: np.array(relation_ids)})
        return node_ids, relation_ids, noise_embedding, dis_node_embedding, dis_relation_embedding

    def evaluate_paper_cluster(self):
        modes = [self.generator, self.discriminator]
        scores = []
        for i in range(2):
            embedding_matrix = self.sess.run(modes[i].node_embedding_matrix)
            score = self.aminer_evaluation.evaluate_paper_cluster_using_Tsne(
                embedding_matrix)
            scores.append(score)

        return scores

    def write_embeddings_to_file(self):
        modes = [self.generator, self.discriminator]
        for i in range(2):
            embedding_matrix = self.sess.run(modes[i].node_embedding_matrix)
            embedding_matrix = np.array(embedding_matrix)

            if i == 0:
                np.save(cfg.emb_filenames_gen+self.name +
                        '_gen.npy', np.array(embedding_matrix))
            else:
                np.save(cfg.emb_filenames_dis+self.name +
                        '_dis.npy', np.array(embedding_matrix))


if __name__ == '__main__':
    # pid2idx_by_name, names = pid2idxMapping()
    # generate graph by name
    name = 'xu_shen'
    n_node = 353
    trainer = Trainer(name, n_node, author_file=cfg.TRAIN_AUTHOR_PATH,
                      feature_file='../dataset/features/test/xu_shen.npy')
    # trainer.restore_model()
    trainer.train(n_epoch=20)
    # trainer.train()
    # trainer.write_embeddings_to_file()

# NOTE: deprecated Graph AutoEncoder
'''
if __name__ == "__main__":
    # word2vec = Word2Vec()
    # word2vec.train()
    # word2vec.save()

    graph = load_json(rfpath=cfg.VAL_GRAPH_PATH)
    print(graph)
    exit(0)
    feats = load_pub_features(rfpath=cfg.VAL_PUB_FEATURES_PATH)
    author_pubs_raw = load_json(rfpath=cfg.VAL_AUTHOR_PATH)
    relation_features = {}

    graph = graphMapping(graph, rfpath=cfg.VAL_AUTHOR_PATH)
    for idx, name in enumerate(graph):
        coo_node_list = graph[name]
        n_node = len(author_pubs_raw[name])
        Y = np.empty([0, EMBEDDING_SIZE])
        for i, pid in enumerate(author_pubs_raw[name]):
            # print(i)
            Y = np.append(Y, np.reshape(
                feats[pid], [1, EMBEDDING_SIZE]), axis=0)

        adj_norm, adj = preprocess_graph(coo_node_list, n_node)
        adj_label = adj + sp.eye(n_node)
        adj_label = torch.FloatTensor(adj_label.toarray())

        pos_weight = float(n_node * n_node - adj.sum()) / adj.sum()
        norm = n_node * n_node / \
            float((n_node * n_node - adj.sum()) * 2)

        Y = torch.from_numpy(Y).float()

        trainer = AutoEncoderTrainer(lr=0.003)
        print('params to optimize: ')
        print(trainer.model.parameters())
        trainer.train(
            Y=Y,
            normed_A=adj_norm,
            labels_A=adj_label,
            n_node=n_node,
            norm=norm,
            pos_weight=torch.tensor(pos_weight, dtype=torch.float)
        )

        embedding = trainer.generate_embedding(Y, adj_norm)
        paper_to_embedding_dict = {}
        for i, pid in enumerate(author_pubs_raw[name]):
            paper_to_embedding_dict[pid] = Y[i]

        relation_features[name] = paper_to_embedding_dict

    dump_data(relation_features, wfname=cfg.VAL_RELATION_FEATURES_PATH)
'''
