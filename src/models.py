import math
'''
import torch
from torch.nn.parameter import Parameter
from torch.nn import Module
from torch import nn
'''
from gensim.models import word2vec
from gensim.models.keyedvectors import KeyedVectors
import config as cfg
from utils import *

import tensorflow as tf
import config as cfg


class Generator():
    def __init__(self, n_node, n_relation, node_emd_init, relation_emd_init):
        self.n_node = n_node
        self.n_relation = n_relation
        self.node_emd_init = node_emd_init
        self.relation_emd_init = relation_emd_init
        self.emd_dim = node_emd_init.shape[1]

        # with tf.variable_scope('generator'):
        self.node_embedding_matrix = tf.get_variable(name="gen_node_embedding",
                                                     shape=self.node_emd_init.shape,
                                                     initializer=tf.constant_initializer(
                                                         self.node_emd_init),
                                                     trainable=True)
        self.relation_embedding_matrix = tf.get_variable(name="gen_relation_embedding",
                                                         shape=[
                                                             self.n_relation, self.emd_dim, self.emd_dim],
                                                         initializer=tf.contrib.layers.xavier_initializer(
                                                             uniform=False),
                                                         trainable=True)

        self.gen_w_1 = tf.get_variable(name='gen_w',
                                       shape=[self.emd_dim, self.emd_dim],
                                       initializer=tf.contrib.layers.xavier_initializer(
                                           uniform=False),
                                       trainable=True)
        self.gen_b_1 = tf.get_variable(name='gen_b',
                                       shape=[self.emd_dim],
                                       initializer=tf.contrib.layers.xavier_initializer(
                                           uniform=False),
                                       trainable=True)
        self.gen_w_2 = tf.get_variable(name='gen_w_2',
                                       shape=[self.emd_dim, self.emd_dim],
                                       initializer=tf.contrib.layers.xavier_initializer(
                                           uniform=False),
                                       trainable=True)
        self.gen_b_2 = tf.get_variable(name='gen_b_2',
                                       shape=[self.emd_dim],
                                       initializer=tf.contrib.layers.xavier_initializer(
                                           uniform=False),
                                       trainable=True)
        #self.bias_vector = tf.Variable(tf.zeros([self.n_node]))

        self.node_id = tf.placeholder(tf.int32, shape=[None])
        self.relation_id = tf.placeholder(tf.int32, shape=[None])
        self.noise_embedding = tf.placeholder(
            tf.float32, shape=[None, self.emd_dim])

        self.dis_node_embedding = tf.placeholder(
            tf.float32, shape=[None, self.emd_dim])
        self.dis_relation_embedding = tf.placeholder(
            tf.float32, shape=[None, self.emd_dim, self.emd_dim])

        self.node_embedding = tf.nn.embedding_lookup(
            self.node_embedding_matrix, self.node_id)
        self.relation_embedding = tf.nn.embedding_lookup(
            self.relation_embedding_matrix, self.relation_id)
        self.node_neighbor_embedding = self.generate_node(
            self.node_embedding, self.relation_embedding, self.noise_embedding)

        t = tf.reshape(tf.matmul(tf.expand_dims(
            self.dis_node_embedding, 1), self.dis_relation_embedding), [-1, self.emd_dim])
        self.score = tf.reduce_sum(tf.multiply(
            t, self.node_neighbor_embedding), axis=1)

        self.loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.score) * (1.0 - cfg.label_smooth), logits=self.score)) \
            + cfg.lambda_gen * (tf.nn.l2_loss(self.node_embedding) + tf.nn.l2_loss(
                self.relation_embedding) + tf.nn.l2_loss(self.gen_w_1))

        optimizer = tf.train.AdamOptimizer(cfg.lr_gen)
        self.g_updates = optimizer.minimize(self.loss)

    def generate_node(self, node_embedding, relation_embedding, noise_embedding):
        #node_embedding = tf.nn.embedding_lookup(self.node_embedding_matrix, node_id)
        #relation_embedding = tf.nn.embedding_lookup(self.relation_embedding_matrix, relation_id)

        input = tf.reshape(tf.matmul(tf.expand_dims(
            node_embedding, 1), relation_embedding), [-1, self.emd_dim])
        #input = tf.concat([input, noise_embedding], axis = 1)
        input = input + noise_embedding

        output = tf.nn.leaky_relu(
            tf.matmul(input, self.gen_w_1) + self.gen_b_1)
        # input = tf.nn.leaky_relu(tf.matmul(input, self.gen_w_1) + self.gen_b_1)# +  relation_embedding
        #output = tf.nn.leaky_relu(tf.matmul(input, self.gen_w_2) + self.gen_b_2)
        #output = node_embedding + relation_embedding + noise_embedding

        return output


class Discriminator():
    def __init__(self, n_node, n_relation, node_emd_init, relation_emd_init):
        self.n_node = n_node
        self.n_relation = n_relation
        self.node_emd_init = node_emd_init
        self.relation_emd_init = relation_emd_init
        self.emd_dim = node_emd_init.shape[1]

        # with tf.variable_scope('disciminator'):
        self.node_embedding_matrix = tf.get_variable(name='dis_node_embedding',
                                                     shape=self.node_emd_init.shape,
                                                     initializer=tf.constant_initializer(
                                                         self.node_emd_init),
                                                     trainable=True)
        self.relation_embedding_matrix = tf.get_variable(name='dis_relation_embedding',
                                                         shape=[
                                                             self.n_relation, self.emd_dim, self.emd_dim],
                                                         initializer=tf.contrib.layers.xavier_initializer(
                                                             uniform=False),
                                                         trainable=True)

        self.pos_node_id = tf.placeholder(tf.int32, shape=[None])
        self.pos_relation_id = tf.placeholder(tf.int32, shape=[None])
        self.pos_node_neighbor_id = tf.placeholder(tf.int32, shape=[None])

        self.neg_node_id_1 = tf.placeholder(tf.int32, shape=[None])
        self.neg_relation_id_1 = tf.placeholder(tf.int32, shape=[None])
        self.neg_node_neighbor_id_1 = tf.placeholder(tf.int32, shape=[None])

        self.neg_node_id_2 = tf.placeholder(tf.int32, shape=[None])
        self.neg_relation_id_2 = tf.placeholder(tf.int32, shape=[None])
        self.node_fake_neighbor_embedding = tf.placeholder(
            tf.float32, shape=[None, self.emd_dim])

        self.pos_node_embedding = tf.nn.embedding_lookup(
            self.node_embedding_matrix, self.pos_node_id)
        self.pos_node_neighbor_embedding = tf.nn.embedding_lookup(
            self.node_embedding_matrix, self.pos_node_neighbor_id)
        self.pos_relation_embedding = tf.nn.embedding_lookup(
            self.relation_embedding_matrix, self.pos_relation_id)

        self.neg_node_embedding_1 = tf.nn.embedding_lookup(
            self.node_embedding_matrix, self.neg_node_id_1)
        self.neg_node_neighbor_embedding_1 = tf.nn.embedding_lookup(
            self.node_embedding_matrix, self.neg_node_neighbor_id_1)
        self.neg_relation_embedding_1 = tf.nn.embedding_lookup(
            self.relation_embedding_matrix, self.neg_relation_id_1)

        self.neg_node_embedding_2 = tf.nn.embedding_lookup(
            self.node_embedding_matrix, self.neg_node_id_2)
        self.neg_relation_embedding_2 = tf.nn.embedding_lookup(
            self.relation_embedding_matrix, self.neg_relation_id_2)

        # pos loss
        t = tf.reshape(tf.matmul(tf.expand_dims(
            self.pos_node_embedding, 1), self.pos_relation_embedding), [-1, self.emd_dim])
        self.pos_score = tf.reduce_sum(tf.multiply(
            t, self.pos_node_neighbor_embedding), axis=1)
        self.pos_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(self.pos_score), logits=self.pos_score))

        # neg loss_1
        t = tf.reshape(tf.matmul(tf.expand_dims(
            self.neg_node_embedding_1, 1), self.neg_relation_embedding_1), [-1, self.emd_dim])
        self.neg_score_1 = tf.reduce_sum(tf.multiply(
            t, self.neg_node_neighbor_embedding_1), axis=1)
        self.neg_loss_1 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(self.neg_score_1), logits=self.neg_score_1))

        # neg loss_2
        t = tf.reshape(tf.matmul(tf.expand_dims(
            self.neg_node_embedding_2, 1), self.neg_relation_embedding_2), [-1, self.emd_dim])
        self.neg_score_2 = tf.reduce_sum(tf.multiply(
            t, self.node_fake_neighbor_embedding), axis=1)
        self.neg_loss_2 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(self.neg_score_2), logits=self.neg_score_2))

        self.loss = self.pos_loss + self.neg_loss_1 + self.neg_loss_2

        optimizer = tf.train.AdamOptimizer(cfg.lr_dis)
        #optimizer = tf.train.GradientDescentOptimizer(config.lr_dis)
        #optimizer = tf.train.RMSPropOptimizer(config.lr_dis)
        self.d_updates = optimizer.minimize(self.loss)


class Word2Vec():

    def __init__(self):
        return

    def train(self):
        sentences = word2vec.Text8Corpus(cfg.ALL_TEXT_PATH)
        self.model = word2vec.Word2Vec(
            sentences, size=100, negative=5, min_count=1, window=5)

    def save(self):
        self.model.save(cfg.WORD_EMBEDDING_MODEL_PATH)

    def load(self):
        self.model = KeyedVectors.load(cfg.WORD_EMBEDDING_MODEL_PATH)


# ------------GpraphConvolution Layer --------------
'''
class GraphConvolution(Module):
    def __init__(self, n_in_features, n_out_features, bias=False):
        
        # n_in_features: input #features of each nodes in graph
        # n_out_features: output #features of each nodes in graph
        # bias: bool 
        
        super(GraphConvolution, self).__init__()
        self.n_in_features = n_in_features
        self.n_out_features = n_out_features
        # kernel W
        self.weight = Parameter(torch.FloatTensor(
            n_in_features, n_out_features))
        # bias b
        if bias:
            self.bias = Parameter(torch.FloatTensor(n_out_features))
        else:
            self.register_parameter('bias', None)
        self.initialize()

    def initialize(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, normed_A):
        
        # input: input node static features
        #     shape: [n_papers_in_candidate, n_in_features] 
        # normed_A: normalized adjency matrix A
        #     shape: [n_papers_in_candidate, n_papers_in_candidate] 
        

        # support = Y * W
        # print('Y:')
        # print(input)
        support = torch.matmul(input, self.weight)
        # print('normed_A:')
        # output = A * support
        # print(normed_A)
        output = torch.sparse.mm(normed_A, support)
        # print('output:')
        # print(output)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


# ----------------Encoder GCN ----------------------
# two layers GCN
# Encoder: g1(Y, A) = A_normed * ReLU(A_normed * Y * W0)* W1
#          A_normed = sqrt(D) * A * 1/sqrt(D)
#          A: adjency matrix, D: degree matrix(diag)

class Encoder(Module):
    def __init__(self, n_input_features, n_hidden_features, n_output_features, dropout_rate):
        
        # n_input_features: input #features of each nodes in graph
        # n_hidden_features: hidden #features of each nodes in graph
        # n_output_features: output #features of each nodes in graph
        # dropout_rate: between [0,1]
        
        super(Encoder, self).__init__()
        self.gcn1 = GraphConvolution(n_input_features, n_hidden_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.gcn2 = GraphConvolution(n_hidden_features, n_output_features)
        self.gcn3 = GraphConvolution(n_hidden_features, n_output_features)

    def forward(self, Y, normed_A):
        
        # Y: ndoe sematic features
        #    shape: [n_papers_in_candidate, n_input_features]
        # normed_A: normalized adjency matrix A
        #    shape: [n_papers_in_candidate, n_papers_in_candidate] 
        
        gcn1_output = self.gcn1.forward(Y, normed_A)
        # print(gcn1_output)

        relu_output = self.relu(gcn1_output)
        # print(relu_output)

        dropout_output = self.dropout(relu_output)
        # print(dropout_output)
        # exit(0)
        mu = self.relu(self.gcn2.forward(dropout_output, normed_A))
        logvar = self.relu(self.gcn3.forward(dropout_output, normed_A))
        return mu, logvar

# ----------------Decoder -------------------
# Decoder: g2(Z) = sigmoid(ZZ')


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        
        # input: output of graph features
        #     shape: [n_papers_in_candidate, n_out_features]
        # return: shape: [n_papers_in_candidate, n_papers_in_candidate]
        
        # print(input)
        output = self.sigmoid(torch.mm(input, input.t()))
        # print(output)
        # exit(0)
        return output


class AutoEncoder(nn.Module):
    def __init__(self, training=True):
        super(AutoEncoder, self).__init__()
        self.training = training
        self.encoder = Encoder(
            n_input_features=100,
            n_hidden_features=128,
            n_output_features=100,
            dropout_rate=0
        )
        self.decoder = Decoder()

    def sample(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, Y, normed_A):
        mu, logvar = self.encoder(Y, normed_A)
        #print(mu, logvar)
        z = self.sample(mu, logvar)
        # print(z)
        # exit(0)
        return self.decoder(z), mu, logvar
'''
