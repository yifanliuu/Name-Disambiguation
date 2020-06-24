import math
import torch
from torch.nn.parameter import Parameter
from torch.nn import Module
from torch import nn
from gensim.models import word2vec
from gensim.models.keyedvectors import KeyedVectors
import config as cfg
from utils import *

# ------------ Word2Vec Class --------------


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


class GraphConvolution(Module):
    def __init__(self, n_in_features, n_out_features, bias=False):
        '''
        n_in_features: input #features of each nodes in graph
        n_out_features: output #features of each nodes in graph
        bias: bool 
        '''
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
        '''
        input: input node static features
            shape: [n_papers_in_candidate, n_in_features] 
        normed_A: normalized adjency matrix A
            shape: [n_papers_in_candidate, n_papers_in_candidate] 
        '''

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
        '''
        n_input_features: input #features of each nodes in graph
        n_hidden_features: hidden #features of each nodes in graph
        n_output_features: output #features of each nodes in graph
        dropout_rate: between [0,1]
        '''
        super(Encoder, self).__init__()
        self.gcn1 = GraphConvolution(n_input_features, n_hidden_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.gcn2 = GraphConvolution(n_hidden_features, n_output_features)
        self.gcn3 = GraphConvolution(n_hidden_features, n_output_features)

    def forward(self, Y, normed_A):
        '''
        Y: ndoe sematic features
            shape: [n_papers_in_candidate, n_input_features]
        normed_A: normalized adjency matrix A
            shape: [n_papers_in_candidate, n_papers_in_candidate] 
        '''
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
        '''
        input: output of graph features
            shape: [n_papers_in_candidate, n_out_features]
        return: shape: [n_papers_in_candidate, n_papers_in_candidate]
        '''
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
