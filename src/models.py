import math
import torch
from torch.nn.parameter import Parameter
from torch.nn import Module
from torch import nn

# TODO: global embedding related models

# TODO: local embedding related models


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
        support = torch.matmul(input, self.weight)
        # output = A * Y *
        output = torch.sparse.mm(normed_A, support)
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

    def forward(self, Y, normed_A):
        '''
        Y: ndoe sematic features
            shape: [n_papers_in_candidate, n_input_features]
        normed_A: normalized adjency matrix A
            shape: [n_papers_in_candidate, n_papers_in_candidate] 
        '''
        gcn1_output = self.gcn1.forward(Y, normed_A)
        relu_output = self.relu(gcn1_output)
        dropout_output = self.dropout(relu_output)
        output = self.gcn2.forward(dropout_output, normed_A)

        return output


# ----------------Decoder -------------------
# Decoder: g2(Z) = sigmoid(Z'Z)
