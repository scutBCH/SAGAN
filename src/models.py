# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import math
from logging import getLogger
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from .utils import load_embeddings, normalize_embeddings
from torch.autograd import Variable

logger = getLogger()
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.diag(torch.ones(self.in_features)))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        # self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.copy_(torch.diag(torch.ones(self.in_features)))

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, params, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.params = params
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        # self.dropout = dropout

    def forward(self, x, adj):
        # x = F.relu(self.gc1(x, adj))

        x0 = x
        x1 = self.gc1(x, adj)
        # x2 = self.gc1(x1, adj)
        # x3 = self.gc1(x2, adj)
        # pairNorm_SI = "renorm,center,renorm"
        # normalize_embeddings(x1, pairNorm_SI)

        # normalize_embeddings(x2, pairNorm_SI)
        # normalize_embeddings(x3, pairNorm_SI)

        # x = F.dropout(x, self.dropout, training=self.training)
        # x2 = self.gc2(x1, adj)
        # x = F.relu(self.gc2(x, adj))

        output = (0.95 * x0 + 0.05 * x1)
        return output

    def set_eval(self, requires_grad=False):
        for param in self.parameters():
            param.requires_grad = requires_grad
        if requires_grad:
            self.train()
        else:
            self.eval()

    def set_train(self):
        self.set_eval(True)


class Discriminator(nn.Module):

    def __init__(self, params):
        super(Discriminator, self).__init__()

        self.emb_dim = params.emb_dim
        self.dis_layers = params.dis_layers
        self.dis_hid_dim = params.dis_hid_dim
        self.dis_dropout = params.dis_dropout
        self.dis_input_dropout = params.dis_input_dropout

        layers = [nn.Dropout(self.dis_input_dropout)]
        for i in range(self.dis_layers + 1):
            input_dim = self.emb_dim if i == 0 else self.dis_hid_dim
            output_dim = 1 if i == self.dis_layers else self.dis_hid_dim
            layers.append(nn.Linear(input_dim, output_dim))
            if i < self.dis_layers:
                layers.append(nn.LeakyReLU(0.2))
                layers.append(nn.Dropout(self.dis_dropout))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        assert x.dim() == 2 and x.size(1) == self.emb_dim
        return self.layers(x).view(-1)

    def set_eval(self, requires_grad=False):
        for param in self.parameters():
            param.requires_grad = requires_grad
        if requires_grad:
            self.train()
        else:
            self.eval()

    def set_train(self):
        self.set_eval(True)


def build_model(params, with_dis, isEval=False):
    """
    Build all components of the model.
    """
    # source embeddings
    src_dico, _src_emb = load_embeddings(params, source=True)
    params.src_dico = src_dico
    origin_src_emb = nn.Embedding(len(src_dico), params.emb_dim, sparse=True)
    origin_src_emb.weight.data.copy_(_src_emb)

    # target embeddings
    if params.tgt_lang:
        tgt_dico, _tgt_emb = load_embeddings(params, source=False)
        params.tgt_dico = tgt_dico
        origin_tgt_emb = nn.Embedding(len(tgt_dico), params.emb_dim, sparse=True)
        origin_tgt_emb.weight.data.copy_(_tgt_emb)
    else:
        origin_tgt_emb = None

    # normalize embeddings
    params.src_mean = normalize_embeddings(origin_src_emb.weight.data, params.normalize_embeddings)
    if params.tgt_lang:
        params.tgt_mean = normalize_embeddings(origin_tgt_emb.weight.data, params.normalize_embeddings)

    src_emb = nn.Embedding(len(src_dico), params.emb_dim, sparse=True)
    tgt_emb = nn.Embedding(len(tgt_dico), params.emb_dim, sparse=True)

    # mapping
    mapping = nn.Linear(params.emb_dim, params.emb_dim, bias=False)
    if getattr(params, 'map_id_init', True):
        mapping.weight.data.copy_(torch.diag(torch.ones(params.emb_dim)))

    # discriminator
    discriminator = Discriminator(params) if with_dis else None

    if params.cuda:
        src_emb.cuda()
        origin_src_emb.cuda()
        if params.tgt_lang:
            tgt_emb.cuda()
            origin_tgt_emb.cuda()
        mapping.cuda()
        if with_dis:
            discriminator.cuda()

    adj_a = None
    adj_b = None
    gnn_model = None
    if not isEval:
        gnn_model = GCN(params=params, nfeat=params.emb_dim, nhid=params.emb_dim, nclass=params.emb_dim, dropout=0.1)
        adj_a = torch.load(params.adj_a)
        adj_b = torch.load(params.adj_b)
        print("adj_a shape", adj_a.size())
        if params.cuda:
            gnn_model.cuda()
            adj_a = adj_a.cuda()
            adj_b = adj_b.cuda()
        src_gnn = gnn_model(_src_emb, adj_a)
        src_emb.weight.data.copy_(src_gnn)
        tgt_gnn = gnn_model(_tgt_emb, adj_b)
        tgt_emb.weight.data.copy_(tgt_gnn)

        adj_b.cpu()
        adj_a.cpu()

    # determine  ablation
    if params.control_gnn == 1:
        return origin_src_emb, origin_tgt_emb, adj_a, adj_b, src_emb, tgt_emb, gnn_model, mapping, discriminator
    else:
        logger.info("ablation GCN")
        return origin_src_emb, origin_tgt_emb, adj_a, adj_b, origin_src_emb, origin_tgt_emb, gnn_model, mapping, discriminator
