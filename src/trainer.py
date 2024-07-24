# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import copy
import itertools
from logging import getLogger
import scipy
import scipy.linalg
import torch
from torch.autograd import Variable
from torch.nn import functional as F
import  numpy as np
from .utils import get_optimizer, load_embeddings, normalize_embeddings, export_embeddings
from .utils import clip_parameters
from .dico_builder import build_dictionary, build_dictionary_artet
from .evaluation.word_translation import DIC_EVAL_PATH, load_identical_char_dico, load_dictionary

logger = getLogger()


class Trainer(object):

    def __init__(self, os, ot, adj_a, adj_b, src_emb, tgt_emb, gnn_model, mapping, discriminator, params):
        """
        Initialize trainer script.
        """
        self.src_emb = src_emb
        self.tgt_emb = tgt_emb
        self.src_dico = params.src_dico
        self.tgt_dico = getattr(params, 'tgt_dico', None)
        self.gnn_model = gnn_model
        self.mapping = mapping
        self.discriminator = discriminator
        self.params = params
        self.adj_a = adj_a
        self.adj_b = adj_b
        self.os = os
        self.ot = ot

        # optimizers
        if hasattr(params, 'map_optimizer'):
            optim_fn, optim_params = get_optimizer(params.map_optimizer)
            self.map_optimizer = optim_fn(mapping.parameters(), **optim_params)
        if hasattr(params, 'dis_optimizer'):
            optim_fn, optim_params = get_optimizer(params.dis_optimizer)
            self.dis_optimizer = optim_fn(discriminator.parameters(), **optim_params)
        else:
            assert discriminator is None


        # best validation score
        self.best_valid_metric = -1e12

        self.decrease_lr = False

        dis_label = torch.FloatTensor(self.params.batch_size * 2).zero_()
        dis_label = Variable(dis_label.cuda() if self.params.cuda else dis_label)
        dis_label[:self.params.batch_size] = 1 - self.params.dis_smooth
        dis_label[self.params.batch_size:] = self.params.dis_smooth
        self.y = dis_label

    def get_dis_xy(self, volatile):
        """
        Get discriminator input batch / output target.
        """
        # select random word IDs
        bs = self.params.batch_size
        src_mf = self.params.src_dis_most_frequent
        tgt_mf = self.params.tgt_dis_most_frequent
        assert src_mf <= len(self.src_dico)
        assert tgt_mf <= len(self.tgt_dico)
        src_ids = torch.LongTensor(bs).random_(len(self.src_dico) if src_mf == 0 else src_mf)
        tgt_ids = torch.LongTensor(bs).random_(len(self.tgt_dico) if tgt_mf == 0 else tgt_mf)
        if self.params.cuda:
            src_ids = src_ids.cuda()
            tgt_ids = tgt_ids.cuda()

        # get word embeddings
        src_emb = self.src_emb(Variable(src_ids, volatile=True))
        src_emb = self.mapping(Variable(src_emb.data, volatile=volatile))
        tgt_emb = self.tgt_emb(Variable(tgt_ids, volatile=True))
        tgt_emb = Variable(tgt_emb.data, volatile=volatile)

        # os = self.os(Variable(src_ids, volatile=True))
        # src_emb = self.mapping(Variable(os.data, volatile=volatile))
        # ot = self.ot(Variable(tgt_ids, volatile=True))
        # ot = Variable(ot.data, volatile=volatile)

        # input / target
        # x = torch.cat([src_emb,os, tgt_emb,ot], 0)
        # y = torch.FloatTensor(4 * bs).zero_()
        x = torch.cat([src_emb, tgt_emb], 0)
        # y = torch.FloatTensor(2 * bs).zero_()
        # y[:bs] = 1 - self.params.dis_smooth
        # y[bs:] = self.params.dis_smooth
        # y = Variable(y.cuda() if self.params.cuda else y)

        return x, self.y  # , y



    def dis_step(self, stats):
        """
        Train the discriminator.
        """
        self.discriminator.set_train()
        self.gnn_model.set_eval()
        self.mapping.weight.requires_grad = False

        self
        # loss
        x, y = self.get_dis_xy(volatile=True)
        preds = self.discriminator(Variable(x.data))
        loss = F.binary_cross_entropy(preds, y)
        stats['DIS_COSTS'].append(loss.data.item())

        # check NaN
        if (loss != loss).data.any():
            logger.error("NaN detected (discriminator)")
            exit()

        # optim
        self.dis_optimizer.zero_grad()
        loss.backward()
        self.dis_optimizer.step()
        clip_parameters(self.discriminator, self.params.dis_clip_weights)

    def mapping_step(self, stats):
        """
        Fooling discriminator training step.
        """
        if self.params.dis_lambda == 0:
            return 0

        self.discriminator.set_eval()
        self.gnn_model.set_eval()
        self.mapping.weight.requires_grad = True

        # loss
        x, y = self.get_dis_xy(volatile=False)

        preds = self.discriminator(x)
        loss = F.binary_cross_entropy(preds, 1 - y)
        loss = self.params.dis_lambda * loss

        # check NaN
        if (loss != loss).data.any():
            logger.error("NaN detected (fool discriminator)")
            exit()

        # optim
        self.map_optimizer.zero_grad()
        loss.backward()
        self.map_optimizer.step()
        # self.orthogonalize()

        return 2 * self.params.batch_size

    def load_training_dico(self, dico_train):
        """
        Load training dictionary.
        """
        word2id1 = self.src_dico.word2id
        word2id2 = self.tgt_dico.word2id

        # identical character strings
        if dico_train == "identical_char":
            self.dico = load_identical_char_dico(word2id1, word2id2)
        # use one of the provided dictionary
        elif dico_train == "default":
            filename = '%s-%s.0-5000.txt' % (self.params.src_lang, self.params.tgt_lang)
            self.dico = load_dictionary(
                os.path.join(DIC_EVAL_PATH, filename),
                word2id1, word2id2
            )
        # dictionary provided by the user
        else:
            self.dico = load_dictionary(dico_train, word2id1, word2id2)

        # cuda
        if self.params.cuda:
            self.dico = self.dico.cuda()

    def build_dictionary(self):
        """
        Build a dictionary from aligned embeddings.
        """
        src_emb = self.mapping(self.os.weight).data
        tgt_emb = self.ot.weight.data
        src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
        tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)
        self.dico = build_dictionary(src_emb, tgt_emb, self.params)



    def build_dictionary_rcr(self):
        """
        Build a dictionary from aligned embeddings.
        """
        src_emb = self.mapping(self.os.weight[:self.params.refine_max_vocab_src]).data
        tgt_emb = self.ot.weight.data[:self.params.refine_max_vocab_tgt]
        print("********************************************")
        print("src emb size:", src_emb.size(), flush=True)
        print("tgt emb size:", tgt_emb.size(), flush=True)

        # normalize_embeddings(src_emb, "renorm,center,renorm")
        # tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)

        self.dico = build_dictionary(src_emb, tgt_emb, self.params)


    def build_dictionary_gcn_emb(self):
        """
        Build a dictionary from aligned embeddings.
        """
        src_emb = self.mapping(self.src_emb.weight[:self.params.refine_max_vocab_src]).data
        tgt_emb = self.tgt_emb.weight.data[:self.params.refine_max_vocab_tgt]
        print("********************************************")
        print("src emb size:", src_emb.size(), flush=True)
        print("tgt emb size:", tgt_emb.size(), flush=True)
        self.dico = build_dictionary(src_emb, tgt_emb, self.params)



    def procrustes(self):
        """
        Find the best orthogonal matrix mapping using the Orthogonal Procrustes problem
        https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
        """
        A = self.os.weight.data[self.dico[:, 0]]
        B = self.ot.weight.data[self.dico[:, 1]]
        W = self.mapping.weight.data
        M = B.transpose(0, 1).mm(A).cpu().numpy()
        U, S, V_t = scipy.linalg.svd(M, full_matrices=True)
        W.copy_(torch.from_numpy(U.dot(V_t)).type_as(W))

    def procrustes_gcn_emb(self):
        """
        Find the best orthogonal matrix mapping using the Orthogonal Procrustes problem
        https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
        """
        A = self.src_emb.weight.data[self.dico[:, 0]]
        B = self.tgt_emb.weight.data[self.dico[:, 1]]
        W = self.mapping.weight.data
        M = B.transpose(0, 1).mm(A).cpu().numpy()
        U, S, V_t = scipy.linalg.svd(M, full_matrices=True)
        W.copy_(torch.from_numpy(U.dot(V_t)).type_as(W))

    def orthogonalize(self):
        """
        Orthogonalize the mapping.
        """
        if self.params.map_beta > 0:
            W = self.mapping.weight.data
            beta = self.params.map_beta
            W.copy_((1 + beta) * W - beta * W.mm(W.transpose(0, 1).mm(W)))

    def update_lr(self, to_log, metric):
        """
        Update learning rate when using SGD.
        """
        if 'sgd' not in self.params.map_optimizer:
            return
        old_lr = self.map_optimizer.param_groups[0]['lr']
        new_lr = max(self.params.min_lr, old_lr * self.params.lr_decay)
        if new_lr < old_lr:
            logger.info("Decreasing learning rate: %.8f -> %.8f" % (old_lr, new_lr))
            self.map_optimizer.param_groups[0]['lr'] = new_lr

        if self.params.lr_shrink < 1 and to_log[metric] >= -1e7:
            if to_log[metric] < self.best_valid_metric:
                logger.info("Validation metric is smaller than the best: %.5f vs %.5f"
                            % (to_log[metric], self.best_valid_metric))
                # decrease the learning rate, only if this is the
                # second time the validation metric decreases
                if self.decrease_lr:
                    old_lr = self.map_optimizer.param_groups[0]['lr']
                    self.map_optimizer.param_groups[0]['lr'] *= self.params.lr_shrink
                    logger.info("Shrinking the learning rate: %.5f -> %.5f"
                                % (old_lr, self.map_optimizer.param_groups[0]['lr']))
                    self.decrease_lr = False
                else:
                    self.decrease_lr = True
            else:
                self.decrease_lr = False

    def save_best(self, to_log, metric, isRefine=False):
        """
        Save the best model for the given validation metric.
        """
        # best mapping for the given validation criterion
        if to_log[metric] > self.best_valid_metric:
            # new best mapping
            self.best_valid_metric = to_log[metric]
            logger.info('* Best value for "%s": %.5f' % (metric, to_log[metric]))
            # save the mapping
            W = self.mapping.weight.data.cpu().numpy()
            if isRefine == False:
                path = os.path.join(self.params.exp_path, 'best_mapping.pth')
            else:
                path = os.path.join(self.params.exp_path, 'best_refine.pth')
            logger.info('* Saving the mapping to %s ...' % path)
            torch.save(W, path)
            return True
        else:
            return False

    def reload_best(self, isRefine=False):
        """
        Reload the best mapping.
        """
        if isRefine:
            path = os.path.join(self.params.exp_path, 'best_refine.pth')
        else:
            path = os.path.join(self.params.exp_path, 'best_mapping.pth')
        logger.info('* Reloading the best model from %s ...' % path)
        # reload the model
        assert os.path.isfile(path)
        to_reload = torch.from_numpy(torch.load(path))
        W = self.mapping.weight.data
        assert to_reload.size() == W.size()
        W.copy_(to_reload.type_as(W))

    def export(self):
        """
        Export embeddings.
        """
        params = self.params

        # load all embeddings
        logger.info("Reloading all embeddings for mapping ...")
        params.src_dico, src_emb = load_embeddings(params, source=True, full_vocab=True)
        params.tgt_dico, tgt_emb = load_embeddings(params, source=False, full_vocab=True)

        # apply same normalization as during training
        normalize_embeddings(src_emb, params.normalize_embeddings, mean=params.src_mean)
        normalize_embeddings(tgt_emb, params.normalize_embeddings, mean=params.tgt_mean)

        # map source embeddings to the target space
        bs = 4096
        logger.info("Map source embeddings to the target space ...")
        for i, k in enumerate(range(0, len(src_emb), bs)):
            x = Variable(src_emb[k:k + bs], volatile=True)
            src_emb[k:k + bs] = self.mapping(x.cuda() if params.cuda else x).data.cpu()

        # write embeddings to the disk
        export_embeddings(src_emb, tgt_emb, params)
