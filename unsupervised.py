# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import time
import json
import argparse
from collections import OrderedDict
import numpy as np
import torch
import scipy

from src.utils import bool_flag, initialize_exp
from src.models import build_model
from src.trainer import Trainer
from src.evaluation import Evaluator


VALIDATION_METRIC = 'mean_cosine-csls_knn_10-S2T-10000'


# main
parser = argparse.ArgumentParser(description='Unsupervised training')
parser.add_argument("--seed", type=int, default=-1, help="Initialization seed")
parser.add_argument("--verbose", type=int, default=2, help="Verbose level (2:debug, 1:info, 0:warning)")
parser.add_argument("--exp_path", type=str, default="", help="Where to store experiment logs and models")
parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
parser.add_argument("--cuda", type=bool_flag, default=True, help="Run on GPU")
parser.add_argument("--export", type=str, default="txt", help="Export embeddings after training (txt / pth)")
# data
parser.add_argument("--src_lang", type=str, default='en', help="Source language")
parser.add_argument("--tgt_lang", type=str, default='de', help="Target language")
parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
parser.add_argument("--max_vocab", type=int, default=200000, help="Maximum vocabulary size (-1 to disable)")
# mapping
parser.add_argument("--map_id_init", type=bool_flag, default=True, help="Initialize the mapping as an identity matrix")
parser.add_argument("--map_beta", type=float, default=0.001, help="Beta for orthogonalization")
# discriminator
parser.add_argument("--dis_layers", type=int, default=2, help="Discriminator layers")
parser.add_argument("--dis_hid_dim", type=int, default=2048, help="Discriminator hidden layer dimensions")
parser.add_argument("--dis_dropout", type=float, default=0., help="Discriminator dropout")
parser.add_argument("--dis_input_dropout", type=float, default=0.1, help="Discriminator input dropout")
parser.add_argument("--dis_steps", type=int, default=5, help="Discriminator steps") #3
parser.add_argument("--dis_lambda", type=float, default=1, help="Discriminator loss feedback coefficient")
parser.add_argument("--src_dis_most_frequent", type=int, default=75000, help="Select embeddings of the k most frequent words for discrimination (0 to disable),75000")
parser.add_argument("--tgt_dis_most_frequent", type=int, default=75000, help="Select embeddings of the k most frequent words for discrimination (0 to disable),75000")
parser.add_argument("--dis_smooth", type=float, default=0.1, help="Discriminator smooth predictions")
parser.add_argument("--dis_clip_weights", type=float, default=0, help="Clip discriminator weights (0 to disable)")

parser.add_argument("--gnn_optimizer", type=str, default="sgd,lr=0.1", help="Mapping optimizer")
parser.add_argument("--adj_a", type=str, default=r"C:\Users\12425\Desktop\gcn_algin\outputs\en_20w_n50_matrix", help="Adjacent matrix")
parser.add_argument("--adj_b", type=str, default=r"C:\Users\12425\Desktop\gcn_algin\outputs\de_20w_n50_matrix", help="Adjacent matrix")


# training adversarial
parser.add_argument("--adversarial", type=bool_flag, default=True, help="Use adversarial training")
parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs")
parser.add_argument("--epoch_size", type=int, default=200000, help="Iterations per epoch")  #20w
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--map_optimizer", type=str, default="sgd,lr=0.1", help="Mapping optimizer")
parser.add_argument("--dis_optimizer", type=str, default="sgd,lr=0.1", help="Discriminator optimizer")
parser.add_argument("--lr_decay", type=float, default=0.98, help="Learning rate decay (SGD only)")
parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate (SGD only)")
parser.add_argument("--lr_shrink", type=float, default=0.5, help="Shrink the learning rate if the validation metric decreases (1 to disable)")
# training refinement
parser.add_argument("--n_refinement", type=int, default=10, help="Number of refinement iterations (0 to disable the refinement procedure)")
parser.add_argument("--refine_max_vocab_src", type=int, default=20000, help="Maximum dictionary words  ")
parser.add_argument("--refine_max_vocab_tgt", type=int, default=20000, help="Maximum dictionary words  ")
parser.add_argument("--refine_max_vocab_eval_acc", type=int, default=200000, help="Maximum dictionary words  ")



# dictionary creation parameters (for refinement)
parser.add_argument("--dico_eval", type=str, default="default", help="Path to evaluation dictionary")
parser.add_argument("--dico_method", type=str, default='csls_knn_10', help="Method used for dictionary generation (nn/invsm_beta_30/csls_knn_10)")
parser.add_argument("--dico_build", type=str, default='S2T&T2S', help="S2T,T2S,S2T|T2S,S2T&T2S")
parser.add_argument("--dico_threshold", type=float, default=0, help="Threshold confidence for dictionary generation")
parser.add_argument("--dico_max_rank", type=int, default=20000, help="Maximum dictionary words rank (0 to disable)")
parser.add_argument("--dico_min_size", type=int, default=0, help="Minimum generated dictionary size (0 to disable)")
parser.add_argument("--dico_max_size", type=int, default=0, help="Maximum generated dictionary size (0 to disable)")
# reload pre-trained embeddings
parser.add_argument("--src_emb", type=str, default=r"D:\dataset\muse\wiki.en.vec", help="Reload source embeddings")
parser.add_argument("--tgt_emb", type=str, default=r"D:\dataset\muse\wiki.de.vec", help="Reload target embeddings")
parser.add_argument("--normalize_embeddings", type=str, default="renorm,center,renorm", help="Normalize embeddings before training")

parser.add_argument("--control_gnn", type=int, default=1)

# parse parameters
params = parser.parse_args()

# check parameters
assert not params.cuda or torch.cuda.is_available()
assert 0 <= params.dis_dropout < 1
assert 0 <= params.dis_input_dropout < 1
assert 0 <= params.dis_smooth < 0.5
assert params.dis_lambda > 0 and params.dis_steps > 0
assert 0 < params.lr_shrink <= 1
assert os.path.isfile(params.src_emb)
assert os.path.isfile(params.tgt_emb)
assert params.dico_eval == 'default' or os.path.isfile(params.dico_eval)
assert params.export in ["", "txt", "pth"]

# build model / trainer / evaluator
logger = initialize_exp(params)
origin_s,origin_t,Adj_A,Adj_B, src_emb, tgt_emb, gnn_model, mapping, discriminator = build_model(params, True)
trainer = Trainer(origin_s,origin_t,Adj_A,Adj_B, src_emb, tgt_emb, gnn_model, mapping, discriminator, params)
evaluator = Evaluator(trainer)


"""
Learning loop for Adversarial Training
"""
if params.adversarial:
    logger.info('----> ADVERSARIAL TRAINING <----\n\n')

    # training loop
    for n_epoch in range(params.n_epochs):#params.n_epochs

        logger.info('Starting adversarial training epoch %i...' % n_epoch)
        tic = time.time()
        n_words_proc = 0
        stats = {'DIS_COSTS': []}

        for n_iter in range(0, params.epoch_size, params.batch_size):

            # discriminaton_refinementr training
            for _ in range(params.dis_steps):
                trainer.dis_step(stats)

            # mapping training (discriminator fooling)
            n_words_proc += trainer.mapping_step(stats)

            # log stats
            if n_iter % 500 == 0:
                stats_str = [('DIS_COSTS', 'Discriminator loss')]
                stats_log = ['%s: %.4f' % (v, np.mean(stats[k]))
                             for k, v in stats_str if len(stats[k]) > 0]
                cost = time.time()-tic
                cost = cost if cost > 0 else 2
                stats_log.append('%i samples/s' % int(n_words_proc / cost))
                logger.info(('%06i - ' % n_iter) + ' - '.join(stats_log))

                # reset
                tic = time.time()
                n_words_proc = 0
                for k, _ in stats_str:
                    del stats[k][:]

        # embeddings / discriminator evaluation
        to_log = OrderedDict({'n_epoch': n_epoch})
        # evaluator.all_eval(to_log)
        evaluator.dist_mean_cosine_test1(to_log)
        # evaluator.word_translation(to_log)
        evaluator.eval_dis(to_log)

        W = trainer.mapping.weight.data.cpu().numpy()
        path = os.path.join(params.exp_path, 'best_mapping%i.pth' % n_epoch)
        logger.info('* Saving the mapping to %s ...' % path)
        torch.save(W, path)



        # JSON log / save best model / end of epoch
        logger.info("__log__:%s" % json.dumps(to_log))
        trainer.save_best(to_log, VALIDATION_METRIC)
        logger.info('End of epoch %i.\n\n' % n_epoch)

        # update the learning rate (stop if too small)
        trainer.update_lr(to_log, VALIDATION_METRIC)
        if trainer.map_optimizer.param_groups[0]['lr'] < params.min_lr:
            logger.info('Learning rate < 1e-6. BREAK.')
            break




saved_acc = 0.
saved_iter = 0
highest_acc = 0.
highest_iter = 0
"""
Learning loop for Procrustes Iterative Refinement
"""
if params.n_refinement > 0:
    # Get the best mapping according to VALIDATION_METRIC
    logger.info('----> ITERATIVE PROCRUSTES REFINEMENT <----\n\n')
    trainer.reload_best()
    trainer.best_valid_metric = -1e9
    # training loop
    for n_iter in range(params.n_refinement):

        logger.info('Starting refinement iteration %i...' % n_iter)

        # build a dictionary from aligned embeddings
        trainer.build_dictionary_rcr()

        # apply the Procrustes solution
        trainer.procrustes()


        # embeddings evaluation
        to_log = OrderedDict({'n_iter': n_iter})
        evaluator.all_eval(to_log)

        # JSON log / save best model / end of epoch
        logger.info("__log__:%s" % json.dumps(to_log))
        isSaved = trainer.save_best(to_log, VALIDATION_METRIC, isRefine=True)

        if isSaved:
            saved_acc = to_log["precision_at_1-csls_knn_10"]
            saved_iter = n_iter
        if to_log["precision_at_1-csls_knn_10"] > highest_acc:
            highest_acc = to_log["precision_at_1-csls_knn_10"]
            highest_iter = n_iter

        logger.info('End of refinement iteration %i.\n\n' % n_iter)



trainer.reload_best(isRefine=True)
trainer.build_dictionary_rcr()
XW = trainer.os.weight.data.cpu().numpy()
ZW = trainer.ot.weight.data.cpu().numpy()
trainer.dico = trainer.dico.cpu().numpy()

# step1: whitening
A = XW[trainer.dico[:, 0]]
B = ZW[trainer.dico[:, 1]]
u, s, vt = scipy.linalg.svd(A, full_matrices=True)
WX1 = vt.T.dot(np.diag(1/s)).dot(vt)
u, s, vt = scipy.linalg.svd(B, full_matrices=True)
WZ1 = vt.T.dot(np.diag(1/s)).dot(vt)

XW = XW.dot(WX1)
ZW = ZW.dot(WZ1)

# step 2: orthogonal mapping
WX2, s, WZ2_t = scipy.linalg.svd(XW[trainer.dico[:, 0]].T.dot(ZW[trainer.dico[:, 1]]), full_matrices=True)
WZ2 = WZ2_t.T
XW = XW.dot(WX2)
ZW = ZW.dot(WZ2)

# step 3: Re-weighting
XW *= s**0.5
ZW *= s**0.5

# step 4: De-whitening
XW = XW.dot(WX2.T.dot(np.linalg.inv(WX1)).dot(WX2))
ZW = ZW.dot(WZ2.T.dot(np.linalg.inv(WZ1)).dot(WZ2))


trainer.os.weight.data.copy_(torch.from_numpy(XW).type_as(trainer.os.weight.data))
trainer.ot.weight.data.copy_(torch.from_numpy(ZW).type_as(trainer.ot.weight.data))

W_x = trainer.mapping.weight.data
W_x.copy_(torch.diag(torch.ones(trainer.params.emb_dim)))
to_log = OrderedDict({'n_iter': 0})
evaluator.dist_mean_cosine_test1(to_log)
evaluator.word_translation(to_log)
final_acc = to_log["precision_at_1-csls_knn_10"]


save_path = params.exp_path + "/final_acc_{}_saved_acc_{}_atIter_{}_highest_acc_{}_atIter_{:.3f}".format(final_acc,saved_acc, saved_iter, highest_acc, highest_iter)
if not os.path.exists(save_path):
    os.makedirs(save_path)


src_path = os.path.join(params.exp_path, 'mapped-vectors-%s.pth' % params.src_lang)
tgt_path = os.path.join(params.exp_path, 'mapped-vectors-%s.pth' % params.tgt_lang)
logger.info('Writing source embeddings to %s ...' % src_path)
torch.save({'dico': params.src_dico, 'vectors': trainer.os.weight.data.cpu()}, src_path)
logger.info('Writing target embeddings to %s ...' % tgt_path)
torch.save({'dico': params.tgt_dico, 'vectors': trainer.ot.weight.data.cpu()}, tgt_path)

