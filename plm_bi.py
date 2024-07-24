

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

from src.utils import bool_flag, initialize_exp
from src.models import build_model
from src.trainer import Trainer
from src.evaluation import Evaluator
from src.evaluation.wordsim import  get_crosslingual_wordsim_scores
from src.evaluation.word_translation import get_word_translation_accuracy
from src.evaluation.sent_translation import load_europarl_data, get_sent_translation_accuracy
from src.dico_builder import build_dictionary
from src.utils import get_idf

def sent_translation(to_log):
    """
    Evaluation on sentence translation.
    Only available on Europarl, for en - {de, es, fr, it} language pairs.
    """
    lg1 = trainer.src_dico.lang
    lg2 = trainer.tgt_dico.lang

    # parameters
    n_keys = 200000
    n_queries = 2000
    n_idf = 300000

    # load europarl data
    europarl_data = load_europarl_data(
        lg1, lg2, n_max=(n_keys + 2 * n_idf)
    )

    # if no Europarl data for this language pair
    if not europarl_data:
        return

    # mapped word embeddings
    src_emb = trainer.os.weight.data
    tgt_emb = trainer.ot.weight.data

    # get idf weights
    idf = get_idf(europarl_data, lg1, lg2, n_idf=n_idf)

    for method in ['nn', 'csls_knn_10']:

        # source <- target sentence translation
        results = get_sent_translation_accuracy(
            europarl_data,
            trainer.src_dico.lang, trainer.src_dico.word2id, src_emb,
            trainer.tgt_dico.lang, trainer.tgt_dico.word2id, tgt_emb,
            n_keys=n_keys, n_queries=n_queries,
            method=method, idf=idf
        )
        to_log.update([('tgt_to_src_%s-%s' % (k, method), v) for k, v in results])

        # target <- source sentence translation
        results = get_sent_translation_accuracy(
            europarl_data,
            trainer.tgt_dico.lang, trainer.tgt_dico.word2id, tgt_emb,
            trainer.src_dico.lang, trainer.src_dico.word2id, src_emb,
            n_keys=n_keys, n_queries=n_queries,
            method=method, idf=idf
        )
        to_log.update([('src_to_tgt_%s-%s' % (k, method), v) for k, v in results])



def get_dictionary(_r_src_emb, _r_tgt_emb):
    """
    get the nearst neighbor as  align pairs
    """
    if params.control_plm_dico_cutoff != 1:
        logger.info("ablation PLM: dico frquency cutoff")
        _src_emb = _r_src_emb.data
        _tgt_emb = _r_tgt_emb.data  

    else:
        _src_emb = _r_src_emb[:trainer.params.refine_max_vocab_src].data
        _tgt_emb = _r_tgt_emb[:trainer.params.refine_max_vocab_tgt].data 

        
    # build dictionary
    trainer.dico =  build_dictionary(_src_emb, _tgt_emb, trainer.params)
    return trainer.dico




def get_closest_cosine(r_src_emb, align_src_emb, k):
    emb1 = r_src_emb
    emb2 = align_src_emb
    emb1 = emb1 / emb1.norm(2, 1, keepdim=True).expand_as(emb1)
    emb2 = emb2 / emb2.norm(2, 1, keepdim=True).expand_as(emb2)
    scores = emb1.mm(emb2.transpose(0, 1))
    #scores = (torch.exp( 10 * scores) - 1)
    top_matched_content, top_matched_idx = scores.topk(k, 1, True)
    return top_matched_content, top_matched_idx


@torch.no_grad()
def plm_torch(params, r_src_emb, align_src_emb, align_tgt_emb, k_closest=70, m=10, step_size=0.1):
    if params.control_plm_dico_scale !=1:
        logger.info("ablation PLM: scale")
    if params.control_plm_step_size !=1:
        logger.info("ablation PLM: step_size")
        step_size = 1.0

    print("dtype: ", r_src_emb.dtype)
    diff = (-align_src_emb + align_tgt_emb)
    logger.info("computing bias")
    bs = 512 #4096
    total = r_src_emb.shape[0]
    move_vector = torch.empty([0, r_src_emb.shape[1]])
    if params.cuda:
        move_vector = move_vector.cuda()
    for i in range(0, total, bs):
        top_matched_weight_part, top_matched_idx_part = get_closest_cosine(r_src_emb[i:i + bs], align_src_emb,
                                                                           k=k_closest)

        if params.control1 == 1:
            # logger.info("use threshold")
            a = torch.zeros(top_matched_weight_part.shape)
            if params.cuda:
                a = a.cuda()
            top_matched_weight_part = torch.where(top_matched_weight_part < params.local_mapping_threshold, a,
                                                  top_matched_weight_part)

        # logger.info(top_matched_weight_part)
        # bias shape:(bs,k_closest,300)
        bias = diff[top_matched_idx_part.type(torch.int64)]
        # csls_weight_part shape:(bs,300)
        csls_weight_part = torch.exp(
            m * top_matched_weight_part) if params.control_plm_dico_scale == 1 else top_matched_weight_part + 0.00001


        csls_weight_part = csls_weight_part.div_(csls_weight_part.norm(1, 1, keepdim=True).expand_as(csls_weight_part))
        # move_vector_part shape:(bs,300)
        move_vector_part = torch.sum(csls_weight_part.unsqueeze(2) * bias, dim=1)
        # print("move_vector_part_shape: ",move_vector_part.shape)
        # print("weight_shape: ",move_vector.shape)
        move_vector = torch.cat((move_vector, move_vector_part), 0)

    logger.info("updating the emb")
    result = []
    # for j in step_size:
    # print(j)
    new_embs_src = r_src_emb + step_size * move_vector
    result.append(new_embs_src.cpu() if params.cuda else new_embs_src)
    return result


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

parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
parser.add_argument("--max_vocab", type=int, default=200000, help="Maximum vocabulary size (-1 to disable)") #200000
# mapping
parser.add_argument("--map_id_init", type=bool_flag, default=True, help="Initialize the mapping as an identity matrix")
parser.add_argument("--map_beta", type=float, default=0.001, help="Beta for orthogonalization")
# discriminator
parser.add_argument("--dis_layers", type=int, default=2, help="Discriminator layers")
parser.add_argument("--dis_hid_dim", type=int, default=2048, help="Discriminator hidden layer dimensions")
parser.add_argument("--dis_dropout", type=float, default=0., help="Discriminator dropout")
parser.add_argument("--dis_input_dropout", type=float, default=0.1, help="Discriminator input dropout")
parser.add_argument("--dis_steps", type=int, default=3, help="Discriminator steps")
parser.add_argument("--dis_lambda", type=float, default=1, help="Discriminator loss feedback coefficient")
parser.add_argument("--src_dis_most_frequent", type=int, default=75000, help="Select embeddings of the k most frequent words for discrimination (0 to disable),75000")
parser.add_argument("--tgt_dis_most_frequent", type=int, default=75000, help="Select embeddings of the k most frequent words for discrimination (0 to disable),75000")
parser.add_argument("--dis_smooth", type=float, default=0.1, help="Discriminator smooth predictions")
parser.add_argument("--dis_clip_weights", type=float, default=0, help="Clip discriminator weights (0 to disable)")

parser.add_argument("--gnn_optimizer", type=str, default="sgd,lr=0.1", help="Mapping optimizer")
parser.add_argument("--adj_a", type=str, default=r"D:\coding\hbc\gcn-align\outputs\A_20w_n50_matrix", help="Adjacent matrix")
parser.add_argument("--adj_b", type=str, default=r"D:\coding\hbc\gcn-align\outputs\zh_20w_n50_matrix", help="Adjacent matrix")



# training adversarial
parser.add_argument("--adversarial", type=bool_flag, default=True, help="Use adversarial training")
parser.add_argument("--n_epochs", type=int, default=20, help="Number of epochs")
parser.add_argument("--epoch_size", type=int, default=200000, help="Iterations per epoch")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--map_optimizer", type=str, default="sgd,lr=0.1", help="Mapping optimizer")
parser.add_argument("--dis_optimizer", type=str, default="sgd,lr=0.1", help="Discriminator optimizer")
parser.add_argument("--lr_decay", type=float, default=0.98, help="Learning rate decay (SGD only)")
parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate (SGD only)")
parser.add_argument("--lr_shrink", type=float, default=0.5, help="Shrink the learning rate if the validation metric decreases (1 to disable)")
# training refinement
parser.add_argument("--n_refinement", type=int, default=5, help="Number of refinement iterations (0 to disable the refinement procedure)")
# dictionary creation parameters (for refinement)
parser.add_argument("--dico_eval", type=str, default="default", help="Path to evaluation dictionary")
parser.add_argument("--dico_method", type=str, default='csls_knn_10', help="Method used for dictionary generation (nn/invsm_beta_30/csls_knn_10)")
parser.add_argument("--dico_build", type=str, default='S2T&T2S', help="S2T,T2S,S2T|T2S,S2T&T2S")
parser.add_argument("--dico_threshold", type=float, default=0, help="Threshold confidence for dictionary generation")
parser.add_argument("--dico_max_rank", type=int, default=20000, help="Maximum dictionary words rank (0 to disable)") #2w, 5w   15000
parser.add_argument("--dico_min_size", type=int, default=0, help="Minimum generated dictionary size (0 to disable)")
parser.add_argument("--dico_max_size", type=int, default=0, help="Maximum generated dictionary size (0 to disable)")

parser.add_argument("--refine_max_vocab_src", type=int, default=20000, help="Maximum dictionary words  ")
parser.add_argument("--refine_max_vocab_tgt", type=int, default=20000, help="Maximum dictionary words  ")
parser.add_argument("--refine_max_vocab_eval_acc", type=int, default=200000, help="Maximum dictionary words  ")



parser.add_argument("--src_lang", type=str, default='en', help="Source language")
parser.add_argument("--tgt_lang", type=str, default='zh', help="Target language")
parser.add_argument("--src_emb", type=str, default=r"D:\dataset\muse\wiki.en.vec", help="Reload source embeddings")
parser.add_argument("--tgt_emb", type=str, default=r"D:\dataset\muse\wiki.zh.vec", help="Reload target embeddings")

parser.add_argument("--normalize_embeddings", type=str, default="renorm,center,renorm", help="Normalize embeddings before training")



parser.add_argument("--step_size", type=float, default=0.02) # for most languages:0.02    for en-ro: 0.01
parser.add_argument("--iter", type=int, default=5)
parser.add_argument("--m", type=int, default=10)
parser.add_argument("--k_closest", type=int, default=100)  # 70
parser.add_argument("--control1", type=int, default=1)
parser.add_argument("--control2", type=int, default=1)
parser.add_argument("--local_mapping_threshold", type=float, default=0.4)

# determine  ablation
parser.add_argument("--control_gnn", type=int, default=1)

parser.add_argument("--control_plm_dico_cutoff", type=int, default=1)
parser.add_argument("--control_plm_dico_scale", type=int, default=1)
parser.add_argument("--control_plm_step_size", type=int, default=1)
parser.add_argument("--control_plm_dico_or_operation", type=int, default=1)
parser.add_argument("--control_plm_bi_direction", type=int, default=1)

parser.add_argument("--tau", type=float, default=10.0)
# parse parameters
params = parser.parse_args()

params.m = 100.0/params.tau

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

origin_s,origin_t,Adj_A,Adj_B, src_emb, tgt_emb, gnn_model, mapping, discriminator = build_model(params, True, isEval=True)
trainer = Trainer(origin_s,origin_t,Adj_A,Adj_B, src_emb, tgt_emb, gnn_model, mapping, discriminator, params)
evaluator = Evaluator(trainer)



logger.info("m:%f"%params.m)

### eval the performance before PLM

W_x = trainer.mapping.weight.data
W_x.copy_(torch.diag(torch.ones(trainer.params.emb_dim)))
to_log = OrderedDict({'n_iter': 0})
# evaluator.dist_mean_cosine_test1(to_log)
# evaluator.word_translation_refine_test1(to_log)



logger.info("eval from %s to %s"%(trainer.src_dico.lang,trainer.tgt_dico.lang))
for method in [ 'csls_knn_10']:
    results = get_word_translation_accuracy(
        trainer.src_dico.lang, trainer.src_dico.word2id, trainer.os.weight.data,
        trainer.tgt_dico.lang, trainer.tgt_dico.word2id, trainer.ot.weight.data,

        method=method,
        dico_eval=params.dico_eval
    )
logger.info("\n")
logger.info("eval from %s to %s"%(trainer.tgt_dico.lang,trainer.src_dico.lang))
for method in ['csls_knn_10']:
    results = get_word_translation_accuracy(
        trainer.tgt_dico.lang, trainer.tgt_dico.word2id, trainer.ot.weight.data,
        trainer.src_dico.lang, trainer.src_dico.word2id, trainer.os.weight.data,
        method=method,
        dico_eval=params.dico_eval
    )

logger.info("\n\n\n")
### local mapping from here

if params.control_plm_dico_or_operation != 1:
    logger.info("ablation PLM: dico-or-operation")
    params.dico_build = 'S2T'
else:
    params.dico_build = 'S2T|T2S'


for iter in range(params.iter):
    generate_dico = get_dictionary(trainer.os.weight, trainer.ot.weight)
    generate_dico = generate_dico.cpu()
    align_src_ids = torch.LongTensor(generate_dico[:, 0])
    align_tgt_ids = torch.LongTensor(generate_dico[:, 1])

    # get word embeddings
    src_emb = trainer.os.weight.data[align_src_ids]
    tgt_emb = trainer.ot.weight.data[align_tgt_ids]
    align_src_emb = src_emb.data
    align_tgt_emb = tgt_emb.data
    outputs = plm_torch(params, trainer.os.weight.data, align_src_emb, align_tgt_emb, k_closest=params.k_closest,
                        m=params.m, step_size=params.step_size)
    # result = torch.from_numpy(outputs[0])
    result = outputs[0].float()
    result = result.cuda()
    trainer.os.weight.data.copy_(result.data)



    if params.control_plm_bi_direction != 1:
        logger.info("ablation PLM: bi-directions")
    else:
        generate_dico = get_dictionary(trainer.ot.weight, trainer.os.weight)
        generate_dico = generate_dico.cpu()
        align_src_ids = torch.LongTensor(generate_dico[:, 1])
        align_tgt_ids = torch.LongTensor(generate_dico[:, 0])

        src_emb = trainer.os.weight.data[align_src_ids]
        tgt_emb = trainer.ot.weight.data[align_tgt_ids]
        align_src_emb = src_emb.data
        align_tgt_emb = tgt_emb.data

        outputs = plm_torch(params, trainer.ot.weight.data, align_tgt_emb, align_src_emb, k_closest=params.k_closest,
                            m=params.m, step_size=params.step_size)
        # result = torch.from_numpy(outputs[0])
        result = outputs[0].float()
        result = result.cuda()
        trainer.ot.weight.data.copy_(result.data)


    # eval
    if (iter == params.iter-1):
        logger.info("eval from %s to %s"%(trainer.src_dico.lang,trainer.tgt_dico.lang))
        for method in [ 'csls_knn_10']:
            results = get_word_translation_accuracy(
                trainer.src_dico.lang, trainer.src_dico.word2id, trainer.os.weight.data,
                trainer.tgt_dico.lang, trainer.tgt_dico.word2id, trainer.ot.weight.data,

                method=method,
                dico_eval=params.dico_eval
            )
            to_log.update([('%s-%s' % (k, method), v) for k, v in results])

        logger.info("\n")
        logger.info("eval from %s to %s"%(trainer.tgt_dico.lang,trainer.src_dico.lang))
        for method in ['csls_knn_10']:
            results = get_word_translation_accuracy(
                trainer.tgt_dico.lang, trainer.tgt_dico.word2id, trainer.ot.weight.data,
                trainer.src_dico.lang, trainer.src_dico.word2id, trainer.os.weight.data,
                method=method,
                dico_eval=params.dico_eval
            )

    logger.info('End of epoch %i.\n\n======================================================' % iter)




# print("==========")


# src_path = os.path.join(params.exp_path, 'lm-vectors-%s.pth' % params.src_lang)
# tgt_path = os.path.join(params.exp_path, 'lm-vectors-%s.pth' % params.tgt_lang)
# logger.info('Writing source embeddings to %s ...' % src_path)
# torch.save({'dico': params.src_dico, 'vectors': trainer.os.weight.data.cpu()}, src_path)
# logger.info('Writing target embeddings to %s ...' % tgt_path)
# torch.save({'dico': params.tgt_dico, 'vectors': trainer.ot.weight.data.cpu()}, tgt_path)



acc = to_log["precision_at_1-csls_knn_10"]
save_path = params.exp_path + "/final_acc_s2t_{}".format(acc)
if not os.path.exists(save_path):
    os.makedirs(save_path)
