import logging
import math

import numpy as np
import tensorflow as tf
from sklearn import metrics
from utils import getLogger
from utils import ProgressBar


def compute_auc(all_label, all_pred):
    return metrics.roc_auc_score(all_label, all_pred)


def compute_accuracy(all_label, all_pred):
    all_pred[all_pred > 0.5] = 1.0
    all_pred[all_pred <= 0.5] = 0.0
    return metrics.accuracy_score(all_label, all_pred)


def binaryEntropy(label, pred, mod="avg"):
    loss = label * np.log(np.maximum(1e-10, pred)) + \
           (1.0 - label) * np.log(np.maximum(1e-10, 1.0 - pred))
    if mod == 'avg':
        return np.average(loss) * (-1.0)
    elif mod == 'sum':
        return - loss.sum()
    else:
        # 将程序引发 AssertionError。
        assert False


def run_model(model, args, q_data, qa_data, t_data, mode="train"):
    '''
     Run one epoch
    :param model: the KVFGKTModel
    :param args:  the main function args
    :param q_data:  the question information
    :param qa_data:  the question answer information
    :param t_data:   the time information
    :param mode:     train or not train
    :return:    loss,accuracy,auc
    '''

    #  用于对输入进来的q  data和 qa_data进行乱序   对于带有原先时间顺序的排除该操作， 因为t_data本身就是带有相应时间顺序的
    # shuffle_index = np.random.permutation(q_data.shape[0])
    # q_data_shuffled = q_data[shuffle_index]
    # qa_data_shuffled = qa_data[shuffle_index]

    # get the iteration number
    global bar
    training_step = q_data.shape[0] // args.batch_size

    # judge the args.show
    if args.show:
        bar = ProgressBar(mode, mmax=training_step)

    pred_list = list()
    label_list = list()
    for step in range(training_step):
        if args.show:
            bar.next()
        q_data_batch = q_data[step * args.batch_size:(step + 1) * args.batch_size, :]
        qa_data_batch = qa_data[step * args.batch_size:(step + 1) * args.batch_size, :]
        t_data_batch = t_data[step * args.batch_size:(step + 1) * args.batch_size, :]

        # qa : exercise index + answer(0 or 1)*exercies_number
        label = qa_data_batch[:, :]
        label = label.astype(np.int)
        label_batch = (label - 1) // args.n_questions  # convert to {-1, 0, 1}
        label_batch = label_batch.astype(np.float)

        feed_dict = {
            model.q_data: q_data_batch,
            model.qa_data: qa_data_batch,
            model.label: label_batch,
            model.t_data: t_data_batch
        }

        if mode == "train":
            pred_, _ = model.sess.run([model.pred, model.train_op], feed_dict=feed_dict)
        else:
            pred_ = model.sess.run([model.pred], feed_dict=feed_dict)

        label_flat = np.asarray(label_batch).reshape((-1))
        pred_flat = np.asarray(pred_).reshape((-1))
        # remove the wrong label and pred
        index_flat = np.flatnonzero(label_flat != -1.).tolist()

        label_list.append(label_flat[index_flat])
        pred_list.append(pred_flat[index_flat])

    if args.show:
        bar.finish()

    # np.concatenate
    all_label = np.concatenate(label_list, axis=0)
    # print("all_label has the nan ? === >  {}".format(np.isnan(all_label).any()))
    all_pred = np.concatenate(pred_list, axis=0)
    #过滤掉nan值
    all_pred = [0. if math.isnan(x) else x for x in all_pred]
    #将列表变换为np
    all_pred = np.array(all_pred)

    # compute the auc、acc、loss
    auc = compute_auc(all_label, all_pred)
    accuracy = compute_accuracy(all_label, all_pred)
    loss = binaryEntropy(all_label, all_pred)

    return loss, accuracy, auc
