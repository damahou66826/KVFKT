import argparse
import datetime
import logging
import numpy as np
import tensorflow as tf
import os
from dataLoad import DataLoader
from Ablation_experiment import model_exclude_F,model_exclude_D,model_exclude_G,model_exclude_IRT
from runModel import run_model
from utils import getLogger
from ablation_config import ModelConfigFactory


# argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='NeurIPS',
                    help="'assist2017','assist2012', 'Junyi', 'fsai','NeurIPS','EdNet'")

parser.add_argument('--save', type=bool, default=False)
parser.add_argument('--cpu', type=bool, default=False)
parser.add_argument('--n_epochs', type=int, default=None)
parser.add_argument('--batch_size', type=int, default=None)
parser.add_argument('--train', type=bool, default=None)
parser.add_argument('--show', type=bool, default=None)
parser.add_argument('--learning_rate', type=float, default=None)
parser.add_argument('--max_grad_norm', type=float, default=None)
parser.add_argument('--use_ogive_model', type=bool, default=False)

# parameter for the dataset
parser.add_argument('--seq_len', type=int, default=None)
# 问题个数
parser.add_argument('--n_questions', type=int, default=None)
parser.add_argument('--data_dir', type=str, default=None)
parser.add_argument('--data_name', type=str, default=None)

# parameter for the DKVMN model
parser.add_argument('--memory_size', type=int, default=None)
parser.add_argument('--key_memory_state_dim', type=int, default=None)
parser.add_argument('--value_memory_state_dim', type=int, default=None)
parser.add_argument('--forget_memory_state_dim', type=int, default=None)
parser.add_argument('--summary_vector_output_dim', type=int, default=None)
parser.add_argument('--forget_cycle', type=int, default=None)

# parameter for forget matrix
parser.add_argument('--max_random_time', type=int, default=None)
parser.add_argument('--min_random_time', type=int, default=None)

# ablation_experiment parameter
parser.add_argument('--ablation_experiment_parameter', type=str, default='D')

_args = parser.parse_args()
args = ModelConfigFactory.create_model_config(_args)

# set logger
logger = getLogger('ablation-KVFKT-model' + str(args.ablation_experiment_parameter))


logger.info("Model Config: {}".format(args))

# create directory
# 检查是否创建对应存储目录
for directory in [args.checkpoint_dir, args.result_log_dir, args.tensorboard_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)


def valid(model, valid_q_data, valid_qa_data, valid_t_data, result_log_path, args):
    saver = tf.train.Saver()
    best_loss = 1e6
    best_rmse = 1e6
    best_acc = 0.0
    best_auc = 0.0
    best_epoch = 0.0
    with open(result_log_path, 'w') as f:
        result_msg = "{},{},{},{},{}\n".format(
            'epoch', 'valid_auc', 'valid_accuracy', 'valid_rmse','valid_loss'
        )
        f.write(result_msg)

        # 测试集： 损失， 准确率 ，AUC
        valid_loss, valid_accuracy, valid_auc, valid_rmse = run_model(
            model, args, valid_q_data, valid_qa_data, valid_t_data, mode='valid'
        )

        # add to log
        # %\t 是对 \t进行转义
        msg = "\nValidation result:    AUC: {:.2f}%\t Acc: {:.2f}%\t RMSE: {:.2f}%\t Loss: {:.4f}".format(
            valid_auc * 100, valid_accuracy * 100, valid_rmse * 100, valid_loss
        )
        logger.info(msg)

        # write epoch result
        with open(result_log_path, 'a') as f:
            result_msg = "{},{},{},{},{}\n".format(
                1, valid_auc, valid_accuracy, valid_rmse, valid_loss
            )
            f.write(result_msg)

        # save the model if the loss is lower
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_acc = valid_accuracy
            best_auc = valid_auc
            best_rmse = valid_rmse
            best_epoch = 1  # 因为epoch从0开始的

        msg = "\nValidation result best result: AUC: {:.2f}\t Accuracy: {:.2f}\t RMSE: {:.2f}\t Loss: {:.4f}".format(
             best_auc * 100, best_acc * 100, best_rmse * 100, best_loss
        )
        logger.info(msg)
        logger.info("===============================================next cross validation========================================================")
        return best_auc, best_acc, best_rmse, best_loss


def train(model, train_q_data, train_qa_data, train_t_data, args):
    saver = tf.train.Saver()
    best_loss = 1e6
    best_rmse = 1e6
    best_acc = 0.0
    best_auc = 0.0
    best_epoch = 0.0

    for epoch in range(args.n_epochs):
        # 核心步骤， 跑模型   拿结果
        # 训练集： 损失 ， 准确率 ， AUC
        train_loss, train_accuracy, train_auc, train_rmse = run_model(
            model, args, train_q_data, train_qa_data, train_t_data, mode='train'
        )

        msg = "\n[Epoch {}/{}] Training result:      AUC: {:.2f}%\t Acc: {:.2f}%\t RMSE: {:.2f}%\t Loss: {:.4f}".format(
            epoch + 1, args.n_epochs, train_auc * 100, train_accuracy * 100, train_rmse * 100, train_loss
        )
        logger.info(msg)

        # add to tensorboard
        # 允许训练程序调用方法直接从训练循环中将数据添加到文件中，而不会减慢训练的速度
        tf_summary = tf.Summary(
            value=[
                tf.Summary.Value(tag="train_loss", simple_value=train_loss),
                tf.Summary.Value(tag="train_auc", simple_value=train_auc),
                tf.Summary.Value(tag="train_accuracy", simple_value=train_accuracy),
                tf.Summary.Value(tag="train_rmse", simple_value=train_rmse)
            ]
        )
        model.tensorboard_writer.add_summary(tf_summary, epoch)

        # save the model if the loss is lower
        if train_loss < best_loss:
            best_loss = train_loss
            best_acc = train_accuracy
            best_auc = train_auc
            best_rmse = train_rmse
            best_epoch = epoch + 1  # 因为epoch从0开始的

            if args.save:
                # 从参数中读取是否保存每一次训练后的模型
                model_dir = "ep{:03d}-auc{:.0f}-acc{:.0f}".format(
                    epoch + 1, train_auc * 100, train_accuracy * 100,
                )
                model_name = "KVFKT"
                save_path = os.path.join(args.checkpoint_dir, model_dir, model_name)
                # 保存模型
                saver.save(sess=model.sess, save_path=save_path)

                logger.info("Model improved. Save model to {}".format(save_path))
            else:
                logger.info("Model improved.")



    # print out the final result
    msg = "Best result at epoch {}: AUC: {:.2f}\t Accuracy: {:.2f}\t RMSE: {:.2f}\t Loss: {:.4f}".format(
        best_epoch, best_auc * 100, best_acc * 100, best_rmse * 100, best_loss
    )
    logger.info(msg)
    return model


def cross_validation():
    # tf.set_random_seed(1234)   deprecated
    tf.compat.v1.set_random_seed(1234)
    # config = tf.ConfigProto()   deprecated
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    aucs, accs, rmses, losses = list(), list(), list(), list()
    # 跑5组对应的训练集与验证集    取平均 (5-fold cross_validation)
    for i in range(5):  # 默认为5
        # 每次都会重新清除已有的node
        tf.compat.v1.reset_default_graph()
        logger.info("Cross Validation {}".format(i + 1))
        result_csv_path = os.path.join(args.result_log_dir, 'fold-{}-result'.format(i) + '.csv')

        # 创建一个tensorflow容器
        with tf.Session(config=config) as sess:
            data_loader = DataLoader(args.n_questions, args.seq_len, ',')  # 拿到数据加载器
            model = getModel(sess)
            sess.run(tf.global_variables_initializer())  # 开始跑  初始化参数
            if args.train:
                for j in range(5):
                    # 正好抽出第i组来做验证
                    if i == j:
                        continue
                    # 每次加载第i组数据进行训练
                    logger.info("training the {}th dataset".format(j))
                    train_data_path = os.path.join(args.data_dir, args.data_name + '_{}.csv'.format(j))
                    logger.info("Reading {}".format(train_data_path))
                    train_q_data, train_qa_data, train_t_data = data_loader.load_data(train_data_path)
                    model = train(
                        # 核心逻辑！！！！
                        model,
                        train_q_data, train_qa_data, train_t_data,
                        args=args
                    )

            valid_data_path = os.path.join(args.data_dir, args.data_name + '_{}.csv'.format(i))
            valid_q_data, valid_qa_data, valid_t_data = data_loader.load_data(valid_data_path)
            # 拿余下的一组数据进行验证
            auc, acc, rmse, loss = valid(model, valid_q_data, valid_qa_data, valid_t_data,
                                       result_log_path=result_csv_path, args=args)
            aucs.append(auc)
            accs.append(acc)
            losses.append(loss)
            rmses.append(rmse)



    cross_validation_msg = "Cross Validation Result:\n"
    cross_validation_msg += "AUC: {:.2f} +/- {:.2f}\n".format(np.average(aucs) * 100, np.std(aucs) * 100)
    cross_validation_msg += "Accuracy: {:.2f} +/- {:.2f}\n".format(np.average(accs) * 100, np.std(accs) * 100)
    cross_validation_msg += "RMSE: {:.2f} +/- {:.2f}\n".format(np.average(rmses) * 100, np.std(rmses) * 100)
    cross_validation_msg += "Loss: {:.2f} +/- {:.2f}\n".format(np.average(losses), np.std(losses))
    logger.info(cross_validation_msg)

    # write result
    result_msg = writeResult(aucs, accs, rmses, losses)
    with open('results/all_result.csv', 'a') as f:
        f.write(result_msg)


def getModel(sess):
    tempStr = args.ablation_experiment_parameter
    if tempStr == 'D':
        return model_exclude_D.KVFKTModelExcludeDifficult(args, sess)
    elif tempStr == 'F':
        return model_exclude_F.KVFKTModelExcludeForget(args, sess)
    elif tempStr == 'G':
        return model_exclude_G.KVFKTModelExcludeGuess(args, sess)
    elif tempStr == 'IRT':
        return model_exclude_IRT.KVFKTModelExcludeIRT(args, sess)


def writeResult(aucs, accs, rmse, losses):
    result_msg = datetime.datetime.now().strftime("%Y-%m-%dT%H%M") + ','
    result_msg += str(args.dataset) + ','
    result_msg += str(args.memory_size) + ','
    result_msg += str(args.key_memory_state_dim) + ','
    result_msg += str(args.value_memory_state_dim) + ','
    result_msg += str(args.summary_vector_output_dim) + ','
    result_msg += str(args.forget_cycle) + ','
    result_msg += str(args.learning_rate) + ','
    result_msg += str(np.average(aucs) * 100) + ','
    result_msg += str(np.std(aucs) * 100) + ','
    result_msg += str(np.average(accs) * 100) + ','
    result_msg += str(np.std(accs) * 100) + ','
    result_msg += str(np.average(rmse) * 100) + ','
    result_msg += str(np.std(rmse) * 100) + ','
    result_msg += str(np.average(losses)) + ','
    result_msg += str(np.std(losses)) + '\n'
    return result_msg


if __name__ == '__main__':
    cross_validation()
