import json

import torch
import time
import logging
import argparse
import os
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

from bo_rep import Learner
from data_loader import SNLIDataset, Sent140Dataset,  collate_pad
from torch.utils.data import DataLoader

import random
import numpy as np
torch.backends.cudnn.enabled = False
def random_seed(value):
    torch.backends.cudnn.deterministic=True
    torch.manual_seed(value)
    torch.cuda.manual_seed(value)
    np.random.seed(value)
    random.seed(value)

def main():
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", default='sentment140', type=str,
                        help="dataset: [news_data, snli, sentment140]", )

    parser.add_argument("--data_path", default='../data/news-data/dataset.json', type=str,
                        help="Path to dataset file")

    parser.add_argument("--batch_size", default=512, type=int,
                        help="batch_size", )

    parser.add_argument("--save_direct", default='logs/', type=str,
                        help="Path to save file")

    parser.add_argument("--word2vec", default="data/news-data/wordvec.pkl", type=str,
                        help="Path to word2vec file")


    parser.add_argument("--num_labels", default=2, type=int,
                        help="Number of class for classification")

    parser.add_argument("--epoch", default=25, type=int,
                        help="Number of outer interation")


    parser.add_argument("--outer_batch_size", default=20, type=int,
                        help="Batch of task size")
    
    parser.add_argument("--inner_batch_size", default=50, type=int,
                        help="Training batch size in inner iteration")


    parser.add_argument("--outer_update_lr", default= 5e-2, type=float,
                        help="Meta learning rate")

    parser.add_argument("--inner_update_lr", default=5e-2, type=float,
                        help="Inner update learning rate")
    
    parser.add_argument("--inner_update_step", default=1, type=int,
                        help="Number of interation in the inner loop during train time")

    parser.add_argument("--outer_update_step", default=1, type=int,
                        help="Number of interation in the outer loop during train time")
    
    parser.add_argument("--num_task_test", default=50, type=int,
                        help="Total number of tasks for testing")

    parser.add_argument("--grad_normalized", default=True, type=bool,
                        help="whether grad normalized or not")


    parser.add_argument("--seed", default=2, type=int,
                        help="random seed")

    # single loops parameters
    parser.add_argument("--beta", default=0.90, type=float,
                        help="momentum parameters")

    parser.add_argument("--nu", default=1e-2, type=float,
                        help="learning rate of z")


    parser.add_argument("--y_warm_start", default=10, type=int,
                        help="update steps for y")

    parser.add_argument("--interval", default=2, type=int,
                        help="update interval for y")


    # RNN hyperparameter settings
    parser.add_argument("--word_embed_dim", default=300, type=int,
                        help="word embedding dimensions")

    parser.add_argument("--encoder_dim", default=4096, type=int,
                        help="encodding dimensions")

    parser.add_argument("--n_enc_layers", default=2, type=int,
                        help="encoding layers")

    parser.add_argument("--fc_dim", default=1024, type=int,
                        help="dimension of fully-connected layer")

    parser.add_argument("--n_classes", default=2, type=int,
                        help="classes of targets")

    parser.add_argument("--linear_fc", default=False, type=bool,
                        help="classes of targets")

    parser.add_argument("--pool_type", default="max", type=str,
                        help="type of pooling")

    parser.add_argument("--noise_rate", default=0.3, type=float,
                        help="rate for label noise")

    args = parser.parse_args()
    random_seed(args.seed)
    if args.data == 'sentment140':
        train = Sent140Dataset("../data", "train", noise_rate=args.noise_rate)
        test = Sent140Dataset("../data", "test")
        args.n_labels = 2
        args.n_classes = 2
        training_size = train.__len__()
    else:
        print('Do not support this data')

    st = time.time()
    learner = Learner(args, training_size)
    acc_all_test = []
    loss_all_test = []

    for epoch in range(args.epoch):
        print(f"[epoch/epochs]:{epoch}/{args.epoch}")
        train_loader = DataLoader(train, shuffle=True, batch_size=args.batch_size, collate_fn=collate_pad)
        test_loader = DataLoader(test, batch_size=args.batch_size, collate_fn=collate_pad)
        for j in range(args.outer_update_step):
            print(f'--------- outer step {j} -----------')
            learner(train_loader, training=True, task_id=epoch)

        print("---------- Testing Mode -------------")
        acc, loss = learner.test(test_loader)
        acc_all_test.append(acc)
        loss_all_test.append(loss)

        print('Test loss:', loss_all_test)
        print('Test Acc:', acc_all_test)

    date = time.strftime('%Y-%m-%d-%H-%M',time.localtime(time.time()))
    file_name = f'{args.methods}_outlr{args.outer_update_lr}_inlr{args.inner_update_lr}_seed{args.seed}_{date}'

    if not os.path.exists('logs/'+args.save_direct):
        os.mkdir('logs/'+args.save_direct)
    save_path = 'logs/'+args.save_direct
    total_time = (time.time() - st) / 3600
    files = open(os.path.join(save_path, file_name)+'.txt', 'w')
    files.write(str({'Exp configuration': str(args), 'AVG Test LOSS': str(loss_all_test),
               'AVG Test ACC': str(acc_all_test),'time': total_time}))
    files.close()
    torch.save((acc_all_test,  loss_all_test), os.path.join(save_path, file_name))
    print(f'time:{total_time} h')
if __name__ == "__main__":
    main()
