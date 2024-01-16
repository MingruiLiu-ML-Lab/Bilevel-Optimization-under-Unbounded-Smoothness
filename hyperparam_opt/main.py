import json
import torch
import time
import logging
import argparse
import os
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from bo_rep import Learner
from data_processing import data_loader
import random
import numpy as np
torch.backends.cudnn.enabled = False
def random_seed(value):
    torch.backends.cudnn.deterministic=True
    torch.manual_seed(value)
    torch.cuda.manual_seed(value)
    np.random.seed(value)
    random.seed(value)

def create_batch_of_tasks(taskset, is_shuffle = True, batch_size = 4):
    idxs = list(range(0,len(taskset)))
    if is_shuffle:
        random.shuffle(idxs)
    for i in range(0,len(idxs), batch_size):
        yield [taskset[idxs[i]] for i in range(i, min(i + batch_size,len(taskset)))]

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data", default='news_data', type=str,
                        help="dataset: [news_data, snli]", )

    parser.add_argument("--data_path", default='../data/news-data/dataset.json', type=str,
                        help="Path to dataset file")

    parser.add_argument("--save_direct", default='logs/', type=str,
                        help="Path to save file")

    parser.add_argument("--word2vec", default="../data/news-data/wordvec.pkl", type=str,
                        help="Path to word2vec file")

    parser.add_argument("--num_labels", default=2, type=int,
                        help="Number of class for classification")

    parser.add_argument("--epoch", default=1, type=int,
                        help="Number of outer interation")

    parser.add_argument("--k_spt", default=50, type=int,
                        help="Number of support samples per task")

    parser.add_argument("--k_qry", default=50, type=int,
                        help="Number of query samples per task")

    parser.add_argument("--outer_update_lr", default= 1e-4, type=float,
                        help="Meta learning rate")

    parser.add_argument("--outer_batch_size", default= 10, type=int,
                        help="batch size for tasks")

    parser.add_argument("--inner_batch_size", default= 50, type=int,
                        help="batch size for tasks")


    parser.add_argument("--inner_update_lr", default=2e-3, type=float,
                        help="Inner update learning rate")

    parser.add_argument("--inner_update_step", default=1, type=int,
                        help="Number of interation in the inner loop during train time")

    parser.add_argument("--outer_update_step", default=1, type=int,
                        help="Number of interation in the outer loop during train time")

    parser.add_argument("--inner_update_step_eval", default=40, type=int,
                        help="Number of interation in the inner loop during test time")

    parser.add_argument("--num_task_train", default=200, type=int,
                        help="Total number of meta tasks for training")

    parser.add_argument("--num_task_test", default=50, type=int,
                        help="Total number of tasks for testing")

    parser.add_argument("--grad_clip", default=False, type=bool,
                        help="whether grad clipping or not")

    parser.add_argument("--grad_normalized", default=True, type=bool,
                        help="whether grad normalized or not")

    parser.add_argument("--seed", default=0, type=int,
                        help="random seed")

    # single loops parameters
    parser.add_argument("--beta", default=0.90, type=float,
                        help="momentum parameters")

    parser.add_argument("--nu", default=1e-2, type=float,
                        help="learning rate of z")

    parser.add_argument("--y_warm_start", default=3, type=int,
                        help="update steps for y")

    parser.add_argument("--interval", default=3, type=int,
                        help="interval for updating y")


    # RNN hyperparameter settings
    parser.add_argument("--word_embed_dim", default=300, type=int,
                        help="word embedding dimensions")

    parser.add_argument("--encoder_dim", default=2048, type=int,
                        help="encodding dimensions")

    parser.add_argument("--n_enc_layers", default=2, type=int,
                        help="encoding layers")

    parser.add_argument("--fc_dim", default=512, type=int,
                        help="dimension of fully-connected layer")

    parser.add_argument("--n_classes", default=2, type=int,
                        help="classes of targets")

    parser.add_argument("--linear_fc", default=False, type=bool,
                        help="classes of targets")

    parser.add_argument("--pool_type", default="max", type=str,
                        help="type of pooling")


    args = parser.parse_args()
    random_seed(args.seed)
    if args.data == 'news_data':
        args.data_path = '../data/news-data/dataset.json'
        reviews = json.load(open(args.data_path))
        low_resource_domains = ["office_products", "automotive", "computer_&_video_games",
        'outdoor_living','software',  'sports_&_outdoors','toys_&_games','video']
        train_examples = [r for r in reviews if r['domain'] not in low_resource_domains]
        test_examples = [r for r in reviews if r['domain'] in low_resource_domains]
        print(len(train_examples), len(test_examples))
    st = time.time()
    learner = Learner(args)

    test_data = data_loader(test_examples, num_task = args.num_task_test, k_support=args.k_spt,
                    k_query=args.k_qry, word2vec=args.word2vec)
    train_data = data_loader(train_examples, num_task = args.num_task_train, k_support=args.k_spt,
                     k_query=args.k_qry, word2vec=args.word2vec)

    global_step = 0
    acc_all_test = []
    loss_all_test = []

    for epoch in range(args.epoch):
        print(f"[epoch/epochs]:{epoch}/{args.epoch}")
        db = create_batch_of_tasks(train_data, is_shuffle=True, batch_size=args.outer_batch_size)
        print(f'\n--------- Epoch {epoch} -----------')
        for step, train_d in enumerate(db):
            learner(train_d,  training = True, task_id = step)

            print("---------- Testing Mode -------------")
            acc, loss = learner(test_data, training = False,  task_id = epoch)
            acc_all_test.append(acc)
            loss_all_test.append(loss)

            print('Step:', step, 'Test loss:', loss_all_test)
            print('Step:', step, 'Test Acc:', acc_all_test)
            global_step += 1

    date = time.strftime('%Y-%m-%d-%H-%M',time.localtime(time.time()))
    file_name = f'meta_learning_{args.methods}_lr{args.inner_update_lr}_outlr{args.outer_update_lr}_seed{args.seed}_{date}'

    if not os.path.exists('logs/'+args.save_direct):
        os.mkdir('logs/'+args.save_direct)
    save_path = 'logs/'+args.save_direct

    total_time = (time.time() - st) / 3600
    files = open(os.path.join(save_path, file_name)+'.txt', 'w')
    files.write(str({'Exp configuration': str(args), 'AVG Test LOSS': str(loss_all_test),
                     'AVG Test ACC': str(acc_all_test),  'time': total_time}))
    files.close()
    torch.save((acc_all_test, loss_all_test), os.path.join(save_path, file_name))
    print(f'time:{total_time} h')
if __name__ == "__main__":
    main()
