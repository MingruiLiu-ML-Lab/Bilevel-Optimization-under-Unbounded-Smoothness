import json
import torch
import time
import logging
import argparse
import os
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)
from bo_rep import Learner
from task import MetaTask
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

    parser.add_argument("--data_path", default='data/news-data/dataset.json', type=str,
                        help="Path to dataset file")


    parser.add_argument("--save_direct", default='soba', type=str,
                        help="Path to save file")

    parser.add_argument("--word2vec", default="data/news-data/wordvec.pkl", type=str,
                        help="Path to word2vec file")

    parser.add_argument("--num_labels", default=2, type=int,
                        help="Number of class for classification")

    parser.add_argument("--epoch", default=1, type=int,
                        help="Number of outer interation")
    
    parser.add_argument("--k_spt", default=20, type=int,
                        help="Number of support samples per task")
    
    parser.add_argument("--k_qry", default=20, type=int,
                        help="Number of query samples per task")

    parser.add_argument("--outer_batch_size", default=20, type=int,
                        help="Batch of task size")
    
    parser.add_argument("--inner_batch_size", default=50, type=int,
                        help="Training batch size in inner iteration")
    
    parser.add_argument("--outer_update_lr", default= 1e-2, type=float,
                        help="Meta learning rate")

    parser.add_argument("--inner_update_lr", default=1e-3, type=float,
                        help="Inner update learning rate")

    parser.add_argument("--hessian_q", default=5, type=int,
                        help="Q steps for hessian-inverse-vector product")

    parser.add_argument("--inner_update_step", default=1, type=int,
                        help="Number of interation in the inner loop during train time")

    parser.add_argument("--outer_update_step", default=1, type=int,
                        help="Number of interation in the outer loop during train time")
    
    parser.add_argument("--num_task_train", default=400, type=int,
                        help="Total number of meta tasks for training")
    
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

    parser.add_argument("--y_warm_start", default=3, type=int,
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


    args = parser.parse_args()
    random_seed(args.seed)
    if args.data == 'news_data':
        args.data_path = '../data/news-data/dataset.json'
        # args.data_path = 'data/news-data/dataset.json'
        reviews = json.load(open(args.data_path))
        low_resource_domains = ["office_products", "automotive", "computer_&_video_games"]
        train_examples = [r for r in reviews if r['domain'] not in low_resource_domains]
        test_examples = [r for r in reviews if r['domain'] in low_resource_domains]
        print(len(train_examples), len(test_examples))
        test = MetaTask(test_examples, num_task=args.num_task_test, k_support=args.k_spt,
                        k_query=args.k_qry, word2vec=args.word2vec)
        train = MetaTask(train_examples, num_task=args.num_task_train, k_support=args.k_spt,
                         k_query=args.k_qry, word2vec=args.word2vec)
    if args.data == 'snli':
        train = torch.load('../data/snli_1.0/SNLI/train_Data.pkl')
        test = torch.load('../data/snli_1.0/SNLI/test_Data.pkl')

    st = time.time()
    args.inner_batch_size = args.num_task_test
    learner = Learner(args)

    acc_all_test = []
    acc_loss_test = []

    for epoch in range(args.epoch):
        print(f"[epoch/epochs]:{epoch}/{args.epoch}")

        db = create_batch_of_tasks(train, is_shuffle = True, batch_size = args.outer_batch_size)

        for step, task_batch in enumerate(db):
            print(f'\n--------- Step {step} -----------')
            for j in range(args.outer_update_step):
                learner(task_batch, training = True, task_id = step)

            print("---------- Testing Mode -------------")
            db_test = create_batch_of_tasks(test, is_shuffle = False, batch_size = args.inner_batch_size)

            for test_batch in db_test:
                acc, loss = learner(test_batch, training = False,  task_id = step)
                acc_all_test.append(acc)
                acc_loss_test.append(loss)

            print('Step:', step, 'Test loss:', acc_loss_test)
            print('Step:', step, 'Test Acc:', acc_all_test)


    date = time.strftime('%Y-%m-%d-%H-%M',time.localtime(time.time()))
    file_name = f'meta_learning_{args.methods}_lr_{args.outer_update_lr}_seed{args.seed}_{date}'

    if not os.path.exists('logs/'+args.save_direct):
        os.mkdir('logs/'+args.save_direct)
    save_path = 'logs/'+args.save_direct
    total_time = (time.time() - st) / 3600
    files = open(os.path.join(save_path, file_name)+'.txt', 'w')
    files.write(str({'Exp configuration': str(args), 'AVG Test LOSS': str(acc_loss_test),
               'AVG Test ACC': str(acc_all_test), 'time': total_time}))
    files.close()
    torch.save((acc_all_test, acc_loss_test), os.path.join(save_path, file_name))
    print(f'time:{total_time} h')
if __name__ == "__main__":
    main()
