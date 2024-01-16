from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler
from torch.optim import SGD
import gc
import torch
from sklearn.metrics import accuracy_score
import numpy as np
from RNN_net import RNN
import copy
import math
GLOVE_DIM = 300

class Learner(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, training_size):
        """
        :param args:
        """
        super(Learner, self).__init__()
        self.args = args
        self.data=args.data
        self.outer_batch_size = args.outer_batch_size
        self.inner_batch_size = args.inner_batch_size
        self.outer_update_lr = args.outer_update_lr
        self.old_outer_update_lr = args.outer_update_lr
        self.inner_update_lr = args.inner_update_lr
        self.inner_update_step = args.inner_update_step
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.collate_pad_ = self.collate_pad if args.data=='news_data' else self.collate_pad_snli
        self.training_size = training_size
        self.interval = args.interval
        if args.data == 'news_data':
            self.meta_model = RNN(
                word_embed_dim=args.word_embed_dim,
                encoder_dim=args.encoder_dim,
                n_enc_layers=args.n_enc_layers,
                dpout_model=0.0,
                dpout_fc=0.0,
                fc_dim=args.fc_dim,
                n_classes=args.n_classes,
                pool_type=args.pool_type,
                linear_fc=args.linear_fc
            )
        if args.data == 'snli' or 'sentment140':
            self.inner_model = RNN(
                word_embed_dim=args.word_embed_dim,
                encoder_dim=args.encoder_dim,
                n_enc_layers=args.n_enc_layers,
                dpout_model=0.0,
                dpout_fc=0.0,
                fc_dim=args.fc_dim,
                n_classes=args.n_classes,
                pool_type=args.pool_type,
                linear_fc=args.linear_fc
            )
        self.lambda_x =  torch.ones((self.training_size)).to(self.device)
        self.lambda_x.requires_grad=True
        param_count = 0
        for param in self.inner_model.parameters():
            param_count += param.numel()
        self.z_params = torch.randn(param_count, 1)
        self.z_params = nn.init.xavier_uniform_(self.z_params).to(self.device)
        self.hyper_momentum = torch.zeros(self.training_size).to(self.device)
        self.outer_optimizer = SGD([self.lambda_x], lr=self.outer_update_lr)
        self.inner_optimizer = SGD(self.inner_model.parameters(), lr=self.inner_update_lr)
        self.inner_stepLR = torch.optim.lr_scheduler.StepLR(self.inner_optimizer, step_size=args.epoch, gamma=0.2)
        self.outer_stepLR = torch.optim.lr_scheduler.StepLR(self.outer_optimizer, step_size=args.epoch, gamma=0.2)
        self.inner_model.train()
        self.beta = args.beta
        self.nu = args.nu
        self.y_warm_start = args.y_warm_start
        self.normalized = args.grad_normalized
        self.criterion = nn.CrossEntropyLoss(reduction='none').to(self.device)

    def forward(self, train_loader, training=True, task_id=0):
        task_accs = []
        task_loss = []
        self.inner_model.to(self.device)
        for step, data in enumerate(train_loader):
            num_inner_update_step = self.y_warm_start if step%self.interval==0  else self.inner_update_step
            all_loss = []

            input, label_id, data_indx = next(iter(train_loader))
            outputs = predict(self.inner_model, input)
            inner_loss = torch.mean(torch.sigmoid(self.lambda_x[data_indx]) * self.criterion(outputs, label_id.to(
                self.device))) + 0.0001 * sum(
                [x.norm().pow(2) for x in self.inner_model.parameters()]).sqrt()
            inner_loss.backward(retain_graph=True, create_graph=True)
            all_loss.append(inner_loss.item())
            g_grad = [g_param.grad.view(-1) for g_param in self.inner_model.parameters()]
            g_grad_flat = torch.unsqueeze(torch.reshape(torch.hstack(g_grad), [-1]), 1)

            jacob = torch.autograd.grad(g_grad_flat, self.inner_model.parameters(), grad_outputs=self.z_params)
            if step%self.interval == 0:
                self.inner_optimizer.step()
                for i in range(0, num_inner_update_step):
                    input, label_id, data_indx = next(iter(train_loader))
                    outputs = predict(self.inner_model, input)
                    inner_loss = torch.mean(torch.sigmoid(self.lambda_x[data_indx])*self.criterion(outputs, label_id.to(self.device))) + 0.0001 * sum(
                        [x.norm().pow(2) for x in self.inner_model.parameters()]).sqrt()
                    inner_loss.backward()
                    self.inner_optimizer.step()

            self.inner_optimizer.zero_grad()
            jacob = [j_param.detach().view(-1) for j_param in jacob]
            jacob_flat = torch.unsqueeze(torch.reshape(torch.hstack(jacob), [-1]), 1)
            q_input, q_label_id, q_data_indx = data
            q_outputs = predict(self.inner_model, q_input)
            q_loss = torch.mean(self.criterion(q_outputs, q_label_id.to(self.device)))
            valid_batch = next(iter(train_loader))
            self.hypergradient(self.args, jacob_flat, q_loss, valid_batch)
            self.lambda_x.grad = self.hyper_momentum[0]
            grad_l2_norm_sq = 0.0

            grad_l2_norm_sq += torch.sum(self.lambda_x.grad* self.lambda_x.grad)
            grad_l2_norm = torch.sqrt(grad_l2_norm_sq).item()
            momentum_coeff = 1.0 / (1e-10 + grad_l2_norm)
            old_out_lr = copy.copy(self.outer_optimizer.param_groups[0]['lr'])
            if self.normalized:
                # print('---------- grad normalized --------')
                self.outer_optimizer.param_groups[0][ 'lr'] = old_out_lr * momentum_coeff

            self.outer_optimizer.step()
            self.outer_optimizer.zero_grad()

            self.outer_optimizer.param_groups[0][ 'lr'] = old_out_lr
            q_logits = F.softmax(q_outputs, dim=1)
            pre_label_id = torch.argmax(q_logits, dim=1)
            pre_label_id = pre_label_id.detach().cpu().numpy().tolist()
            q_label_id = q_label_id.detach().cpu().numpy().tolist()
            self.outer_optimizer.zero_grad()
            acc = accuracy_score(pre_label_id, q_label_id)
            task_accs.append(acc)
            task_loss.append(q_loss.detach().cpu())
            torch.cuda.empty_cache()
            print(f'Task loss: {np.mean(task_loss):.4f}, Task acc: {np.mean(task_accs):.4f}')
        self.inner_stepLR.step()
        self.outer_stepLR.step()
        return np.mean(task_accs),  np.mean(task_loss)

    def test(self, test_loader):
        task_accs = []
        task_loss = []

        self.inner_model.to(self.device)
        for step, data in enumerate(test_loader):
            q_input, q_label_id, q_data_indx = data
            q_outputs = predict(self.inner_model, q_input)
            q_loss = torch.mean(self.criterion(q_outputs, q_label_id.to(self.device)))

            q_logits = F.softmax(q_outputs, dim=1)
            pre_label_id = torch.argmax(q_logits, dim=1)
            pre_label_id = pre_label_id.detach().cpu().numpy().tolist()
            q_label_id = q_label_id.detach().cpu().numpy().tolist()
            acc = accuracy_score(pre_label_id, q_label_id)
            task_accs.append(acc)
            task_loss.append(q_loss.detach().cpu())
            torch.cuda.empty_cache()
            print(f'Task loss: {np.mean(task_loss):.4f}, Task acc: {np.mean(task_accs):.4f}')
        return np.mean(task_accs),  np.mean(task_loss)

    def hypergradient(self, args, jacob_flat, loss, query_batch):
        val_data, val_labels, data_idx = query_batch
        loss.backward()

        Fy_gradient = [g_param.grad.detach().view(-1) for g_param in self.inner_model.parameters()]
        Fy_gradient_flat = torch.unsqueeze(torch.reshape(torch.hstack(Fy_gradient), [-1]), 1)
        self.z_params -= args.nu * (jacob_flat - Fy_gradient_flat)

        # Gyx_gradient
        output = predict(self.inner_model, val_data)
        loss = torch.mean(
            torch.sigmoid(self.lambda_x[data_idx]) * F.cross_entropy(output, val_labels.cuda(), reduction='none')) + 0.0001 * sum(
            [x.norm().pow(2) for x in self.inner_model.parameters()]).sqrt()
        Gy_gradient = torch.autograd.grad(loss, self.inner_model.parameters(), retain_graph=True, create_graph=True)
        Gy_params = [Gy_param.view(-1) for Gy_param in Gy_gradient]
        Gy_gradient_flat = torch.reshape(torch.hstack(Gy_params), [-1])
        Gyxz_gradient = torch.autograd.grad(-torch.matmul(Gy_gradient_flat, self.z_params.detach()), self.lambda_x)
        self.hyper_momentum = [args.beta * h + (1 - args.beta) *  Gyxz for (h, Gyxz) in
                          zip(self.hyper_momentum,  Gyxz_gradient)]


    def collate_pad(self, data_points):
        """ Pad data points with zeros to fit length of longest data point in batch. """
        s_embeds = data_points[0] if type(data_points[0]) == list or type(data_points[0]) == tuple else data_points[1]
        targets = data_points[1] if type(data_points[0]) == list or type(data_points[0]) == tuple  else data_points[0]

        # Get sentences for batch and their lengths.
        s_lens = np.array([sent.shape[0] for sent in s_embeds])
        max_s_len = np.max(s_lens)
        # Encode sentences as glove vectors.
        bs = len(data_points[0])
        s_embed = np.zeros((max_s_len, bs, GLOVE_DIM))
        for i in range(bs):
            e = s_embeds[i]
            if len(e) <= 0:
                s_lens[i] = 1
            s_embed[: len(e), i] = e.copy()
        embeds = torch.from_numpy(s_embed).float().to(self.device)
        targets = torch.LongTensor(targets).to(self.device)
        return (embeds, s_lens), targets

    def collate_pad_snli(self, data_points):
        """ Pad data points with zeros to fit length of longest data point in batch. """
        s_embeds = data_points[0] if type(data_points[0]) == list or type(data_points[0]) == tuple else data_points[1]

        targets = data_points[1] if type(data_points[0]) == list or type(data_points[0]) == tuple else data_points[0]

        s1_embeds = [x for x in s_embeds[0]]
        s2_embeds = [x for x in s_embeds[1]]

        # Get sentences for batch and their lengths.
        s1_lens = np.array([sent.shape[0] for sent in s1_embeds])
        max_s1_len = np.max(s1_lens)
        s2_lens = np.array([sent.shape[0] for sent in s2_embeds])
        max_s2_len = np.max(s2_lens)
        lens = (s1_lens, s2_lens)

        # Encode sentences as glove vectors.
        bs = len(targets)
        s1_embed = np.zeros((max_s1_len, bs, GLOVE_DIM))
        s2_embed = np.zeros((max_s2_len, bs, GLOVE_DIM))
        for i in range(bs):
            e1 = s1_embeds[i]
            e2 = s2_embeds[i]
            s1_embed[: len(e1), i] = e1.copy()
            s2_embed[: len(e2), i] = e2.copy()
            if len(e1) <= 0:
                s1_lens[i] = 1
            if len(e2) <= 0:
                s2_lens[i] = 1
        embeds = (
            torch.from_numpy(s1_embed).float().to(self.device), torch.from_numpy(s2_embed).float().to(self.device)
        )

        # Convert targets to tensor.
        targets = torch.LongTensor(targets).to(self.device)

        return (embeds, lens), targets

def predict(net, inputs):
    """ Get predictions for a single batch. """
    # snli dataaset
    # (s1_embed, s2_embed), (s1_lens, s2_lens) = inputs
    # outputs = net((s1_embed.cuda(), s1_lens), (s2_embed.cuda(), s2_lens))
    s_embed, s_lens = inputs
    outputs = net((s_embed.cuda(), s_lens))
    return outputs



