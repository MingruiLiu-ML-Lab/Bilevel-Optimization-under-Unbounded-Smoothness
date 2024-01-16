from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler
from torch.optim import SGD
import gc
import torch
from sklearn.metrics import accuracy_score
import numpy as np
from RNN_net import RNN
GLOVE_DIM = 300

class Learner(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args):
        """
        :param args:
        """
        super(Learner, self).__init__()
        self.args = args
        self.inner_batch_size = args.inner_batch_size
        self.outer_update_lr = args.outer_update_lr
        self.old_outer_update_lr = args.outer_update_lr
        self.inner_update_lr = args.inner_update_lr
        self.inner_update_step = args.inner_update_step
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.rnn_model = RNN(
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
        self.hyperparamter = torch.zeros(1).to(self.device)
        self.hyperparamter.requires_grad = True
        param_count = 0
        for param in self.rnn_model.parameters():
            param_count += param.numel()
        self.z_params = torch.randn(param_count, 1)
        self.z_params = nn.init.xavier_uniform_(self.z_params).to(self.device)

        self.hyper_momentum = torch.zeros(self.hyperparamter.size())
        self.outer_optimizer = SGD([self.hyperparamter], lr=self.outer_update_lr)
        self.inner_optimizer = SGD(self.rnn_model.parameters(), lr=self.inner_update_lr)
        self.stepLR = torch.optim.lr_scheduler.StepLR(self.inner_optimizer, step_size=20, gamma=0.8)
        self.interval = args.interval
        self.beta = args.beta
        self.nu = args.nu
        self.y_warm_start = args.y_warm_start
        self.normalized = args.grad_normalized
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def forward(self, batch_tasks, training=True, task_id=0):
        task_accs = []
        task_loss = []
        for step, task in enumerate(batch_tasks):
            self.inner_optimizer.param_groups[0][
                'lr'] = self.inner_update_lr if training else self.inner_update_lr * 10
            num_inner_update_step =  self.y_warm_start if step%self.interval==0  else self.inner_update_step
            support = task[0]
            query = task[1]
            self.rnn_model.to(self.device)
            support_dataloader = DataLoader(support, sampler=RandomSampler(support),
                                            batch_size=len(support), collate_fn=self.collate_pad_)
            all_loss = []

            if step % self.interval == 0 or not training:
                for i in range(0, num_inner_update_step):
                    for inner_step, batch in enumerate(support_dataloader):
                        input, label_id = batch
                        outputs = predict(self.rnn_model, input)
                        inner_loss = self.criterion(outputs, label_id) + self.hyperparamter/2 * sum(
                            [x.norm().pow(2) for x in self.rnn_model.parameters()]).sqrt()
                        inner_loss.backward()
                        self.inner_optimizer.step()

            if training:
                input, label_id = next(iter(support_dataloader))
                outputs = predict(self.rnn_model, input)
                inner_loss = self.criterion(outputs, label_id) + self.hyperparamter / 2 * sum(
                    [x.norm().pow(2) for x in self.rnn_model.parameters()]).sqrt()
                inner_loss.backward(retain_graph=True, create_graph=True)
                all_loss.append(inner_loss.item())
                g_grad = [g_param.grad.view(-1) for g_param in self.rnn_model.parameters()]
                g_grad_flat = torch.unsqueeze(torch.reshape(torch.hstack(g_grad), [-1]), 1)
                jacob = torch.autograd.grad(g_grad_flat, self.rnn_model.parameters(), grad_outputs=self.z_params)

            self.inner_optimizer.zero_grad()
            self.outer_optimizer.zero_grad()
            if training:
                jacob = [j_param.detach().view(-1) for j_param in jacob]
                jacob_flat = torch.unsqueeze(torch.reshape(torch.hstack(jacob), [-1]), 1)

            query_dataloader = DataLoader(query, sampler=None, batch_size=len(query), collate_fn=self.collate_pad_)
            query_batch = iter(query_dataloader).__next__()
            q_input, q_label_id = query_batch
            q_outputs = predict(self.rnn_model, q_input)
            q_loss = self.criterion(q_outputs, q_label_id)
            if training:
                self.hyper_momentum, self.z_params = hypergradient(self.args, jacob_flat, self.hyper_momentum, self.z_params, \
                                                                   q_loss, self.hyperparamter, self.rnn_model, query_batch)
                self.hyperparamter.grad = self.hyper_momentum[0]
                if self.normalized:
                    momentum_coeff = 1.0 / (1e-10 + self.hyper_momentum[0].norm())
                    self.outer_optimizer.param_groups[0]['lr'] = self.outer_update_lr * momentum_coeff.item()
                self.outer_optimizer.step()
                self.outer_optimizer.zero_grad()
            self.outer_optimizer.param_groups[0]['lr'] = self.old_outer_update_lr
            q_logits = F.softmax(q_outputs, dim=1)
            pre_label_id = torch.argmax(q_logits, dim=1)
            pre_label_id = pre_label_id.detach().cpu().numpy().tolist()
            q_label_id = q_label_id.detach().cpu().numpy().tolist()
            acc = accuracy_score(pre_label_id, q_label_id)
            task_accs.append(acc)
            task_loss.append(q_loss.detach().cpu())
            torch.cuda.empty_cache()
            print(f'Task loss: {np.mean(task_loss):.4f}')
        self.stepLR.step()
        return np.mean(task_accs),  np.mean(task_loss)

    def collate_pad_(self, data_points):
        """ Pad data points with zeros to fit length of longest data point in batch. """
        s_embeds = data_points[0] if type(data_points[0]) == list else data_points[1]
        targets = data_points[1] if type(data_points[0]) == list else data_points[0]

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


def predict(net, inputs):
    """ Get predictions for a single batch. """
    s_embed, s_lens = inputs
    outputs = net((s_embed, s_lens))
    return outputs


def hypergradient(args, jacob_flat, hyper_momentum, z_params, loss, hyperparameter, rnn_model, query_batch):
    val_data, val_labels = query_batch
    loss.backward()

    Fy_gradient = [g_param.grad.detach().view(-1) for g_param in rnn_model.parameters()]
    Fy_gradient_flat = torch.unsqueeze(torch.reshape(torch.hstack(Fy_gradient), [-1]), 1)
    z_params -= args.nu * (jacob_flat - Fy_gradient_flat)

    # Gyx_gradient
    output = predict(rnn_model, val_data)
    loss = F.cross_entropy(output, val_labels) + hyperparameter/2 * sum(
        [x.norm().pow(2) for x in rnn_model.parameters()]).sqrt()
    Gy_gradient = torch.autograd.grad(loss, rnn_model.parameters(), retain_graph=True, create_graph=True)
    Gy_params = [Gy_param.view(-1) for Gy_param in Gy_gradient]
    Gy_gradient_flat = torch.reshape(torch.hstack(Gy_params), [-1])
    Gyxz_gradient = torch.autograd.grad(-torch.matmul(Gy_gradient_flat, z_params.detach()), hyperparameter)
    hyper_momentum = [args.beta * h + (1-args.beta) *  Gyxz for (h,  Gyxz) in zip(hyper_momentum, Gyxz_gradient)]

    return hyper_momentum, z_params

