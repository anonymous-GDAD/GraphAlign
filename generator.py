import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import match_loss, regularization, row_normalize_tensor
from utils_graphsaint import DataGraphSAINT
import deeprobust.graph.utils as utils
from models.gcn import GCN
from models.sgc import SGC
from models.sgc_multi import SGC as SGC1
from models.parametrized_adj import PGE
from torch_sparse import SparseTensor
from tqdm import tqdm, trange
from collections import Counter
import time
from dual_gnn.models.augmentation import *
import os
import matplotlib.pyplot as plt
from torch_geometric.nn.inits import uniform
import scipy.sparse as sp
class Discriminator(nn.Module):
    def __init__(self, hidden_dim):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        size = self.weight.size(0)
        uniform(size, self.weight)

    def forward(self, x, summary):
        h = torch.matmul(summary, self.weight).reshape(-1, 1)
        return torch.sum(x * h, dim=1)



class MyGenerator:
    def __init__(self, data: DataGraphSAINT, args, device='cuda', **kwargs):
        self.data = data
        self.args = args
        self.device = device
        self.dataset = args.dataset

        n = int(len(data.idx_train) * args.reduction_rate)
        d = data.feat_train.shape[1]
        self.nnodes_syn = n  # 'syn' means 'synthetic'.
        self.feat_syn = nn.Parameter(torch.FloatTensor(n, d).to(device))

        # 'pge' means 'parameterized graph embedding'.
        self.pge = PGE(nfeat=d, nnodes=n, device=device, args=args).to(device)
        self.num_class_dict = dict()
        self.syn_class_indices = dict()
        self.labels_syn = torch.LongTensor(self.generate_labels_syn(data)).to(device)
        self.reset_parameters()
        self.optimizer_feat = torch.optim.Adam([self.feat_syn], lr=args.lr_feat)
        self.optimizer_pge = torch.optim.Adam(self.pge.parameters(), lr=args.lr_adj)

        self.max_performance = None
        self.max_adj_syn = None
        self.max_feat_syn = None
        self.max_labels_syn = None
        print('adj_syn:', (n, n), 'feat_syn:', self.feat_syn.shape)

    def reset_parameters(self):
        self.feat_syn.data.copy_(torch.randn(self.feat_syn.size()))

    def generate_labels_syn(self, data: DataGraphSAINT) -> list:
        """The distribution of the synthetic labels is (almost) the same as the original label distribution.
        """
        counter = Counter(data.labels_train)
        n = len(data.labels_train)

        sorted_counter = sorted(counter.items(), key=lambda x: x[1])  # descending order.
        sum_ = 0
        labels_syn = []

        for ix, (c, num) in enumerate(sorted_counter):
            if ix == len(sorted_counter) - 1:
                self.num_class_dict[c] = int(n * self.args.reduction_rate) - sum_
                self.syn_class_indices[c] = [len(labels_syn), len(labels_syn) + self.num_class_dict[c]]
                labels_syn += [c] * self.num_class_dict[c]
            else:
                self.num_class_dict[c] = max(int(num * self.args.reduction_rate), 1)
                sum_ += self.num_class_dict[c]
                self.syn_class_indices[c] = [len(labels_syn), len(labels_syn) + self.num_class_dict[c]]
                labels_syn += [c] * self.num_class_dict[c]

        return labels_syn

    def test_with_multival(self, generation_epoch, performance):
        res = []
        args = self.args
        data, device = self.data, self.device
        dropout = 0.5 if self.args.dataset in ['reddit'] else 0
        if args.source == "cora":
            model = GCN(nfeat=data.feat_train.shape[1], nhid=256, dropout=0.3,
                        weight_decay=2e-2, nlayers=2,
                        nclass=data.nclass, device=device).to(device)
            print("hh")
        else:
            model = GCN(nfeat=data.feat_train.shape[1], nhid=self.args.hidden, dropout=dropout,
                        weight_decay=5e-4, nlayers=2,
                        nclass=data.nclass, device=device).to(device)
        for i in range(generation_epoch):
            iteration = "iteration" + str(i)
            max_performance = performance[i]
            adj_syn = torch.load(
                f'./{args.savefile}/{iteration}_adj_{args.source}_{args.target}_{args.reduction_rate}_{args.seed}_{max_performance:.4f}.pt')
            feat_syn = torch.load(
                f'./{args.savefile}/{iteration}_feat_{args.source}_{args.target}_{args.reduction_rate}_{args.seed}_{max_performance:.4f}.pt')
            labels_syn = torch.load(
                f'./{args.savefile}/{iteration}_labels_{args.source}_{args.target}_{args.reduction_rate}_{args.seed}_{max_performance:.4f}.pt')

            # with_bn = True if args.dataset in ['ogbn-arxiv'] else False
            noval = True
            model.fit_with_train(feat_syn, adj_syn, labels_syn, data,
                                 train_iters=200, normalize=True, verbose=False, noval=noval)
        model.eval()
        labels_test = torch.LongTensor(data.labels_test).cuda()
        output = model.predict(data.feat_test, data.adj_test)
        loss_test = F.nll_loss(output, labels_test)
        acc_test = utils.accuracy(output, labels_test)

        res.append(acc_test.item())

        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))

        return res

    def get_performance(self, iteration):
        print(iteration)
        args = self.args
        file_names = os.listdir(f'./{args.savefile}/')
        desired_files = [file for file in file_names if
                         f'{iteration}_adj_{args.source}_{args.target}_{args.reduction_rate}_{args.seed}' in file]
        performance_list = [file.split('_')[-1][:-3] for file in desired_files]
        return float(performance_list[-2])

    def tune_with_multival(self, generation_epoch, performance):
        res = []
        args = self.args
        data, device = self.data, self.device
        dropout = 0.5 if self.args.dataset in ['reddit'] else 0
        model = GCN(nfeat=data.feat_train.shape[1], nhid=self.args.hidden, dropout=0.0,
                    weight_decay=5e-4, nlayers=2,
                    nclass=data.nclass, device=device, lr=0.01
                    ).to(device)
        for i in range(1, generation_epoch):
            iteration = "iteration" + str(i)
            # max_performance = performance[i]
            max_performance = self.get_performance(iteration)
            print(max_performance)
            adj_syn = torch.load(
                f'./{args.savefile}/{iteration}_adj_{args.source}_{args.target}_{args.reduction_rate}_{args.seed}_{max_performance:.4f}.pt')
            feat_syn = torch.load(
                f'./{args.savefile}/{iteration}_feat_{args.source}_{args.target}_{args.reduction_rate}_{args.seed}_{max_performance:.4f}.pt')
            labels_syn = torch.load(
                f'./{args.savefile}/{iteration}_labels_{args.source}_{args.target}_{args.reduction_rate}_{args.seed}_{max_performance:.4f}.pt')

            noval = True
            model.fit(feat_syn, adj_syn, labels_syn, data,
                      train_iters=500, normalize=True, verbose=False, noval=noval)
        model.eval()
        labels_test = torch.LongTensor(data.labels_test).cuda()
        output = model.predict(data.feat_test, data.adj_test)
        loss_test = F.nll_loss(output, labels_test)
        acc_test = utils.accuracy(output, labels_test)

        res.append(acc_test.item())

        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))

        return res

    def tune_alldata(self):
        res = []
        args = self.args
        data, device = self.data, self.device

        feat_full, adj_full = data.feat_train, data.adj_train
        feat_full, adj_full = utils.to_tensor(feat_full, adj_full, device=self.device)
        adj_full_norm = utils.normalize_adj_tensor(adj_full, sparse=True)
        labels_train = torch.LongTensor(data.labels_train).to(self.device)

        0.5 if self.args.dataset in ['reddit'] else 0
        model = GCN(nfeat=data.feat_train.shape[1], nhid=self.args.hidden, dropout=0.0,
                    weight_decay=5e-4, nlayers=2,
                    nclass=data.nclass, device=device, lr=0.01
                    ).to(device)
        iteration = "iteration" + str(1)
        max_performance = self.get_performance(iteration)
        print(max_performance)
        adj_syn = torch.load(
            f'./{args.savefile}/{iteration}_adj_{args.source}_{args.target}_{args.reduction_rate}_{args.seed}_{max_performance:.4f}.pt')
        feat_syn = torch.load(
            f'./{args.savefile}/{iteration}_feat_{args.source}_{args.target}_{args.reduction_rate}_{args.seed}_{max_performance:.4f}.pt')
        labels_syn = torch.load(
            f'./{args.savefile}/{iteration}_labels_{args.source}_{args.target}_{args.reduction_rate}_{args.seed}_{max_performance:.4f}.pt')

        noval = True
        model.fit(feat_full, adj_full_norm, labels_train, data,
                  train_iters=200, normalize=True, verbose=False, noval=noval)
        model.eval()
        labels_test = torch.LongTensor(data.labels_test).cuda()
        output = model.predict(data.feat_test, data.adj_test)
        loss_test = F.nll_loss(output, labels_test)
        acc_test = utils.accuracy(output, labels_test)

        res.append(acc_test.item())

        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))

        return res

    def test_with_train(self, iteration, verbose=True):
        res = []
        args = self.args
        data, device = self.data, self.device
        feat_syn, pge, labels_syn = self.feat_syn.detach(), \
            self.pge, self.labels_syn
        # with_bn = True if args.dataset in ['ogbn-arxiv'] else False
        dropout = 0.5 if self.args.dataset in ['reddit'] else 0
        if args.source == "cora":
            model = GCN(nfeat=data.feat_train.shape[1], nhid=256, dropout=0.3,
                        weight_decay=2e-2, nlayers=2,
                        nclass=data.nclass, device=device).to(device)
        elif args.version == "old":
            model = GCN(nfeat=data.feat_train.shape[1], nhid=args.hidden, dropout=0.2,
                        weight_decay=5e-3, nlayers=5,
                        nclass=data.nclass, device=device, lr=0.001, plot=args.plot
                        ).to(device)
        else:
            model = GCN(nfeat=data.feat_train.shape[1], nhid=self.args.hidden, dropout=dropout,
                        weight_decay=5e-4, nlayers=2,
                        nclass=data.nclass, device=device).to(device)

        adj_syn = pge.inference(feat_syn)
        args = self.args

        noval = True
        model.fit_with_train(feat_syn, adj_syn, labels_syn, data,
                             train_iters=600, normalize=True, verbose=False, noval=noval)

        model.eval()
        labels_test = torch.LongTensor(data.labels_test).cuda()
        output = model.predict(data.feat_test, data.adj_test)
        loss_test = F.nll_loss(output, labels_test)
        acc_test = utils.accuracy(output, labels_test)

        res.append(acc_test.item())
        if verbose:
            print("Test set results:",
                  "loss= {:.4f}".format(loss_test.item()),
                  "accuracy= {:.4f}".format(acc_test.item()))
        print("Sum of adj_syn: {:.4f}, Average of adj_syn: {:.4f}".format(
            adj_syn.sum().item(), (adj_syn.sum() / (adj_syn.shape[0] ** 2)).item()))

        if self.max_performance is None or self.max_performance < acc_test.item():
            self.max_performance = acc_test.item()
            self.max_adj_syn = adj_syn
            self.max_feat_syn = feat_syn
            self.max_labels_syn = labels_syn

        if args.save:
            # these saved synthetic data can be used in GADS.
            torch.save(adj_syn,
                       f'./saved_ours/adj_{args.dataset}_{args.reduction_rate}_{args.seed}_{iteration}_{acc_test.item():.4f}.pt')
            torch.save(feat_syn,
                       f'./saved_ours/feat_{args.dataset}_{args.reduction_rate}_{args.seed}_{iteration}_{acc_test.item():.4f}.pt')
            torch.save(labels_syn,
                       f'./saved_ours/labels_{args.dataset}_{args.reduction_rate}_{args.seed}_{iteration}_{acc_test.item():.4f}.pt')

        return res, loss_test.item(), acc_test.item()

    def save(self):
        args = self.args
        if not os.path.exists(f'./{args.savefile}'):
            os.makedirs(f'./{args.savefile}')
            print(f"Directory '{args.savefile}' created.")
        torch.save(self.max_adj_syn,
                   f'./{args.savefile}/adj_{args.source}_{args.target}_{args.reduction_rate}_{args.seed}_{self.max_performance:.4f}.pt')
        torch.save(self.max_feat_syn,
                   f'./{args.savefile}/feat_{args.source}_{args.target}_{args.reduction_rate}_{args.seed}_{self.max_performance:.4f}.pt')
        torch.save(self.max_labels_syn,
                   f'./{args.savefile}/labels_{args.source}_{args.target}_{args.reduction_rate}_{args.seed}_{self.max_performance:.4f}.pt')

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        sparserow = torch.LongTensor(sparse_mx.row).unsqueeze(1)
        sparsecol = torch.LongTensor(sparse_mx.col).unsqueeze(1)
        sparseconcat = torch.cat((sparserow, sparsecol), 1)
        sparsedata = torch.FloatTensor(sparse_mx.data)
        return torch.sparse.FloatTensor(sparseconcat.t(), sparsedata, torch.Size(sparse_mx.shape))

    def train(self, method=None, verbose=True):
        args = self.args
        data = self.data
        feat_syn, pge, labels_syn = self.feat_syn, self.pge, self.labels_syn
        features, adj, labels, adj_full, feat_full, labels_full = data.feat_train, data.adj_train, data.labels_train, data.adj_full, data.feat_full, data.labels_full
        syn_class_indices = self.syn_class_indices
        features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        features_full, adj_full, labels_full = utils.to_tensor(feat_full, adj_full, labels_full, device=self.device)

        feat_sub, _ = self.get_sub_adj_feat(features, args.init_method)
        self.feat_syn.data.copy_(feat_sub)

        if self.dataset == "cora":
            adj = adj.to_sparse()

        if utils.is_sparse_tensor(adj):
            adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            adj_full_norm = utils.normalize_adj_tensor(adj_full, sparse=True)
        else:
            adj_norm = utils.normalize_adj_tensor(adj)
            adj_full_norm = utils.normalize_adj_tensor(adj_full)

        adj_full = adj_full_norm
        adj = SparseTensor(row=adj._indices()[0], col=adj._indices()[1],
                           value=adj._values(), sparse_sizes=adj.size()).t()
        adj_full = SparseTensor(row=adj_full._indices()[0], col=adj_full._indices()[1],
                                value=adj_full._values(), sparse_sizes=adj_full.size()).t()

        outer_loop, inner_loop = get_loops(args)

        loss_list, acc_list = [], []
        for it in range(args.epochs + 1):
            loss_avg = 0
            if args.sgc == 1:
                model = SGC(nfeat=data.feat_train.shape[1], nhid=args.hidden,
                            nclass=data.nclass, dropout=args.dropout,
                            nlayers=args.nlayers, with_bn=False,
                            device=self.device).to(self.device)
            elif args.sgc == 2:
                model = SGC1(nfeat=data.feat_train.shape[1], nhid=args.hidden,
                             nclass=data.nclass, dropout=args.dropout,
                             nlayers=args.nlayers, with_bn=False,
                             device=self.device).to(self.device)
            else:
                model = GCN(nfeat=data.feat_train.shape[1], nhid=args.hidden,
                            nclass=data.nclass, dropout=args.dropout, nlayers=args.nlayers,
                            device=self.device).to(self.device)

            if args.surrogate is True:
                model1 = SGC(nfeat=data.feat_train.shape[1], nhid=args.hidden,
                             nclass=data.nclass, dropout=args.dropout,
                             nlayers=args.nlayers, with_bn=False,
                             device=self.device).to(self.device)
                # discriminator = Discriminator(data.feat_train.shape[1])
                discriminator = Discriminator(256).to(self.device)
                discriminator.to()

            model.initialize()
            model_parameters = list(model.parameters())
            optimizer_model = torch.optim.Adam(model_parameters, lr=args.lr_model)
            model.train()

            if args.surrogate is True:
                model1.initialize()
                model1_parameters = list(model1.parameters()) + list(discriminator.parameters())
                optimizer_model1 = torch.optim.Adam(model1_parameters, lr=args.lr_model)
                model1.train()

            t1 = time.time()
            for ol in tqdm(range(outer_loop)):
                adj_syn = pge(self.feat_syn)  # use MLP to construct the structure of the synthetic graph.
                adj_syn_norm = utils.normalize_adj_tensor(adj_syn, sparse=False)

                BN_flag = False
                for module in model.modules():
                    if 'BatchNorm' in module._get_name():  # BatchNorm
                        BN_flag = True
                if BN_flag:
                    model.train()  # for updating the mu, sigma of BatchNorm
                    output_real = model.forward(features, adj_norm)
                    for module in model.modules():
                        if 'BatchNorm' in module._get_name():  # BatchNorm
                            module.eval()  # fix mu and sigma of every BatchNorm layer

                if args.surrogate is True:
                    for module in model1.modules():
                        if 'BatchNorm' in module._get_name():  # BatchNorm
                            BN_flag = True
                    if BN_flag:
                        model1.train()  # for updating the mu, sigma of BatchNorm
                        output_real = model1.forward(features, adj_norm)
                        for module in model1.modules():
                            if 'BatchNorm' in module._get_name():  # BatchNorm
                                module.eval()  # fix mu and sigma of every BatchNorm layer

                loss = torch.tensor(0.0).to(self.device)  # condensation loss.
                grad_record = 0
                mmd_record = 0
                for c in range(data.nclass):
                    if c not in self.num_class_dict:
                        continue

                    batch_size, n_id, adjs = data.retrieve_class_sampler(
                        c, adj, transductive=False, num=256, args=args)

                    if args.nlayers == 1:
                        adjs = [adjs]
                    adjs = [adj.to(self.device) for adj in adjs]
                    output = model.forward_sampler(features[n_id], adjs)
                    loss_real = F.nll_loss(output, labels[n_id[:batch_size]])
                    gw_real = torch.autograd.grad(loss_real, model_parameters)
                    gw_real = list((_.detach().clone() for _ in gw_real))

                    ind = syn_class_indices[c]
                    if args.nlayers == 1:
                        adj_syn_norm_list = [adj_syn_norm[ind[0]: ind[1]]]
                    else:
                        adj_syn_norm_list = [adj_syn_norm] * (args.nlayers - 1) + \
                                            [adj_syn_norm[ind[0]: ind[1]]]

                    output_syn = model.forward_sampler_syn(feat_syn, adj_syn_norm_list)
                    loss_syn = F.nll_loss(output_syn, labels_syn[ind[0]: ind[1]])

                    gw_syn = torch.autograd.grad(loss_syn, model_parameters, create_graph=True)
                    coeff = self.num_class_dict[c] / max(self.num_class_dict.values())
                    loss += coeff * match_loss(gw_syn, gw_real, args, device=self.device)

                    if method == "target":
                        target_batch_size, target_n_id, target_adjs = data.retrieve_target_sampler(
                            c, adj_full, transductive=False, num=256, args=args)

                        if args.nlayers == 1:
                            target_adjs = [target_adjs]
                        target_adjs = [adj.to(self.device) for adj in target_adjs]
                        target_output = model.forward_sampler(features_full[target_n_id], target_adjs)
                        target_real = F.nll_loss(target_output, labels_full[n_id[:batch_size]])
                        target_gw_real = torch.autograd.grad(target_real, model_parameters)
                        target_gw_real = list((_.detach().clone() for _ in target_gw_real))
                        loss += coeff * match_loss(gw_syn, target_gw_real, args, device=self.device)

                if method == "mmd":
                    idx_target = torch.LongTensor(np.random.choice(data.idx_test, size=self.nnodes_syn)).to(self.device)
                    loss_distance = MMD(self.feat_syn, features_full[idx_target])
                    loss += self.args.beta * loss_distance
                elif method == "mmd-un":
                    output_1 = model1.forward_sampler(features[n_id], adjs)
                    output_syn_1 = model1.forward_sampler_syn(feat_syn, adj_syn_norm_list)
                    loss_distance = MMD(output_1, output_syn_1)
                    loss += self.args.beta * loss_distance
                    mmd_record += (self.args.beta * loss_distance).item()
                elif method == "mmd-sup":
                    output_1 = model.forward_sampler(features[n_id], adjs)
                    output_syn_1 = model.forward_sampler_syn(feat_syn, adj_syn_norm_list)
                    loss_distance = MMD(output_1, output_syn_1)
                    loss += self.args.beta * loss_distance
                    mmd_record += (self.args.beta * loss_distance).item()
                elif method == "mmd-t":
                    idx_target = torch.LongTensor(np.random.choice(data.idx_test, size=self.nnodes_syn)).to(self.device)
                    loss_distance = MMD(self.feat_syn, features_full[idx_target])
                    loss = self.args.beta * loss_distance
                elif method == "cmd":
                    idx_target = torch.LongTensor(np.random.choice(data.idx_test, size=self.nnodes_syn)).to(self.device)
                    loss_distance = CMD(self.feat_syn, features_full[idx_target])
                    loss += self.args.beta * loss_distance
                elif method == "ot":
                    idx_target = torch.LongTensor(np.random.choice(data.idx_test, size=self.nnodes_syn)).to(self.device)
                    loss_distance = OT_dist(self.feat_syn, features_full[idx_target], self.device)
                    loss += self.args.beta * loss_distance
                elif method == "l2":
                    idx_target = torch.LongTensor(np.random.choice(data.idx_test, size=self.nnodes_syn)).to(self.device)
                    loss_distance = L2_dist(self.feat_syn, features_full[idx_target])
                    loss += self.args.beta * loss_distance
                elif method == "nce":
                    target_batch_size, target_n_id, target_adjs = data.retrieve_target_class_sampler(
                        c, adj_full, transductive=False, num=256, args=args)

                    if args.nlayers == 1:
                        target_adjs = [target_adjs]
                    target_adjs = [adj.to(self.device) for adj in target_adjs]
                    target_output = model.forward_sampler(features_full[target_n_id], target_adjs)

                    output_syn_nce = model.forward_sampler_syn(feat_syn, adj_syn_norm_list)
                    loss_syn_nce = infonce(output_syn_nce, self.device)
                    target_real_nce = infonce(target_output, self.device)

                    gw_syn_nce = torch.autograd.grad(loss_syn_nce, model_parameters)
                    gw_syn_nce = list((_.detach().clone() for _ in gw_syn_nce))

                    target_gw_real_nce = torch.autograd.grad(target_real_nce, model_parameters)
                    target_gw_real_nce = list((_.detach().clone() for _ in target_gw_real_nce))

                    loss += coeff * match_loss(gw_syn_nce, target_gw_real_nce, args, device=self.device)

                loss_avg += loss.item()
                # TODO: regularize
                if args.alpha > 0:
                    loss_reg = args.alpha * regularization(adj_syn, self.tensor2onehot(labels_syn)).to(self.device)
                else:
                    loss_reg = torch.tensor(0).to(self.device)

                loss = loss + loss_reg

                # update synthetic graph
                self.optimizer_feat.zero_grad()
                self.optimizer_pge.zero_grad()
                loss.backward()  # the gradients are based on the condensation loss.

                # update the synthetic 'features' and the 'MLP' alternatively.
                if it % 50 < 10:
                    self.optimizer_pge.step()
                else:
                    self.optimizer_feat.step()

                if args.debug and ol % 5 == 0:
                    print('Gradient matching loss:', loss.item())

                if ol == outer_loop - 1:
                    break

                feat_syn_inner = feat_syn.detach()
                adj_syn_inner = pge.inference(feat_syn)
                adj_syn_inner_norm = utils.normalize_adj_tensor(adj_syn_inner, sparse=False)
                feat_syn_inner_norm = feat_syn_inner
                for j in range(inner_loop):
                    optimizer_model.zero_grad()
                    output_syn_inner = model.forward(feat_syn_inner_norm, adj_syn_inner_norm)
                    loss_syn_inner = F.nll_loss(output_syn_inner, labels_syn)
                    loss_syn_inner.backward()
                    optimizer_model.step()  # update gnn param

                    if args.surrogate is True:
                        optimizer_model1.zero_grad()
                        output_emb = model.forward_sampler(features[n_id], adjs)
                        shuffled_indices = torch.randperm(features[n_id].size(0))
                        output_emb_sf = model.forward_sampler(features[n_id][shuffled_indices], adjs)

                        positive_score = discriminator(output_emb, torch.mean(output_emb, dim=1))
                        negative_score = discriminator(output_emb_sf, torch.mean(output_emb_sf, dim=1))

                        func = nn.BCEWithLogitsLoss()
                        loss_infonce = func(positive_score, torch.ones_like(positive_score)) + func(
                            negative_score, torch.zeros_like(negative_score))
                        loss_infonce.backward()
                        optimizer_model1.step()

            loss_avg /= (data.nclass * outer_loop)
            if it % 10 == 0:
                print('Epoch {}, loss_avg: {:.4f}, time gap: {:.4f}'.format(it, loss_avg, time.time() - t1))

            # if verbose and it % 1 == 0:
            #     _, loss, acc = self.test_with_train(it)
            #     loss_list.append(loss)
            #     acc_list.append(acc)

        # if args.plot is True:
        #     plt.plot([i for i in range(len(loss_list))], loss_list)
        #     plt.savefig('./tune/loss_list.jpg')
        #     plt.close()
        #     plt.plot([i for i in range(len(acc_list))], acc_list)
        #     plt.savefig('./tune/acc_list.jpg')
        #     plt.close()

    def ppr(self, adj, alpha=0.15, normalization="symmetric"):
        if sp.issparse(adj):
            adj = adj.toarray()
        elif isinstance(adj, np.ndarray):
            pass
        else:
            raise ValueError(f"adj tead)")
        eps = 1e-6
        deg = adj.sum(1) + eps
        deg_inv = np.power(deg, -1)

        num_nodes = adj.shape[0]
        if normalization == "right":
            M = np.eye(num_nodes) - (1 - alpha) * adj * deg_inv[:, None]
        elif normalization == "symmetric":
            deg_inv_root = np.power(deg_inv, 0.5)
            M = (np.eye(num_nodes) - (1 - alpha) * deg_inv_root[None, :] * adj * deg_inv_root[:, None])

        return alpha * np.linalg.inv(M)

    def get_sub_adj_feat(self, features):
        data = self.data
        args = self.args
        idx_selected = []

        from collections import Counter
        counter = Counter(self.labels_syn.cpu().numpy())

        for c in range(data.nclass):
            tmp = data.retrieve_class(c, num=counter[c])
            tmp = list(tmp)
            idx_selected = idx_selected + tmp
        idx_selected = np.array(idx_selected).reshape(-1)
        features = features[idx_selected]

        adj_knn = 0
        return features, adj_knn

    def tensor2onehot(self, labels):
        """Convert label tensor to label onehot tensor.
        Parameters
        ----------
        labels : torch.LongTensor
            node labels
        Returns
        -------
        torch.LongTensor
            onehot labels tensor
        """
        labels = labels.long()
        eye = torch.eye(labels.max() + 1).to(labels.device)
        onehot_mx = eye[labels]
        return onehot_mx.to(labels.device)

def get_loops(args):
    if args.one_step:
        return 10, 0

    if args.dataset in ['arxiv']:
        return 10, 0
    if args.dataset in ['reddit']:
        return args.outer, args.inner
    if args.dataset in ['flickr']:
        return args.outer, args.inner
    if args.dataset in ['cora']:
        return 20, 10
    if args.dataset in ['citeseer']:
        return 20, 5  # at least 200 epochs
    else:
        return args.outer, args.inner
