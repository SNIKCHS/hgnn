#!/usr/bin/env/python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
from utils import *
from dataset.GraphDataset import GraphDataset
from dataset.SyntheticDataset import SyntheticDataset

import numpy as np
from torch.utils.data.dataloader import default_collate, DataLoader
from task.GraphPrediction import GraphPrediction


def collate_fn(batch):
    max_neighbor_num = -1
    for data in batch:
        for row in data['adj_mat']:
            max_neighbor_num = max(max_neighbor_num, len(row))

    for data in batch:
        # pad the adjacency list
        data['adj_mat'] = pad_sequence(data['adj_mat'], maxlen=max_neighbor_num)
        data['weight'] = pad_sequence(data['weight'], maxlen=max_neighbor_num)

        data['node'] = np.array(data['node']).astype(np.float32)
        data['adj_mat'] = np.array(data['adj_mat']).astype(np.int32)
        data['weight'] = np.array(data['weight']).astype(np.float32)
        data['label'] = np.array(data['label'])
    return default_collate(batch)


class BaseTask(object):
    """
    A base class that supports loading datasets, early stop and reporting statistics
    """

    def __init__(self, args, logger, criterion='max'):
        """
        criterion: min/max
        """
        self.args = args
        self.logger = logger
        self.early_stop = EarlyStoppingCriterion(self.args.patience, criterion)

    def reset_epoch_stats(self, epoch, prefix):
        """
        prefix: train/dev/test
        """
        self.epoch_stats = {
            'prefix': prefix,
            'epoch': epoch,
            'loss': 0,
            'num_correct': 0,
            'num_total': 0,
        }

    def update_epoch_stats(self, loss, score, label, is_regression=False):
        with torch.no_grad():
            self.epoch_stats['loss'] += loss.item()
            self.epoch_stats['num_total'] += label.size(0)
            if not is_regression:
                self.epoch_stats['num_correct'] += torch.sum(torch.eq(torch.argmax(score, dim=1), label)).item()

    def report_epoch_stats(self):
        if self.epoch_stats['prefix'] == 'train':
            statistics = [self.epoch_stats['num_correct'], self.epoch_stats['num_total'], self.epoch_stats['loss']]
        else:
            # aggregate the results from all nodes
            statistics = torch.tensor(
                [self.epoch_stats['num_correct'], self.epoch_stats['num_total'], self.epoch_stats['loss']],
                dtype=torch.float32
            ).cuda()

            # if self.args.dist_method == 'reduce':
            #     dist.reduce(tensor=statistics, dst=0, op=dist.ReduceOp.SUM, group=group)
            # elif self.args.dist_method == 'all_gather':
            #     all_statistics = [th.zeros((1, 3)).cuda() for _ in range(self.args.world_size)]
            #     dist.all_gather(tensor=statistics, tensor_list=all_statistics, group=group)
            #     statistics = th.sum(th.cat(all_statistics, dim=0), dim=0).cpu().numpy()

        accuracy = float(statistics[0]) / statistics[1]
        loss = statistics[2] / statistics[1]
        if self.epoch_stats['prefix'] != 'test':
            self.logger.info(
                "%s phase of epoch %d: accuracy %.6f, loss %.6f, num_correct %d, total %d" % (
                    self.epoch_stats['prefix'],
                    self.epoch_stats['epoch'],
                    accuracy,
                    loss,
                    statistics[0],
                    statistics[1]))
        return accuracy, loss

    def report_best(self):
        self.logger.info("best dev %.6f, best test %.6f"
                         % (self.early_stop.best_dev_score, self.early_stop.best_test_score))

    def load_dataset(self, dataset_class, collate_fn):
        train_dataset = dataset_class(self.args, self.logger, split='train')
        dev_dataset = dataset_class(self.args, self.logger, split='dev')
        test_dataset = dataset_class(self.args, self.logger, split='test')

        train_loader = DataLoader(train_dataset, batch_size=1, collate_fn=collate_fn,
                                  num_workers=0)
        dev_loader = DataLoader(dev_dataset, batch_size=1, collate_fn=collate_fn,
                                num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn,
                                 num_workers=0)
        self.logger.info("train data size: %d" % len(train_dataset))
        self.logger.info("dev data size: %d" % len(dev_dataset))
        self.logger.info("test data size: %d" % len(test_dataset))
        return train_loader, dev_loader, test_loader


class GraphPredictionTask(BaseTask):

    def __init__(self, args, logger, rgnn, manifold):
        if args.is_regression:
            super(GraphPredictionTask, self).__init__(args, logger, criterion='min')
        else:
            super(GraphPredictionTask, self).__init__(args, logger, criterion='max')
        self.hyperbolic = False if args.select_manifold == "euclidean" else True
        self.rgnn = rgnn
        self.manifold = manifold
        self.device = torch.device(args.device)

    def forward(self, model, sample, loss_function):
        mask = sample['mask'].int() if 'mask' in sample else torch.Tensor([sample['adj_mat'].size(1)]).cuda()
        scores = model(
            sample['node'].cuda().float(),
            sample['adj_mat'].cuda().long(),
            sample['weight'].cuda().float(),
            mask)
        if self.args.is_regression:
            loss = loss_function(
                scores.view(-1) * self.args.std[self.args.prop_idx] + self.args.mean[self.args.prop_idx],
                torch.Tensor([sample['label'].view(-1)[self.args.prop_idx]]).float().cuda())
        else:
            loss = loss_function(scores, torch.Tensor([sample['label'].view(-1)[self.args.prop_idx]]).long().cuda())
        return scores, loss

    def run_gnn(self):
        train_loader, dev_loader, test_loader = self.load_data()

        model = GraphPrediction(self.args, self.logger, self.rgnn, self.manifold).cuda()
        model = model.to(self.device)
        if self.args.is_regression:
            loss_function = nn.MSELoss(reduction='sum')
        else:
            loss_function = nn.CrossEntropyLoss(reduction='sum')

        optimizer, lr_scheduler, hyperbolic_optimizer, hyperbolic_lr_scheduler = \
            set_up_optimizer_scheduler(self.hyperbolic, self.args, model)

        for epoch in range(self.args.max_epochs):
            self.reset_epoch_stats(epoch, 'train')
            model.train()
            for i, sample in enumerate(train_loader):
                model.zero_grad()
                scores, loss = self.forward(model, sample, loss_function)
                loss.backward(retain_graph=True)

                if self.args.grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.grad_clip)

                optimizer.step()
                if self.hyperbolic and len(self.args.hyp_vars) != 0:
                    hyperbolic_optimizer.step()
                if self.args.is_regression and self.args.metric == "mae":
                    loss = torch.sqrt(loss)
                self.update_epoch_stats(loss, scores, sample['label'].cuda(), is_regression=self.args.is_regression)
                if i % 400 == 0:
                    self.report_epoch_stats()

            dev_acc, dev_loss = self.evaluate(epoch, dev_loader, 'dev', model, loss_function)
            test_acc, test_loss = self.evaluate(epoch, test_loader, 'test', model, loss_function)

            if self.args.is_regression and not self.early_stop.step(dev_loss, test_loss, epoch):
                break
            elif not self.args.is_regression and not self.early_stop.step(dev_acc, test_acc, epoch):
                break

            lr_scheduler.step()
            if self.hyperbolic and len(self.args.hyp_vars) != 0:
                hyperbolic_lr_scheduler.step()
            torch.cuda.empty_cache()
        self.report_best()

    def evaluate(self, epoch, data_loader, prefix, model, loss_function):
        model.eval()
        with torch.no_grad():
            self.reset_epoch_stats(epoch, prefix)
            for i, sample in enumerate(data_loader):
                scores, loss = self.forward(model, sample, loss_function)
                if self.args.is_regression and self.args.metric == "mae":
                    loss = torch.sqrt(loss)
                self.update_epoch_stats(loss, scores, sample['label'].cuda(), is_regression=self.args.is_regression)
            accuracy, loss = self.report_epoch_stats()
        if self.args.is_regression and self.args.metric == "rmse":
            loss = np.sqrt(loss)
        return accuracy, loss

    def load_data(self):
        if self.args.task == 'synthetic':
            return self.load_dataset(SyntheticDataset, collate_fn)
        else:
            return self.load_dataset(GraphDataset, collate_fn)
