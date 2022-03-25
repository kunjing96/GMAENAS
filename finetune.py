import os, sys
import argparse
import time
import math
import torch
import random
import tqdm
import nni
import numpy as np
from scipy import stats
from typing import Optional
from torch_geometric.data import DataLoader
from tensorboardX import SummaryWriter
from torch.distributions.bernoulli import Bernoulli

from sp import SimpleParam
from data import DataIterator, Dataset
from model import PairWiseLearning_BARLOW_TWINS, PairWiseLearning_BARLOW_TWINS_L2L, PairWiseLearning_GRACE, PairWiseLearning_GRAPHCL, PairWiseLearning_BGRL, PairWiseLearning_MVGRL, MAELearning, Predictor
from utils import get_base_model, get_activation, AvgrageMeter
#from ChildTuningOptimizer import ChildTuningAdamW


class MSELoss(torch.nn.Module):

    def __init__(self, exp_weighted=False):
        super(MSELoss, self).__init__()
        self.exp_weighted = exp_weighted

    def forward(self, input, target):
        if self.exp_weighted:
            N = input.size(0)
            return 1 / (N * (math.e - 1)) * torch.sum((torch.exp(target) - 1) * (input - target)**2)
        else:
            return torch.nn.functional.mse_loss(input, target)


class HingePairwiseLoss(torch.nn.Module):

    def __init__(self, exp_weighted=False, m=0.01):
        super(HingePairwiseLoss, self).__init__()
        self.exp_weighted = exp_weighted
        self.m = m

    def forward(self, input, target):
        N = input.size(0)
        total_loss = 0
        for i in range(N):
            indices = (target>target[i])
            x = (self.m - (input[indices] - input[i]))
            if self.exp_weighted:
                x = (torch.exp(target[i]) - 1) * (torch.exp(target[indices]) - 1) * x
            else:
                x = x
            x[x<0] = 0
            total_loss += torch.sum(x)
        if self.exp_weighted:
            return 2 / (N * (math.e - 1))**2 * total_loss
        else:
            return 2 / N**2                  * total_loss


class BPRLoss(torch.nn.Module):

    def __init__(self, exp_weighted=False):
        super(BPRLoss, self).__init__()
        self.exp_weighted = exp_weighted

    def forward(self, input, target):
        N = input.size(0)
        total_loss = 0
        for i in range(N):
            indices = (target>target[i])
            x = torch.log(1 + torch.exp(-(input[indices] - input[i])))
            if self.exp_weighted:
                x = (torch.exp(target[i]) - 1) * (torch.exp(target[indices]) - 1) * x
            else:
                x = x
            total_loss += torch.sum(x)
        if self.exp_weighted:
            return 2 / (N * (math.e - 1))**2 * total_loss
        else:
            return 2 / N**2                  * total_loss


class MSEHingeLoss(torch.nn.Module):

    def __init__(self, exp_weighted=False, m=0.01, alpha1=0.5, alpha2=0.5):
        super(MSEHingeLoss, self).__init__()
        self.exp_weighted = exp_weighted
        self.m = m
        self.alpha1 = alpha1
        self.alpha2 = alpha2

    def forward(self, input, target):
        N = input.size(0)
        total_loss = 0
        for i in range(N):
            indices = (target>target[i])
            x = self.alpha1 * (self.m - (input[indices] - input[i])) + self.alpha2 * (input[i] - target[i])**2
            if self.exp_weighted:
                x = (torch.exp(target[i]) - 1) * (torch.exp(target[indices]) - 1) * x
            else:
                x = x
            x[x<0] = 0
            total_loss += torch.sum(x)
        if self.exp_weighted:
            return 2 / (N * (math.e - 1))**2 * total_loss
        else:
            return 2 / N**2                  * total_loss


class MSEBPRLoss(torch.nn.Module):

    def __init__(self, exp_weighted=False, alpha1=0.5, alpha2=0.5 ):
        super(MSEBPRLoss, self).__init__()
        self.exp_weighted = exp_weighted
        self.alpha1 = alpha1
        self.alpha2 = alpha2

    def forward(self, input, target):
        N = input.size(0)
        total_loss = 0
        for i in range(N):
            indices = (target>target[i])
            x = self.alpha1 * torch.log(1 + torch.exp(-(input[indices] - input[i]))) + self.alpha2 * (input[i] - target[i])**2
            if self.exp_weighted:
                x = (torch.exp(target[i]) - 1) * (torch.exp(target[indices]) - 1) * x
            else:
                x = x
            x[x<0] = 0
            total_loss += torch.sum(x)
        if self.exp_weighted:
            return 2 / (N * (math.e - 1))**2 * total_loss
        else:
            return 2 / N**2                  * total_loss


class MyLoss(torch.nn.Module):

    def __init__(self, exp_weighted=False):
        super(MyLoss, self).__init__()
        self.exp_weighted = exp_weighted

    def forward(self, input, target):
        N = input.size(0)
        total_loss = 0
        for i in range(N):
            indices = (target>target[i])
            x = torch.sigmoid(-(input[indices] - input[i] + 1))
            if self.exp_weighted:
                x = (torch.exp(target[i]) - 1) * (torch.exp(target[indices]) - 1) * x
            else:
                x = x
            x[x<0] = 0
            total_loss += torch.sum(x)
        if self.exp_weighted:
            return 2 / (N * (math.e - 1))**2 * total_loss
        else:
            return 2 / N**2                  * total_loss


def calculate_fisher(loader, model, predictor, criterion, reserve_p):
    '''
    Calculate Fisher Information for different parameters
    '''
    gradient_mask = dict()
    model.train()
    predictor.train()

    for params in model.parameters():
        gradient_mask[params] = params.new_zeros(params.size())

    N = len(loader)
    for step, batch in enumerate(loader):
        if len(batch) == 2: # 101 201
            batch_x, batch_y = batch
            x            = batch_x.x.cuda(non_blocking=True)
            edge_index_x = batch_x.edge_index.cuda(non_blocking=True)
            ptr_x        = batch_x.ptr.cuda(non_blocking=True)
            y            = batch_y.x.cuda(non_blocking=True)
            edge_index_y = batch_y.edge_index.cuda(non_blocking=True)
            ptr_y        = batch_y.ptr.cuda(non_blocking=True)
            target       = batch_x.y.cuda(non_blocking=True)[ptr_x[:-1]]
        elif len(batch) == 4: # 301 darts
            batch_x, batch_y, batch_x_reduce, batch_y_reduce = batch
            x            = batch_x.x.cuda(non_blocking=True)
            edge_index_x = batch_x.edge_index.cuda(non_blocking=True)
            ptr_x        = batch_x.ptr.cuda(non_blocking=True)
            y            = batch_y.x.cuda(non_blocking=True)
            edge_index_y = batch_y.edge_index.cuda(non_blocking=True)
            ptr_y        = batch_y.ptr.cuda(non_blocking=True)
            reduce_x            = batch_x_reduce.x.cuda(non_blocking=True)
            reduce_edge_index_x = batch_x_reduce.edge_index.cuda(non_blocking=True)
            reduce_ptr_x        = batch_x_reduce.ptr.cuda(non_blocking=True)
            reduce_y            = batch_y_reduce.x.cuda(non_blocking=True)
            reduce_edge_index_y = batch_y_reduce.edge_index.cuda(non_blocking=True)
            reduce_ptr_y        = batch_y_reduce.ptr.cuda(non_blocking=True)
            target       = batch_x.y.cuda(non_blocking=True)[ptr_x[:-1]]
        else:
            raise ValueError('The length of batch must be 1 or 2!')
        n = target.size(0)

        z, g = model(x, edge_index_x, ptr_x, y, edge_index_y, ptr_y)
        if len(batch) == 4:
            reduce_z, reduce_g = model(reduce_x, reduce_edge_index_x, reduce_ptr_x, reduce_y, reduce_edge_index_y, reduce_ptr_y)
            z = torch.cat([z, reduce_z], -1)
            g = torch.cat([g, reduce_g], -1)

        if isinstance(criterion, torch.nn.BCEWithLogitsLoss):
            output = []
            losses_ = 0
            for i in range(n):
                input_  = torch.cat([g[torch.arange(n)!=i], g[[i]].repeat(n-1, 1)], -1)
                target_ = (target[torch.arange(n)!=i] > target[[i]].repeat(n-1)).float()
                output_ = predictor(input_)
                loss_   = criterion(output_.squeeze(), target_.squeeze())
                losses_  = losses_ + loss_
                output.append(output_.squeeze().unsqueeze(0))
            loss = losses_ / n
            output = torch.cat(output)
        else:
            output = predictor(g)
            loss = criterion(output.squeeze(), target.squeeze())

        loss.backward()

        for params in model.parameters():
            if params.grad is not None:
                torch.nn.utils.clip_grad_norm_(params, 5)
                gradient_mask[params] += (params.grad ** 2) / N
        model.zero_grad()

    print('Calculate Fisher Information')

    # Numpy
    r = None
    for k, v in gradient_mask.items():
        v = v.view(-1).cpu().numpy()
        if r is None:
            r = v
        else:
            r = np.append(r, v)
    polar = np.percentile(r, (1-reserve_p)*100)
    for k in gradient_mask:
        gradient_mask[k] = gradient_mask[k] >= polar
    print('Polar => {}'.format(polar))

    # TODO: pytorch: torch.kthvalue

    return gradient_mask


def train(loader, batch_size, model, predictor, optimizer, criterion, finetune_fixed_encoder=True, opt_mode=None, opt_reserve_p=1.0, gradient_mask=None):
    losses = AvgrageMeter()
    batch_times = AvgrageMeter()
    taus = AvgrageMeter()

    if finetune_fixed_encoder:
        model.eval()
    else:
        model.train()
    predictor.train()
    optimizer.zero_grad()

    outputs = []
    targets = []
    for step, batch in enumerate(loader):
        if len(batch) == 2: # 101 201
            batch_x, batch_y = batch
            x            = batch_x.x.cuda(non_blocking=True)
            edge_index_x = batch_x.edge_index.cuda(non_blocking=True)
            ptr_x        = batch_x.ptr.cuda(non_blocking=True)
            y            = batch_y.x.cuda(non_blocking=True)
            edge_index_y = batch_y.edge_index.cuda(non_blocking=True)
            ptr_y        = batch_y.ptr.cuda(non_blocking=True)
            target       = batch_x.y.cuda(non_blocking=True)[ptr_x[:-1]]
        elif len(batch) == 4: # 301 darts
            batch_x, batch_y, batch_x_reduce, batch_y_reduce = batch
            x            = batch_x.x.cuda(non_blocking=True)
            edge_index_x = batch_x.edge_index.cuda(non_blocking=True)
            ptr_x        = batch_x.ptr.cuda(non_blocking=True)
            y            = batch_y.x.cuda(non_blocking=True)
            edge_index_y = batch_y.edge_index.cuda(non_blocking=True)
            ptr_y        = batch_y.ptr.cuda(non_blocking=True)
            reduce_x            = batch_x_reduce.x.cuda(non_blocking=True)
            reduce_edge_index_x = batch_x_reduce.edge_index.cuda(non_blocking=True)
            reduce_ptr_x        = batch_x_reduce.ptr.cuda(non_blocking=True)
            reduce_y            = batch_y_reduce.x.cuda(non_blocking=True)
            reduce_edge_index_y = batch_y_reduce.edge_index.cuda(non_blocking=True)
            reduce_ptr_y        = batch_y_reduce.ptr.cuda(non_blocking=True)
            target       = batch_x.y.cuda(non_blocking=True)[ptr_x[:-1]]
        else:
            raise ValueError('The length of batch must be 1 or 2!')
        n = target.size(0)

        b_start = time.time()
        optimizer.zero_grad()

        if finetune_fixed_encoder:
            with torch.no_grad():
                z, g = model(x, edge_index_x, ptr_x, y, edge_index_y, ptr_y)
                if len(batch) == 4:
                    reduce_z, reduce_g = model(reduce_x, reduce_edge_index_x, reduce_ptr_x, reduce_y, reduce_edge_index_y, reduce_ptr_y)
                    z = torch.cat([z, reduce_z], -1)
                    g = torch.cat([g, reduce_g], -1)
            g = g.detach()
        else:
            z, g = model(x, edge_index_x, ptr_x, y, edge_index_y, ptr_y)
            if len(batch) == 4:
                reduce_z, reduce_g = model(reduce_x, reduce_edge_index_x, reduce_ptr_x, reduce_y, reduce_edge_index_y, reduce_ptr_y)
                z = torch.cat([z, reduce_z], -1)
                g = torch.cat([g, reduce_g], -1)
        if isinstance(criterion, torch.nn.BCEWithLogitsLoss):
            output = []
            losses_ = 0
            for i in range(n):
                input_  = torch.cat([g[torch.arange(n)!=i], g[[i]].repeat(n-1, 1)], -1)
                target_ = (target[torch.arange(n)!=i] > target[[i]].repeat(n-1)).float()
                output_ = predictor(input_)
                loss_   = criterion(output_.squeeze(), target_.squeeze())
                losses_  = losses_ + loss_
                output.append(output_.squeeze().unsqueeze(0))
            loss = losses_ / n
            output = torch.cat(output)
            batch_tau = stats.kendalltau(target.squeeze().detach().cpu().numpy(), np.sum(output.squeeze().detach().cpu().numpy()<0, 1), nan_policy='omit')[0]
            taus.update(batch_tau, 1)
        else:
            output = predictor(g)
            loss = criterion(output.squeeze(), target.squeeze())

        loss.backward()
        #if finetune_fixed_encoder:
            #torch.nn.utils.clip_grad_norm_(predictor.parameters(), 5)
        #else:
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            #torch.nn.utils.clip_grad_norm_(predictor.parameters(), 5)
        # =================== HACK BEGIN =====================
        if opt_mode is not None and not finetune_fixed_encoder and opt_reserve_p < 1:
            for p in model.parameters():
                if p.grad is None:
                    continue
                if opt_mode == 'D' and gradient_mask is not None:
                    if p in gradient_mask:
                        p.grad *= gradient_mask[p] / opt_reserve_p
                else: 
                    # F
                    grad_mask = Bernoulli(p.grad.new_full(size=p.grad.size(), fill_value=opt_reserve_p))
                    p.grad *= grad_mask.sample() / opt_reserve_p
        # =================== HACK END =======================
        optimizer.step()

        batch_times.update(time.time() - b_start)
        losses.update(loss.data.item(), n)

        outputs.append(output.squeeze().detach())
        targets.append(target.squeeze().detach())

    if isinstance(criterion, torch.nn.BCEWithLogitsLoss):
        tau = taus.avg
    else:
        outputs = torch.cat(outputs).cpu().numpy()
        targets = torch.cat(targets).cpu().numpy()
        tau = stats.kendalltau(targets, outputs, nan_policy='omit')[0]

    return losses.avg, tau, batch_times.avg


def test(all_loader, top_loader, batch_size, model, predictor, criterion):
    all_losses = AvgrageMeter()
    all_batch_times = AvgrageMeter()
    top_losses = AvgrageMeter()
    top_batch_times = AvgrageMeter()
    all_taus = AvgrageMeter()
    top_taus = AvgrageMeter()

    model.eval()
    predictor.eval()

    outputs = []
    targets = []
    for step, batch in enumerate(all_loader):
        if len(batch) == 2: # 101 201
            batch_x, batch_y = batch
            x            = batch_x.x.cuda(non_blocking=True)
            edge_index_x = batch_x.edge_index.cuda(non_blocking=True)
            ptr_x        = batch_x.ptr.cuda(non_blocking=True)
            y            = batch_y.x.cuda(non_blocking=True)
            edge_index_y = batch_y.edge_index.cuda(non_blocking=True)
            ptr_y        = batch_y.ptr.cuda(non_blocking=True)
            target       = batch_x.y.cuda(non_blocking=True)[ptr_x[:-1]]
        elif len(batch) == 4: # 301 darts
            batch_x, batch_y, batch_x_reduce, batch_y_reduce = batch
            x            = batch_x.x.cuda(non_blocking=True)
            edge_index_x = batch_x.edge_index.cuda(non_blocking=True)
            ptr_x        = batch_x.ptr.cuda(non_blocking=True)
            y            = batch_y.x.cuda(non_blocking=True)
            edge_index_y = batch_y.edge_index.cuda(non_blocking=True)
            ptr_y        = batch_y.ptr.cuda(non_blocking=True)
            reduce_x            = batch_x_reduce.x.cuda(non_blocking=True)
            reduce_edge_index_x = batch_x_reduce.edge_index.cuda(non_blocking=True)
            reduce_ptr_x        = batch_x_reduce.ptr.cuda(non_blocking=True)
            reduce_y            = batch_y_reduce.x.cuda(non_blocking=True)
            reduce_edge_index_y = batch_y_reduce.edge_index.cuda(non_blocking=True)
            reduce_ptr_y        = batch_y_reduce.ptr.cuda(non_blocking=True)
            target       = batch_x.y.cuda(non_blocking=True)[ptr_x[:-1]]
        else:
            raise ValueError('The length of batch must be 1 or 2!')
        n = target.size(0)

        b_start = time.time()

        with torch.no_grad():
            z, g = model(x, edge_index_x, ptr_x, y, edge_index_y, ptr_y)
            if len(batch) == 4:
                reduce_z, reduce_g = model(reduce_x, reduce_edge_index_x, reduce_ptr_x, reduce_y, reduce_edge_index_y, reduce_ptr_y)
                z = torch.cat([z, reduce_z], -1)
                g = torch.cat([g, reduce_g], -1)
            g = g.detach()
            if isinstance(criterion, torch.nn.BCEWithLogitsLoss):
                output = []
                losses_ = 0
                for i in range(n):
                    input_  = torch.cat([g[torch.arange(n)!=i], g[[i]].repeat(n-1, 1)], -1)
                    target_ = (target[torch.arange(n)!=i] > target[[i]].repeat(n-1)).float()
                    output_ = predictor(input_)
                    loss_   = criterion(output_.squeeze(), target_.squeeze())
                    losses_  = losses_ + loss_
                    output.append(output_.squeeze().unsqueeze(0))
                loss = losses_ / n
                output = torch.cat(output)
                batch_tau = stats.kendalltau(target.squeeze().detach().cpu().numpy(), np.sum(output.squeeze().detach().cpu().numpy()<0, 1), nan_policy='omit')[0]
                all_taus.update(batch_tau, 1)
            else:
                output = predictor(g)
                loss = criterion(output.squeeze(), target.squeeze())
    
        all_batch_times.update(time.time() - b_start)
        all_losses.update(loss.data.item(), n)
        
        outputs.append(output.squeeze().detach())
        targets.append(target.squeeze().detach())

    if isinstance(criterion, torch.nn.BCEWithLogitsLoss):
        all_tau = all_taus.avg
    else:
        outputs = torch.cat(outputs).cpu().numpy()
        targets = torch.cat(targets).cpu().numpy()
        all_tau = stats.kendalltau(targets, outputs, nan_policy='omit')[0]

    outputs = []
    targets = []
    for step, batch in enumerate(top_loader):
        if len(batch) == 2: # 101 201
            batch_x, batch_y = batch
            x            = batch_x.x.cuda(non_blocking=True)
            edge_index_x = batch_x.edge_index.cuda(non_blocking=True)
            ptr_x        = batch_x.ptr.cuda(non_blocking=True)
            y            = batch_y.x.cuda(non_blocking=True)
            edge_index_y = batch_y.edge_index.cuda(non_blocking=True)
            ptr_y        = batch_y.ptr.cuda(non_blocking=True)
            target       = batch_x.y.cuda(non_blocking=True)[ptr_x[:-1]]
        elif len(batch) == 4: # 301 darts
            batch_x, batch_y, batch_x_reduce, batch_y_reduce = batch
            x            = batch_x.x.cuda(non_blocking=True)
            edge_index_x = batch_x.edge_index.cuda(non_blocking=True)
            ptr_x        = batch_x.ptr.cuda(non_blocking=True)
            y            = batch_y.x.cuda(non_blocking=True)
            edge_index_y = batch_y.edge_index.cuda(non_blocking=True)
            ptr_y        = batch_y.ptr.cuda(non_blocking=True)
            reduce_x            = batch_x_reduce.x.cuda(non_blocking=True)
            reduce_edge_index_x = batch_x_reduce.edge_index.cuda(non_blocking=True)
            reduce_ptr_x        = batch_x_reduce.ptr.cuda(non_blocking=True)
            reduce_y            = batch_y_reduce.x.cuda(non_blocking=True)
            reduce_edge_index_y = batch_y_reduce.edge_index.cuda(non_blocking=True)
            reduce_ptr_y        = batch_y_reduce.ptr.cuda(non_blocking=True)
            target       = batch_x.y.cuda(non_blocking=True)[ptr_x[:-1]]
        else:
            raise ValueError('The length of batch must be 1 or 2!')
        n = target.size(0)

        b_start = time.time()

        with torch.no_grad():
            z, g = model(x, edge_index_x, ptr_x, y, edge_index_y, ptr_y)
            if len(batch) == 4:
                reduce_z, reduce_g = model(reduce_x, reduce_edge_index_x, reduce_ptr_x, reduce_y, reduce_edge_index_y, reduce_ptr_y)
                z = torch.cat([z, reduce_z], -1)
                g = torch.cat([g, reduce_g], -1)
            g = g.detach()
            if isinstance(criterion, torch.nn.BCEWithLogitsLoss):
                output = []
                losses_ = 0
                for i in range(n):
                    input_  = torch.cat([g[torch.arange(n)!=i], g[[i]].repeat(n-1, 1)], -1)
                    target_ = (target[torch.arange(n)!=i] > target[[i]].repeat(n-1)).float()
                    output_ = predictor(input_)
                    loss_   = criterion(output_.squeeze(), target_.squeeze())
                    losses_  = losses_ + loss_
                    output.append(output_.squeeze().unsqueeze(0))
                loss = losses_ / n
                output = torch.cat(output)
                batch_tau = stats.kendalltau(target.squeeze().detach().cpu().numpy(), np.sum(output.squeeze().detach().cpu().numpy()<0, 1), nan_policy='omit')[0]
                top_taus.update(batch_tau, 1)
            else:
                output = predictor(g)
                loss = criterion(output.squeeze(), target.squeeze())
    
        top_batch_times.update(time.time() - b_start)
        top_losses.update(loss.data.item(), n)
        
        outputs.append(output.squeeze().detach())
        targets.append(target.squeeze().detach())

    if isinstance(criterion, torch.nn.BCEWithLogitsLoss):
        top_tau = top_taus.avg
    else:
        outputs = torch.cat(outputs).cpu().numpy()
        targets = torch.cat(targets).cpu().numpy()
        top_tau = stats.kendalltau(targets, outputs, nan_policy='omit')[0]

    return all_losses.avg, all_tau, all_batch_times.avg, top_losses.avg, top_tau, top_batch_times.avg


def finetune(space, train_loader, val_all_loader, val_top_loader, batch_size, model, criterion, exp_weighted, device, fixed_encoder, epochs=1000, optimizer='SGD', lr=0.01, weight_decay=0.0, opt_reserve_p=1.0, opt_mode=None, verbose=False, writer=None, use_nni=False):
    
    # get model
    d_model = model.d_model
    if criterion == 'bce':
        d_model *= 2
    predictor = Predictor(d_model).to(device)

    # get optimizer
    if fixed_encoder:
        params = [
            {'params': predictor.parameters(), 'lr': lr[1]},
        ]
    else:
        params = [
            {'params': model.parameters(), 'lr': lr[0]},
            {'params': predictor.parameters(), 'lr': lr[1]},
        ]
    #optimizer = ChildTuningAdamW(params, lr=0, weight_decay=weight_decay, reserve_p=opt_reserve_p, mode=opt_mode)
    if optimizer == 'SGD':
        optimizer = torch.optim.SGD(params, lr=0, momentum=0.9, weight_decay=weight_decay)
    elif optimizer =='AdamW':
        optimizer = torch.optim.AdamW(params, lr=0, weight_decay=weight_decay)

    # get scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-4)
    #scheduler = None

    # get criterion
    if criterion == 'mse':
        criterion = MSELoss(exp_weighted=exp_weighted)
    elif criterion == 'hinge':
        criterion = HingePairwiseLoss(exp_weighted=exp_weighted)
    elif criterion == 'bpr':
        criterion = BPRLoss(exp_weighted=exp_weighted)
    elif criterion == 'bce':
        criterion = torch.nn.BCEWithLogitsLoss()
    elif criterion == 'mine':
        criterion = MyLoss(exp_weighted=exp_weighted)
    elif criterion == 'mse+hinge':
        criterion = MSEHingeLoss(exp_weighted=exp_weighted)
    elif criterion == 'mse+bpr':
        criterion = MSEBPRLoss(exp_weighted=exp_weighted)
    else:
        raise ValueError('No implementation!')
    
    # =================== HACK BEGIN =====================
    if opt_mode == 'D':
        gradient_mask = calculate_fisher(train_loader, model, predictor, criterion, opt_reserve_p)
    else:
        gradient_mask = None
    # =================== HACK END =======================
    
    best_all_val_tau = -1
    best_top_val_tau = -1
    if verbose:
        iters = range(1, epochs+1)
    else:
        iters = tqdm.tqdm(range(1, epochs+1))
    for epoch in iters:
        if scheduler:
            scheduler.step()
        
        train_loss, train_tau, train_batch_time = train(train_loader, batch_size, model, predictor, optimizer, criterion, fixed_encoder, opt_mode, opt_reserve_p, gradient_mask)
        if verbose:
            print(f'(FT) | Epoch={epoch:04d}, train_loss={train_loss:.4f}, train_tau={train_tau:.4f}, train_batch_time={train_batch_time:.2f}s')
            if not use_nni and writer:
                writer.add_scalar('train_loss', train_loss, global_step=epoch)
                writer.add_scalar('train_tau', train_tau, global_step=epoch)

        if epoch % (epochs//10) == 0:
            if len(val_all_loader) != 0 and len(val_top_loader) != 0:
                all_val_loss, all_val_tau, all_val_batch_time, top_val_loss, top_val_tau, top_val_batch_time = test(val_all_loader, val_top_loader, batch_size, model, predictor, criterion)
                if verbose and use_nni:
                    nni.report_intermediate_result(all_val_tau)
                is_best = False                
                if best_all_val_tau < all_val_tau:
                    best_all_val_tau = all_val_tau
                    is_best = True
                if best_top_val_tau < top_val_tau:
                    best_top_val_tau = top_val_tau
                    is_best = True
                if verbose:
                    print(('*' if is_best else '') + f'(FV) | Epoch={epoch:04d}, all_val_loss={all_val_loss:.4f}, all_val_tau={all_val_tau:.4f}, all_val_batch_time={all_val_batch_time:.2f}s, top_val_loss={top_val_loss:.4f}, top_val_tau={top_val_tau:.4f}, top_val_batch_time={top_val_batch_time:.2f}s')
                    if not use_nni and writer:
                        writer.add_scalar('all_val_loss', all_val_loss, global_step=epoch)
                        writer.add_scalar('all_val_tau', all_val_tau, global_step=epoch)
                        writer.add_scalar('top_val_loss', top_val_loss, global_step=epoch)
                        writer.add_scalar('top_val_tau', top_val_tau, global_step=epoch)
                        torch.save(model.state_dict(), os.path.join(args.save, 'finetune_encoder_{}.pt'.format(args.seed)))
                        torch.save(predictor.state_dict(), os.path.join(args.save, 'finetune_predictor_{}.pt'.format(args.seed)))
                        if is_best:
                            torch.save(model.state_dict(), os.path.join(args.save, 'best_finetune_encoder_{}.pt'.format(args.seed)))
                            torch.save(predictor.state_dict(), os.path.join(args.save, 'best_finetune_predictor_{}.pt'.format(args.seed)))
            else:
                torch.save(model.state_dict(), os.path.join(args.save, 'finetune_encoder_{}.pt'.format(args.seed)))
                torch.save(predictor.state_dict(), os.path.join(args.save, 'finetune_predictor_{}.pt'.format(args.seed)))

    if verbose and use_nni:
        nni.report_final_result(all_val_tau)#best_all_val_tau)

    return best_all_val_tau, best_top_val_tau


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--space', type=str, default='nasbench101')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--param', type=str, default='default')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--save', type=str, default='finetune_models')
    parser.add_argument('--model_state_dict', type=str, default=None)
    parser.add_argument('--device', type=str, default=None)
    default_param = {
        'finetune_train_samples': 423,
        'finetune_batch_size': 43,
        'finetune_epochs': 75,
        'finetune_criterion': 'bpr',
        'finetune_exp_weighted': False,
        'finetune_fixed_encoder': False,
        'finetune_graph_encoder_learning_rate': 0.01,
        'finetune_learning_rate': 0.01,
        'finetune_weight_decay': 0.0001,
        'finetune_opt_reserve_p': 0.5,
        'finetune_opt_mode': 'F',
        'finetune_graph_encoder_dropout': 0.2,
        'finetune_dropout': 0.2,
        'finetune_diffusion': 0.0,
        'finetune_optimizer': 'AdamW',
        'n_layers': 16,
        'd_model': 32,
        'activation': 'prelu',
        'base_model': 'GATConv',
        'projector_layers': '64',
        'model_type': 'mae', # barlow_twins barlow_twins_l2l grace graphcl bgrl mvgrl
    }
    # add hyper-parameters into parser
    param_keys = default_param.keys()
    for key in param_keys:
        if type(default_param[key]) is not bool:
            parser.add_argument(f'--{key}', type=type(default_param[key]), nargs='?')
        else:
            parser.add_argument(f'--{key}', action='store_true')
    args = parser.parse_args()

    # parse param
    sp = SimpleParam(default=default_param)
    param = sp(source=args.param, preprocess='nni')

    ### nni
    #if 'nni' in args.param:
        #param['finetune_graph_encoder_learning_rate'] = param['finetune_learning_rate']
        #param['finetune_graph_encoder_dropout'] = param['finetune_dropout']
    ### nni

    # merge cli arguments and parsed param
    for key in param_keys:
        if getattr(args, key) is not None:
            param[key] = getattr(args, key)

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    use_nni = args.param == 'nni'

    if not use_nni:
        writer = SummaryWriter(args.save)
    else:
        writer = None

    # set device
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.device)
    # set seed
    if args.seed is None:
        args.seed = random.randint(0, 65535)
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] =str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    #torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    print(args)
    print(param)

    # get dataloader
    finetune_dataset = Dataset(space=args.space, dataset=args.dataset, root='~/datasets', diffusion_rate=param['finetune_diffusion'], seed=args.seed)
    finetune_indices = list(range(len(finetune_dataset)))
    random.shuffle(finetune_indices)
    finetune_train_indices   = finetune_indices[:param['finetune_train_samples']]
    finetune_val_indices_all = finetune_indices[param['finetune_train_samples']:param['finetune_train_samples']+int(len(finetune_indices)*0.01)]
    finetune_val_indices_top = finetune_dataset.top_indices[:int(len(finetune_indices)*0.01)]
    finetune_train_loader     = DataLoader(finetune_dataset[finetune_train_indices], batch_size=param['finetune_batch_size'], shuffle=True)
    finetune_val_all_loader   = DataLoader(finetune_dataset[finetune_val_indices_all],   batch_size=param['finetune_batch_size'], shuffle=False)
    finetune_val_top_loader   = DataLoader(finetune_dataset[finetune_val_indices_top],   batch_size=param['finetune_batch_size'], shuffle=False)

    # get model
    if param['model_type'] == 'barlow_twins':
        PairWiseLearning = PairWiseLearning_BARLOW_TWINS
    elif param['model_type'] == 'barlow_twins_l2l':
        PairWiseLearning = PairWiseLearning_BARLOW_TWINS_L2L
    elif param['model_type'] == 'grace':
        PairWiseLearning = PairWiseLearning_GRACE
    elif param['model_type'] == 'graphcl':
        PairWiseLearning = PairWiseLearning_GRAPHCL
    elif param['model_type'] == 'bgrl':
        PairWiseLearning = PairWiseLearning_BGRL
    elif param['model_type'] == 'mvgrl':
        PairWiseLearning = PairWiseLearning_MVGRL
    elif param['model_type'] == 'mae':
        PairWiseLearning = MAELearning
    else:
        raise ValueError('The model_type {:} is undefined!'.format(param['model_type']))
    model = PairWiseLearning(len(finetune_dataset.available_ops), param['n_layers'], param['d_model'], get_activation(param['activation']), get_base_model(param['base_model']), param['finetune_graph_encoder_dropout'], param['finetune_dropout'], 0, 0, 0, 0, param['projector_layers']).to(args.device)
    if args.model_state_dict is not None:
        model.load_state_dict(torch.load(os.path.join(args.save, args.model_state_dict)))

    best_all_val_tau, best_top_val_tau = finetune(args.space, finetune_train_loader, finetune_val_all_loader, finetune_val_top_loader, param['finetune_batch_size'], model, param['finetune_criterion'], param['finetune_exp_weighted'], args.device, param['finetune_fixed_encoder'], epochs=param['finetune_epochs'], optimizer=param['finetune_optimizer'], lr=(param['finetune_graph_encoder_learning_rate'], param['finetune_learning_rate']), weight_decay=param['finetune_weight_decay'], opt_reserve_p=param['finetune_opt_reserve_p'], opt_mode=None if param['finetune_opt_mode']=='none' else param['finetune_opt_mode'], verbose=True, writer=writer, use_nni=use_nni)
    print(f'***(best) | best_all_val_tau={best_all_val_tau:.4f}, best_top_val_tau={best_top_val_tau:.4f}')
