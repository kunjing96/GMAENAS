import os
import time
import nni
import copy
import torch
import numpy as np
import random
import argparse
from tensorboardX import SummaryWriter

from sp import SimpleParam
from data import DataIterator, Dataset
from torch_geometric.data import DataLoader
from model import PairWiseLearning_BARLOW_TWINS, PairWiseLearning_BARLOW_TWINS_L2L, PairWiseLearning_GRACE, PairWiseLearning_GRAPHCL, PairWiseLearning_BGRL, PairWiseLearning_MVGRL, MAELearning
from finetune import finetune
from utils import get_base_model, get_activation, AvgrageMeter


def train(loader, batch_size, model, optimizer, mm, param):
    model.train()

    batch = loader.sample(batch_size)
    if len(batch) == 2: # 101 201
        batch_x, batch_y = batch
        x            = batch_x.x.cuda(non_blocking=True)
        edge_index_x = batch_x.edge_index.cuda(non_blocking=True)
        ptr_x        = batch_x.ptr.cuda(non_blocking=True)
        y            = batch_y.x.cuda(non_blocking=True)
        edge_index_y = batch_y.edge_index.cuda(non_blocking=True)
        ptr_y        = batch_y.ptr.cuda(non_blocking=True)
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
    else:
        raise ValueError('The length of batch must be 2 or 4!')

    b_start = time.time()
    optimizer.zero_grad()

    loss = model.loss(x, edge_index_x, ptr_x, y, edge_index_y, ptr_y, pretrain_target=param['pretrain_target'])
    if len(batch) == 4:
        reduce_loss = model.loss(reduce_x, reduce_edge_index_x, reduce_ptr_x, reduce_y, reduce_edge_index_y, reduce_ptr_y)
        loss = (loss + reduce_loss) / 2

    loss.backward()
    torch.nn.utils.clip_grad_norm(model.parameters(), 5)
    optimizer.step()
    # update target network
    if hasattr(model, 'update_target_network'):
        model.update_target_network(mm)

    batch_time = time.time() - b_start

    return loss.data.item(), batch_time


def test(space, train_loader, val_all_loader, val_top_loader, batch_size, model, finetune_criterion, finetune_exp_weighted, finetune_epochs, finetune_optimizer, finetune_graph_encoder_learning_rate, finetune_learning_rate, finetune_weight_decay, finetune_opt_reserve_p, finetune_opt_mode, finetune_fixed_encoder, voc_size, n_layers, d_model, activation, base_model, finetune_graph_encoder_dropout, finetune_dropout, projector_layers):
    device = list(model.parameters())[0].device
    total_all_val_tau = []
    total_top_val_tau = []
    for _ in range(10):
        copy_model = type(model)(voc_size, n_layers, d_model, get_activation(activation), get_base_model(base_model), finetune_graph_encoder_dropout, finetune_dropout, 0, 0, 0, 0, projector_layers).to(args.device)
        copy_model.load_state_dict(model.state_dict())
        best_all_val_tau, best_top_val_tau = finetune(space, train_loader, val_all_loader, val_top_loader, batch_size, copy_model, finetune_criterion, finetune_exp_weighted, device, finetune_fixed_encoder, epochs=finetune_epochs, optimizer=finetune_optimizer, lr=(finetune_graph_encoder_learning_rate, finetune_learning_rate), weight_decay=finetune_weight_decay, opt_reserve_p=finetune_opt_reserve_p, opt_mode=finetune_opt_mode)
        total_all_val_tau.append(best_all_val_tau)
        total_top_val_tau.append(best_top_val_tau)
    total_all_val_tau = np.array(total_all_val_tau)
    total_top_val_tau = np.array(total_top_val_tau)

    return total_all_val_tau.mean(), total_all_val_tau.std(), total_top_val_tau.mean(), total_top_val_tau.std()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--space', type=str, default='nasbench101')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--param', type=str, default='default')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--save', type=str, default='pretrained_models')
    parser.add_argument('--device', type=str, default=None)
    default_param = {
        'batch_size': 2048,
        'num_iterations': 10000,
        'learning_rate': 0.0001,
        'weight_decay': 0.1,
        'graph_encoder_dropout': 0.0,
        'dropout': 0.75,
        'diffusion': 0.0,
        'drop_edge_rate_1': 0.0,
        'drop_edge_rate_2': 0.0,
        'drop_feature_rate_1': 0.0,
        'drop_feature_rate_2': 0.0,
        'mm': 0.99,
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
        'pretrain_target': 'masked', # masked all
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
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    print(args)
    print(param)

    # get dataloader
    if args.space == 'nasbench101':
        from nasbench import api as NASBench101API
        data_path = os.path.join(os.path.expanduser(os.path.normpath('~/datasets')), 'nasbench_full.tfrecord')
        nasbench = NASBench101API.NASBench(data_path)
    elif args.space == 'nasbench201':
        from nas_201_api import NASBench201API
        data_path = os.path.join(os.path.expanduser(os.path.normpath('~/datasets')), 'NAS-Bench-201-v1_1-096897.pth')
        nasbench = NASBench201API(data_path, verbose=False)
    elif args.space == 'nasbench301':
        import nasbench301  as NASBench301API
        data_path = os.path.join(os.path.expanduser(os.path.normpath('~')), 'datasets', 'nb_models', 'xgb_v1.0')
        nasbench = NASBench301API.load_ensemble(data_path)
    else:
        nasbench = None
    # for pretrain
    loader = DataIterator(space=args.space, dataset=args.dataset, root='~/datasets', nasbench=nasbench, diffusion_rate=param['finetune_diffusion'], seed=args.seed)
    # for finetune
    if args.space != 'darts':
        finetune_dataset = Dataset(space=args.space, dataset=args.dataset, root='~/datasets', nasbench=nasbench, diffusion_rate=param['finetune_diffusion'], seed=args.seed)
        finetune_indices = list(range(len(finetune_dataset)))
        random.shuffle(finetune_indices)
        finetune_train_indices   = finetune_indices[:param['finetune_train_samples']]
        finetune_val_indices_all = finetune_indices[param['finetune_train_samples']:param['finetune_train_samples']+10000]
        finetune_val_indices_top = finetune_dataset.top_indices[:1000]
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
    model = PairWiseLearning(len(loader.available_ops), param['n_layers'], param['d_model'], get_activation(param['activation']), get_base_model(param['base_model']), param['graph_encoder_dropout'], param['dropout'], param['drop_edge_rate_1'], param['drop_edge_rate_2'], param['drop_feature_rate_1'], param['drop_feature_rate_2'], param['projector_layers']).to(args.device)

    # get optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=param['learning_rate'],
        weight_decay=param['weight_decay']
    )

    # get scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, param['num_iterations'])

    if args.space != 'darts':
        all_val_tau, all_val_tau_std, top_val_tau, top_val_tau_std = test(args.space, finetune_train_loader, finetune_val_all_loader, finetune_val_top_loader, param['finetune_batch_size'], model, param['finetune_criterion'], param['finetune_exp_weighted'], param['finetune_epochs'], param['finetune_optimizer'], param['finetune_graph_encoder_learning_rate'], param['finetune_learning_rate'], param['finetune_weight_decay'], param['finetune_opt_reserve_p'], None if param['finetune_opt_mode']=='none' else param['finetune_opt_mode'], param['finetune_fixed_encoder'], len(loader.available_ops), param['n_layers'], param['d_model'], param['activation'], param['base_model'], param['finetune_graph_encoder_dropout'], param['finetune_dropout'], param['projector_layers'])
        if use_nni:
            nni.report_intermediate_result(all_val_tau)
        print(f'(PV) | Step=0000, all_val_tau = {all_val_tau:.4f}, top_val_tau = {top_val_tau:.4f}')
        if not use_nni and writer:
            writer.add_scalar('all_val_tau', all_val_tau, global_step=0)
            writer.add_scalar('top_val_tau', top_val_tau, global_step=0)

    best_tau = -1
    best_loss = np.inf
    warmup_steps = 0
    for step in range(1, param['num_iterations'] + 1):
        scheduler.step()
        mm = 1 - (1 - param['mm']) * (1 + np.cos((step - warmup_steps) * np.pi / (param['num_iterations'] - warmup_steps))) / 2

        loss, batch_time = train(loader, param['batch_size'], model, optimizer, mm, param)
        print(f'(PT) | Step={step:04d}, loss={loss:.4f}, batch_time={batch_time:.2f}s')
        if not use_nni and writer:
            writer.add_scalar('loss', loss, global_step=step)

        if step % (param['num_iterations']//10) == 0:
            if args.space != 'darts':
                all_val_tau, all_val_tau_std, top_val_tau, top_val_tau_std = test(args.space, finetune_train_loader, finetune_val_all_loader, finetune_val_top_loader, param['finetune_batch_size'], model, param['finetune_criterion'], param['finetune_exp_weighted'], param['finetune_epochs'], param['finetune_optimizer'], param['finetune_graph_encoder_learning_rate'], param['finetune_learning_rate'], param['finetune_weight_decay'], param['finetune_opt_reserve_p'], None if param['finetune_opt_mode']=='none' else param['finetune_opt_mode'], param['finetune_fixed_encoder'], len(loader.available_ops), param['n_layers'], param['d_model'], param['activation'], param['base_model'], param['finetune_graph_encoder_dropout'], param['finetune_dropout'], param['projector_layers'])
                if use_nni:
                    nni.report_intermediate_result(all_val_tau)
                is_best = False
                if best_tau < all_val_tau:
                    best_tau = all_val_tau
                    is_best = True
                print(('*' if is_best else '') + f'(PV) | Step={step:04d}, all_val_tau = {all_val_tau:.4f}, all_val_tau_std = {all_val_tau_std:.4f}, top_val_tau = {top_val_tau:.4f}, top_val_tau_std = {top_val_tau_std:.4f}')
                if not use_nni and writer:
                    writer.add_scalar('all_val_tau', all_val_tau, global_step=step)
                    writer.add_scalar('top_val_tau', top_val_tau, global_step=step)
                if not use_nni:
                    torch.save(model.state_dict(), os.path.join(args.save, 'model_{}.pt'.format(args.seed)))
                    torch.save(model.state_dict(), os.path.join(args.save, 'model_{}_e{}.pt'.format(args.seed, step)))
                    if is_best:
                        torch.save(model.state_dict(), os.path.join(args.save, 'best_model_{}.pt'.format(args.seed)))
            else:
                if use_nni:
                    nni.report_intermediate_result(loss)
                is_best = False
                if best_loss > loss:
                    best_loss = loss
                    is_best = True
                if not use_nni:
                    torch.save(model.state_dict(), os.path.join(args.save, 'model_{}.pt'.format(args.seed)))
                    if is_best:
                        torch.save(model.state_dict(), os.path.join(args.save, 'best_model_{}.pt'.format(args.seed)))

    if use_nni:
        if args.space != 'darts':
            nni.report_final_result(all_val_tau)
        else:
            nni.report_final_result(loss)
