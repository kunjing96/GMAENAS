import os, sys, time, random, argparse, collections, copy, logging, PIL, json
import torch
from torch.distributions import Categorical, Bernoulli
import numpy as np
from tqdm import tqdm
from functools import cmp_to_key
from nasbench import api as NASBench101API
from nas_201_api import NASBench201API
import nasbench301 as NASBench301API
from itertools import chain, repeat
from torch_geometric.data import InMemoryDataset, Data, Batch, DataLoader

from data import sample_darts_arch, nasbench201_to_nasbench101, darts_to_nasbench101, Genotype, Dataset
from utils import get_base_model, get_activation
from model import MAELearning, Predictor
from finetune import BPRLoss, train, calculate_fisher
from model import PairWiseLearning_BARLOW_TWINS, PairWiseLearning_BARLOW_TWINS_L2L, PairWiseLearning_GRACE, PairWiseLearning_GRAPHCL, PairWiseLearning_BGRL, PairWiseLearning_MVGRL, MAELearning, Predictor
from darts.cnn.train_search import Train


def prepare_logger(args):
    args = copy.deepcopy(args)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save_dir, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info('Main Function with logger : {:}'.format(logging))
    logging.info('Arguments : -------------------------------')
    for name, value in args._get_kwargs():
        logging.info('{:16} : {:}'.format(name, value))
    logging.info("Python  Version  : {:}".format(sys.version.replace('\n', ' ')))
    logging.info("Pillow  Version  : {:}".format(PIL.__version__))
    logging.info("PyTorch Version  : {:}".format(torch.__version__))
    logging.info("cuDNN   Version  : {:}".format(torch.backends.cudnn.version()))
    logging.info("CUDA available   : {:}".format(torch.cuda.is_available()))
    logging.info("CUDA GPU numbers : {:}".format(torch.cuda.device_count()))
    logging.info("CUDA_VISIBLE_DEVICES : {:}".format(os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ else 'None'))


def parse_arguments():
    parser = argparse.ArgumentParser(description='Neural Architecture Search')
    parser.add_argument('--search_space', default=None, type=str, choices=['nasbench101', 'nasbench201', 'nasbench301', 'darts'], help='search space')
    parser.add_argument('--search_algo', default=None, type=str, choices=['r', 'ae', 'rl', 'bo'], help='search algorithm')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset to use [cifar10, cifar100, ImageNet16-120, ImageNet1k]')
    parser.add_argument('--outdir', default='./', type=str, help='output directory')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to work on')
    parser.add_argument('--seed', type=int, default=42, help='pytorch manual seed')
    parser.add_argument('--N', default=150, type=int, help='the number of searched archs')
    parser.add_argument('--population_size', default=20, type=int)
    parser.add_argument('--tournament_size', default=5, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--EMA_momentum', default=0.9, type=float)
    parser.add_argument('--flops_limit', default=600e6, type=float)
    parser.add_argument('--encoder_model_state_dict', default=None, type=str)
    parser.add_argument('--predictor_model_state_dict', default=None, type=str)
    args = parser.parse_args()
    args.device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")
    if args.search_space == 'nasbench101':
        args.api_loc = './data/nasbench_full.tfrecord'
    elif args.search_space == 'nasbench201':
        args.api_loc = './data/NAS-Bench-201-v1_1-096897.pth'
    elif args.search_space == 'nasbench301':
        args.api_loc = './data/nb_models/xgb_v1.0'
    else:
        args.api_loc = None
    if args.search_space == 'nasbench101':
        with open('param/end2end_finetune_nasbench101.json') as f:
            args.net_params = json.load(f)
    elif args.search_space == 'nasbench201':
        with open('param/end2end_finetune_nasbench201.json') as f:
            args.net_params = json.load(f)
    elif args.search_space == 'nasbench301':
        with open('param/end2end_finetune_nasbench301.json') as f:
            args.net_params = json.load(f)
    else:
        with open('param/end2end_finetune_nasbench301.json') as f:
            args.net_params = json.load(f)
    args.save_dir = os.path.join(args.outdir, f'{args.search_space}_{args.search_algo}_N{args.N}_seed{args.seed}')
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    return args


class ExponentialMovingAverage(object):
    """Class that maintains an exponential moving average."""

    def __init__(self, momentum):
        self._numerator = 0
        self._denominator = 0
        self._momentum = momentum

    def update(self, value):
        self._numerator = (
            self._momentum * self._numerator + (1 - self._momentum) * value
        )
        self._denominator = self._momentum * self._denominator + (1 - self._momentum)

    def value(self):
        """Return the current value of the moving average"""
        return self._numerator / self._denominator


class NAS(object):

    def __init__(self, N, search_space, dataset, flops_limit, api_loc=None, encoder_model_state_dict=None, predictor_model_state_dict=None, net_params=None, device='cpu', seed=None):
        self.N = N
        self.search_space = search_space
        self.dataset = dataset
        self.flops_limit = flops_limit
        self.net_params = net_params
        self.device = device
        self.seed = seed
        self.visited = []
        if self.search_space == 'nasbench101':
            self.nasbench = NASBench101API.NASBench(api_loc)
            self.available_ops = ['input', 'conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3', 'output']
            self.max_num_vertices = 7
            self.max_num_edges = 9
        elif self.search_space == 'nasbench201':
            self.nasbench = NASBench201API(api_loc, verbose=False)
            self.available_ops = ['input', 'none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3', 'output']
            self.max_num_vertices = 8
        elif self.search_space == 'nasbench301':
            self.nasbench = NASBench301API.load_ensemble(api_loc)
            self.available_ops = ['input1', 'input2', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5', 'output']
            self.max_num_vertices = 11
        elif self.search_space == 'darts':
            self.nasbench = Train()
            self.available_ops = ['input1', 'input2', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5', 'output']
            self.max_num_vertices = 11
        else:
            raise ValueError('There is no {:} search space.'.format(self.search_space))
            
        if self.net_params['model_type'] == 'barlow_twins':
            PairWiseLearning = PairWiseLearning_BARLOW_TWINS
        elif self.net_params['model_type'] == 'barlow_twins_l2l':
            PairWiseLearning = PairWiseLearning_BARLOW_TWINS_L2L
        elif self.net_params['model_type'] == 'grace':
            PairWiseLearning = PairWiseLearning_GRACE
        elif self.net_params['model_type'] == 'graphcl':
            PairWiseLearning = PairWiseLearning_GRAPHCL
        elif self.net_params['model_type'] == 'bgrl':
            PairWiseLearning = PairWiseLearning_BGRL
        elif self.net_params['model_type'] == 'mvgrl':
            PairWiseLearning = PairWiseLearning_MVGRL
        elif self.net_params['model_type'] == 'mae':
            PairWiseLearning = MAELearning
        else:
            raise ValueError('The model_type {:} is undefined!'.format(param['model_type']))
        self.encoder = PairWiseLearning(len(self.available_ops), self.net_params['n_layers'], self.net_params['d_model'], get_activation(self.net_params['activation']), get_base_model(self.net_params['base_model']), self.net_params['finetune_graph_encoder_dropout'], self.net_params['finetune_dropout'], 0, 0, 0, 0, self.net_params['projector_layers']).to(args.device)
        if encoder_model_state_dict:
            self.encoder.load_state_dict(torch.load(encoder_model_state_dict))
        d_model = self.encoder.d_model
        if self.net_params['finetune_criterion'] == 'bce':
            d_model *= 2
        self.predictor = Predictor(d_model).to(device)
        if predictor_model_state_dict:
            self.predictor.load_state_dict(torch.load(predictor_model_state_dict))

    def sample_arch(self):
        if self.search_space == 'nasbench101':
            hash_list = list(self.nasbench.hash_iterator())
            hash_value = random.choice(hash_list)
            fixed_statistic = self.nasbench.fixed_statistics[hash_value]
            sampled_arch = (hash_value, fixed_statistic['module_adjacency'], fixed_statistic['module_operations'])
        elif self.search_space == 'nasbench201':
            arch_list = list(enumerate(self.nasbench))
            sampled_arch = random.choice(arch_list)
        elif self.search_space == 'nasbench301':
            genotype = sample_darts_arch(self.available_ops[2:-1])
            sampled_arch = genotype
        elif self.search_space == 'darts':
            genotype = sample_darts_arch(self.available_ops[2:-1])
            sampled_arch = genotype
        else:
            raise ValueError('No implementation!')

        return sampled_arch

    def eval_arch(self, arch, use_val_acc=False, encoder=None, predictor=None):
        start_time = time.time()
        if use_val_acc:
            if self.search_space == 'nasbench101':
                info = self.nasbench.computed_statistics[arch[0]][108][np.random.randint(3)]
                val_acc = info['final_validation_accuracy']
                test_acc = info['final_test_accuracy']
                total_eval_time = info['final_training_time']
                return val_acc, test_acc, total_eval_time
            elif self.search_space == 'nasbench201':
                dataset = self.dataset if self.dataset != 'cifar10' else 'cifar10-valid'
                info = self.nasbench.get_more_info(arch[0], dataset, iepoch=None, hp="200", is_random=True)
                val_acc = info['valid-accuracy'] / 100.0
                dataset = self.dataset if self.dataset != 'cifar10' else 'cifar10'
                info = self.nasbench.get_more_info(arch[0], dataset, iepoch=None, hp="200", is_random=True)
                test_acc = info['test-accuracy'] / 100.0
                total_eval_time = (info["train-all-time"] + info["valid-per-time"])
                return val_acc, test_acc, total_eval_time
            elif self.search_space == 'nasbench301':
                acc = self.nasbench.predict(config=arch, representation='genotype', with_noise=False) / 100
                total_eval_time = time.time() - start_time
                return acc, total_eval_time
            elif self.search_space == 'darts':
                val_accs, test_accs = self.nasbench.main(42, arch)
                val_acc  = np.mean(list(zip(*val_accs ))[1])
                test_acc = np.mean(list(zip(*test_accs))[1])
                total_eval_time = time.time() - start_time
                return test_acc, total_eval_time
            else:
                raise ValueError('Arch in {:} search space have to be trained or this space does not exist.'.format(self.search_space))
        else:
            if self.search_space == 'nasbench101':
                arch = ({'adj': arch[1],
                         'ops': arch[2]}, )
            elif self.search_space == 'nasbench201':
                arch = nasbench201_to_nasbench101(self.nasbench.str2lists(arch[1]))
                arch = (arch, )
            elif self.search_space == 'nasbench301':
                arch = darts_to_nasbench101(arch)
                arch = (arch, )
            elif self.search_space == 'darts':
                arch = darts_to_nasbench101(arch)
                arch = (arch, )
            else:
                raise ValueError('There is no {:} search space.'.format(self.search_space))
            batch = tuple()
            for i in range(len(arch)):
                _arch = self.arch2Data(arch[i])
                batch = (*batch, Batch.from_data_list([_arch]))
            if len(batch) == 1: # 101 201
                batch_x = batch[0]
                x            = batch_x.x.cuda(non_blocking=True)
                edge_index_x = batch_x.edge_index.cuda(non_blocking=True)
                ptr_x        = batch_x.ptr.cuda(non_blocking=True)
            elif len(batch) == 2: # 301 darts
                batch_x, batch_x_reduce = batch
                x            = batch_x.x.cuda(non_blocking=True)
                edge_index_x = batch_x.edge_index.cuda(non_blocking=True)
                ptr_x        = batch_x.ptr.cuda(non_blocking=True)
                reduce_x            = batch_x_reduce.x.cuda(non_blocking=True)
                reduce_edge_index_x = batch_x_reduce.edge_index.cuda(non_blocking=True)
                reduce_ptr_x        = batch_x_reduce.ptr.cuda(non_blocking=True)
            else:
                raise ValueError('The length of batch must be 1 or 2!')
            with torch.no_grad():
                if encoder:
                    z, g = encoder(x, edge_index_x, ptr_x)
                else:
                    z, g = self.encoder(x, edge_index_x, ptr_x)
                if len(batch) == 2:
                    if encoder:
                        reduce_z, reduce_g = encoder(reduce_x, reduce_edge_index_x, reduce_ptr_x)
                    else:
                        reduce_z, reduce_g = self.encoder(reduce_x, reduce_edge_index_x, reduce_ptr_x)
                    z = torch.cat([z, reduce_z], -1)
                    g = torch.cat([g, reduce_g], -1)
                g = g.detach()
                if predictor:
                    output = predictor(g)
                else:
                    output = self.predictor(g)
                measure = output.squeeze().detach().cpu().item()
            total_eval_time = time.time() - start_time
            return measure, total_eval_time, g

    def arch2Data(self, arch):
        x = torch.tensor([self.available_ops.index(x) for x in arch['ops']], dtype=torch.long)
        if 'acc' in arch.keys():
            y = torch.ones_like(x) * arch['acc']
        else:
            y = None
        forward_edges = [[(i, j) for j, x in enumerate(xs) if x > 0] for i, xs in enumerate(arch['adj'])]
        forward_edges = np.array(list(chain(*forward_edges)))
        backward_edges = forward_edges[::-1, ::-1]
        edges = np.concatenate([forward_edges, backward_edges])
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        data = Data(x=x, y=y, edge_index=edge_index)
        return data

    def cmp(self, x, y):
        ret = x[1] - y[1]
        if ret<0: return -1
        elif ret>0: return 1
        else: return 0
    
    def train(self, history, encoder, predictor):
        batch_size = (len(history) - 1) // 10 + 1
        # data prepare
        archs = []
        for h in history:
            if self.search_space == 'nasbench101':
                arch = {'adj': h[0][1], 'ops': h[0][2]}
            elif self.search_space == 'nasbench201':
                arch = nasbench201_to_nasbench101(self.nasbench.str2lists(h[0][1]))
            elif self.search_space == 'nasbench301':
                arch = darts_to_nasbench101(h[0])
            elif self.search_space == 'darts':
                arch = darts_to_nasbench101(h[0])
            else:
                raise ValueError('There is no {:} search space.'.format(self.search_space))
            arch.update({'acc': h[1]})
            archs.append((arch, ))
        finetune_dataset = Dataset(space=self.search_space, dataset=self.dataset, root='~/datasets', diffusion_rate=self.net_params['finetune_diffusion'], seed=self.seed, archs=archs)
        finetune_train_loader = DataLoader(finetune_dataset, batch_size=batch_size, shuffle=True)

        # get optimizer
        if self.net_params['finetune_fixed_encoder']:
            params = [
                {'params': predictor.parameters(), 'lr': self.net_params['finetune_learning_rate']},
            ]
        else:
            params = [
                {'params': encoder.parameters(), 'lr': self.net_params['finetune_graph_encoder_learning_rate']},
                {'params': predictor.parameters(), 'lr': self.net_params['finetune_learning_rate']},
            ]
        if self.net_params['finetune_optimizer'] == 'SGD':
            optimizer = torch.optim.SGD(params, lr=0, momentum=0.9, weight_decay=self.net_params['finetune_weight_decay'])
        elif self.net_params['finetune_optimizer'] =='AdamW':
            optimizer = torch.optim.AdamW(params, lr=0, weight_decay=self.net_params['finetune_weight_decay'])

        # get scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.net_params['finetune_epochs'], eta_min=1e-4)

        # get criterion
        if self.net_params['finetune_criterion'] == 'mse':
            criterion = MSELoss(exp_weighted=self.net_params['finetune_exp_weighted'])
        elif self.net_params['finetune_criterion'] == 'hinge':
            criterion = HingePairwiseLoss(exp_weighted=self.net_params['finetune_exp_weighted'])
        elif self.net_params['finetune_criterion'] == 'bpr':
            criterion = BPRLoss(exp_weighted=self.net_params['finetune_exp_weighted'])
        elif self.net_params['finetune_criterion'] == 'bce':
            criterion = torch.nn.BCEWithLogitsLoss()
        elif self.net_params['finetune_criterion'] == 'mine':
            criterion = MyLoss(exp_weighted=self.net_params['finetune_exp_weighted'])
        elif self.net_params['finetune_criterion'] == 'mse+hinge':
            criterion = MSEHingeLoss(exp_weighted=self.net_params['finetune_exp_weighted'])
        elif self.net_params['finetune_criterion'] == 'mse+bpr':
            criterion = MSEBPRLoss(exp_weighted=self.net_params['finetune_exp_weighted'])
        else:
            raise ValueError('No implementation!')

        # =================== HACK BEGIN =====================
        if self.net_params['finetune_opt_mode'] == 'D':
            gradient_mask = calculate_fisher(finetune_train_loader, encoder, predictor, criterion, self.net_params['finetune_opt_reserve_p'])
        else:
            gradient_mask = None
        # =================== HACK END =======================

        best_all_val_tau = -1
        best_top_val_tau = -1
        iters = tqdm(range(1, self.net_params['finetune_epochs']+1))
        for epoch in iters:
            scheduler.step()
            train_loss, train_tau, train_batch_time = train(finetune_train_loader, batch_size, encoder, predictor, optimizer, criterion, self.net_params['finetune_fixed_encoder'], self.net_params['finetune_opt_mode'], self.net_params['finetune_opt_reserve_p'], gradient_mask)

        return encoder, predictor
    
    def predict(self, candidates, encoder, predictor):
        predictions = []
        for c in candidates:
            measure, _, _ = self.eval_arch(c, use_val_acc=False, encoder=encoder, predictor=predictor)
            predictions.append(measure)
        return predictions


class Random_NAS(NAS):

    def __init__(self, N, search_space, dataset, flops_limit, api_loc=None, encoder_model_state_dict=None, predictor_model_state_dict=None, num_init_archs=20, K=10, net_params=None, device='cpu', seed=None):
        super(Random_NAS, self).__init__(N, search_space, dataset, flops_limit, api_loc=api_loc, encoder_model_state_dict=encoder_model_state_dict, predictor_model_state_dict=predictor_model_state_dict, net_params=net_params, device=device, seed=seed)
        self.num_init_archs = num_init_archs
        self.K = K

    def get_candidates(self):
        patience_factor = 5
        num = 100
        candidates = []
        for _ in range(patience_factor):
            for _ in range(int(num)):
                arch = self.sample_arch()
                if arch[0] not in self.visited:
                    candidates.append(arch)
                    self.visited.append(arch[0])
                if len(candidates) >= num:
                    return candidates
        return candidates

    def run(self):
        total_eval_time = 0
        history  = []
        while len(history) < self.num_init_archs:
            arch = self.sample_arch()
            if arch[0] not in self.visited:
                if self.search_space in ['nasbench101', 'nasbench201']:
                    valid_acc, test_acc, eval_time = self.eval_arch(arch, use_val_acc=True)
                    cur = (arch, valid_acc, test_acc, eval_time)
                else:
                    acc, eval_time = self.eval_arch(arch, use_val_acc=True)
                    cur = (arch, acc, eval_time)
                total_eval_time += eval_time
                history.append(cur)
                self.visited.append(arch[0])
        while len(history) < self.N:
            candidates = self.get_candidates()
            encoder = copy.deepcopy(self.encoder)
            predictor = copy.deepcopy(self.predictor)
            encoder, predictor = self.train(history, encoder, predictor)
            candidate_predictions = self.predict(candidates, encoder, predictor)
            candidate_indices = np.argsort(candidate_predictions)
            for i in candidate_indices[-self.K:]:
                arch = candidates[i]
                if self.search_space in ['nasbench101', 'nasbench201']:
                    valid_acc, test_acc, eval_time = self.eval_arch(arch, use_val_acc=True)
                    cur = (arch, valid_acc, test_acc, eval_time)
                else:
                    acc, eval_time = self.eval_arch(arch, use_val_acc=True)
                    cur = (arch, acc, eval_time)
                total_eval_time += eval_time
                history.append(cur)
        best = max(history, key=lambda x: x[1])
        return best, history, total_eval_time


class RL_NAS(NAS):

    def __init__(self, N, search_space, dataset, flops_limit, lr, EMA_momentum, api_loc=None, encoder_model_state_dict=None, predictor_model_state_dict=None, num_init_archs=20, K=10, net_params=None, device='cpu', seed=None):
        super(RL_NAS, self).__init__(N, search_space, dataset, flops_limit, api_loc=api_loc, encoder_model_state_dict=encoder_model_state_dict, predictor_model_state_dict=predictor_model_state_dict, net_params=net_params, device=device, seed=seed)
        self.num_init_archs = num_init_archs
        self.K = K
        if self.search_space == 'nasbench101':
            self.op_parameters = torch.nn.Parameter(1e-3 * torch.randn(self.max_num_vertices-2, len(self.available_ops)-2))
            #self.edge_parameters = torch.nn.Parameter(1e-3 * torch.randn(self.max_num_vertices, self.max_num_vertices))
            #self.edge_parameters = torch.nn.Parameter(1e-3 * torch.randn(self.max_num_vertices, self.max_num_vertices) - np.log((self.max_num_vertices-1)*self.max_num_vertices//2/self.max_num_edges-1))
            #self.edge_parameters = torch.nn.Parameter(1e-3 * torch.randn(self.max_num_edges, self.max_num_vertices * (self.max_num_vertices-1) // 2))
            self.edge_parameters = torch.nn.Parameter(1e-3 * torch.randn(self.max_num_vertices * (self.max_num_vertices-1) // 2))
            self.optimizer = torch.optim.Adam([self.op_parameters, self.edge_parameters], lr=lr)
        elif self.search_space == 'nasbench201':
            self.arch_parameters = torch.nn.Parameter(1e-3 * torch.randn(self.max_num_vertices-2, len(self.available_ops)-2))
            self.optimizer = torch.optim.Adam([self.arch_parameters], lr=lr)
        elif self.search_space == 'nasbench301':
            self.op_parameters = torch.nn.Parameter(1e-3 * torch.randn(2*(self.max_num_vertices-3), len(self.available_ops)-3))
            self.edge_parameters = [torch.nn.Parameter(1e-3 * torch.randn(2*2, 2+i)) for i in range((self.max_num_vertices-3)//2)]
            self.optimizer = torch.optim.Adam([self.op_parameters] + self.edge_parameters, lr=lr)
        elif self.search_space == 'darts':
            self.op_parameters = torch.nn.Parameter(1e-3 * torch.randn(2*(self.max_num_vertices-3), len(self.available_ops)-3))
            self.edge_parameters = [torch.nn.Parameter(1e-3 * torch.randn(2*2, 2+i)) for i in range((self.max_num_vertices-3)//2)]
            self.optimizer = torch.optim.Adam([self.op_parameters] + self.edge_parameters, lr=lr)
        else:
            raise ValueError('There is no {:} search space.'.format(self.search_space))
        self.baseline = ExponentialMovingAverage(EMA_momentum)

    def select_action(self):
        if self.search_space == 'nasbench101':
            while True:
                op_probs = torch.nn.functional.softmax(self.op_parameters, dim=-1)
                m_op = Categorical(op_probs)
                ops = m_op.sample()
                ops_ = ['input'] + [self.available_ops[1:-1][op] for op in ops] + ['output']
                #matrix_probs = torch.nn.functional.sigmoid(self.edge_parameters).triu(1)
                #m_mat = Bernoulli(matrix_probs)
                #matrix = m_mat.sample()
                #matrix_ = np.int8(matrix.cpu().numpy())
                #matrix_probs = torch.nn.functional.softmax(self.edge_parameters, dim=-1)
                #m_mat = Categorical(matrix_probs)
                #matrix = m_mat.sample()
                #idx_to_ij = {int(i*(self.max_num_vertices-1)-i*(i-1)/2+(j-i-1)): (i, j) for i in range(self.max_num_vertices) for j in range(i+1, self.max_num_vertices)}
                #matrix_ = np.zeros((self.max_num_vertices, self.max_num_vertices), dtype=np.int8)
                #for idx in matrix:
                    #ij = idx_to_ij[int(idx)]
                    #matrix_[ij[0], ij[1]] = 1
                matrix_probs = torch.nn.functional.softmax(self.edge_parameters, dim=-1)
                m_mat = Categorical(matrix_probs)
                matrix = []
                for _ in range(self.max_num_edges):
                    matrix.append(m_mat.sample())
                matrix = torch.tensor(matrix)
                idx_to_ij = {int(i*(self.max_num_vertices-1)-i*(i-1)/2+(j-i-1)): (i, j) for i in range(self.max_num_vertices) for j in range(i+1, self.max_num_vertices)}
                matrix_ = np.zeros((self.max_num_vertices, self.max_num_vertices), dtype=np.int8)
                for idx in matrix:
                    ij = idx_to_ij[int(idx)]
                    matrix_[ij[0], ij[1]] = 1
                spec = NASBench101API.ModelSpec(matrix=matrix_, ops=ops_)
                if self.nasbench.is_valid(spec):
                    break
            return (m_op.log_prob(ops), m_mat.log_prob(matrix)), (ops, matrix)
        elif self.search_space == 'nasbench201':
            probs = torch.nn.functional.softmax(self.arch_parameters, dim=-1)
            m = Categorical(probs)
            action = m.sample()
            return m.log_prob(action), action.cpu().tolist()
        elif self.search_space == 'nasbench301':
            op_probs = torch.nn.functional.softmax(self.op_parameters, dim=-1)
            m_op = Categorical(op_probs)
            ops = m_op.sample()
            edges = []
            log_prob_edges = []
            for edge_parameters in self.edge_parameters:
                edge_prob = torch.nn.functional.softmax(edge_parameters, dim=-1)
                m_edge = Categorical(edge_prob)
                while True:
                    edge = m_edge.sample()
                    if edge[0] != edge[1] and edge[2] != edge[3]:
                        break
                edges.append(edge)
                log_prob_edges.append(m_edge.log_prob(edge))
            return (m_op.log_prob(ops), tuple(log_prob_edges)), (ops, tuple(edges))
        elif self.search_space == 'darts':
            op_probs = torch.nn.functional.softmax(self.op_parameters, dim=-1)
            m_op = Categorical(op_probs)
            ops = m_op.sample()
            edges = []
            log_prob_edges = []
            for edge_parameters in self.edge_parameters:
                edge_prob = torch.nn.functional.softmax(edge_parameters, dim=-1)
                m_edge = Categorical(edge_prob)
                while True:
                    edge = m_edge.sample()
                    if edge[0] != edge[1] and edge[2] != edge[3]:
                        break
                edges.append(edge)
                log_prob_edges.append(m_edge.log_prob(edge))
            return (m_op.log_prob(ops), tuple(log_prob_edges)), (ops, tuple(edges))
        else:
            raise ValueError('There is no {:} search space.'.format(self.search_space))

    def generate_arch(self, actions):
        if self.search_space == 'nasbench101':
            ops = ['input'] + [self.available_ops[1:-1][action] for action in actions[0]] + ['output']
            #matrix = np.int8(actions[1].cpu().numpy())
            #idx_to_ij = {int(i*(self.max_num_vertices-1)-i*(i-1)/2+(j-i-1)): (i, j) for i in range(self.max_num_vertices) for j in range(i+1, self.max_num_vertices)}
            #matrix = np.zeros((self.max_num_vertices, self.max_num_vertices), dtype=np.int8)
            #for idx in actions[1]:
                #ij = idx_to_ij[int(idx)]
                #matrix[ij[0], ij[1]] = 1
            idx_to_ij = {int(i*(self.max_num_vertices-1)-i*(i-1)/2+(j-i-1)): (i, j) for i in range(self.max_num_vertices) for j in range(i+1, self.max_num_vertices)}
            matrix = np.zeros((self.max_num_vertices, self.max_num_vertices), dtype=np.int8)
            for idx in actions[1]:
                ij = idx_to_ij[int(idx)]
                matrix[ij[0], ij[1]] = 1
            spec = NASBench101API.ModelSpec(matrix=matrix, ops=ops)
            spec_hash = spec.hash_spec(self.available_ops[1:-1])
            return (spec_hash, matrix, ops)
        elif self.search_space == 'nasbench201':
            spec = [self.available_ops[1:-1][action] for action in actions]
            arch_str = '|{:}~0|+|{:}~0|{:}~1|+|{:}~0|{:}~1|{:}~2|'.format(*spec)
            i = self.nasbench.query_index_by_arch(arch_str)
            return (i, arch_str)
        elif self.search_space == 'nasbench301':
            geno = []
            for c in range(2):
                cell = []
                for i in range((self.max_num_vertices-3)//2):
                    cell.extend([(self.available_ops[2:-1][actions[0][c*8+i*2]], actions[1][i][2*c].item()), (self.available_ops[2:-1][actions[0][c*8+i*2+1]], actions[1][i][2*c+1].item())])
                geno.append(cell)
            genotype = Genotype(normal=geno[0], normal_concat=list(range(2,2+(self.max_num_vertices-3)//2)), reduce=geno[1], reduce_concat=list(range(2,2+(self.max_num_vertices-3)//2)))
            return genotype
        elif self.search_space == 'darts':
            geno = []
            for c in range(2):
                cell = []
                for i in range((self.max_num_vertices-3)//2):
                    cell.extend([(self.available_ops[2:-1][actions[0][c*8+i*2]], actions[1][i][2*c].item()), (self.available_ops[2:-1][actions[0][c*8+i*2+1]], actions[1][i][2*c+1].item())])
                geno.append(cell)
            genotype = Genotype(normal=geno[0], normal_concat=list(range(2,2+(self.max_num_vertices-3)//2)), reduce=geno[1], reduce_concat=list(range(2,2+(self.max_num_vertices-3)//2)))
            return genotype
        else:
            raise ValueError('There is no {:} search space.'.format(self.search_space))

    def select_generate(self):
        log_prob, action = self.select_action()
        arch = self.generate_arch(action)
        return log_prob, arch

    def get_candidates(self):                
        patience_factor = 5
        num = 100
        candidates = []
        log_probs  = []
        for _ in range(patience_factor):
            for _ in range(int(num)):
                log_prob, arch = self.select_generate()
                if arch[0] not in self.visited:
                    candidates.append(arch)
                    log_probs.append(log_prob)
                    self.visited.append(arch[0])
                if len(candidates) >= num:
                    return candidates, log_probs
        return candidates, log_probs
    
    def run(self):
        runs = []
        for _ in tqdm(range(200)):
            arch = self.sample_arch()
            res = self.eval_arch(arch, use_val_acc=True)
            runs.append(res[0])
        runs = np.array(runs)
        mean_value = runs.mean()
        std_value  = runs.std()

        total_eval_time = 0
        history  = []
        while len(history) < self.num_init_archs:
            arch = self.sample_arch()
            if arch[0] not in self.visited:
                if self.search_space in ['nasbench101', 'nasbench201']:
                    valid_acc, test_acc, eval_time = self.eval_arch(arch, use_val_acc=True)
                    cur = (arch, valid_acc, test_acc, eval_time)
                else:
                    acc, eval_time = self.eval_arch(arch, use_val_acc=True)
                    cur = (arch, acc, eval_time)
                total_eval_time += eval_time
                history.append(cur)
                self.visited.append(arch[0])
        while len(history) < self.N:
            candidates, log_probs = self.get_candidates()
            encoder = copy.deepcopy(self.encoder)
            predictor = copy.deepcopy(self.predictor)
            encoder, predictor = self.train(history, encoder, predictor)
            candidate_predictions = self.predict(candidates, encoder, predictor)
            candidate_indices = np.argsort(candidate_predictions)
            reward = 0
            for i in candidate_indices[-self.K:]:
                arch = candidates[i]
                if self.search_space in ['nasbench101', 'nasbench201']:
                    valid_acc, test_acc, eval_time = self.eval_arch(arch, use_val_acc=True)
                    cur = (arch, valid_acc, test_acc, eval_time)
                else:
                    acc, eval_time = self.eval_arch(arch, use_val_acc=True)
                    cur = (arch, acc, eval_time)
                total_eval_time += eval_time
                history.append(cur)
                reward += cur[1]
            reward = (reward / self.K - mean_value) / std_value
            if not (np.isnan(reward) or np.isinf(reward)):
                self.baseline.update(reward)
                policy_loss = 0
                for log_prob in [log_probs[idx] for idx in candidate_indices[-self.K:]]:
                    if self.search_space == 'nasbench101':
                        policy_loss += (-log_prob[0] * (reward - self.baseline.value())).sum() + (-log_prob[1] * (reward - self.baseline.value())).sum()
                    elif self.search_space == 'nasbench201':
                        policy_loss += (-log_prob * (reward - self.baseline.value())).sum()
                    elif self.search_space == 'nasbench301':
                        policy_loss += sum([(-log_prob[0] * (reward - self.baseline.value())).sum()] + [(-x * (reward - self.baseline.value())).sum() for x in log_prob[1]])
                    elif self.search_space == 'darts':
                        policy_loss += sum([(-log_prob[0] * (reward - self.baseline.value())).sum()] + [(-x * (reward - self.baseline.value())).sum() for x in log_prob[1]])
                    else:
                        raise ValueError('There is no {:} search space.'.format(self.search_space))
                policy_loss = policy_loss / self.K
                self.optimizer.zero_grad()
                policy_loss.backward()
                self.optimizer.step()
            else:
                print('No updating!')
        best = max(history, key=lambda x: x[1])
        return best, history, total_eval_time


class Evolved_NAS(NAS):

    def __init__(self, N, search_space, population_size, tournament_size, dataset, flops_limit, api_loc=None, encoder_model_state_dict=None, predictor_model_state_dict=None, K=10, net_params=None, device='cpu', seed=None):
        super(Evolved_NAS, self).__init__(N, search_space, dataset, flops_limit, api_loc=api_loc, encoder_model_state_dict=encoder_model_state_dict, predictor_model_state_dict=predictor_model_state_dict, net_params=net_params, device=device, seed=seed)
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.K = K

    def mutate(self, parent, p):
        if self.search_space == 'nasbench101':
            if random.random() < p:
                while True:
                    old_matrix, old_ops = parent[1], parent[2]
                    idx_to_change = random.randrange(len(old_ops[1:-1])) + 1
                    entry_to_change = old_ops[idx_to_change]
                    possible_entries = [x for x in self.available_ops[1:-1] if x != entry_to_change]
                    new_entry = random.choice(possible_entries)
                    new_ops = copy.deepcopy(old_ops)
                    new_ops[idx_to_change] = new_entry
                    idx_to_change = random.randrange(sum(range(1, len(old_matrix))))
                    new_matrix = copy.deepcopy(old_matrix)
                    num_node = len(old_matrix)
                    idx_to_ij = {int(i*(num_node-1)-i*(i-1)/2+(j-i-1)): (i, j) for i in range(num_node) for j in range(i+1, num_node)}
                    i, j = idx_to_ij[idx_to_change]
                    new_matrix[i][j] = 1 if new_matrix[i][j] == 0 else 0
                    new_spec = NASBench101API.ModelSpec(matrix=new_matrix, ops=new_ops)
                    if self.nasbench.is_valid(new_spec):
                        spec_hash = new_spec.hash_spec(self.available_ops[1:-1])
                        child = (spec_hash, new_matrix, new_ops)
                        break
            else:
                child = parent
        elif self.search_space == 'nasbench201':
            if random.random() < p:
                nodes = parent[1].split('+')
                nodes = [node[1:-1].split('|') for node in nodes]
                nodes = [[op_and_input.split('~')[0]  for op_and_input in node] for node in nodes]
                old_spec = [op for node in nodes for op in node]
                idx_to_change = random.randrange(len(old_spec))
                entry_to_change = old_spec[idx_to_change]
                possible_entries = [x for x in self.available_ops[1:-1] if x != entry_to_change]
                new_entry = random.choice(possible_entries)
                new_spec = copy.deepcopy(old_spec)
                new_spec[idx_to_change] = new_entry
                arch_str = '|{:}~0|+|{:}~0|{:}~1|+|{:}~0|{:}~1|{:}~2|'.format(*new_spec)
                i = self.nasbench.query_index_by_arch(arch_str)
                child = (i, arch_str)
            else:
                child = parent
        elif self.search_space == 'nasbench301':
            if random.random() < p:
                old_all = parent.normal + parent.reduce
                idx_to_change = random.randrange(len(old_all))
                entry_to_change = old_all[idx_to_change][0]
                possible_entries = [x for x in self.available_ops[2:-1] if x != entry_to_change]
                new_entry = random.choice(possible_entries)
                new_all = copy.deepcopy(old_all)
                new_all[idx_to_change] = (new_entry, old_all[idx_to_change][1])
                idx_to_change = random.randrange(len(old_all))
                entry_to_change = old_all[idx_to_change][1]
                entry_to_change_neibor = old_all[idx_to_change+1][1] if idx_to_change%2 == 0 else old_all[idx_to_change-1][1]
                possible_entries = [x for x in range(2+idx_to_change%len(parent.normal)//2) if x != entry_to_change and x != entry_to_change_neibor]
                if len(possible_entries) != 0:
                    new_entry = random.choice(possible_entries)
                    new_all[idx_to_change] = (old_all[idx_to_change][0], new_entry)
                child = Genotype(normal=new_all[:len(new_all)//2], normal_concat=list(range(2,2+(self.max_num_vertices-3)//2)), reduce=new_all[len(new_all)//2:], reduce_concat=list(range(2,2+(self.max_num_vertices-3)//2)))
            else:
                child = parent
        elif self.search_space == 'darts':
            if random.random() < p:
                old_all = parent.normal + parent.reduce
                idx_to_change = random.randrange(len(old_all))
                entry_to_change = old_all[idx_to_change][0]
                possible_entries = [x for x in self.available_ops[2:-1] if x != entry_to_change]
                new_entry = random.choice(possible_entries)
                new_all = copy.deepcopy(old_all)
                new_all[idx_to_change] = (new_entry, old_all[idx_to_change][1])
                idx_to_change = random.randrange(len(old_all))
                entry_to_change = old_all[idx_to_change][1]
                entry_to_change_neibor = old_all[idx_to_change+1][1] if idx_to_change%2 == 0 else old_all[idx_to_change-1][1]
                possible_entries = [x for x in range(2+idx_to_change%len(parent.normal)//2) if x != entry_to_change and x != entry_to_change_neibor]
                if len(possible_entries) != 0:
                    new_entry = random.choice(possible_entries)
                    new_all[idx_to_change] = (old_all[idx_to_change][0], new_entry)
                child = Genotype(normal=new_all[:len(new_all)//2], normal_concat=list(range(2,2+(self.max_num_vertices-3)//2)), reduce=new_all[len(new_all)//2:], reduce_concat=list(range(2,2+(self.max_num_vertices-3)//2)))
            else:
                child = parent
        else:
            raise ValueError('There is no {:} search space.'.format(self.search_space))
        return child


    def get_candidates(self, arch_pool):
        '''if i < self.N / 2:
            lambd = (self.N - 2*i) / self.N
            p = lambd + 0.1 * (1 - lambd)
        else:
            p = 0.1'''
        p = 1.0
        num_arches_to_mutate = 1
        patience_factor = 5
        num = 100
        candidates = []
        for _ in range(patience_factor):
            samples  = random.sample(arch_pool, self.tournament_size)
            parents = [arch[0] for arch in sorted(samples, key=cmp_to_key(self.cmp), reverse=True)[:num_arches_to_mutate]]
            for parent in parents:
                for _ in range(int(num / num_arches_to_mutate)):
                    child = self.mutate(parent, p)
                    if child[0] not in self.visited:
                        candidates.append(child)
                        self.visited.append(child[0])
                    if len(candidates) >= num:
                        return candidates
        return candidates

    def run(self):
        total_eval_time = 0
        history  = []
        population = collections.deque()
        while len(history) < self.population_size:
            arch = self.sample_arch()
            if arch[0] not in self.visited:
                if self.search_space in ['nasbench101', 'nasbench201']:
                    valid_acc, test_acc, eval_time = self.eval_arch(arch, use_val_acc=True)
                    cur = (arch, valid_acc, test_acc, eval_time)
                else:
                    acc, eval_time = self.eval_arch(arch, use_val_acc=True)
                    cur = (arch, acc, eval_time)
                total_eval_time += eval_time
                population.append(cur)
                history.append(cur)
                self.visited.append(arch[0])
        while len(history) < self.N:
            candidates = self.get_candidates(population)
            encoder = copy.deepcopy(self.encoder)
            predictor = copy.deepcopy(self.predictor)
            encoder, predictor = self.train(history, encoder, predictor)
            candidate_predictions = self.predict(candidates, encoder, predictor)
            candidate_indices = np.argsort(candidate_predictions)
            for i in candidate_indices[-self.K:]:
                arch = candidates[i]
                if self.search_space in ['nasbench101', 'nasbench201']:
                    valid_acc, test_acc, eval_time = self.eval_arch(arch, use_val_acc=True)
                    cur = (arch, valid_acc, test_acc, eval_time)
                else:
                    acc, eval_time = self.eval_arch(arch, use_val_acc=True)
                    cur = (arch, acc, eval_time)
                total_eval_time += eval_time
                population.append(cur)
                history.append(cur)
                population.popleft()
        best = max(history, key=lambda x: x[1])
        return best, history, total_eval_time


class BANANAS(NAS):

    def __init__(self, N, search_space, dataset, flops_limit, api_loc=None, encoder_model_state_dict=None, predictor_model_state_dict=None, num_init_archs=20, K=10, net_params=None, num_ensemble=5, device='cpu', seed=None):
        super(BANANAS, self).__init__(N, search_space, dataset, flops_limit, api_loc=api_loc, encoder_model_state_dict=encoder_model_state_dict, predictor_model_state_dict=predictor_model_state_dict, net_params=net_params, device=device, seed=seed)
        self.num_init_archs = num_init_archs
        self.K = K
        self.num_ensemble = num_ensemble

    def mutate(self, parent, p):
        if self.search_space == 'nasbench101':
            if random.random() < p:
                while True:
                    old_matrix, old_ops = parent[1], parent[2]
                    idx_to_change = random.randrange(len(old_ops[1:-1])) + 1
                    entry_to_change = old_ops[idx_to_change]
                    possible_entries = [x for x in self.available_ops[1:-1] if x != entry_to_change]
                    new_entry = random.choice(possible_entries)
                    new_ops = copy.deepcopy(old_ops)
                    new_ops[idx_to_change] = new_entry
                    idx_to_change = random.randrange(sum(range(1, len(old_matrix))))
                    new_matrix = copy.deepcopy(old_matrix)
                    num_node = len(old_matrix)
                    idx_to_ij = {int(i*(num_node-1)-i*(i-1)/2+(j-i-1)): (i, j) for i in range(num_node) for j in range(i+1, num_node)}
                    i, j = idx_to_ij[idx_to_change]
                    new_matrix[i][j] = 1 if new_matrix[i][j] == 0 else 0
                    new_spec = NASBench101API.ModelSpec(matrix=new_matrix, ops=new_ops)
                    if self.nasbench.is_valid(new_spec):
                        spec_hash = new_spec.hash_spec(self.available_ops[1:-1])
                        child = (spec_hash, new_matrix, new_ops)
                        break
            else:
                child = parent
        elif self.search_space == 'nasbench201':
            if random.random() < p:
                nodes = parent[1].split('+')
                nodes = [node[1:-1].split('|') for node in nodes]
                nodes = [[op_and_input.split('~')[0]  for op_and_input in node] for node in nodes]
                old_spec = [op for node in nodes for op in node]
                idx_to_change = random.randrange(len(old_spec))
                entry_to_change = old_spec[idx_to_change]
                possible_entries = [x for x in self.available_ops[1:-1] if x != entry_to_change]
                new_entry = random.choice(possible_entries)
                new_spec = copy.deepcopy(old_spec)
                new_spec[idx_to_change] = new_entry
                arch_str = '|{:}~0|+|{:}~0|{:}~1|+|{:}~0|{:}~1|{:}~2|'.format(*new_spec)
                i = self.nasbench.query_index_by_arch(arch_str)
                child = (i, arch_str)
            else:
                child = parent
        elif self.search_space == 'nasbench301':
            if random.random() < p:
                old_all = parent.normal + parent.reduce
                idx_to_change = random.randrange(len(old_all))
                entry_to_change = old_all[idx_to_change][0]
                possible_entries = [x for x in self.available_ops[2:-1] if x != entry_to_change]
                new_entry = random.choice(possible_entries)
                new_all = copy.deepcopy(old_all)
                new_all[idx_to_change] = (new_entry, old_all[idx_to_change][1])
                idx_to_change = random.randrange(len(old_all))
                entry_to_change = old_all[idx_to_change][1]
                entry_to_change_neibor = old_all[idx_to_change+1][1] if idx_to_change%2 == 0 else old_all[idx_to_change-1][1]
                possible_entries = [x for x in range(2+idx_to_change%len(parent.normal)//2) if x != entry_to_change and x != entry_to_change_neibor]
                if len(possible_entries) != 0:
                    new_entry = random.choice(possible_entries)
                    new_all[idx_to_change] = (old_all[idx_to_change][0], new_entry)
                child = Genotype(normal=new_all[:len(new_all)//2], normal_concat=list(range(2,2+(self.max_num_vertices-3)//2)), reduce=new_all[len(new_all)//2:], reduce_concat=list(range(2,2+(self.max_num_vertices-3)//2)))
            else:
                child = parent
        elif self.search_space == 'darts':
            if random.random() < p:
                old_all = parent.normal + parent.reduce
                idx_to_change = random.randrange(len(old_all))
                entry_to_change = old_all[idx_to_change][0]
                possible_entries = [x for x in self.available_ops[2:-1] if x != entry_to_change]
                new_entry = random.choice(possible_entries)
                new_all = copy.deepcopy(old_all)
                new_all[idx_to_change] = (new_entry, old_all[idx_to_change][1])
                idx_to_change = random.randrange(len(old_all))
                entry_to_change = old_all[idx_to_change][1]
                entry_to_change_neibor = old_all[idx_to_change+1][1] if idx_to_change%2 == 0 else old_all[idx_to_change-1][1]
                possible_entries = [x for x in range(2+idx_to_change%len(parent.normal)//2) if x != entry_to_change and x != entry_to_change_neibor]
                if len(possible_entries) != 0:
                    new_entry = random.choice(possible_entries)
                    new_all[idx_to_change] = (old_all[idx_to_change][0], new_entry)
                child = Genotype(normal=new_all[:len(new_all)//2], normal_concat=list(range(2,2+(self.max_num_vertices-3)//2)), reduce=new_all[len(new_all)//2:], reduce_concat=list(range(2,2+(self.max_num_vertices-3)//2)))
            else:
                child = parent
        else:
            raise ValueError('There is no {:} search space.'.format(self.search_space))
        return child

    def get_candidates(self, arch_pool):
        num_arches_to_mutate = 1
        patience_factor = 5
        num = 100
        best_arches = [arch[0] for arch in sorted(arch_pool, key=cmp_to_key(self.cmp), reverse=True)[:num_arches_to_mutate * patience_factor]]
        candidates = []
        for arch in best_arches:
            for i in range(int(num / num_arches_to_mutate)):
                mutated = self.mutate(arch, p=1.0)
                if mutated[0] not in self.visited:
                    candidates.append(mutated)
                    self.visited.append(mutated[0])
                if len(candidates) >= num:
                    return candidates
        return candidates

    def acq_fn(self, predictions, ytrain=None, stds=None, explore_type='its'):
        predictions = np.array(predictions)

        if stds is None:
            stds = np.sqrt(np.var(predictions, axis=0))

        # Upper confidence bound (UCB) acquisition function
        if explore_type == 'ucb':
            explore_factor = 0.5
            mean = np.mean(predictions, axis=0)
            ucb = mean - explore_factor * stds
            sorted_indices = np.argsort(ucb)

        # Expected improvement (EI) acquisition function
        elif explore_type == 'ei':
            ei_calibration_factor = 5.
            mean = list(np.mean(predictions, axis=0))
            factored_stds = list(stds / ei_calibration_factor)
            min_y = ytrain.min()
            gam = [(min_y - mean[i]) / factored_stds[i] for i in range(len(mean))]
            ei = [-1 * factored_stds[i] * (gam[i] * norm.cdf(gam[i]) + norm.pdf(gam[i]))
                  for i in range(len(mean))]
            sorted_indices = np.argsort(ei)

        # Probability of improvement (PI) acquisition function
        elif explore_type == 'pi':
            mean = list(np.mean(predictions, axis=0))
            stds = list(stds)
            min_y = ytrain.min()
            pi = [-1 * norm.cdf(min_y, loc=mean[i], scale=stds[i]) for i in range(len(mean))]
            sorted_indices = np.argsort(pi)

        # Thompson sampling (TS) acquisition function
        elif explore_type == 'ts':
            rand_ind = np.random.randint(predictions.shape[0])
            ts = predictions[rand_ind,:]
            sorted_indices = np.argsort(ts)

        # Top exploitation 
        elif explore_type == 'percentile':
            min_prediction = np.min(predictions, axis=0)
            sorted_indices = np.argsort(min_prediction)

        # Top mean
        elif explore_type == 'mean':
            mean = np.mean(predictions, axis=0)
            sorted_indices = np.argsort(mean)

        elif explore_type == 'confidence':
            confidence_factor = 2
            mean = np.mean(predictions, axis=0)
            conf = mean + confidence_factor * stds
            sorted_indices = np.argsort(conf)

        # Independent Thompson sampling (ITS) acquisition function
        elif explore_type == 'its':
            mean = np.mean(predictions, axis=0)
            samples = np.random.normal(mean, stds)
            sorted_indices = np.argsort(samples)

        else:
            print('{} is not a valid exploration type'.format(explore_type))
            raise NotImplementedError()

        return sorted_indices

    def run(self):
        total_eval_time = 0
        history = []
        while len(history) < self.num_init_archs:
            arch = self.sample_arch()
            if arch[0] not in self.visited:
                if self.search_space in ['nasbench101', 'nasbench201']:
                    valid_acc, test_acc, eval_time = self.eval_arch(arch, use_val_acc=True)
                    cur = (arch, valid_acc, test_acc, eval_time)
                else:
                    acc, eval_time = self.eval_arch(arch, use_val_acc=True)
                    cur = (arch, acc, eval_time)
                total_eval_time += eval_time
                history.append(cur)
                self.visited.append(arch[0])
        while len(history) < self.N:
            candidates = self.get_candidates(history)
            candidate_predictions = []
            for e in range(self.num_ensemble):
                encoder = copy.deepcopy(self.encoder)
                predictor = copy.deepcopy(self.predictor)
                encoder, predictor = self.train(history, encoder, predictor)
                candidate_predictions.append(self.predict(candidates, encoder, predictor))
            candidate_indices = self.acq_fn(predictions=candidate_predictions, explore_type='its')
            # add the k arches with the maximum acquisition function values
            for i in candidate_indices[-self.K:]:
                arch = candidates[i]
                if self.search_space in ['nasbench101', 'nasbench201']:
                    valid_acc, test_acc, eval_time = self.eval_arch(arch, use_val_acc=True)
                    cur = (arch, valid_acc, test_acc, eval_time)
                else:
                    acc, eval_time = self.eval_arch(arch, use_val_acc=True)
                    cur = (arch, acc, eval_time)
                total_eval_time += eval_time
                history.append(cur)
        best = max(history, key=lambda x: x[1])
        return best, history, total_eval_time


if __name__ == '__main__':
    args = parse_arguments()
    prepare_logger(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if args.search_algo == 'r':
        nas = Random_NAS(args.N, args.search_space, args.dataset, args.flops_limit, api_loc=args.api_loc, encoder_model_state_dict=args.encoder_model_state_dict, predictor_model_state_dict=args.predictor_model_state_dict, net_params=args.net_params, device=args.device)
    elif args.search_algo == 'ae':
        nas = Evolved_NAS(args.N, args.search_space, args.population_size, args.tournament_size, args.dataset, args.flops_limit, api_loc=args.api_loc, encoder_model_state_dict=args.encoder_model_state_dict, predictor_model_state_dict=args.predictor_model_state_dict, net_params=args.net_params, device=args.device)
    elif args.search_algo == 'rl':
        nas = RL_NAS(args.N, args.search_space, args.dataset, args.flops_limit, args.lr, args.EMA_momentum, api_loc=args.api_loc, encoder_model_state_dict=args.encoder_model_state_dict, predictor_model_state_dict=args.predictor_model_state_dict, net_params=args.net_params, device=args.device)
    elif args.search_algo == 'bo':
        nas = BANANAS(args.N, args.search_space, args.dataset, args.flops_limit, api_loc=args.api_loc, encoder_model_state_dict=args.encoder_model_state_dict, predictor_model_state_dict=args.predictor_model_state_dict, net_params=args.net_params, device=args.device)
    else:
        raise ValueError('There is no {:} algorithm.'.format(args.search_algo))

    begin_time = time.time()
    best, history, total_eval_time = nas.run()
    end_time = time.time()

    logging.info('The search history:')
    for i, h in enumerate(history):
        logging.info('{:5d} {:}'.format(i, h))
    logging.info('\n' + '-' * 100)
    logging.info('The best architectures:\n{:}'.format(best))
    logging.info(
        '{:} : search {:} architectures, total cost {:.1f} s, total evalution cost {:.1f} s, the best is {:}, the measure is {:}.'.format(
            args.search_algo, args.N, end_time - begin_time, total_eval_time, best[0], best[1]
        )
    )
    if args.search_space == 'nasbench101':
        logging.info('{:}\n{:}'.format(nas.nasbench.fixed_statistics[best[0][0]], nas.nasbench.computed_statistics[best[0][0]][108]))
    elif args.search_space == 'nasbench201':
        logging.info('{:}'.format(nas.nasbench.query_by_arch(best[0][1], "200")))
    elif args.search_space == 'nasbench301':
        logging.info('Valid accuracy {:}'.format(nas.nasbench.predict(config=best[0], representation='genotype', with_noise=False)))
    elif args.search_space == 'darts':
        logging.info('Train and test DARTS model using train_DARTS.py and test_DARTS.py!')
    else:
        raise ValueError('There is no {:} search space.'.format(args.search_space))

    torch.save({'best': best, 'history': history}, os.path.join(args.save_dir, 'history.ckp'))
