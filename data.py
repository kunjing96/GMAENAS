import os
import random
import torch
import numpy as np
from itertools import chain, repeat
from torch_geometric.data import InMemoryDataset, Data, Batch
import copy
from collections import namedtuple
from collections.abc import Sequence


def floyed(r, diffusion_rate):
    r = copy.deepcopy(r)
    r = r.astype(bool)
    N = r.shape[0]
    mask = np.random.random(r.shape) < diffusion_rate
    for k in range(N):
        for i in range(N):
            for j in range(N):
                r[i, j] |= ((mask[i, j] & r[i, k] & r[k, j]) | ((~mask[i, j]) & r[i, j]))
                #if r[i, k] > 0 and r[k, j] > 0:
                    #r[i, j] = 1
    return r.astype(int)


class Dataset(InMemoryDataset):

    def __init__(self, space=None, dataset=None, root=None, transform=None, pre_transform=None, nasbench=None, diffusion_rate=0.0, seed=None, archs=None):
        self.space = space
        self.dataset = dataset
        if isinstance(root, str):
            root = os.path.expanduser(os.path.normpath(root))
        self.root = root
        self.transform = transform
        self.pre_transform = pre_transform
        self._indices = None
        self.nasbench = nasbench
        self.diffusion_rate = diffusion_rate
        self.seed = seed
        self.archs = archs
        self.process()

    def process(self):
        if self.archs is None:
            if self.space == 'nasbench101':
                from nasbench import api as NASBench101API
                if self.nasbench is None:
                    data_path = os.path.join(self.root, 'nasbench_full.tfrecord')
                    self.nasbench = NASBench101API.NASBench(data_path, seed=self.seed)
                archs = []
                for hash_value in self.nasbench.hash_iterator():
                    fixed_statistic = self.nasbench.fixed_statistics[hash_value]
                    computed_statistic = self.nasbench.computed_statistics[hash_value]
                    archs.append(({'adj': fixed_statistic['module_adjacency'],
                                  'ops': fixed_statistic['module_operations'],
                                  'acc': computed_statistic[108][np.random.randint(3)]['final_validation_accuracy']}, ))
                self.available_ops = ['input', 'conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3', 'output']
                self.max_num_vertices = 7
            elif self.space == 'nasbench201':
                from nas_201_api import NASBench201API
                if self.nasbench is None:
                    data_path = os.path.join(self.root, 'NAS-Bench-201-v1_1-096897.pth')
                    self.nasbench = NASBench201API(data_path, verbose=False)
                archs = []
                for arch in self.nasbench:
                    acc = self.nasbench.get_more_info(arch, 'cifar10-valid' if self.dataset=='cifar10' else self.dataset, hp='200')['valid-accuracy'] / 100
                    arch = self.nasbench.str2lists(arch)
                    arch = nasbench201_to_nasbench101(arch)
                    arch.update({'acc': acc})
                    archs.append((arch, ))
                self.available_ops = ['input', 'none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3', 'output']
                self.max_num_vertices = 8
            elif self.space == 'nasbench301':
                import nasbench301  as NASBench301API
                if self.nasbench is None:
                    data_path = os.path.join(self.root, 'nb_models/xgb_v1.0')
                    self.nasbench = NASBench301API.load_ensemble(data_path)
                self.available_ops = ['input1', 'input2', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5', 'output']
                self.max_num_vertices = 11
                '''archs = []
                for _ in range(100000):
                    genotype = sample_darts_arch(self.available_ops[2:-1])
                    arch     = darts_to_nasbench101(genotype)
                    acc      = self.nasbench.predict(config=genotype, representation='genotype', with_noise=True) / 100
                    arch[0].update({'acc': acc})
                    arch[1].update({'acc': acc})
                    archs.append(arch)'''
                archs = torch.load(os.path.join(self.root, 'nasbench301_proxy.pt'))
            elif self.space == 'darts':
                self.nasbench = None
                self.available_ops = ['input1', 'input2', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5', 'output']
                self.max_num_vertices = 11
                archs_perfs = torch.load(os.path.join(self.root, 'darts_archs_perfs.pth'))
                archs = []
                for _, (genotype, perf, _) in enumerate(archs_perfs):
                    arch     = darts_to_nasbench101(genotype)
                    acc      = perf / 100
                    arch.update({'acc': acc})
                    archs.append((arch, ))
            else:
                raise ValueError('No implementation!')
        else:
            if self.space == 'nasbench101':
                self.available_ops = ['input', 'conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3', 'output']
                self.max_num_vertices = 7
            elif self.space == 'nasbench201':
                self.available_ops = ['input', 'none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3', 'output']
                self.max_num_vertices = 8
            elif self.space == 'nasbench301':
                self.available_ops = ['input1', 'input2', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5', 'output']
                self.max_num_vertices = 11
            elif self.space == 'darts':
                self.nasbench = None
                self.available_ops = ['input1', 'input2', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5', 'output']
                self.max_num_vertices = 11
            else:
                raise ValueError('No implementation!')
            archs = self.archs

        def arch2Data(arch):
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
        
        def prepare_augmented_graph(arch):
            arch_x = copy.deepcopy(arch)            
            arch_y = copy.deepcopy(arch)
            if self.diffusion_rate > 0:
                arch_y['adj'] = floyed(arch_y['adj'], self.diffusion_rate)
            return arch_x, arch_y

        self.data = tuple()
        self.slices = tuple()
        for i in range(len(archs[0])):
            acc_list  = []
            data_x_list = []
            data_y_list = []
            for arch in archs:
                arch_x, arch_y = prepare_augmented_graph(arch[i])
                data_x = arch2Data(arch_x)
                data_y = arch2Data(arch_y)
                #x = torch.tensor([self.available_ops.index(x) for x in arch[i]['ops']], dtype=torch.long)
                #y = torch.ones_like(x) * arch[i]['acc']
                acc_list.append(arch[i]['acc'])
                #forward_edges = [[(k, j) for j, x in enumerate(xs) if x > 0] for k, xs in enumerate(arch[i]['adj'])]
                #forward_edges = np.array(list(chain(*forward_edges)))
                #backward_edges = forward_edges[::-1, ::-1]
                #edges = np.concatenate([forward_edges, backward_edges])
                #edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                #data = Data(x=x, y=y, edge_index=edge_index)
                if self.pre_transform is not None:
                    data_x = self.pre_transform(data_x)
                    data_y = self.pre_transform(data_y)
                data_x_list.append(data_x)
                data_y_list.append(data_y)
            top_indices = np.argsort(acc_list)[::-1].tolist()
            data_x, slices_x = self.collate(data_x_list)
            data_y, slices_y = self.collate(data_y_list)
            self.data = (*self.data, data_x, data_y)
            self.slices = (*self.slices, slices_x, slices_y)
        self.top_indices = top_indices

    def len(self):
        for item in self.slices[0].values():
            return len(item) - 1
        return 0

    def __getitem__(self, idx):
        if (isinstance(idx, (int, np.integer)) or (isinstance(idx, torch.Tensor) and idx.dim() == 0) or (isinstance(idx, np.ndarray) and np.isscalar(idx))):
            data = self.get(self.indices()[idx])
            ret_data = tuple()
            for i in range(len(data)):
                _data = data[i] if self.transform is None else self.transform(data[i])
                ret_data = (*ret_data, _data)
            return ret_data
        else:
            return self.index_select(idx)

    def get(self, idx):
        if hasattr(self, '_data_list'):
            if self._data_list is None:
                self._data_list = self.len() * [None]
            else:
                return copy.copy(self._data_list[idx])
        data = tuple()
        for i in range(len(self.data)):
            _data = self.data[i].__class__()
            if hasattr(self.data[i], '__num_nodes__'):
                _data.num_nodes = self.data[i].__num_nodes__[idx]
            for key in self.data[i].keys:
                item, slices = self.data[i][key], self.slices[i][key]
                start, end = slices[idx].item(), slices[idx + 1].item()
                if torch.is_tensor(item):
                    s = list(repeat(slice(None), item.dim()))
                    cat_dim = self.data[i].__cat_dim__(key, item)
                    if cat_dim is None:
                        cat_dim = 0
                    s[cat_dim] = slice(start, end)
                elif start + 1 == end:
                    s = slices[start]
                else:
                    s = slice(start, end)
                _data[key] = item[s]
            data = (*data, _data)
        if hasattr(self, '_data_list'):
            self._data_list[idx] = copy.copy(data)
        return data
    
    def index_select(self, idx):
        indices = self.indices()
        if isinstance(idx, slice):
            indices = indices[idx]
        elif isinstance(idx, torch.Tensor) and idx.dtype == torch.long:
            return self.index_select(idx.flatten().tolist())
        elif isinstance(idx, torch.Tensor) and idx.dtype == torch.bool:
            idx = idx.flatten().nonzero(as_tuple=False)
            return self.index_select(idx.flatten().tolist())
        elif isinstance(idx, np.ndarray) and idx.dtype == np.int64:
            return self.index_select(idx.flatten().tolist())
        elif isinstance(idx, np.ndarray) and idx.dtype == np.bool:
            idx = idx.flatten().nonzero()[0]
            return self.index_select(idx.flatten().tolist())
        elif isinstance(idx, Sequence) and not isinstance(idx, str):
            indices = [indices[i] for i in idx]
        else:
            raise IndexError(
                f"Only slices (':'), list, tuples, torch.tensor and "
                f"np.ndarray of dtype long or bool are valid indices (got "
                f"'{type(idx).__name__}')")

        dataset = copy.copy(self)
        dataset._indices = indices
        return dataset

    def __repr__(self):
        return '{}(space={})'.format(self.__class__.__name__, self.space)


class DataIterator(object):

    def __init__(self, space=None, dataset=None, root=None, nasbench=None, diffusion_rate=0.0, seed=None):
        self.space = space
        self.dataset = dataset
        if isinstance(root, str):
            root = os.path.expanduser(os.path.normpath(root))
        self.root = root
        self.nasbench = nasbench
        self.diffusion_rate = diffusion_rate
        self.seed = seed
        self.process()

    def process(self):
        if self.space == 'nasbench101':
            from nasbench import api as NASBench101API
            if self.nasbench is None:
                data_path = os.path.join(self.root, 'nasbench_full.tfrecord')
                self.nasbench = NASBench101API.NASBench(data_path, seed=self.seed)
            self.available_ops = ['input', 'conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3', 'output']
            self.max_num_vertices = 7
        elif self.space == 'nasbench201':
            from nas_201_api import NASBench201API
            if self.nasbench is None:
                data_path = os.path.join(self.root, 'NAS-Bench-201-v1_1-096897.pth')
                self.nasbench = NASBench201API(data_path, verbose=False)
            self.available_ops = ['input', 'none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3', 'output']
            self.max_num_vertices = 8
        elif self.space == 'nasbench301':
            import nasbench301  as NASBench301API
            if self.nasbench is None:
                data_path = os.path.join(self.root, 'nb_models/xgb_v1.0')
                self.nasbench = NASBench301API.load_ensemble(data_path)
            self.available_ops = ['input1', 'input2', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5', 'output']
            self.max_num_vertices = 11
        elif self.space == 'darts':
            self.nasbench = None
            self.available_ops = ['input1', 'input2', 'none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5', 'output']
            self.max_num_vertices = 11
        else:
            raise ValueError('No implementation!')

    def sample(self, batch_size):
        archs = []
        if self.space == 'nasbench101':
            hash_list = list(self.nasbench.hash_iterator())
            for _ in range(batch_size):
                hash_value = random.choice(hash_list)
                fixed_statistic = self.nasbench.fixed_statistics[hash_value]
                #computed_statistic = self.nasbench.computed_statistics[hash_value]
                archs.append(({'adj': fixed_statistic['module_adjacency'],
                              'ops': fixed_statistic['module_operations'],},))
                              #'acc': computed_statistic[108][np.random.randint(3)]['final_validation_accuracy']},))
        elif self.space == 'nasbench201':
            arch_list = list(self.nasbench)
            for _ in range(batch_size):
                arch = random.choice(arch_list)
                #acc = self.nasbench.get_more_info(arch, 'cifar10-valid' if self.dataset=='cifar10' else self.dataset, hp='200')['valid-accuracy'] / 100
                arch = self.nasbench.str2lists(arch)
                arch = nasbench201_to_nasbench101(arch)
                #arch.update({'acc': acc})
                archs.append((arch,))
        elif self.space == 'nasbench301':
            for _ in range(batch_size):
                genotype = sample_darts_arch(self.available_ops[2:-1])
                arch     = darts_to_nasbench101(genotype)
                #acc      = self.nasbench.predict(config=genotype, representation='genotype', with_noise=True) / 100
                #arch[0].update({'acc': acc})
                #arch[1].update({'acc': acc})
                archs.append((arch,))
        elif self.space == 'darts':
            for _ in range(batch_size):
                genotype = sample_darts_arch(self.available_ops[2:-1])
                arch     = darts_to_nasbench101(genotype)
                archs.append((arch,))
        else:
            raise ValueError('No implementation!')

        def arch2Data(arch):
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
        
        def prepare_augmented_graph(arch):
            arch_x = copy.deepcopy(arch)            
            arch_y = copy.deepcopy(arch)
            if self.diffusion_rate > 0:
                arch_y['adj'] = floyed(arch_y['adj'], self.diffusion_rate)
            return arch_x, arch_y

        ret = tuple()
        for i in range(len(archs[0])):
            data_x_list = []
            data_y_list = []
            for arch in archs:
                arch_x, arch_y = prepare_augmented_graph(arch[i])
                data_x_list.append(arch2Data(arch_x))
                data_y_list.append(arch2Data(arch_y))
            ret = (*ret, Batch.from_data_list(data_x_list), Batch.from_data_list(data_y_list))
        
        return ret

    def __repr__(self):
        return '{}(space={})'.format(self.__class__.__name__, self.space)


def nasbench201_to_nasbench101(arch_list):
    num_ops = sum(range(1, 1 + len(arch_list))) + 2
    adj = np.zeros((num_ops, num_ops), dtype=np.uint8)
    ops = ['input', 'output']
    node_lists = [[0]]
    for node_201 in arch_list:
        node_list = []
        for node in node_201:
            node_idx = len(ops) - 1
            adj[node_lists[node[1]], node_idx] = 1
            ops.insert(-1, node[0])
            node_list.append(node_idx)
        node_lists.append(node_list)
    adj[-(1+len(arch_list)):-1, -1] = 1
    arch = {'adj': adj,
            'ops': ops,}
    return arch


def darts_to_nasbench101(genotype):
    arch = []
    for arch_list, concat in [(genotype.normal, genotype.normal_concat), (genotype.reduce, genotype.reduce_concat)]:
        num_ops = len(arch_list) + 3
        adj = np.zeros((num_ops, num_ops), dtype=np.uint8)
        ops = ['input1', 'input2', 'output']
        node_lists = [[0], [1] , [2, 3], [4, 5], [6, 7], [8, 9], [10]]
        for node in arch_list:
            node_idx = len(ops) - 1
            adj[node_lists[node[1]], node_idx] = 1
            ops.insert(-1, node[0])
        adj[[x for c in concat for x in node_lists[c]], -1] = 1
        cell = {'adj': adj,
                'ops': ops,}
        arch.append(cell)
    adj = np.zeros((num_ops*2, num_ops*2), dtype=np.uint8)
    adj[:num_ops, :num_ops] = arch[0]['adj']
    adj[num_ops:, num_ops:] = arch[1]['adj']
    ops = arch[0]['ops'] + arch[1]['ops']
    arch = {'adj': adj,
            'ops': ops,}
    return arch


Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')


def sample_darts_arch(available_ops):
    geno = []
    for _ in range(2):
        cell = []
        for i in range(4):
            ops_normal = np.random.choice(available_ops, 2)
            nodes_in_normal = sorted(np.random.choice(range(i+2), 2, replace=False))
            cell.extend([(ops_normal[0], nodes_in_normal[0]), (ops_normal[1], nodes_in_normal[1])])
        geno.append(cell)
    genotype = Genotype(normal=geno[0], normal_concat=[2, 3, 4, 5], reduce=geno[1], reduce_concat=[2, 3, 4, 5])
    return genotype
