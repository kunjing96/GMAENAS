import nasbench301  as NASBench301API
from data import sample_darts_arch, darts_to_nasbench101
import tqdm
import torch
import os

data_path = os.path.join(os.path.expanduser(os.path.normpath('~/datasets')), 'nb_models/xgb_v1.0')
nasbench = NASBench301API.load_ensemble(data_path)
available_ops = ['input1', 'input2', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5', 'output']
archs = []
for _ in tqdm.tqdm(range(100000)):
    genotype = sample_darts_arch(available_ops[2:-1])
    arch     = darts_to_nasbench101(genotype)
    acc      = nasbench.predict(config=genotype, representation='genotype', with_noise=True) / 100
    #arch[0].update({'acc': acc})
    #arch[1].update({'acc': acc})
    arch.update({'acc': acc})
    archs.append((arch,))

torch.save(archs, 'nasbench301_proxy.pt')
