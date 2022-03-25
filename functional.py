import torch
import networkx as nx
from torch_geometric.utils import degree, to_undirected


def compute_pr(edge_index, damp: float = 0.85, k: int = 10):
    num_nodes = edge_index.max().item() + 1
    deg_out = degree(edge_index[0])
    x = torch.ones((num_nodes, )).to(edge_index.device).to(torch.float32)
    for i in range(k):
        edge_msg = x[edge_index[0]] / deg_out[edge_index[0]]
        agg_msg = scatter(edge_msg, edge_index[1], reduce='sum')
        x = (1 - damp) * x + damp * agg_msg
    return x


def eigenvector_centrality(data):
    graph = to_networkx(data)
    x = nx.eigenvector_centrality_numpy(graph)
    x = [x[i] for i in range(data.num_nodes)]
    return torch.tensor(x, dtype=torch.float32).to(data.edge_index.device)


def feature_drop_weights(dataset, drop_scheme):
    edge_index = dataset.edge_index.clone().detach()
    x          = dataset.x.clone().detach()
    device = dataset.x.device
    if drop_scheme in ['degree', 'pr', 'evc']:
        if param['drop_scheme'] == 'degree':
            node_c = degree(to_undirected(edge_index)[1])
        elif param['drop_scheme'] == 'pr':
            node_c = compute_pr(edge_index)
        elif param['drop_scheme'] == 'evc':
            node_c = eigenvector_centrality(dataset)
        x = x.to(torch.bool).to(torch.float32)
        w = x.t() @ node_c
        w = w.log()
        s = (w.max() - w) / (w.max() - w.mean())
    else:
        s = None
    return s


def drop_feature_weighted(x, feature_weights, p, threshold=1.):
    if feature_weights is None:
        feature_weights = torch.ones((x.size(1),)).to(x.device)

    feature_weights = feature_weights / feature_weights.mean() * p
    feature_weights = feature_weights.where(feature_weights < threshold, torch.ones_like(feature_weights) * threshold)
    drop_mask = torch.bernoulli(feature_weights).to(torch.bool)

    x = x.clone()
    x[:, drop_mask] = 0.
    return x


def degree_drop_weights(edge_index):
    edge_index_ = to_undirected(edge_index)
    deg = degree(edge_index_[1])
    deg_col = deg[edge_index[1]].to(torch.float32)
    s_col = torch.log(deg_col)
    weights = (s_col.max() - s_col) / (s_col.max() - s_col.mean())
    return weights


def pr_drop_weights(edge_index, aggr='mean', k=200):
    pv = compute_pr(edge_index, k=k)
    pv_row = pv[edge_index[0]].to(torch.float32)
    pv_col = pv[edge_index[1]].to(torch.float32)
    s_row = torch.log(pv_row)
    s_col = torch.log(pv_col)
    if aggr == 'sink':
        s = s_col
    elif aggr == 'source':
        s = s_row
    elif aggr == 'mean':
        s = (s_col + s_row) * 0.5
    else:
        s = (s_col + s_row) * 0.5
    weights = (s.max() - s) / (s.max() - s.mean())
    return weights


def evc_drop_weights(data):
    evc = eigenvector_centrality(data)
    evc = evc.where(evc > 0, torch.zeros_like(evc))
    evc = evc + 1e-8
    s = evc.log()
    edge_index = data.edge_index
    s_row, s_col = s[edge_index[0]], s[edge_index[1]]
    s = s_col
    return (s.max() - s) / (s.max() - s.mean())


def drop_edge_weighted(edge_index, edge_weights, p, threshold=1.):
    if edge_weights is None:
        edge_weights = torch.ones((edge_index.size(1),)).to(edge_index.device)

    edge_weights = edge_weights / edge_weights.mean() * p
    edge_weights = edge_weights.where(edge_weights < threshold, torch.ones_like(edge_weights) * threshold)
    sel_mask = torch.bernoulli(1. - edge_weights).to(torch.bool)

    return edge_index[:, sel_mask].contiguous()
