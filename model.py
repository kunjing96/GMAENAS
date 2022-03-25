import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dropout_adj


class AvgPooling(torch.nn.Module):

    def __init__(self):
        super(AvgPooling, self).__init__()

    def forward(self, x, ptr):
        g = []
        for i in range(ptr.size(0)-1):
            g.append(torch.mean(x[ptr[i]:ptr[i+1]], 0, True))
        return torch.cat(g, 0)


class GraphEncoder(torch.nn.Module):
    def __init__(self, embedding_dim, out_channels, activation, base_model=GCNConv, k=2, dropout=0.0, skip=True, use_bn=True):
        super(GraphEncoder, self).__init__()
        self.base_model = base_model
        self.num_hidden = out_channels

        assert k >= 2
        self.k = k
        self.skip = skip
        self.use_bn = use_bn
        self.activation = activation
        self.dropout = nn.Dropout(p = dropout)
        self.readout = AvgPooling()
        if self.skip:
            self.fc_skip = torch.nn.Linear(embedding_dim, out_channels)
            self.conv = [base_model(embedding_dim, out_channels)]
            if self.use_bn:
                self.bn = [torch.nn.LayerNorm(out_channels)]
            for _ in range(1, k):
                self.conv.append(base_model(out_channels, out_channels))
                if self.use_bn:
                    self.bn.append(torch.nn.LayerNorm(out_channels))
            self.conv = torch.nn.ModuleList(self.conv)
            if self.use_bn:
                self.bn = torch.nn.ModuleList(self.bn)
        else:
            self.conv = [base_model(embedding_dim, 2 * out_channels)]
            if self.use_bn:
                self.bn = [torch.nn.LayerNorm(2 * out_channels)]
            for _ in range(1, k - 1):
                self.conv.append(base_model(2 * out_channels, 2 * out_channels))
                if self.use_bn:
                    self.bn.append(torch.nn.LayerNorm(2 * out_channels))
            self.conv.append(base_model(2 * out_channels, out_channels))
            if self.use_bn:
                self.bn.append(torch.nn.LayerNorm(out_channels))
            self.conv = torch.nn.ModuleList(self.conv)
            if self.use_bn:
                self.bn = torch.nn.ModuleList(self.bn)

    def forward(self, x, edge_index, ptr):
        if self.skip:
            if self.use_bn:
                h = self.dropout(self.activation(self.bn[0](self.conv[0](x, edge_index))))
            else:
                h = self.dropout(self.activation(self.conv[0](x, edge_index)))
            hs = [self.fc_skip(x), h]
            for i in range(1, self.k):
                u = sum(hs)
                if self.use_bn:
                    hs.append(self.dropout(self.activation(self.bn[i](self.conv[i](u, edge_index)))))
                else:
                    hs.append(self.dropout(self.activation(self.conv[i](u, edge_index))))
            return hs[-1], self.readout(hs[-1], ptr)
        else:
            for i in range(self.k):
                if self.use_bn:
                    x = self.dropout(self.activation(self.bn[i](self.conv[i](x, edge_index))))
                else:
                    x = self.dropout(self.activation(self.conv[i](x, edge_index)))
            return x, self.readout(x, ptr)


def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0
    return x

        
def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class PairWiseLearning_BARLOW_TWINS(nn.Module):
    def __init__(self, n_vocab, n_layers, d_model, activation, base_model, graph_encoder_dropout, dropout, drop_edge_rate_1, drop_edge_rate_2, drop_feature_rate_1, drop_feature_rate_2, projector_layers):
        super(PairWiseLearning_BARLOW_TWINS, self).__init__()
        self.d_model = d_model
        self.drop_edge_rate_1 = drop_edge_rate_1
        self.drop_edge_rate_2 = drop_edge_rate_2
        self.drop_feature_rate_1 = drop_feature_rate_1
        self.drop_feature_rate_2 = drop_feature_rate_2
        # Embedding Layer
        self.opEmb = nn.Embedding(n_vocab, d_model)
        self.dropout_op = nn.Dropout(p=dropout)
        # Graph Encoder
        self.graph_encoder = GraphEncoder(d_model, d_model, activation, base_model=base_model, k=n_layers, dropout=graph_encoder_dropout, skip=True, use_bn=True)
        # projector
        sizes = [d_model] + list(map(int, projector_layers.split('-')))
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.LayerNorm(sizes[i]))
            layers.append(nn.PReLU())
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        self.projector = nn.Sequential(*layers)
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, x, edge_index_x, ptr_x, y, edge_index_y, ptr_y):
        emb_x = self.dropout_op(self.opEmb(x))
        h_x, g_x = self.graph_encoder(emb_x, edge_index_x, ptr_x)
        return h_x, g_x

    def ssl_loss(self, z1, z2, lambd):
        c = self.bn(z1).T @ self.bn(z2)
        c.div_(z1.size(0))
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + lambd * off_diag
        return loss

    def loss(self, x, edge_index_x, ptr_x, y, edge_index_y, ptr_y, lambd=0.0051, batch_size=0, pretrain_target=None):
        emb_x = self.dropout_op(self.opEmb(x))
        emb_y = self.dropout_op(self.opEmb(y))
        edge_index_x = dropout_adj(edge_index_x, p=self.drop_edge_rate_1)[0]
        edge_index_y = dropout_adj(edge_index_x, p=self.drop_edge_rate_2)[0]
        emb_x = drop_feature(emb_x, self.drop_feature_rate_1)
        emb_y = drop_feature(emb_x, self.drop_feature_rate_2)

        h_x, g_x = self.graph_encoder(emb_x, edge_index_x, ptr_x)
        h_y, g_y = self.graph_encoder(emb_y, edge_index_y, ptr_y)

        # compute loss
        z1 = self.projector(g_x)
        z2 = self.projector(g_y)
        loss = self.ssl_loss(z1, z2, lambd)
        return loss


class PairWiseLearning_BARLOW_TWINS_L2L(nn.Module):
    def __init__(self, n_vocab, n_layers, d_model, activation, base_model, graph_encoder_dropout, dropout, drop_edge_rate_1, drop_edge_rate_2, drop_feature_rate_1, drop_feature_rate_2, projector_layers):
        super(PairWiseLearning_BARLOW_TWINS_L2L, self).__init__()
        self.d_model = d_model
        self.drop_edge_rate_1 = drop_edge_rate_1
        self.drop_edge_rate_2 = drop_edge_rate_2
        self.drop_feature_rate_1 = drop_feature_rate_1
        self.drop_feature_rate_2 = drop_feature_rate_2
        # Embedding Layer
        self.opEmb = nn.Embedding(n_vocab, d_model)
        self.dropout_op = nn.Dropout(p=dropout)
        # Graph Encoder
        self.graph_encoder = GraphEncoder(d_model, d_model, activation, base_model=base_model, k=n_layers, dropout=graph_encoder_dropout, skip=True, use_bn=True)
        # projector
        sizes = [d_model] + list(map(int, projector_layers.split('-')))
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.LayerNorm(sizes[i]))
            layers.append(nn.PReLU())
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        self.projector = nn.Sequential(*layers)
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, x, edge_index_x, ptr_x, y, edge_index_y, ptr_y):
        emb_x = self.dropout_op(self.opEmb(x))
        h_x, g_x = self.graph_encoder(emb_x, edge_index_x, ptr_x)
        return h_x, g_x

    def ssl_loss(self, z1, z2, lambd):
        c = self.bn(z1).T @ self.bn(z2)
        c.div_(z1.size(0))
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + lambd * off_diag
        return loss

    def loss(self, x, edge_index_x, ptr_x, y, edge_index_y, ptr_y, lambd=0.0051, batch_size=0, pretrain_target=None):
        emb_x = self.dropout_op(self.opEmb(x))
        emb_y = self.dropout_op(self.opEmb(y))
        edge_index_x = dropout_adj(edge_index_x, p=self.drop_edge_rate_1)[0]
        edge_index_y = dropout_adj(edge_index_x, p=self.drop_edge_rate_2)[0]
        emb_x = drop_feature(emb_x, self.drop_feature_rate_1)
        emb_y = drop_feature(emb_x, self.drop_feature_rate_2)

        h_x, g_x = self.graph_encoder(emb_x, edge_index_x, ptr_x)
        h_y, g_y = self.graph_encoder(emb_y, edge_index_y, ptr_y)

        # compute loss
        z1 = self.projector(h_x)
        z2 = self.projector(h_y)
        loss = self.ssl_loss(z1, z2, lambd)
        return loss


class PairWiseLearning_GRACE(nn.Module):
    def __init__(self, n_vocab, n_layers, d_model, activation, base_model, graph_encoder_dropout, dropout, drop_edge_rate_1, drop_edge_rate_2, drop_feature_rate_1, drop_feature_rate_2, projector_layers, tau=0.5):
        super(PairWiseLearning_GRACE, self).__init__()
        self.d_model = d_model
        self.drop_edge_rate_1 = drop_edge_rate_1
        self.drop_edge_rate_2 = drop_edge_rate_2
        self.drop_feature_rate_1 = drop_feature_rate_1
        self.drop_feature_rate_2 = drop_feature_rate_2
        self.tau = tau
        # Embedding Layer
        self.opEmb = nn.Embedding(n_vocab, d_model)
        self.dropout_op = nn.Dropout(p=dropout)
        # Graph Encoder
        self.graph_encoder = GraphEncoder(d_model, d_model, activation, base_model=base_model, k=n_layers, dropout=graph_encoder_dropout, skip=True, use_bn=True)
        # projector
        sizes = [d_model] + list(map(int, projector_layers.split('-')))
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.LayerNorm(sizes[i]))
            layers.append(nn.PReLU())
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        self.projector = nn.Sequential(*layers)

    def forward(self, x, edge_index_x, ptr_x, y, edge_index_y, ptr_y):
        emb_x = self.dropout_op(self.opEmb(x))
        h_x, g_x = self.graph_encoder(emb_x, edge_index_x, ptr_x)
        return h_x, g_x

    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def ssl_loss(self, z1, z2):        
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))
        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_ssl_loss(self, z1, z2, batch_size):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []
        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))
        return torch.cat(losses)

    def loss(self, x, edge_index_x, ptr_x, y, edge_index_y, ptr_y, batch_size=0, pretrain_target=None):
        emb_x = self.dropout_op(self.opEmb(x))
        emb_y = self.dropout_op(self.opEmb(y))
        edge_index_x = dropout_adj(edge_index_x, p=self.drop_edge_rate_1)[0]
        edge_index_y = dropout_adj(edge_index_x, p=self.drop_edge_rate_2)[0]
        emb_x = drop_feature(emb_x, self.drop_feature_rate_1)
        emb_y = drop_feature(emb_x, self.drop_feature_rate_2)

        h_x, g_x = self.graph_encoder(emb_x, edge_index_x, ptr_x)
        h_y, g_y = self.graph_encoder(emb_y, edge_index_y, ptr_y)

        # compute loss
        z1 = self.projector(h_x)
        z2 = self.projector(h_y)
        if batch_size == 0:
            l1 = self.ssl_loss(z1, z2)
            l2 = self.ssl_loss(z2, z1)
        else:
            l1 = self.batched_ssl_loss(z1, z2, batch_size)
            l2 = self.batched_ssl_loss(z2, z1, batch_size)
        loss = (l1 + l2) * 0.5
        loss = loss.mean()
        return loss


class PairWiseLearning_GRAPHCL(nn.Module):
    def __init__(self, n_vocab, n_layers, d_model, activation, base_model, graph_encoder_dropout, dropout, drop_edge_rate_1, drop_edge_rate_2, drop_feature_rate_1, drop_feature_rate_2, projector_layers):
        super(PairWiseLearning_GRAPHCL, self).__init__()
        self.d_model = d_model
        self.drop_edge_rate_1 = drop_edge_rate_1
        self.drop_edge_rate_2 = drop_edge_rate_2
        self.drop_feature_rate_1 = drop_feature_rate_1
        self.drop_feature_rate_2 = drop_feature_rate_2
        # Embedding Layer
        self.opEmb = nn.Embedding(n_vocab, d_model)
        self.dropout_op = nn.Dropout(p=dropout)
        # Graph Encoder
        self.graph_encoder = GraphEncoder(d_model, d_model, activation, base_model=base_model, k=n_layers, dropout=graph_encoder_dropout, skip=True, use_bn=True)
        # projector
        sizes = [d_model] + list(map(int, projector_layers.split('-')))
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.LayerNorm(sizes[i]))
            layers.append(nn.PReLU())
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        self.projector = nn.Sequential(*layers)

    def forward(self, x, edge_index_x, ptr_x, y, edge_index_y, ptr_y):
        emb_x = self.dropout_op(self.opEmb(x))
        h_x, g_x = self.graph_encoder(emb_x, edge_index_x, ptr_x)
        return h_x, g_x

    def ssl_loss(self, z1, z2, T):
        n, _ = z1.size()
        z1_abs = z1.norm(dim=1)
        z2_abs = z2.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', z1, z2) / torch.einsum('i,j->ij', z1_abs, z2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(n), range(n)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        return - torch.log(loss)
    
    def batched_ssl_loss(self, z1, z2, T, batch_size):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []
        z1_abs = z1.norm(dim=1)
        z2_abs = z2.norm(dim=1)
        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            n, _ = z1[mask].size()
            sim_matrix = torch.einsum('ik,jk->ij', z1[mask], z2) / torch.einsum('i,j->ij', z1_abs[mask], z2_abs)
            sim_matrix = torch.exp(sim_matrix / T)
            pos_sim = sim_matrix[:, i * batch_size:(i + 1) * batch_size]
            loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
            losses.append(- torch.log(loss))
        return torch.cat(losses)

    def loss(self, x, edge_index_x, ptr_x, y, edge_index_y, ptr_y, T=0.2, batch_size=0, pretrain_target=None):
        emb_x = self.dropout_op(self.opEmb(x))
        emb_y = self.dropout_op(self.opEmb(y))
        edge_index_x = dropout_adj(edge_index_x, p=self.drop_edge_rate_1)[0]
        edge_index_y = dropout_adj(edge_index_x, p=self.drop_edge_rate_2)[0]
        emb_x = drop_feature(emb_x, self.drop_feature_rate_1)
        emb_y = drop_feature(emb_x, self.drop_feature_rate_2)

        h_x, g_x = self.graph_encoder(emb_x, edge_index_x, ptr_x)
        h_y, g_y = self.graph_encoder(emb_y, edge_index_y, ptr_y)

        # compute loss
        z1 = self.projector(g_x)
        z2 = self.projector(g_y)
        if batch_size == 0:
            loss = self.ssl_loss(z1, z2, T)
        else:
            loss = self.batched_ssl_loss(z1, z2, T, batch_size)
        loss = loss.mean()
        return loss


class PairWiseLearning_BGRL(nn.Module):
    def __init__(self, n_vocab, n_layers, d_model, activation, base_model, graph_encoder_dropout, dropout, drop_edge_rate_1, drop_edge_rate_2, drop_feature_rate_1, drop_feature_rate_2, projector_layers):
        super(PairWiseLearning_BGRL, self).__init__()
        self.d_model = d_model
        self.drop_edge_rate_1 = drop_edge_rate_1
        self.drop_edge_rate_2 = drop_edge_rate_2
        self.drop_feature_rate_1 = drop_feature_rate_1
        self.drop_feature_rate_2 = drop_feature_rate_2
        # Embedding Layer
        self.opEmb = nn.Embedding(n_vocab, d_model)
        self.dropout_op = nn.Dropout(p=dropout)
        # Graph Encoder
        self.graph_encoder = GraphEncoder(d_model, d_model, activation, base_model=base_model, k=n_layers, dropout=graph_encoder_dropout, skip=True, use_bn=True)
        # projector
        sizes = [d_model] + list(map(int, projector_layers.split('-')))
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.LayerNorm(sizes[i]))
            layers.append(nn.PReLU())
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        self.projector = nn.Sequential(*layers)

        # target network
        self.target_graph_encoder = GraphEncoder(d_model, d_model, activation, base_model=base_model, k=n_layers, dropout=graph_encoder_dropout, skip=True, use_bn=True)
        # stop gradient
        for param in self.target_graph_encoder.parameters():
            param.requires_grad = False

    def parameters(self):
        return list(self.opEmb.parameters()) + list(self.graph_encoder.parameters()) + list(self.projector.parameters())

    @torch.no_grad()
    def update_target_network(self, mm):
        r"""Performs a momentum update of the target network's weights.
        Args:
            mm (float): Momentum used in moving average update.
        """
        assert 0.0 <= mm <= 1.0, "Momentum needs to be between 0.0 and 1.0, got %.5f" % mm
        for param_q, param_k in zip(self.graph_encoder.parameters(), self.target_graph_encoder.parameters()):
            param_k.data.mul_(mm).add_(param_q.data, alpha=1. - mm)

    def forward(self, x, edge_index_x, ptr_x, y, edge_index_y, ptr_y):
        emb_x = self.dropout_op(self.opEmb(x))
        h_x, g_x = self.graph_encoder(emb_x, edge_index_x, ptr_x)
        return h_x, g_x

    def ssl_loss(self, q1, q2, y1, y2):
        return 2 - F.cosine_similarity(q1, y2.detach(), dim=-1).mean() - F.cosine_similarity(q2, y1.detach(), dim=-1).mean()

    def batched_ssl_loss(self, q1, q2, y1, y2, batch_size):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []
        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            losses.append(2 - F.cosine_similarity(q1[mask], y2.detach(), dim=-1).mean() - F.cosine_similarity(q2[mask], y1.detach(), dim=-1).mean())
        return losses

    def loss(self, x, edge_index_x, ptr_x, y, edge_index_y, ptr_y, batch_size=0, pretrain_target=None):
        emb_x = self.dropout_op(self.opEmb(x))
        emb_y = self.dropout_op(self.opEmb(y))
        edge_index_x = dropout_adj(edge_index_x, p=self.drop_edge_rate_1)[0]
        edge_index_y = dropout_adj(edge_index_x, p=self.drop_edge_rate_2)[0]
        emb_x = drop_feature(emb_x, self.drop_feature_rate_1)
        emb_y = drop_feature(emb_x, self.drop_feature_rate_2)

        h_x, g_x = self.graph_encoder(emb_x, edge_index_x, ptr_x)
        h_y, g_y = self.target_graph_encoder(emb_y, edge_index_y, ptr_y)
        q1 = self.projector(h_x)
        y2 = h_y

        h_y, g_y = self.graph_encoder(emb_y, edge_index_y, ptr_y)
        h_x, g_x = self.target_graph_encoder(emb_x, edge_index_x, ptr_x)
        q2 = self.projector(h_y)
        y1 = h_x

        # compute loss
        if batch_size == 0:
            loss = self.ssl_loss(q1, q2, y1, y2)
        else:
            loss = self.batched_ssl_loss(q1, q2, y1, y2, batch_size)
            loss = sum(losses) / len(losses)
        return loss


# Borrowed from https://github.com/fanyun-sun/InfoGraph
def get_positive_expectation(p_samples, measure, average=True):
    """Computes the positive part of a divergence / difference.
    Args:
        p_samples: Positive samples.
        measure: Measure to compute for.
        average: Average the result over samples.
    Returns:
        torch.Tensor
    """
    log_2 = np.log(2.)

    if measure == 'GAN':
        Ep = - F.softplus(-p_samples)
    elif measure == 'JSD':
        Ep = log_2 - F.softplus(- p_samples)
    elif measure == 'X2':
        Ep = p_samples ** 2
    elif measure == 'KL':
        Ep = p_samples + 1.
    elif measure == 'RKL':
        Ep = -torch.exp(-p_samples)
    elif measure == 'DV':
        Ep = p_samples
    elif measure == 'H2':
        Ep = 1. - torch.exp(-p_samples)
    elif measure == 'W1':
        Ep = p_samples

    if average:
        return Ep.mean()
    else:
        return Ep


def get_negative_expectation(q_samples, measure, average=True):
    """Computes the negative part of a divergence / difference.
    Args:
        q_samples: Negative samples.
        measure: Measure to compute for.
        average: Average the result over samples.
    Returns:
        torch.Tensor
    """
    log_2 = np.log(2.)

    if measure == 'GAN':
        Eq = F.softplus(-q_samples) + q_samples
    elif measure == 'JSD':
        Eq = F.softplus(-q_samples) + q_samples - log_2
    elif measure == 'X2':
        Eq = -0.5 * ((torch.sqrt(q_samples ** 2) + 1.) ** 2)
    elif measure == 'KL':
        Eq = torch.exp(q_samples)
    elif measure == 'RKL':
        Eq = q_samples - 1.
    elif measure == 'H2':
        Eq = torch.exp(q_samples) - 1.
    elif measure == 'W1':
        Eq = q_samples

    if average:
        return Eq.mean()
    else:
        return Eq


def local_global_loss_(l_enc, g_enc, batch, measure, mask=[]):
    '''
    Args:
        l: Local feature map.
        g: Global features.
        measure: Type of f-divergence. For use with mode `fd`
        mode: Loss mode. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.
    Returns:
        torch.Tensor: Loss.
    '''
    num_graphs = g_enc.shape[0]
    num_nodes = l_enc.shape[0]
    max_nodes = num_nodes // num_graphs

    pos_mask = torch.zeros((num_nodes, num_graphs)).cuda()
    neg_mask = torch.ones((num_nodes, num_graphs)).cuda()
    msk = torch.ones((num_nodes, num_graphs)).cuda()
    for nodeidx, graphidx in enumerate(batch):
        pos_mask[nodeidx][graphidx] = 1.
        neg_mask[nodeidx][graphidx] = 0.

    for idx, m in enumerate(mask):
        msk[idx * max_nodes + m: idx * max_nodes + max_nodes, idx] = 0.

    res = torch.mm(l_enc, g_enc.t()) * msk

    E_pos = get_positive_expectation(res * pos_mask, measure, average=False).sum()
    E_pos = E_pos / num_nodes
    E_neg = get_negative_expectation(res * neg_mask, measure, average=False).sum()
    E_neg = E_neg / (num_nodes * (num_graphs - 1))
    return E_neg - E_pos


def global_global_loss_(g1_enc, g2_enc, measure):
    '''
    Args:
        l: Local feature map.
        g: Global features.
        measure: Type of f-divergence. For use with mode `fd`
        mode: Loss mode. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.
    Returns:
        torch.Tensor: Loss.
    '''
    num_graphs = g1_enc.shape[0]

    pos_mask = torch.zeros((num_graphs, num_graphs)).cuda()
    neg_mask = torch.ones((num_graphs, num_graphs)).cuda()
    for graphidx in range(num_graphs):
        pos_mask[graphidx][graphidx] = 1.
        neg_mask[graphidx][graphidx] = 0.

    res = torch.mm(g1_enc, g2_enc.t())

    E_pos = get_positive_expectation(res * pos_mask, measure, average=False).sum()
    E_pos = E_pos / num_graphs
    E_neg = get_negative_expectation(res * neg_mask, measure, average=False).sum()
    E_neg = E_neg / (num_graphs * (num_graphs - 1))
    return E_neg - E_pos


class MLP(nn.Module):
    def __init__(self, in_ft, out_ft):
        super(MLP, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(in_ft, out_ft),
            nn.LayerNorm(out_ft),
            nn.PReLU(),
            nn.Linear(out_ft, out_ft),
            nn.LayerNorm(out_ft),
            nn.PReLU(),
            nn.Linear(out_ft, out_ft),
            nn.LayerNorm(out_ft),
            nn.PReLU()
        )
        self.linear_shortcut = nn.Linear(in_ft, out_ft)

    def forward(self, x):
        return self.ffn(x) + self.linear_shortcut(x)    


class PairWiseLearning_MVGRL(nn.Module):
    def __init__(self, n_vocab, n_layers, d_model, activation, base_model, graph_encoder_dropout, dropout, drop_edge_rate_1, drop_edge_rate_2, drop_feature_rate_1, drop_feature_rate_2, projector_layers):
        super(PairWiseLearning_MVGRL, self).__init__()
        self.d_model = d_model
        self.drop_edge_rate_1 = drop_edge_rate_1
        self.drop_edge_rate_2 = drop_edge_rate_2
        self.drop_feature_rate_1 = drop_feature_rate_1
        self.drop_feature_rate_2 = drop_feature_rate_2
        # Embedding Layer
        self.opEmb = nn.Embedding(n_vocab, d_model)
        self.dropout_op = nn.Dropout(p=dropout)
        # Graph Encoder
        self.graph_encoder_1 = GraphEncoder(d_model, d_model, activation, base_model=base_model, k=n_layers, dropout=graph_encoder_dropout, skip=True, use_bn=True)
        self.graph_encoder_2 = GraphEncoder(d_model, d_model, activation, base_model=base_model, k=n_layers, dropout=graph_encoder_dropout, skip=True, use_bn=True)
        self.mlp1 = MLP(d_model, d_model)
        self.mlp2 = MLP(d_model, d_model)
        self.bn1  = nn.BatchNorm1d(d_model, affine=False)
        self.bn2  = nn.BatchNorm1d(d_model, affine=False)

    def forward(self, x, edge_index_x, ptr_x, y, edge_index_y, ptr_y):
        emb_x = self.dropout_op(self.opEmb(x))
        emb_y = self.dropout_op(self.opEmb(y))
        edge_index_x = dropout_adj(edge_index_x, p=self.drop_edge_rate_1)[0]
        edge_index_y = dropout_adj(edge_index_x, p=self.drop_edge_rate_2)[0]
        emb_x = drop_feature(emb_x, self.drop_feature_rate_1)
        emb_y = drop_feature(emb_x, self.drop_feature_rate_2)

        h_x, g_x = self.graph_encoder_1(emb_x, edge_index_x, ptr_x)
        h_y, g_y = self.graph_encoder_2(emb_y, edge_index_y, ptr_y)
        h_x = self.mlp1(h_x)
        h_y = self.mlp1(h_y)
        g_x = self.mlp2(g_x)
        g_y = self.mlp2(g_y)
        return (h_x + h_y) / 2, (g_x + g_y) / 2

    def ssl_loss(self, h, g, batch, m):
        return local_global_loss_(h, g, batch, m)

    def batched_ssl_loss(self, h, g, batch, m, batch_size):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = h.device
        num_nodes = h.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        indices = torch.arange(0, num_nodes).to(device)
        losses = []
        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            losses.append(local_global_loss_(h[mask], g, batch[mask], m))
        return losses

    def loss(self, x, edge_index_x, ptr_x, y, edge_index_y, ptr_y, batch_size=0, pretrain_target=None):
        emb_x = self.dropout_op(self.opEmb(x))
        emb_y = self.dropout_op(self.opEmb(y))
        edge_index_x = dropout_adj(edge_index_x, p=self.drop_edge_rate_1)[0]
        edge_index_y = dropout_adj(edge_index_x, p=self.drop_edge_rate_2)[0]
        emb_x = drop_feature(emb_x, self.drop_feature_rate_1)
        emb_y = drop_feature(emb_x, self.drop_feature_rate_2)

        h_x, g_x = self.graph_encoder_1(emb_x, edge_index_x, ptr_x)
        h_y, g_y = self.graph_encoder_2(emb_y, edge_index_y, ptr_y)
        h_x = self.bn1(self.mlp1(h_x))
        h_y = self.bn1(self.mlp1(h_y))
        g_x = self.bn2(self.mlp2(g_x))
        g_y = self.bn2(self.mlp2(g_y))

        # compute loss
        batch = np.zeros(h_x.shape[0])
        for i, p in enumerate(ptr_x):
            batch[p:] = i
        batch = torch.LongTensor(batch).to(h_x.device)
        if batch_size == 0:
            loss1 = self.ssl_loss(h_x, g_y, batch, 'JSD')
            loss2 = self.ssl_loss(h_y, g_x, batch, 'JSD')
        else:
            losses1 = self.batched_ssl_loss(h_x, g_y, batch, 'JSD', batch_size)
            losses2 = self.batched_ssl_loss(h_y, g_x, batch, 'JSD', batch_size)
            loss1 = sum(losses1) / len(losses1)
            loss2 = sum(losses2) / len(losses2)
        #loss1 = local_global_loss_(h_x, g_y, batch, 'JSD')
        #loss2 = local_global_loss_(h_y, g_x, batch, 'JSD')
        # loss3 = global_global_loss_(gv1, gv2, 'JSD')
        loss = loss1 + loss2 #+ loss3
        return loss


class Predictor(torch.nn.Module):
    def __init__(self, graph_dim):
        super(Predictor, self).__init__()
        #self.fc = torch.nn.Linear(graph_dim, 1)
        self.fc = nn.Sequential(
            nn.Linear(graph_dim, graph_dim),
            nn.LayerNorm(graph_dim),
            nn.PReLU(),
            nn.Linear(graph_dim, graph_dim),
            nn.LayerNorm(graph_dim),
            nn.PReLU(),
            nn.Linear(graph_dim, 1)
        )
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, z):
        return self.fc(z)


class CLSHead(nn.Module):
    def __init__(self, n_vocab, d_model, dropout, init_weights=None):
        super(CLSHead, self).__init__()
        self.layer_1 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_2 = nn.Linear(d_model, n_vocab)
        if init_weights is not None:
            self.layer_2.weight = init_weights

    def forward(self, x):
        x = self.dropout(torch.tanh(self.layer_1(x)))
        return F.log_softmax(self.layer_2(x), dim=-1)


class MAELearning(nn.Module):
    def __init__(self, n_vocab, n_layers, d_model, activation, base_model, graph_encoder_dropout, dropout, drop_edge_rate_1, drop_edge_rate_2, drop_feature_rate_1, drop_feature_rate_2, projector_layers):
        super(MAELearning, self).__init__()
        self.d_model = d_model
        self.drop_edge_rate_1 = drop_edge_rate_1
        self.drop_edge_rate_2 = drop_edge_rate_2
        self.drop_feature_rate_1 = drop_feature_rate_1
        self.drop_feature_rate_2 = drop_feature_rate_2
        self.dropout = dropout
        # Embedding Layer
        self.opEmb = nn.Embedding(n_vocab, d_model)
        #self.dropout_op = nn.Dropout(p=dropout)
        # Graph Encoder
        self.graph_encoder = GraphEncoder(d_model, d_model, activation, base_model=base_model, k=n_layers, dropout=graph_encoder_dropout, skip=True, use_bn=True)
        # cls
        self.cls = CLSHead(n_vocab, d_model, graph_encoder_dropout)
        self.ssl_loss = nn.KLDivLoss()

    def forward(self, x, edge_index_x, ptr_x, y=None, edge_index_y=None, ptr_y=None):
        emb_x = self.opEmb(x)
        h_x, g_x = self.graph_encoder(emb_x, edge_index_x, ptr_x)
        return h_x, g_x

    def loss(self, x, edge_index_x, ptr_x, y, edge_index_y, ptr_y, lambd=0.0051, batch_size=0, pretrain_target=None):
        emb_x = self.opEmb(x)
        mask = (torch.rand(emb_x.size(0))<self.dropout).to(emb_x.device)
        emb_x[mask] = 0
        edge_index_x = dropout_adj(edge_index_x, p=self.drop_edge_rate_1)[0]
        emb_x = drop_feature(emb_x, self.drop_feature_rate_1)
        h_x, g_x = self.graph_encoder(emb_x, edge_index_x, ptr_x)
        # compute loss
        output = self.cls(h_x)
        target = torch.zeros_like(output).scatter_(1, x.unsqueeze(-1), 1)
        if pretrain_target == 'masked':
            loss = self.ssl_loss(output[mask], target[mask])
        else:
            loss = self.ssl_loss(output, target)
        return loss
