import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.one_hot_categorical import OneHotCategorical

from model.inference_mlp import InferenceMLP
from model.inference_utils import reparameterize


class InferenceGNN(InferenceMLP):
    def __init__(self, encoder, params):
        self.gnn_params = params.inference_params.gnn_params
        super(InferenceGNN, self).__init__(encoder, params)

    def init_model(self):
        params = self.params
        device = self.device
        gnn_params = self.gnn_params

        # model params
        self.continuous_state = continuous_state = params.continuous_state

        self.action_dim = action_dim = params.action_dim
        self.feature_dim = feature_dim = self.encoder.feature_dim
        self.feature_inner_dim = feature_inner_dim = self.encoder.feature_inner_dim

        self.node_attr_dim = node_attr_dim = gnn_params.node_attr_dim
        self.edge_attr_dim = edge_attr_dim = gnn_params.edge_attr_dim

        # ------------------------------------------------------------------ #
        # Edge existence threshold for masking weak edges
        self.edge_threshold = getattr(gnn_params, 'edge_threshold', 0.5)
        # Temperature for sigmoid during training (lower = sharper decisions)
        self.edge_temperature = getattr(gnn_params, 'edge_temperature', 1.0)
        # ------------------------------------------------------------------ #

        # Node embedders — one per feature dimension
        self.embedders = nn.ModuleList()
        for i in range(feature_dim):
            in_dim = 1 if continuous_state else feature_inner_dim[i]
            embedder_i = []
            for out_dim in gnn_params.embedder_dims:
                embedder_i.append(nn.Linear(in_dim, out_dim))
                embedder_i.append(nn.ReLU())
                in_dim = out_dim
            embedder_i.append(nn.Linear(in_dim, node_attr_dim))
            embedder_i = nn.Sequential(*embedder_i)
            self.embedders.append(embedder_i)

        # ------------------------------------------------------------------ #
        # Edge EXISTENCE predictor: (node_i, node_j) -> scalar score in [0,1]
        # Predicts whether an edge i->j exists
        edge_exist_net = []
        in_dim = node_attr_dim * 2
        for out_dim in gnn_params.edge_net_dims:
            edge_exist_net.append(nn.Linear(in_dim, out_dim))
            edge_exist_net.append(nn.ReLU())
            in_dim = out_dim
        edge_exist_net.append(nn.Linear(in_dim, 1))   # single logit per edge
        self.edge_exist_net = nn.Sequential(*edge_exist_net)
        # ------------------------------------------------------------------ #

        # Edge attribute network: (node_i, node_j) -> edge_attr
        edge_net = []
        in_dim = node_attr_dim * 2
        for out_dim in gnn_params.edge_net_dims:
            edge_net.append(nn.Linear(in_dim, out_dim))
            edge_net.append(nn.ReLU())
            in_dim = out_dim
        edge_net.append(nn.Linear(in_dim, edge_attr_dim))
        self.edge_net = nn.Sequential(*edge_net)

        # Node update network
        node_net = []
        in_dim = node_attr_dim + edge_attr_dim + action_dim
        for out_dim in gnn_params.node_net_dims:
            node_net.append(nn.Linear(in_dim, out_dim))
            node_net.append(nn.ReLU())
            in_dim = out_dim
        node_net.append(nn.Linear(in_dim, node_attr_dim))
        self.node_net = nn.Sequential(*node_net)

        # Output projectors — one per feature dimension
        self.projectors = nn.ModuleList()
        for i in range(feature_dim):
            in_dim = node_attr_dim
            final_dim = 2 if continuous_state or feature_inner_dim[i] == 1 else feature_inner_dim[i]
            projector_i = []
            for out_dim in gnn_params.projector_dims:
                projector_i.append(nn.Linear(in_dim, out_dim))
                projector_i.append(nn.ReLU())
                in_dim = out_dim
            projector_i.append(nn.Linear(in_dim, final_dim))
            projector_i = nn.Sequential(*projector_i)
            self.projectors.append(projector_i)

        # All directed edges i->j where i != j
        adj_full = torch.ones(feature_dim, feature_dim, device=device) - torch.eye(feature_dim, device=device)
        edge_pair = adj_full.nonzero(as_tuple=False)
        self.edge_left_idxes  = edge_pair[:, 0]   # source node indices
        self.edge_right_idxes = edge_pair[:, 1]   # target node indices

        # Store learned adjacency for inspection / graph eval
        # shape: (feature_dim, feature_dim), diagonal = 0
        self.register_buffer(
            'learned_adj',
            torch.zeros(feature_dim, feature_dim, device=device)
        )

    # ---------------------------------------------------------------------- #
    #  Core: compute soft adjacency from current embeddings
    # ---------------------------------------------------------------------- #
    def compute_edge_weights(self, embeddings):
        """
        Compute soft edge existence probabilities for all directed edges.

        :param embeddings: (bs, feature_dim, node_attr_dim)
        :return:
            edge_weights : (bs, feature_dim, feature_dim)  — soft adj matrix,
                           diagonal is always 0
            edge_exist_prob : (bs, num_edges) — per-edge existence probability
        """
        feature_dim = self.feature_dim

        edge_left  = embeddings[..., self.edge_left_idxes,  :]   # (bs, num_edge, node_attr_dim)
        edge_right = embeddings[..., self.edge_right_idxes, :]   # (bs, num_edge, node_attr_dim)
        edge_input = torch.cat([edge_left, edge_right], dim=-1)  # (bs, num_edge, node_attr_dim*2)

        # raw logit for edge existence
        edge_logit = self.edge_exist_net(edge_input).squeeze(-1)  # (bs, num_edge)
        edge_exist_prob = torch.sigmoid(edge_logit / self.edge_temperature)  # (bs, num_edge)

        # scatter into (bs, feature_dim, feature_dim) adjacency matrix
        bs = embeddings.shape[:-2]
        adj = torch.zeros(*bs, feature_dim, feature_dim, device=embeddings.device)
        adj[..., self.edge_left_idxes, self.edge_right_idxes] = edge_exist_prob

        return adj, edge_exist_prob

    # ---------------------------------------------------------------------- #
    #  Get discrete adjacency (for graph eval / logging)
    # ---------------------------------------------------------------------- #
    def get_adjacency(self):
        """
        Returns the current learned adjacency matrix as a (feature_dim, feature_dim)
        tensor of edge existence probabilities (averaged over recent batch).
        Compatible with the graph_eval code in train.py.
        """
        return self.learned_adj

    # ---------------------------------------------------------------------- #
    #  Forward step
    # ---------------------------------------------------------------------- #
    def forward_step(self, feature, action):
        """
        :param feature: (bs, feature_dim) if continuous, else [(bs, fi_dim)] * feature_dim
        :param action:  (bs, action_dim)
        :return: distribution over next state variables
        """
        if self.continuous_state:
            feature_ = torch.split(feature, 1, dim=-1)    # [(bs, 1)] * feature_dim
        else:
            feature_ = feature

        # 1. Embed each node
        embeddings = []
        for feature_i, embedder_i in zip(feature_, self.embedders):
            embeddings.append(embedder_i(feature_i.float()))   # (bs, node_attr_dim)
        embeddings = torch.stack(embeddings, dim=-2)           # (bs, feature_dim, node_attr_dim)

        # 2. Compute soft adjacency (edge existence probabilities)
        adj, edge_exist_prob = self.compute_edge_weights(embeddings)
        # adj: (bs, feature_dim, feature_dim)

        # Update stored learned_adj (mean over batch, detached)
        self.learned_adj = adj.detach().mean(dim=0) if adj.ndim == 3 else adj.detach()

        # 3. Compute edge attributes, weighted by existence probability
        edge_left  = embeddings[..., self.edge_left_idxes,  :]   # (bs, num_edge, node_attr_dim)
        edge_right = embeddings[..., self.edge_right_idxes, :]   # (bs, num_edge, node_attr_dim)
        edge_input = torch.cat([edge_left, edge_right], dim=-1)  # (bs, num_edge, node_attr_dim*2)
        edge_attr  = self.edge_net(edge_input)                   # (bs, num_edge, edge_attr_dim)

        # Weight edge attributes by existence probability
        # edge_exist_prob: (bs, num_edge) -> (bs, num_edge, 1)
        edge_attr = edge_attr * edge_exist_prob.unsqueeze(-1)

        # 4. Aggregate edge attributes into each target node
        bs = edge_attr.shape[:-2]
        feature_dim = self.feature_dim
        # (bs, feature_dim, feature_dim-1, edge_attr_dim)
        edge_attr_agg = edge_attr.reshape(*bs, feature_dim, feature_dim - 1, self.edge_attr_dim)
        edge_attr_agg = edge_attr_agg.sum(dim=-2)              # (bs, feature_dim, edge_attr_dim)

        # 5. Node update
        action_exp = action.unsqueeze(-2).expand(*bs, feature_dim, -1)   # (bs, feature_dim, action_dim)
        node_input = torch.cat([embeddings, action_exp, edge_attr_agg], dim=-1)
        next_node_attr = self.node_net(node_input)             # (bs, feature_dim, node_attr_dim)

        # 6. Project to output distribution
        next_features = []
        next_node_attr_list = torch.unbind(next_node_attr, dim=-2)
        for next_node_attr_i, projector_i in zip(next_node_attr_list, self.projectors):
            next_features.append(projector_i(next_node_attr_i))

        if self.continuous_state:
            next_features = torch.stack(next_features, dim=-2)            # (bs, feature_dim, 2)
            mu, log_std   = torch.unbind(next_features, dim=-1)
            return self.normal_helper(mu, feature, log_std)
        else:
            dist = []
            for feature_i, feature_i_inner_dim, dist_i in zip(feature, self.feature_inner_dim, next_features):
                if feature_i_inner_dim == 1:
                    mu, log_std = torch.split(dist_i, 1, dim=-1)
                    dist.append(self.normal_helper(mu, feature_i, log_std))
                else:
                    dist.append(OneHotCategorical(logits=dist_i))
            return dist

    # ---------------------------------------------------------------------- #
    #  Edge sparsity loss — call this in update() to encourage sparse graphs
    # ---------------------------------------------------------------------- #
    def edge_sparsity_loss(self, edge_exist_prob, lambda_sparse=1e-3):
        """
        L1 penalty on edge probabilities to encourage sparse adjacency.
        Add this to your prediction loss during training.

        :param edge_exist_prob: (bs, num_edges) from compute_edge_weights()
        :param lambda_sparse: sparsity weight
        :return: scalar loss
        """
        return lambda_sparse * edge_exist_prob.mean()