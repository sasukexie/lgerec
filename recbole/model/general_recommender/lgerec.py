# -*- coding: utf-8 -*-
# @Time   : 2021/10/12
# @Author : Tian Zhen
# @Email  : chenyuwuxinn@gmail.com

r"""
Reference code:
    https://github.com/sasukexie/lgerec
"""

import os
import pickle
import random

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType


class DynamicMultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super(DynamicMultiHeadAttention, self).__init__()
        assert embedding_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads."
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        # Linear transformation matrix for generating query, key, and value
        self.q_linear = nn.Linear(embedding_dim, embedding_dim)
        self.k_linear = nn.Linear(embedding_dim, embedding_dim)
        self.v_linear = nn.Linear(embedding_dim, embedding_dim)
        self.out_linear = nn.Linear(embedding_dim, embedding_dim)

        # Introduce trainable weight parameters for each attention head and initialize
        self.head_weights = nn.Parameter(torch.ones(self.num_heads))
        torch.nn.init.normal_(self.head_weights, mean=0, std=0.1)  # Initialize to a small random value

        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(dropout)

        # Layer Normalization to stabilize the training process
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, query, key, value):
        batch_size = query.size(0)

        # Linear transformation and split into multiple heads
        query = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.v_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Calculate attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        weighted_value = torch.matmul(attention_weights, value)

        # Apply head weights and Dropout
        weighted_value = weighted_value * self.head_weights.view(1, -1, 1, 1)
        weighted_value = self.dropout(weighted_value)
        weighted_value = weighted_value.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)

        # Pass the output through linear transformation and Layer Normalization
        output = self.out_linear(weighted_value)
        output = self.layer_norm(output)

        return output.squeeze(1)


class LGERec(GeneralRecommender):
    r"""SGL is a GCN-based recommendation model.

    SGL leverages self-supervised tasks to support classical recommendation tasks and enhances node representation learning through self-distinction.
    Specifically, SGL generates multiple views of nodes, maximizing the similarity between different views of the same node while keeping the views of other nodes distinct.
    SGL designs three operations for generating views—node dropping, edge dropping, and random walks—to alter the graph structure in different ways.

    This model is implemented using pairwise training mode.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(LGERec, self).__init__(config, dataset)
        # Retrieve interaction information between users and items, including all interactions
        self.a_inter_user = dataset.inter_feat[dataset.uid_field]
        self.a_inter_item = dataset.inter_feat[dataset.iid_field]
        self.embedding_size = config["embedding_size"]  # Embedding dimension
        self.n_layers = config["n_layers"]  # Number of GCN layers
        self.drop_ratio = config["dropout"]  # Dropout ratio
        self.cl_temperature = config["cl_temperature"]  # Temperature parameter for contrastive learning
        self.reg_weight = config["reg_weight"]  # Regularization weight
        self.cl_weight = config["cl_weight"]  # Contrastive learning weight
        self.sem_weight = config["sem_weight"]  # Semantic weight
        self.a_cl_weight = config["a_cl_weight"]  # Adaptive contrastive learning weight
        self.require_pow = config["require_pow"]
        self.neg_topk = config["neg_topk"]
        self.llm_embed_weight = config["llm_embed_weight"]

        # Initialize embeddings for users and items
        self.user_embedding = torch.nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = torch.nn.Embedding(self.n_items, self.embedding_size)
        self.reg_loss = EmbLoss()  # Embedding regularization loss
        self.ce_loss = nn.CrossEntropyLoss()
        self.mf_loss = BPRLoss()

        # Construct the complete interaction graph
        self.train_graph = self.build_graph(self.get_adjust_matrix(is_sub=False))
        self.restore_user_e = None  # Cache user embeddings
        self.restore_item_e = None  # Cache item embeddings
        self.apply(xavier_uniform_initialization)  # Apply Xavier uniform initialization
        self.other_parameter_name = ["restore_user_e", "restore_item_e"]

        # mix
        self.is_open_llm = config['is_open_llm']  # Whether to enable LLM embeddings
        self.is_all_embed = config['is_all_embed']  # Whether to use all LLM embeddings
        self.is_open_attention = config['is_open_attention']  # Whether to enable attention mechanism

        # Layer Normalization
        # self.layer_norm = nn.LayerNorm(self.embedding_size)  # Define layer normalization

        if self.is_open_llm:
            # Retrieve user and item IDs, IDs start from 1, and dataset will add 0 by default
            # Using with statement to ensure file is properly closed
            emb_map_path = f"./dataset/llm/{self.config['dataset']}/emb_map.pkl"
            id_map_path = f"./dataset/llm/{self.config['dataset']}/id_map.pkl"
            with open(emb_map_path, 'rb') as f:
                emb_map = pickle.load(f)

            with open(id_map_path, 'rb') as f:
                id_map = pickle.load(f)
            uid_seq1 = dataset.field2id_token['user_id']
            iid_seq1 = dataset.field2id_token['item_id']
            uid_seq = [0]
            iid_seq = [0]
            for i in range(1, len(uid_seq1)):
                id = uid_seq1[i]
                uid_seq.append(id_map['u'][id])
            for i in range(1, len(iid_seq1)):
                id = iid_seq1[i]
                iid_seq.append(id_map['i'][id])

            user_ex_all_embeddings = torch.tensor(emb_map['u'], device=self.device)[uid_seq]
            item_ex_all_embeddings = torch.tensor(emb_map['i'], device=self.device)[iid_seq]
            self.llm_embedding_dim = len(user_ex_all_embeddings[0])  # Adjust according to the embedding dimension of the pretrained model (e.g., llama:4096, qwen:5120)
            self.user_ex_all_embeddings = torch.nn.Embedding(self.n_users, self.llm_embedding_dim).from_pretrained(user_ex_all_embeddings)
            self.item_ex_all_embeddings = torch.nn.Embedding(self.n_items, self.llm_embedding_dim).from_pretrained(item_ex_all_embeddings)

            self.fusion_layer = torch.nn.Linear(self.embedding_size + self.llm_embedding_dim, self.embedding_size)

        if self.is_open_attention:
            # Initialize multi-head attention mechanism
            self.num_heads = config['num_heads']  # Default is 8 heads
            self.dropout = config['dropout']  # Get dropout configuration
            self.multihead_attention = DynamicMultiHeadAttention(self.embedding_size, self.num_heads, self.dropout)
            self.activation = nn.ReLU()

    def get_u_inter_map(self, u_inter_map_path=None):
        if os.path.exists(u_inter_map_path):
            with open(u_inter_map_path, 'rb') as f:
                u_inter_map = pickle.load(f)
            return u_inter_map

        # Construct all positive and negative samples for users
        u_pos_map = {}
        for u, i in zip(self.a_inter_user, self.a_inter_item):
            u, i = u.item(), i.item()
            if not u_pos_map.__contains__(u):
                u_pos_map[u] = set()
            u_pos_map[u].add(i)
        u_pos_map = {k: list(sorted(v)) for k, v in u_pos_map.items()}

        u_neg_map = {}
        a_iids = set(self.a_inter_item.tolist())
        for uid, iids in u_pos_map.items():
            u_neg_map[uid] = list(sorted(a_iids - set(iids)))

        u_inter_map = {'pos': u_pos_map, 'neg': u_neg_map}
        # Save dictionary as a pickle file
        with open(u_inter_map_path, 'wb') as pf:
            pickle.dump(u_inter_map, pf)
        return u_inter_map

    def rand_sample(self, high, size=None, replace=True):
        r"""Randomly select some nodes or edges.
        Args:
            high (int): Upper limit of index values
            size (int): Array size after sampling
        Returns:
            numpy.ndarray: Sampled index array with shape [size]
        """
        a = np.arange(high)
        sample = np.random.choice(a, size=size, replace=replace)
        return sample

    def get_adjust_matrix(self, is_sub: bool):
        r"""Get the normalized interaction matrix between users and items.
        Construct a square matrix from training data and normalize it using Laplacian matrix. If it is a subgraph, it may be processed by node dropping or edge dropping.
        Returns:
            csr_matrix: Normalized interaction matrix.
        """
        if is_sub:
            keep_item = self.rand_sample(
                len(self.a_inter_user),
                size=int(len(self.a_inter_user) * (1 - self.drop_ratio)),
                replace=False,
            )
            user = self.a_inter_user[keep_item]
            item = self.a_inter_item[keep_item]
            matrix = sp.csr_matrix(
                (np.ones_like(user), (user, item + self.n_users)),
                shape=(self.n_users + self.n_items, self.n_users + self.n_items),
            )
        else:
            ratings = np.ones_like(self.a_inter_user, dtype=np.float32)
            matrix = sp.csr_matrix(
                (ratings, (self.a_inter_user, self.a_inter_item + self.n_users)),
                shape=(self.n_users + self.n_items, self.n_users + self.n_items),
            )
        matrix = matrix + matrix.T  # Symmetrize the matrix
        D = np.array(matrix.sum(axis=1)) + 1e-7  # Compute degree matrix and prevent division by zero
        D = np.power(D, -0.5).flatten()
        D = sp.diags(D)
        return D.dot(matrix).dot(D)  # Laplacian normalization

    def build_graph(self, matrix: sp.csr_matrix):
        r"""Convert csr_matrix to tensor.
        Args:
            matrix (scipy.csr_matrix): Sparse matrix to convert.
        Returns:
            torch.sparse.FloatTensor: Converted sparse matrix.
        """
        matrix = matrix.tocoo()
        x = torch.sparse.FloatTensor(
            torch.LongTensor(np.array([matrix.row, matrix.col])),
            torch.FloatTensor(matrix.data.astype(np.float32)),
            matrix.shape,
        ).to(self.device)
        return x

    def get_ego_embeddings(self, is_has_llm=False):
        """ Get embeddings for users and items and combine them into one embedding matrix """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight

        # llm start
        if is_has_llm and self.is_all_embed:
            if self.is_open_llm:
                user_ex_all_embeddings = self.user_ex_all_embeddings.weight
                item_ex_all_embeddings = self.item_ex_all_embeddings.weight
                # Concatenate GCN embeddings with LLaMA text embeddings
                user_combined_embeddings = torch.cat([user_embeddings, self.llm_embed_weight * user_ex_all_embeddings], dim=1)
                item_combined_embeddings = torch.cat([item_embeddings, self.llm_embed_weight * item_ex_all_embeddings], dim=1)

                # Fuse the embeddings through a linear layer
                user_fused_embeddings = self.fusion_layer(user_combined_embeddings)
                item_fused_embeddings = self.fusion_layer(item_combined_embeddings)
                user_embeddings, item_embeddings = user_fused_embeddings, item_fused_embeddings
            if self.is_open_attention:
                # Apply dynamic multi-head attention mechanism to query, key, value
                user_final_embeddings = self.multihead_attention(user_embeddings, user_embeddings, user_embeddings)
                item_final_embeddings = self.multihead_attention(item_embeddings, item_embeddings, item_embeddings)
                user_embeddings, item_embeddings = user_final_embeddings, item_final_embeddings
        # llm end

        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self, graph, is_has_llm=False):
        r"""Forward pass to calculate embeddings for users and items.
        Args:
            graph: Interaction graph or subgraph.
        Returns:
            Embeddings for users and items.
        """
        all_embeddings = self.get_ego_embeddings(is_has_llm)
        embeddings_list = [all_embeddings]
        for i in range(self.n_layers):
            all_embeddings = torch.sparse.mm(graph, all_embeddings)
            embeddings_list.append(all_embeddings)
        embeddings_list = torch.stack(embeddings_list, dim=1)
        embeddings_list = torch.mean(embeddings_list, dim=1, keepdim=False)  # Mean pooling
        user_all_embeddings, item_all_embeddings = torch.split(embeddings_list, [self.n_users, self.n_items], dim=0)
        return user_all_embeddings, item_all_embeddings

    def calculate_loss(self, interaction):
        r"""Calculate the loss, including BPR loss and self-supervised loss."""
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]
        user_all_embeddings, item_all_embeddings = self.forward(self.train_graph)

        if self.is_all_embed:
            fusion_user_all_embeddings, fusion_item_all_embeddings = self.forward(self.train_graph, is_has_llm=True)
            user_embeddings = fusion_user_all_embeddings[user]
            pos_embeddings = fusion_item_all_embeddings[pos_item]
            neg_embeddings = fusion_item_all_embeddings[neg_item]
        else:
            user_embeddings = user_all_embeddings[user]
            pos_embeddings = item_all_embeddings[pos_item]
            neg_embeddings = item_all_embeddings[neg_item]

            # llm start
            if self.is_open_llm:
                user_ex_embeddings = self.user_ex_all_embeddings(user)
                pos_ex_embeddings = self.item_ex_all_embeddings(pos_item)
                neg_ex_embeddings = self.item_ex_all_embeddings(neg_item)

                # Concatenate GCN embeddings with LLaMA text embeddings
                user_combined_embeddings = torch.cat([user_embeddings, self.llm_embed_weight * user_ex_embeddings], dim=1)
                pos_item_combined_embeddings = torch.cat([pos_embeddings, self.llm_embed_weight * pos_ex_embeddings], dim=1)
                neg_item_combined_embeddings = torch.cat([neg_embeddings, self.llm_embed_weight * neg_ex_embeddings], dim=1)

                # Fuse the embeddings through a linear layer
                user_fused_embeddings = self.fusion_layer(user_combined_embeddings)
                pos_item_fused_embeddings = self.fusion_layer(pos_item_combined_embeddings)
                neg_item_fused_embeddings = self.fusion_layer(neg_item_combined_embeddings)
                user_embeddings, pos_embeddings, neg_embeddings = user_fused_embeddings, pos_item_fused_embeddings, neg_item_fused_embeddings
            if self.is_open_attention:
                # Apply dynamic multi-head attention mechanism to query, key, value
                user_final_embeddings = self.multihead_attention(user_embeddings, user_embeddings, user_embeddings)
                pos_item_final_embeddings = self.multihead_attention(pos_embeddings, pos_embeddings, pos_embeddings)
                neg_item_final_embeddings = self.multihead_attention(neg_embeddings, neg_embeddings, neg_embeddings)
                user_embeddings, pos_embeddings, neg_embeddings = user_final_embeddings, pos_item_final_embeddings, neg_item_final_embeddings
            # llm end

        # calculate BPR Loss
        pos_scores = torch.mul(user_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(user_embeddings, neg_embeddings).sum(dim=1)
        loss = self.mf_loss(pos_scores, neg_scores)

        loss += self.cl_weight * self.cl_loss(user_all_embeddings, item_all_embeddings, user, pos_item, neg_item) if self.cl_weight > 0. else 0.
        loss += self.sem_weight * self.sem_loss(user, pos_item, neg_item, user_all_embeddings, item_all_embeddings) if (self.sem_weight > 0.) else 0.

        u_embeddings1 = self.user_embedding(user)
        pos_embeddings1 = self.item_embedding(pos_item)
        neg_embeddings1 = self.item_embedding(neg_item)
        loss += self.reg_weight * self.reg_loss(u_embeddings1, pos_embeddings1, neg_embeddings1, require_pow=self.require_pow)[0]
        return loss

    def cl_loss(self, user_sub, item_sub, user, pos_item, neg_item):
        """
        Calculate contrastive learning loss, bringing user closer to positive items and pushing negative samples away.
        neg_candidates: candidate embeddings for negative samples, used to mine hard negative samples
        user_item_interactions: user interactions with candidate items, to help filter valid negative samples
        """
        # Retrieve negative samples
        u_neg_map = self.u_inter_map['neg']
        u_neg_ids = []
        for uid in user:
            t_top = self.neg_topk * 4
            neg_ids = u_neg_map[uid.item()]
            if len(neg_ids) < t_top:
                t_top_neg_ids = neg_ids
            else:
                t_top_neg_ids = random.sample(neg_ids, t_top)
            u_neg_ids.append(item_sub[t_top_neg_ids])

        # Convert list of tensors to a new tensor using torch.stack
        u_neg_emb = torch.stack(u_neg_ids)
        u_embeddings, pos_embeddings, neg_embeddings = user_sub[user], item_sub[pos_item], item_sub[neg_item]

        # Compute similarity between users and positive samples
        pos_similarity = F.cosine_similarity(u_embeddings, pos_embeddings) / self.cl_temperature

        # Compute similarity between users and negative sample candidates
        neg_similarity = F.cosine_similarity(u_embeddings.unsqueeze(1), u_neg_emb, dim=-1) / self.cl_temperature  # [Batch, neg_count]

        # Filter top hard negative samples with highest similarity to users
        neg_similarity, _ = neg_similarity.topk(k=self.neg_topk, dim=1, largest=True)

        # Average similarity of top-k hard negative samples
        hard_neg_similarity = neg_similarity.mean(dim=1)

        # Compute contrastive loss
        loss = -torch.log(torch.exp(pos_similarity) / (torch.exp(pos_similarity) + torch.exp(hard_neg_similarity)))
        return loss.mean()

    def a_loss(self, user_sub, item_sub, user, pos_item, neg_item):
        # Use embeddings after convolution or attention for prediction
        pos_scores = self.a_scores(user_sub, item_sub, user, pos_item)
        neg_scores = self.a_scores(user_sub, item_sub, user, neg_item)
        loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        return loss

    def a_scores(self, user_sub, item_sub, user, item):
        user_embeddings = user_sub[user]
        item_embeddings = item_sub[item]

        user_ex_embeddings = self.user_ex_all_embeddings[user]
        item_ex_embeddings = self.item_ex_all_embeddings[item]
        # Concatenate GCN embeddings with LLaMA text embeddings
        user_combined_embeddings = torch.cat([user_embeddings, self.llm_embed_weight * user_ex_embeddings], dim=1)
        item_combined_embeddings = torch.cat([item_embeddings, self.llm_embed_weight * item_ex_embeddings], dim=1)

        # Fuse the embeddings through a linear layer
        user_fused_embeddings = self.fusion_layer(user_combined_embeddings)
        item_fused_embeddings = self.fusion_layer(item_combined_embeddings)
        user_embeddings, item_embeddings = user_fused_embeddings, item_fused_embeddings

        # Apply self-attention mechanism to user and item embeddings
        user_attn_output = self.multihead_attention(user_embeddings, user_embeddings, user_embeddings)
        item_attn_output = self.multihead_attention(item_embeddings, item_embeddings, item_embeddings)

        # Non-linear activation
        user_embeddings = self.activation(user_attn_output)
        item_embeddings = self.activation(item_attn_output)

        return (user_embeddings * item_embeddings).sum(dim=1)

    def sem_loss(self,user, pos_item, neg_item, user_all_embeddings, item_all_embeddings):
        """
        Calculate joint semantic similarity loss and graph structure similarity loss, bringing users closer to positive samples and pushing negative samples away.

        Parameters:
        - user_sem_embeddings: User LLM embeddings, shape [batch_size, embed_dim1]
        - user_embeddings: User graph structure embeddings, shape [batch_size, embed_dim]
        - pos_sem_embeddings: Positive sample item LLM embeddings, shape [batch_size, embed_dim1]
        - pos_embeddings: Positive sample item graph structure embeddings, shape [batch_size, embed_dim]
        - neg_sem_embeddings: Negative sample item LLM embeddings, shape [batch_size, embed_dim1]
        - neg_embeddings: Negative sample item graph structure embeddings, shape [batch_size, embed_dim]

        Returns:
        - Semantic similarity loss value
        """

        user_embeddings, pos_embeddings, neg_embeddings = user_all_embeddings[user], item_all_embeddings[pos_item], item_all_embeddings[neg_item]
        user_sem_embeddings, pos_sem_embeddings, neg_sem_embeddings = self.user_ex_all_embeddings[user], self.item_ex_all_embeddings[pos_item], self.item_ex_all_embeddings[neg_item]

        # Compute similarity between user and positive samples for LLM and graph embeddings
        pos_similarity_llm = F.cosine_similarity(user_sem_embeddings, pos_sem_embeddings, dim=-1)
        pos_similarity_graph = F.cosine_similarity(user_embeddings, pos_embeddings, dim=-1)

        # Compute similarity between user and negative samples for LLM and graph embeddings
        neg_similarity_llm = F.cosine_similarity(user_sem_embeddings, neg_sem_embeddings, dim=-1)
        neg_similarity_graph = F.cosine_similarity(user_embeddings, neg_embeddings, dim=-1)

        # Combined similarity, merging semantic similarity with graph structure similarity
        pos_similarity = pos_similarity_llm + pos_similarity_graph
        neg_similarity = neg_similarity_llm + neg_similarity_graph

        # Compute semantic similarity loss, pushing negative samples away and bringing positive samples closer
        loss = -torch.log(torch.exp(pos_similarity) / (torch.exp(pos_similarity) + torch.exp(neg_similarity)))

        return loss.mean()

    def predict(self, interaction):
        r"""Predict user preferences for items."""
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward(self.train_graph)

        user = self.restore_user_e[interaction[self.USER_ID]]
        item = self.restore_item_e[interaction[self.ITEM_ID]]
        return torch.sum(user * item, dim=1)

    def full_sort_predict(self, interaction):
        r"""Predict user preferences for all items."""
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward(self.train_graph)

        user = self.restore_user_e[interaction[self.USER_ID]]
        return user.matmul(self.restore_item_e.T)

    def train(self, mode: bool = True):
        # # Construct all positive and negative samples for users
        # {'pos':{},'neg':{}}
        if self.cl_weight > 0 and not hasattr(self, 'u_inter_map'):
            self.u_inter_map = self.get_u_inter_map(f"{self.config['data_path']}/u_inter_map.pkl")
        r"""Override the base class training method, reconstruct the subgraph at each call."""
        return super().train(mode=mode)

    def train_epoch(self, train_data, epoch_idx):
        # print('train_epoch.epoch_idx:',epoch_idx)
        pass
