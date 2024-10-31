# -*- coding: utf-8 -*-
# @Time   : 2023/10/13
# @Author : OpenAI Assistant
# @Email  : openai_assistant@example.com

"""
GraphSASRec
################################################

融合图神经网络和 SASRec 的序列推荐模型。

模型特点：
- 使用 GNN 捕获序列的局部图结构信息。
- 使用单向 Transformer（SASRec）捕获全局上下文信息。
- 避免复杂的损失设计，确保模型训练稳定。

"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss


class GNNLayer(nn.Module):
    def __init__(self, embedding_size, step=1):
        super(GNNLayer, self).__init__()
        self.embedding_size = embedding_size
        self.step = step
        self.w_ih = nn.Linear(embedding_size, embedding_size)
        self.w_hh = nn.Linear(embedding_size, embedding_size)

    def forward(self, A, hidden):
        for _ in range(self.step):
            m = torch.matmul(A, hidden)
            m = self.w_ih(m)
            hidden = self.w_hh(hidden)
            hidden = torch.relu(m + hidden)
        return hidden


class NewRec(SequentialRecommender):
    def __init__(self, config, dataset):
        super(NewRec, self).__init__(config, dataset)

        # 模型参数
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']
        self.inner_size = config['inner_size']
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']

        self.step = config['step']
        self.device = config['device']
        self.loss_type = config['loss_type']
        self.initializer_range = config['initializer_range']

        # 定义模型的层
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.gnn = GNNLayer(self.hidden_size, self.step)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError(f"Make sure 'loss_type' in ['BPR', 'CE'], but got {self.loss_type}")

        # 参数初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """参数初始化"""
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight.data, mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _construct_graph(self, item_seq):
        batch_size, seq_len = item_seq.size()
        A = []
        for session in item_seq.cpu().numpy():
            node = np.unique(session[session > 0])
            node_len = len(node)
            adj = np.zeros((seq_len, seq_len), dtype=np.float32)
            for i in range(seq_len - 1):
                if session[i] == 0 or session[i + 1] == 0:
                    continue
                u = i
                v = i + 1
                adj[u][v] = 1
            deg = np.sum(adj, axis=1, keepdims=True) + 1e-8  # 防止除以零
            adj = adj / deg
            A.append(adj)
        A = torch.tensor(A, dtype=torch.float32, device=self.device)
        return A

    def forward(self, item_seq, item_seq_len):
        # 构建图结构
        A = self._construct_graph(item_seq)
        item_emb = self.item_embedding(item_seq)  # [B, L, H]
        hidden = self.gnn(A, item_emb)

        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_emb = self.position_embedding(position_ids)
        input_emb = hidden + position_emb.unsqueeze(0)
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        attention_mask = self.get_attention_mask(item_seq)
        trm_output = self.trm_encoder(input_emb, attention_mask, output_all_encoded_layers=False)
        output = trm_output[-1]  # [B, L, H]

        # 获取最后一个有效位置的表示
        seq_output = self.gather_indexes(output, item_seq_len - 1)
        return seq_output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)
            loss = self.loss_fct(pos_score, neg_score)
        else:  # 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
        return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.sum(seq_output * test_item_emb, dim=-1)
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores
