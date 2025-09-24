import torch
import torch.nn as nn
import numpy as np

from recbole.utils import InputType
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
from recbole.model.init import xavier_normal_initialization


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        return outputs


class SASRec(SequentialRecommender):

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(SASRec, self).__init__(config, dataset)

        # load dataset info
        self.n_users = dataset.user_num
        self.n_items = dataset.item_num

        # load parameters info - SASRec parameters
        self.hidden_units = config['hidden_size'] if 'hidden_size' in config else 64
        # Expose fields expected by TTARArec
        self.hidden_size = self.hidden_units
        self.hidden_act = config['hidden_act'] if 'hidden_act' in config else 'gelu'
        self.initializer_range = config['initializer_range'] if 'initializer_range' in config else 0.02
        self.num_heads = config['n_heads'] if 'n_heads' in config else 2
        self.num_blocks = config['n_layers'] if 'n_layers' in config else 2
        self.dropout_rate = config['hidden_dropout_prob'] if 'hidden_dropout_prob' in config else 0.5
        self.max_len = config['MAX_ITEM_LIST_LENGTH'] if 'MAX_ITEM_LIST_LENGTH' in config else 50
        self.norm_first = config['norm_first'] if 'norm_first' in config else True

        # SASRec layers
        self.item_emb = nn.Embedding(self.n_items, self.hidden_units, padding_idx=0)
        # Alias expected by TTARArec
        self.item_embedding = self.item_emb
        self.pos_emb = nn.Embedding(self.max_len + 1, self.hidden_units, padding_idx=0)
        self.emb_dropout = nn.Dropout(p=self.dropout_rate)

        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()

        self.last_layernorm = nn.LayerNorm(self.hidden_units, eps=1e-8)

        for _ in range(self.num_blocks):
            new_attn_layernorm = nn.LayerNorm(self.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = nn.MultiheadAttention(
                self.hidden_units,
                self.num_heads,
                self.dropout_rate,
                batch_first=True
            )
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = nn.LayerNorm(self.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(self.hidden_units, self.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

        # loss function (alias to match TTARArec usage)
        self.loss_fct = nn.CrossEntropyLoss()
        self.loss = self.loss_fct

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def log2feats(self, log_seqs):
        """Convert item sequences to features using embeddings and positional encoding"""
        seqs = self.item_emb(log_seqs)
        seqs *= self.item_emb.embedding_dim ** 0.5
        
        # Create positional encoding
        batch_size, seq_len = log_seqs.shape
        poss = torch.arange(1, seq_len + 1, device=log_seqs.device).unsqueeze(0).expand(batch_size, -1).clone()
        poss = poss * (log_seqs != 0)  # Mask padding positions
        seqs += self.pos_emb(poss)
        seqs = self.emb_dropout(seqs)

        # Create attention mask for causality
        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=seqs.device))

        # Apply transformer blocks
        for i in range(len(self.attention_layers)):
            if self.norm_first:
                x = self.attention_layernorms[i](seqs)
                mha_outputs, _ = self.attention_layers[i](x, x, x, attn_mask=attention_mask)
                seqs = seqs + mha_outputs
                seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
            else:
                mha_outputs, _ = self.attention_layers[i](seqs, seqs, seqs, attn_mask=attention_mask)
                seqs = self.attention_layernorms[i](seqs + mha_outputs)
                seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))

        log_feats = self.last_layernorm(seqs)
        return log_feats

    def encode(self, item_seq, item_seq_len):
        """Encode sequence to a single vector [B, H] expected by TTARArec."""
        # Note: item_seq_len is currently not used because we rely on padding mask inside
        log_feats = self.log2feats(item_seq)
        final_feat = log_feats[:, -1, :]
        return final_feat

    def forward(self, item_seq, item_seq_len):
        """Return sequence embedding [B, hidden_size] for TTARArec."""
        return self.encode(item_seq, item_seq_len)

    def logits_from_seq(self, item_seq, item_seq_len):
        final_feat = self.encode(item_seq, item_seq_len)  # [B, H]
        all_item_embs = self.item_embedding.weight[1:]
        logits = torch.matmul(final_feat, all_item_embs.T)
        return logits

    def calculate_loss(self, interaction):
        """Calculate Cross Entropy loss"""
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN] if self.ITEM_SEQ_LEN in interaction else None
        logits = self.logits_from_seq(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        loss = self.loss_fct(logits, pos_items - 1)
        return loss

    def predict(self, interaction):
        """Predict scores for given user-item pairs"""
        item_seq = interaction[self.ITEM_SEQ]
        item = interaction[self.ITEM_ID]
        
        final_feat = self.encode(item_seq, None)
        item_emb = self.item_embedding(item)  # [batch_size, hidden_units]
        scores = (final_feat * item_emb).sum(dim=-1)  # [batch_size]
        
        return scores

    def full_sort_predict(self, interaction):
        """Predict scores for all items for given users"""
        item_seq = interaction[self.ITEM_SEQ]
        
        final_feat = self.encode(item_seq, None)
        # Get all item embeddings except padding id 0
        valid_item_embs = self.item_embedding.weight[1:]  # [n_items-1, hidden_units]
        
        # Calculate scores for valid items
        scores_valid = torch.matmul(final_feat, valid_item_embs.T)  # [batch_size, n_items-1]
        
        # Prepend a padding column for item id 0 to match framework's tot_item_num
        pad_col = torch.full(
            (scores_valid.size(0), 1),
            fill_value=-1e9,
            device=scores_valid.device,
            dtype=scores_valid.dtype,
        )
        scores = torch.cat([pad_col, scores_valid], dim=1)  # [batch_size, n_items]
        
        return scores