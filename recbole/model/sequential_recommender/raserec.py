# -*- coding: utf-8 -*-
# @Time    : 2025/1/1
# @Author  : Xinping Zhao
# @Email   : zhaoxinping@stu.hit.edu.cn

"""
SASRec
################################################

Reference:
    Wang-Cheng Kang et al. "Self-Attentive Sequential Recommendation." in ICDM 2018.

Reference:
    https://github.com/kang205/SASRec

""" 
import torch, heapq, scipy, faiss, random, math
from faiss import normalize_L2
from torch import nn
import numpy as np
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder, CrossMultiHeadAttention, FeedForward, activation_layer, MLPLayers, MultiHeadAttention
from recbole.model.loss import BPRLoss
import torch.nn.functional as F


class RaSeRec(SequentialRecommender):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, config, dataset):
        super(RaSeRec, self).__init__(config, dataset)

        self.len_lower_bound = config["len_lower_bound"] if "len_lower_bound" in config else -1
        self.len_upper_bound = config["len_upper_bound"] if "len_upper_bound" in config else -1
        self.len_bound_reverse = config["len_bound_reverse"] if "len_bound_reverse" in config else True
        self.nprobe = config['nprobe']
        self.topk = config['top_k']
        self.alpha = config['alpha']
        self.low_popular = config['low_popular'] if 'low_popular' in config else 100
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']

        # RetrieverEncoder相关参数
        self.retriever_layers = config['retriever_layers'] if 'retriever_layers' in config else 1
        self.retriever_temperature = config['retriever_temperature'] if 'retriever_temperature' in config else 0.1
        self.recommendation_temperature = config['recommendation_temperature'] if 'recommendation_temperature' in config else 0.1
        self.retriever_dropout = config['retriever_dropout'] if 'retriever_dropout' in config else 0.1
        self.kl_weight = config['kl_weight'] if 'kl_weight' in config else 0.1

        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        # 检索器编码器 - 简单MLP架构
        self.retriever_mlp = nn.ModuleList()
        self.retriever_layer_norms = nn.ModuleList()
        
        for i in range(self.retriever_layers):
            self.retriever_mlp.append(
                nn.Linear(self.hidden_size, self.hidden_size)
            )
            self.retriever_layer_norms.append(
                nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
            )
        
        # 残差连接和激活函数
        self.retriever_act_fn = activation_layer(self.hidden_act)
        self.retriever_dropout_layer = nn.Dropout(self.retriever_dropout)

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)
        # precached knowledge
        self.dataset = dataset
        
        # 加载预训练融合模块参数
        pretrained_path = config['pretrained_path']
        if pretrained_path:
            self._load_pretrained_weights(pretrained_path)
        
        # 冻结除检索器编码器以外的所有参数
        self._freeze_parameters()

    def _load_pretrained_weights(self, pretrained_path):
        """加载预训练模型参数"""
        try:
            print(f"正在加载预训练模型参数: {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
            
            # 如果checkpoint是字典且包含state_dict
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                pretrained_state_dict = checkpoint['state_dict']
            else:
                pretrained_state_dict = checkpoint
            
            # 过滤掉不存在的参数键（例如被移除的交叉注意力相关参数）
            current_state_dict = self.state_dict()
            filtered_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in current_state_dict}
            
            # 加载预训练参数
            missing_keys, unexpected_keys = self.load_state_dict(filtered_state_dict, strict=False)
            
            print(f"成功加载预训练模型参数!")
            print(f"缺失的参数键: {missing_keys}")
            print(f"多余的参数键: {unexpected_keys}")
        except Exception as e:
            print(f"加载预训练模型参数失败: {e}")

    def _freeze_parameters(self):
        """冻结除检索器编码器以外的所有参数"""
        # 首先冻结所有参数
        for name, param in self.named_parameters():
            param.requires_grad = False
        
        # 然后只解冻检索器编码器的参数
        for name, param in self.retriever_mlp.named_parameters():
            param.requires_grad = True
        
        for name, param in self.retriever_layer_norms.named_parameters():
            param.requires_grad = True
        
        # 统计并打印
        frozen_count = 0
        trainable_count = 0
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_count += 1
                print(f"可训练参数: {name}")
            else:
                frozen_count += 1
            
        print(f"已冻结 {frozen_count} 个参数，保留 {trainable_count} 个可训练参数")

    def precached_knowledge(self):
        length_threshold = 1
        seq_emb_knowledge, tar_emb_knowledge, user_id_list = None, None, None
        item_seq_all = None
        item_seq_len_all = None
        for batch_idx, interaction in enumerate(self.dataset):
            interaction = interaction.to("cuda")
            if self.len_lower_bound != -1 or self.len_upper_bound != -1:
                if self.len_lower_bound != -1 and self.len_upper_bound != -1:
                    look_up_indices = (interaction[self.ITEM_SEQ_LEN]>=self.len_lower_bound) * (interaction[self.ITEM_SEQ_LEN]<=self.len_upper_bound)
                elif self.len_upper_bound != -1:
                    look_up_indices = interaction[self.ITEM_SEQ_LEN]<self.len_upper_bound
                else:
                    look_up_indices = interaction[self.ITEM_SEQ_LEN]>self.len_lower_bound
                if self.len_bound_reverse:
                    look_up_indices = ~look_up_indices
            else:
                look_up_indices = interaction[self.ITEM_SEQ_LEN]>-1
            item_seq = interaction[self.ITEM_SEQ][look_up_indices]
            if item_seq_all==None:
                item_seq_all = item_seq
            else:
                item_seq_all = torch.cat((item_seq_all, item_seq), dim=0)
            item_seq_len = interaction[self.ITEM_SEQ_LEN][look_up_indices]
            item_seq_len_list = list(interaction[self.ITEM_SEQ_LEN][look_up_indices].detach().cpu().numpy())
            if isinstance(item_seq_len_all, list):
                item_seq_len_all.extend(item_seq_len_list)
            else:
                item_seq_len_all = item_seq_len_list
            seq_output = self.forward(item_seq, item_seq_len)
            tar_items = interaction[self.POS_ITEM_ID][look_up_indices]
            tar_items_emb = self.item_embedding(tar_items)
            user_id_cans = list(interaction[self.USER_ID][look_up_indices].detach().cpu().numpy())
            if isinstance(seq_emb_knowledge, np.ndarray):
                seq_emb_knowledge = np.concatenate((seq_emb_knowledge, seq_output.detach().cpu().numpy()), 0)
            else:
                seq_emb_knowledge = seq_output.detach().cpu().numpy()
            
            if isinstance(tar_emb_knowledge, np.ndarray):
                tar_emb_knowledge = np.concatenate((tar_emb_knowledge, tar_items_emb.detach().cpu().numpy()), 0)
            else:
                tar_emb_knowledge = tar_items_emb.detach().cpu().numpy()
            
            if isinstance(user_id_list, list):
                user_id_list.extend(user_id_cans)
            else:
                user_id_list = user_id_cans
        self.user_id_list = user_id_list
        self.item_seq_all = item_seq_all
        self.item_seq_len_all = item_seq_len_all
        self.seq_emb_knowledge = seq_emb_knowledge
        self.tar_emb_knowledge = tar_emb_knowledge
        # faiss
        d = self.hidden_size
        nlist = 128
        seq_emb_knowledge_copy = np.array(seq_emb_knowledge, copy=True)
        normalize_L2(seq_emb_knowledge_copy)
        seq_emb_quantizer = faiss.IndexFlatL2(d) 
        self.seq_emb_index = faiss.IndexIVFFlat(seq_emb_quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT) 
        self.seq_emb_index.train(seq_emb_knowledge_copy)
        self.seq_emb_index.add(seq_emb_knowledge_copy)    
        self.seq_emb_index.nprobe=self.nprobe

        tar_emb_knowledge_copy = np.array(tar_emb_knowledge, copy=True)
        normalize_L2(tar_emb_knowledge_copy)
        tar_emb_quantizer = faiss.IndexFlatL2(d) 
        self.tar_emb_index = faiss.IndexIVFFlat(tar_emb_quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT) 
        self.tar_emb_index.train(tar_emb_knowledge_copy)
        self.tar_emb_index.add(tar_emb_knowledge_copy) 
        self.tar_emb_index.nprobe=self.nprobe

    def precached_knowledge_val(self, val_dataset):
        length_threshold = 1
        seq_emb_knowledge, tar_emb_knowledge, user_id_list = None, None, None
        item_seq_len_all = None
        for batch_idx, interaction in enumerate(self.dataset):
            interaction = interaction.to("cuda")
            if self.len_lower_bound != -1 or self.len_upper_bound != -1:
                if self.len_lower_bound != -1 and self.len_upper_bound != -1:
                    look_up_indices = (interaction[self.ITEM_SEQ_LEN]>=self.len_lower_bound) * (interaction[self.ITEM_SEQ_LEN]<=self.len_upper_bound)
                elif self.len_upper_bound != -1:
                    look_up_indices = interaction[self.ITEM_SEQ_LEN]<self.len_upper_bound
                else:
                    look_up_indices = interaction[self.ITEM_SEQ_LEN]>self.len_lower_bound
                if self.len_bound_reverse:
                    look_up_indices = ~look_up_indices
            else:
                look_up_indices = interaction[self.ITEM_SEQ_LEN]>-1
            item_seq = interaction[self.ITEM_SEQ][look_up_indices]
            item_seq_len = interaction[self.ITEM_SEQ_LEN][look_up_indices]
            item_seq_len_list = list(interaction[self.ITEM_SEQ_LEN][look_up_indices].detach().cpu().numpy())
            if isinstance(item_seq_len_all, list):
                item_seq_len_all.extend(item_seq_len_list)
            else:
                item_seq_len_all = item_seq_len_list
            seq_output = self.forward(item_seq, item_seq_len)
            tar_items = interaction[self.POS_ITEM_ID][look_up_indices]
            tar_items_emb = self.item_embedding(tar_items)
            user_id_cans = list(interaction[self.USER_ID][look_up_indices].detach().cpu().numpy())
            if isinstance(seq_emb_knowledge, np.ndarray):
                seq_emb_knowledge = np.concatenate((seq_emb_knowledge, seq_output.detach().cpu().numpy()), 0)
            else:
                seq_emb_knowledge = seq_output.detach().cpu().numpy()
            if isinstance(tar_emb_knowledge, np.ndarray):
                tar_emb_knowledge = np.concatenate((tar_emb_knowledge, tar_items_emb.detach().cpu().numpy()), 0)
            else:
                tar_emb_knowledge = tar_items_emb.detach().cpu().numpy()
            if isinstance(user_id_list, list):
                user_id_list.extend(user_id_cans)
            else:
                user_id_list = user_id_cans
        length_threshold = 1
        for batch_idx, batched_data in enumerate(val_dataset):
            interaction, history_index, swap_row, swap_col_after, swap_col_before = batched_data
            interaction = interaction.to("cuda")
            item_seq = interaction[self.ITEM_SEQ]
            item_seq_len = interaction[self.ITEM_SEQ_LEN]
            item_seq_len_list = list(interaction[self.ITEM_SEQ_LEN].detach().cpu().numpy())
            if isinstance(item_seq_len_all, list):
                item_seq_len_all.extend(item_seq_len_list)
            else:
                item_seq_len_all = item_seq_len_list

            seq_output = self.forward(item_seq, item_seq_len)
            tar_items = interaction[self.POS_ITEM_ID]
            tar_items_emb = self.item_embedding(tar_items)
            user_id_cans = list(interaction[self.USER_ID].detach().cpu().numpy())
            if isinstance(seq_emb_knowledge, np.ndarray):
                seq_emb_knowledge = np.concatenate((seq_emb_knowledge, seq_output.detach().cpu().numpy()), 0)
            else:
                seq_emb_knowledge = seq_output.detach().cpu().numpy()
            if isinstance(tar_emb_knowledge, np.ndarray):
                tar_emb_knowledge = np.concatenate((tar_emb_knowledge, tar_items_emb.detach().cpu().numpy()), 0)
            else:
                tar_emb_knowledge = tar_items_emb.detach().cpu().numpy()
            if isinstance(user_id_list, list):
                user_id_list.extend(user_id_cans)
            else:
                user_id_list = user_id_cans
        self.user_id_list = user_id_list
        self.item_seq_len_all = item_seq_len_all

        self.seq_emb_knowledge = seq_emb_knowledge
        self.tar_emb_knowledge = tar_emb_knowledge
        # faiss
        d = self.hidden_size  
        nlist = 128
        seq_emb_knowledge_copy = np.array(seq_emb_knowledge, copy=True)
        normalize_L2(seq_emb_knowledge_copy)
        seq_emb_quantizer = faiss.IndexFlatL2(d) 
        self.seq_emb_index = faiss.IndexIVFFlat(seq_emb_quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT) 
        self.seq_emb_index.train(seq_emb_knowledge_copy)
        self.seq_emb_index.add(seq_emb_knowledge_copy)    
        self.seq_emb_index.nprobe=self.nprobe

        tar_emb_knowledge_copy = np.array(tar_emb_knowledge, copy=True)
        normalize_L2(tar_emb_knowledge_copy)
        tar_emb_quantizer = faiss.IndexFlatL2(d) 
        self.tar_emb_index = faiss.IndexIVFFlat(tar_emb_quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT) 
        self.tar_emb_index.train(tar_emb_knowledge_copy)
        self.tar_emb_index.add(tar_emb_knowledge_copy) 
        self.tar_emb_index.nprobe=self.nprobe

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            # module.weight.data = self.truncated_normal_(tensor=module.weight.data, mean=0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def truncated_normal_(self, tensor, mean=0, std=0.09):
        with torch.no_grad():
            size = tensor.shape
            tmp = tensor.new_empty(size + (4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)
            return tensor

    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask
    
    def get_bi_attention_mask(self, item_seq):
        """Generate bidirectional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # bidirectional mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)
        # extended_attention_mask = self.get_bi_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H]

    def seq_augmented(self, seq_output, batch_user_id, batch_seq_len, mode="train"):
        """使用目标嵌入进行序列增强"""
        # 直接使用原始序列表示进行检索
        retrieved_seqs1, retrieved_tars1, retrieved_seqs2, retrieved_tars2 = self.retrieve_seq_tar(seq_output, batch_user_id, batch_seq_len, topk=self.topk, mode=mode)

        # 使用检索器编码器调整原始序列表示
        retriever_encoded_seq = self.retriever_forward(seq_output)
        
        # 计算检索器编码后序列与相似序列的相似度作为权重
        similarities = torch.sum(retriever_encoded_seq.unsqueeze(1) * retrieved_seqs1, dim=-1)  # [B, K]
        weights = torch.softmax(similarities, dim=-1)  # [B, K]
        
        # 使用权重来融合对应的目标嵌入
        weighted_retrieved = torch.sum(weights.unsqueeze(-1) * retrieved_tars1, dim=1)  # [B, H]
        
        # 与原始序列进行加权组合（注意：这里仍然用原始序列，不是编码后的）
        alpha = self.alpha
        seq_output = alpha * seq_output + (1 - alpha) * weighted_retrieved
        return seq_output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        batch_user_id = list(interaction[self.USER_ID].detach().cpu().numpy())
        batch_seq_len = list(item_seq_len.detach().cpu().numpy())
        
        # 直接使用原始序列表示进行检索，不使用检索器编码器
        retrieved_seqs1, retrieved_tars1, retrieved_seqs2, retrieved_tars2 = self.retrieve_seq_tar(
            seq_output,  # 直接使用原始序列表示
            batch_user_id, 
            batch_seq_len,
            topk=self.topk
        )
        
        # 计算检索分布：基于检索序列与目标项的点积相似度
        retrieval_probs = self.compute_retrieval_scores(
            seq_output, retrieved_seqs1, retrieved_tars1, pos_items
        ) # [B, K]
        
        # 计算推荐分布：基于原序列与检索序列的点积相似度
        recommendation_probs = self.compute_recommendation_scores(
            seq_output, retrieved_seqs2, retrieved_tars2
        ) # [B, K]
        
        # 计算KL散度损失
        kl_loss = self.compute_kl_loss(retrieval_probs, recommendation_probs)
        
        # 总损失 = KL散度损失 * 权重
        total_loss = self.kl_weight * kl_loss
        
        return total_loss
    
    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def retrieve_seq_tar(self, queries, batch_user_id, batch_seq_len, topk=5, mode="train"):
        """检索相似序列和对应的目标嵌入"""
        queries_cpu = queries.detach().cpu().numpy()
        normalize_L2(queries_cpu)
        _, I1 = self.seq_emb_index.search(queries_cpu, 4*topk)
        I1_filtered = []
        for i, I_entry in enumerate(I1):
            current_user = batch_user_id[i]
            current_length = batch_seq_len[i]
            filtered_indices = [idx for idx in I_entry if self.user_id_list[idx] != current_user or (self.user_id_list[idx] == current_user and self.item_seq_len_all[idx] < current_length)]
            I1_filtered.append(filtered_indices[:topk])
        I1_filtered = np.array(I1_filtered)
        if mode=="train":
            retrieval_seq1 = self.seq_emb_knowledge[I1_filtered]
            retrieval_tar1 = self.tar_emb_knowledge[I1_filtered]
            retrieval_seq2 = self.seq_emb_knowledge[I1_filtered]
            retrieval_tar2 = self.tar_emb_knowledge[I1_filtered]
        else:
            retrieval_seq1 = self.seq_emb_knowledge[I1_filtered]
            retrieval_tar1 = self.tar_emb_knowledge[I1_filtered]
            retrieval_seq2 = self.seq_emb_knowledge[I1_filtered]
            retrieval_tar2 = self.tar_emb_knowledge[I1_filtered]
        return torch.tensor(retrieval_seq1).to("cuda"), torch.tensor(retrieval_tar1).to("cuda"), torch.tensor(retrieval_seq2).to("cuda"), torch.tensor(retrieval_tar2).to("cuda")

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        batch_user_id = list(interaction[self.USER_ID].detach().cpu().numpy())
        batch_seq_len = list(item_seq_len.detach().cpu().numpy())
        # aug
        seq_output_aug = self.seq_augmented(seq_output, batch_user_id, batch_seq_len, mode="test")
        seq_output_aug = torch.where((item_seq_len > self.low_popular).unsqueeze(-1).repeat(1, self.hidden_size), seq_output, seq_output_aug)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output_aug, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores

    def retriever_forward(self, seq_output):
        """检索器编码器前向传播 - 使用MLP对序列表示进行非线性变换"""
        hidden = seq_output  # [B, H]
        
        # 应用MLP层
        for idx, (layer, layer_norm) in enumerate(zip(self.retriever_mlp, self.retriever_layer_norms)):
            residual = hidden  # 每层的残差连接
            
            # MLP变换
            hidden = layer(hidden)
            hidden = self.retriever_act_fn(hidden)
            hidden = self.retriever_dropout_layer(hidden)
            
            # Layer Norm + 残差连接
            hidden = layer_norm(hidden + residual)
        
        return hidden

    def compute_retrieval_scores(self, seq_output, retrieved_seqs, retrieved_tars, pos_items):
        """计算检索评分 - 基于检索器编码后序列与目标嵌入的融合相似度"""
        batch_size, n_retrieved, hidden_size = retrieved_seqs.size()
        pos_items_emb = self.item_embedding(pos_items)  # [batch_size, hidden_size]
        
        # 使用检索器编码器处理原始序列
        retriever_encoded_seq = self.retriever_forward(seq_output)
        
        # 计算融合目标嵌入后的表示与目标项的相似度
        fusion_scores = []
        
        # 对每个检索到的目标嵌入进行简单融合并计算相似度
        for i in range(n_retrieved):
            # 获取当前检索结果的目标嵌入
            current_tar_emb = retrieved_tars[:, i, :]  # [batch_size, hidden_size]
            
            # 简单的加权融合：检索器编码后序列 + 检索目标嵌入
            fused_rep = self.alpha * retriever_encoded_seq + (1 - self.alpha) * current_tar_emb
            
            # 计算融合后表示与目标项的点积相似度
            similarity_score = torch.sum(fused_rep * pos_items_emb, dim=-1)  # [batch_size]
            fusion_scores.append(similarity_score)
        
        # 将所有检索结果的相似度堆叠
        stacked_scores = torch.stack(fusion_scores, dim=1)  # [batch_size, n_retrieved]
        
        # 应用温度缩放并转换为概率分布
        retrieval_logits = stacked_scores / self.retriever_temperature
        retrieval_probs = torch.softmax(retrieval_logits, dim=1)
        
        return retrieval_probs  # [batch_size, n_retrieved]

    def compute_recommendation_scores(self, seq_output, retrieved_seqs, retrieved_tars):
        """直接模拟DuoRec的full_sort_predict逻辑，但只针对k个检索项
        
        逻辑：
        1. 使用seq_output（已经是DuoRec.forward的输出）
        2. 不是与所有物品计算相似度，而是只与k个检索到的目标项计算
        3. 本质上是full_sort_predict的"局部版本"
        """
        batch_size, n_retrieved, hidden_size = retrieved_tars.size()
        
        with torch.no_grad():
            # 直接使用full_sort_predict的核心逻辑     
            # 重塑retrieved_tars进行批量矩阵乘法从[batch_size, n_retrieved, hidden_size]变为[batch_size, hidden_size, n_retrieved]
            retrieved_tars_t = retrieved_tars.transpose(1, 2)

            # 与full_sort_predict中的torch.matmul(seq_output, test_items_emb.transpose(0, 1))相同
            scores = torch.bmm(seq_output.unsqueeze(1), retrieved_tars_t).squeeze(1)  # [B, n_retrieved]
            
            # 使用softmax将分数转换为概率分布
            recommendation_scores = torch.softmax(scores, dim=-1)
            
        return recommendation_scores

    def compute_kl_loss(self, retrieval_probs, recommendation_probs):
        """计算检索分布与推荐分布之间的KL散度损失
           L = (1/|B|) * sum_x KL(P_R(d|x) || Q_M(d|x,y))
        """     
        # 确保两个分布的维度完全匹配
        assert retrieval_probs.shape == recommendation_probs.shape, \
            f"Shape mismatch: retrieval_probs {retrieval_probs.shape} vs recommendation_probs {recommendation_probs.shape}"
        
        # 避免数值问题
        epsilon = 1e-8
        retrieval_probs = retrieval_probs + epsilon
        recommendation_probs = recommendation_probs + epsilon
        
        # KL散度计算: KL(retrieval_probs || recommendation_probs)
        # 优化检索分布使其接近推荐分布
        kl_div = torch.sum(retrieval_probs * torch.log(retrieval_probs / recommendation_probs), dim=-1)
        
        # 返回批次平均损失
        return kl_div.mean()
