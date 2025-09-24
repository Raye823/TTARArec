# -*- coding: utf-8 -*-
# @Time    : 2025/1/1
# @Author  : Xinping Zhao
# @Email   : zhaoxinping@stu.hit.edu.cn

"""
TTARArec (Text-Time Adaptive Retrieval Augmented Recommender)
################################################

基于检索增强的序列推荐模型，支持动态加载不同类型的预训练模型作为特征提取器。
通过训练检索器编码器来对齐检索分布和推荐分布，实现更好的推荐性能。

"""

import torch
import heapq
import scipy
import faiss
import random
import math
from faiss import normalize_L2
from torch import nn
import numpy as np
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import activation_layer, CrossMultiHeadAttention, FeedForward
from recbole.model.sequential_recommender.pretrained_model_loader import PretrainedModelLoader
import torch.nn.functional as F
from recbole.utils.ttararec_diagnostics import compute_retrieval_effectiveness_vectorized, print_diagnostic_info_optimized


class TTARArec(SequentialRecommender):
    """
    基于检索增强的序列推荐模型
    
    主要特点：
    1. 动态加载不同类型的预训练模型作为特征提取器
    2. 训练专门的检索器编码器
    3. 通过KL散度对齐检索分布和推荐分布
    4. 支持序列增强和检索增强推荐
    """

    def __init__(self, config, dataset):
        super(TTARArec, self).__init__(config, dataset)

        # ========== 1. 加载预训练模型 ==========
        self.pretrained_model = PretrainedModelLoader.load_model(config, dataset)
        self.pretrained_model.requires_grad_(False)
        
        # ========== 2. 从预训练模型获取基础架构参数 ==========
        self.hidden_size = self.pretrained_model.hidden_size
        self.hidden_act = getattr(self.pretrained_model, 'hidden_act', config['hidden_act'] if 'hidden_act' in config else 'gelu')
        self.initializer_range = config['initializer_range']
        # 损失类型（CE 或 BPR）
        self.loss_type = config['loss_type'] if 'loss_type' in config else 'CE'
        self.training_neg_sample_num = config['training_neg_sample_num'] if 'training_neg_sample_num' in config else self.bpr_num_negatives
        # 定义损失函数
        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        else:
            self.loss_fct = nn.CrossEntropyLoss()

        # ========== 3. 设置检索相关参数 ==========
        # 检索配置参数
        self.topk = config['top_k'] if 'top_k' in config else 10
        self.nprobe = config['nprobe'] if 'nprobe' in config else 1
        
        # 序列长度过滤参数
        self.len_lower_bound = config["len_lower_bound"] if "len_lower_bound" in config else -1
        self.len_upper_bound = config["len_upper_bound"] if "len_upper_bound" in config else -1
        self.len_bound_reverse = config["len_bound_reverse"] if "len_bound_reverse" in config else True
        self.low_popular = config['low_popular'] if 'low_popular' in config else 100
        # 检索评分负信号配置
        self.retrieval_num_negatives = config['retrieval_num_negatives'] if 'retrieval_num_negatives' in config else 3

        # ========== 4. 设置训练相关参数 ==========
        # 损失权重
        
        # 新增损失函数权重和融合权重参数
        self.kl_loss_weight = config['kl_loss_weight'] if 'kl_loss_weight' in config else 0.6

        # ========== 5. 构建检索器和融合组件 ==========
        self._build_retrieval_components(config)

        # ========== 6. 初始化检索知识库相关变量 ==========
        self.dataset = dataset
        self.user_id_list = None
        self.item_seq_all = None
        self.item_seq_len_all = None
        self.seq_emb_knowledge = None
        self.item_seq_knowledge = None  # 原始交互序列知识库
        self.tar_emb_knowledge = None
        self.tar_item_knowledge = None  # 目标物品ID知识库
        self.seq_emb_index = None
        self.tar_emb_index = None
        
        # 训练状态控制
        self.use_retrieval = False  # 初始时不使用检索增强

    def _build_retrieval_components(self, config):
        """构建检索器MLP层和交叉注意力融合组件"""
        # 检索器编码器相关参数
        self.retriever_layers = config['retriever_layers'] if 'retriever_layers' in config else 0
        self.retriever_dropout = config['retriever_dropout'] if 'retriever_dropout' in config else 0
                
        # 激活函数和dropout
        self.retriever_act_fn = activation_layer(self.hidden_act)
        self.retriever_dropout_layer = nn.Dropout(self.retriever_dropout)

        # 构建检索器MLP层
        self.retriever_mlp = nn.ModuleList()
        self.retriever_layer_norms = nn.ModuleList()
        
        for i in range(self.retriever_layers):
            self.retriever_mlp.append(
                nn.Linear(self.hidden_size, self.hidden_size)
            )
            self.retriever_layer_norms.append(
                nn.LayerNorm(self.hidden_size)
            )

        # 交叉注意力融合机制参数（独立于预训练模型参数）
        self.fusion_n_heads = config['fusion_n_heads'] if 'fusion_n_heads' in config else 1
        self.fusion_inner_size = config['fusion_inner_size'] if 'fusion_inner_size' in config else 256
        self.fusion_dropout_prob = config['fusion_dropout_prob'] if 'fusion_dropout_prob' in config else 0
        self.fusion_layer_norm_eps = config['fusion_layer_norm_eps'] if 'fusion_layer_norm_eps' in config else 1e-12
        
        # 交叉注意力融合机制组件（使用PyTorch MultiheadAttention）
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.fusion_n_heads,
            dropout=self.fusion_dropout_prob,
            batch_first=True,
            kdim=self.hidden_size,
            vdim=self.hidden_size
        )
        
        self.fusion_ffn = FeedForward(self.hidden_size, self.fusion_inner_size, self.fusion_dropout_prob, self.hidden_act, self.fusion_layer_norm_eps)
        
        # 位置嵌入（用于检索序列）
        self.fusion_position_embedding = nn.Embedding(self.topk, self.hidden_size)
        
        # 构建完成后立即初始化权重
        self._init_component_weights()

    def _init_component_weights(self):
        """初始化检索器和融合组件的权重"""
        # 初始化检索器MLP和LayerNorm
        self.retriever_mlp.apply(self._init_weights)
        for layer_norm in self.retriever_layer_norms:
            self._init_weights(layer_norm)
        
        # 初始化交叉注意力融合机制
        # PyTorch MultiheadAttention 已内置初始化
        self.fusion_ffn.apply(self._init_weights)
        self.fusion_position_embedding.apply(self._init_weights)

    def _init_weights(self, module):
        """权重初始化回调函数"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            
    def get_item_embedding(self, item_ids):
        """获取物品嵌入 - 直接调用预训练模型"""
        with torch.no_grad():
            return self.pretrained_model.item_embedding(item_ids)
    def forward(self, item_seq, item_seq_len):
        """序列编码 - 直接调用预训练模型"""
        with torch.no_grad():
            return self.pretrained_model.forward(item_seq, item_seq_len)

    def retriever_forward(self, seq_output):
        """检索器编码器前向传播 - 使用MLP对序列表示进行非线性变换"""
        hidden = seq_output  # [B, H]
        
        # 应用MLP层
        for idx, (layer, layer_norm) in enumerate(zip(self.retriever_mlp, self.retriever_layer_norms)):
            residual = hidden  # 残差连接
            
            # MLP变换
            hidden = layer(hidden)
            hidden = self.retriever_act_fn(hidden)
            hidden = self.retriever_dropout_layer(hidden)
            
            # Layer Norm + 残差连接
            hidden = layer_norm(hidden + residual)
        
        return hidden
    
    def fusion_forward(self, seq_output, key_sequences, value_sequences=None):
        """交叉注意力融合机制前向传播（使用nn.MultiheadAttention）"""
        if value_sequences is None:
            value_sequences = key_sequences
        # 形状：[B, 1, H] 与 [B, K, H]
        query = seq_output.unsqueeze(1)
        key = key_sequences
        value = value_sequences
        # nn.MultiheadAttention expects (B, L, H) with batch_first=True
        fused_output, attn_weights = self.cross_attn(query, key, value, need_weights=True)
        fused_output = fused_output.squeeze(1)  # [B, H]
        fused_output = self.fusion_ffn(fused_output)
        return fused_output

    def retrieve_seq_tar(self, queries, batch_user_id, batch_seq_len, topk=5, mode="train"):
        """检索相似序列和对应的目标物品ID以及原始交互序列（基于q检索）"""
        queries_cpu = queries.detach().cpu().numpy()
        normalize_L2(queries_cpu)
        _, I1 = self.seq_emb_index.search(queries_cpu, 4 * topk)
        
        # 过滤掉同用户的相同长度序列
        I1_filtered = []
        for i, I_entry in enumerate(I1):
            current_user = batch_user_id[i]
            current_length = batch_seq_len[i]
            filtered_indices = [
                idx for idx in I_entry 
                if self.user_id_list[idx] != current_user or 
                (self.user_id_list[idx] == current_user and self.item_seq_len_all[idx] < current_length)
            ]
            I1_filtered.append(filtered_indices[:topk])
        
        I1_filtered = np.array(I1_filtered)
        
        # 获取检索结果 - 三项内容：序列表征、原始交互序列、目标物品ID
        retrieval_seqs = self.seq_emb_knowledge[I1_filtered]  # 序列表征
        retrieval_item_seqs = self.item_seq_knowledge[I1_filtered]  # 原始交互序列
        retrieval_tar_items = self.tar_item_knowledge[I1_filtered]  # 目标物品ID
        
        return (
            torch.tensor(retrieval_seqs).to("cuda"), 
            torch.tensor(retrieval_item_seqs).to("cuda"),  # 原始交互序列
            torch.tensor(retrieval_tar_items).to("cuda"),  # 目标物品ID
        )

    def compute_entropy(self, logits):
        """计算logits的熵"""
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        return entropy
    
    def compute_alpha_weights(self, logits1, logits2):
        """基于两个logits的熵计算融合权重α"""
        h1 = self.compute_entropy(logits1)  # [B] 或 [B,K]
        h2 = self.compute_entropy(logits2)  # [B] 或 [B,K]
        
        exp_h1 = torch.exp(1.0 / (1.0 + h1))
        exp_h2 = torch.exp(1.0 / (1.0 + h2))
        alpha = exp_h1 / (exp_h1 + exp_h2)
        
        return alpha   
    # ============ 损失计算相关方法 ============
    
    def compute_retrieval_scores(self, retrieved_item_seqs, retrieved_tar_items, pos_items, item_seq, item_seq_len, batch_seq_len, enhanced_sequences=None):
        batch_size = pos_items.size(0)
        n_retrieved = enhanced_sequences.size(1) 
        pos_items_emb = self.get_item_embedding(pos_items)  # [B, H]
        all_items_emb = self.pretrained_model.item_embedding.weight  # [N, H]

        if enhanced_sequences is not None:
            if self.loss_type == 'CE':
                # CE损失对齐：计算每个增强序列与所有物品的logits，然后提取正样本位置的概率
                # enhanced_sequences: [B, K, H], all_items_emb: [N, H]
                logits_k = torch.matmul(enhanced_sequences, all_items_emb.transpose(0, 1))  # [B, K, N]
                
                # 计算每个增强序列的CE损失（负log似然）
                ce_losses_k = F.cross_entropy(
                    logits_k.reshape(-1, logits_k.size(-1)),  # [B*K, N]
                    pos_items.unsqueeze(1).expand(-1, n_retrieved).reshape(-1),  # [B*K]
                    reduction='none'
                ).reshape(batch_size, n_retrieved)  # [B, K]
                
                logits = -ce_losses_k  # 损失越小越好，转为正分数
                tau = float(getattr(self, 'retrieval_tau', 1.0)) if hasattr(self, 'retrieval_tau') else 1
        return torch.softmax(logits / max(tau, 1e-6), dim=1).detach()

    def compute_attention_scores(self, seq_output, key_sequences, value_sequences=None, enhanced_sequences=None):
        """计算注意力评分 - 使用nn.MultiheadAttention提取注意力权重"""
        if enhanced_sequences is not None:
            key_sequences = enhanced_sequences
            value_sequences = enhanced_sequences
        query = seq_output.unsqueeze(1)  # [B, 1, H]
        # 通过cross_attn拿到注意力权重
        _, attn_weights = self.cross_attn(query, key_sequences, value_sequences, need_weights=True, average_attn_weights=False)
        # attn_weights: [B, num_heads, Lq=1, K]
        attn_weights = attn_weights.squeeze(2).mean(dim=1)  # [B, K]
        return attn_weights

    def compute_kl_loss(self, attention_probs, retrieval_probs):
        """计算注意力分布与检索分布之间的KL散度损失 - 使用PyTorch库函数"""
        # 确保两个分布的维度完全匹配
        assert attention_probs.shape == retrieval_probs.shape, \
            f"形状不匹配: attention_probs {attention_probs.shape} vs retrieval_probs {retrieval_probs.shape}"
        
        # 使用PyTorch的kl_div函数，更数值稳定
        # kl_div接受log概率作为第一个参数，目标分布作为第二个参数
        # KL(attention_probs || retrieval_probs)
        kl_div = F.kl_div(torch.log(attention_probs + 1e-8), retrieval_probs, reduction='batchmean')
        
        return kl_div

    def calculate_loss(self, interaction):
        """计算训练损失 - KL散度损失 + 推荐损失"""
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        # 减少CPU-GPU传输：一次性转换所有数据，避免list()转换
        batch_user_id = interaction[self.USER_ID].detach().cpu().numpy()
        batch_seq_len = item_seq_len.detach().cpu().numpy()        

        # 检索相似序列和目标物品ID
        retrieved_seqs, retrieved_item_seqs, retrieved_tar_items = self.retrieve_seq_tar(
            seq_output,
            batch_user_id, 
            batch_seq_len,
            topk=self.topk
        )

        # 直接使用检索到的目标 item 向量进行融合（最小改动，保留原增强序列用于检索对齐与诊断）
        target_embs = self.get_item_embedding(retrieved_tar_items)  # [B, K, H]
        enhanced_sequences=target_embs
        # 使用检索到的目标 item 向量进行序列增强
        seq_output_aug = self.seq_augmented(
            seq_output, batch_user_id, batch_seq_len, 
            enhanced_sequences=enhanced_sequences
        )
            
        # 计算推荐损失（支持CE/BPR）
        if self.loss_type == 'CE':
            # 默认回退CE
            test_item_emb = self.pretrained_model.item_embedding.weight
            logits = torch.matmul(seq_output_aug, test_item_emb.transpose(0, 1))
            rec_loss = self.pretrained_model.loss_fct(logits, pos_items)

        # 计算检索评分：使用预计算的增强序列表征

        retrieval_probs = self.compute_retrieval_scores(
            retrieved_item_seqs, retrieved_tar_items, pos_items, item_seq, item_seq_len, batch_seq_len,
            enhanced_sequences=enhanced_sequences
        )  # [B, K]
        
        # 计算注意力评分：使用目标 item 嵌入
        attention_probs = self.compute_attention_scores(
            seq_output, None, None, enhanced_sequences=enhanced_sequences
        ) 
        
        # 计算KL散度损失（注意力评分向检索评分对齐）
        kl_loss = self.compute_kl_loss(attention_probs, retrieval_probs)

        # ============ 诊断信息输出 ============
    
        # 每33个batch输出一次诊断信息
        if hasattr(self, 'batch_count'):
            self.batch_count += 1
        else:
            self.batch_count = 0
            
        if self.batch_count % 129 == 0:
            # 计算检索效果指标
            retrieval_effectiveness, augment_retrieval_consistency, fusion_retrieval_consistency, top_retrieval_similarity, augment_fusion_consistency = compute_retrieval_effectiveness_vectorized(self,
                retrieved_item_seqs, pos_items, item_seq, item_seq_len, batch_seq_len, retrieved_seqs, retrieved_tar_items, enhanced_sequences
            )
            
            print_diagnostic_info_optimized(self,
                rec_loss, kl_loss, retrieval_probs, attention_probs, 
                seq_output, seq_output_aug, pos_items, retrieval_effectiveness, 
                augment_retrieval_consistency, fusion_retrieval_consistency, top_retrieval_similarity, augment_fusion_consistency,
            )
        
        # 总损失 = KL散度损失 * 权重 + 推荐损失 * 权重
        total_loss = kl_loss * self.kl_loss_weight + rec_loss 
        
        return total_loss

    # ============ 知识库构建相关方法 ============
    
    def precached_knowledge(self):
        """预缓存知识库 - 构建检索索引"""
        print("开始构建检索知识库...")
        seq_emb_knowledge, item_seq_knowledge, tar_emb_knowledge, tar_item_knowledge, user_id_list = None, None, None, None, None
        item_seq_all = None
        item_seq_len_all = None
        
        for batch_idx, interaction in enumerate(self.dataset):
            interaction = interaction.to("cuda")
            
            # 根据序列长度过滤
            if self.len_lower_bound != -1 or self.len_upper_bound != -1:
                if self.len_lower_bound != -1 and self.len_upper_bound != -1:
                    look_up_indices = (interaction[self.ITEM_SEQ_LEN] >= self.len_lower_bound) * \
                                    (interaction[self.ITEM_SEQ_LEN] <= self.len_upper_bound)
                elif self.len_upper_bound != -1:
                    look_up_indices = interaction[self.ITEM_SEQ_LEN] < self.len_upper_bound
                else:
                    look_up_indices = interaction[self.ITEM_SEQ_LEN] > self.len_lower_bound
                    
                if self.len_bound_reverse:
                    look_up_indices = ~look_up_indices
            else:
                look_up_indices = interaction[self.ITEM_SEQ_LEN] > -1
            
            item_seq = interaction[self.ITEM_SEQ][look_up_indices]
            if item_seq_all is None:
                item_seq_all = item_seq
            else:
                item_seq_all = torch.cat((item_seq_all, item_seq), dim=0)
                
            item_seq_len = interaction[self.ITEM_SEQ_LEN][look_up_indices]
            item_seq_len_list = list(item_seq_len.detach().cpu().numpy())
            if isinstance(item_seq_len_all, list):
                item_seq_len_all.extend(item_seq_len_list)
            else:
                item_seq_len_all = item_seq_len_list
                
            # 获取序列表示
            seq_output = self.forward(item_seq, item_seq_len)
            tar_items = interaction[self.POS_ITEM_ID][look_up_indices]
            tar_items_emb = self.get_item_embedding(tar_items)
            user_id_cans = list(interaction[self.USER_ID][look_up_indices].detach().cpu().numpy())
            
            # 累积知识 - 四项内容：序列表征、原始交互序列、目标嵌入、目标物品ID
            if isinstance(seq_emb_knowledge, np.ndarray):
                seq_emb_knowledge = np.concatenate((seq_emb_knowledge, seq_output.detach().cpu().numpy()), 0)
            else:
                seq_emb_knowledge = seq_output.detach().cpu().numpy()
            
            # 累积原始交互序列
            if isinstance(item_seq_knowledge, np.ndarray):
                item_seq_knowledge = np.concatenate((item_seq_knowledge, item_seq.detach().cpu().numpy()), 0)
            else:
                item_seq_knowledge = item_seq.detach().cpu().numpy()
            
            # 累积目标嵌入
            if isinstance(tar_emb_knowledge, np.ndarray):
                tar_emb_knowledge = np.concatenate((tar_emb_knowledge, tar_items_emb.detach().cpu().numpy()), 0)
            else:
                tar_emb_knowledge = tar_items_emb.detach().cpu().numpy()
            
            # 累积目标物品ID
            if isinstance(tar_item_knowledge, np.ndarray):
                tar_item_knowledge = np.concatenate((tar_item_knowledge, tar_items.detach().cpu().numpy()), 0)
            else:
                tar_item_knowledge = tar_items.detach().cpu().numpy()
            
            if isinstance(user_id_list, list):
                user_id_list.extend(user_id_cans)
            else:
                user_id_list = user_id_cans
        
        # 保存知识库 - 四项内容
        self.user_id_list = user_id_list
        self.item_seq_all = item_seq_all
        self.item_seq_len_all = item_seq_len_all
        self.seq_emb_knowledge = seq_emb_knowledge  # 序列表征
        self.item_seq_knowledge = item_seq_knowledge  # 原始交互序列
        self.tar_emb_knowledge = tar_emb_knowledge  # 目标嵌入
        self.tar_item_knowledge = tar_item_knowledge  # 目标物品ID
        
        # 构建Faiss索引
        self._build_faiss_index()
        
        # 标记知识库已构建
        self.knowledge_built = True
        print(f"知识库构建完成，包含 {len(user_id_list)} 个序列样本")
        print(f"知识库四项内容：序列表征维度 {self.seq_emb_knowledge.shape}，原始序列维度 {self.item_seq_knowledge.shape}，目标嵌入维度 {self.tar_emb_knowledge.shape}，目标物品ID维度 {self.tar_item_knowledge.shape}")

    def precached_knowledge_val(self, val_dataset):
        """为验证集构建知识库"""
        print("为验证集构建检索知识库...")
        seq_emb_knowledge, item_seq_knowledge, tar_emb_knowledge, tar_item_knowledge, user_id_list = None, None, None, None, None
        item_seq_len_all = None
        
        # 处理训练集
        for batch_idx, interaction in enumerate(self.dataset):
            interaction = interaction.to("cuda")
            
            # 序列长度过滤逻辑
            if self.len_lower_bound != -1 or self.len_upper_bound != -1:
                if self.len_lower_bound != -1 and self.len_upper_bound != -1:
                    look_up_indices = (interaction[self.ITEM_SEQ_LEN] >= self.len_lower_bound) * \
                                    (interaction[self.ITEM_SEQ_LEN] <= self.len_upper_bound)
                elif self.len_upper_bound != -1:
                    look_up_indices = interaction[self.ITEM_SEQ_LEN] < self.len_upper_bound
                else:
                    look_up_indices = interaction[self.ITEM_SEQ_LEN] > self.len_lower_bound
                    
                if self.len_bound_reverse:
                    look_up_indices = ~look_up_indices
            else:
                look_up_indices = interaction[self.ITEM_SEQ_LEN] > -1
            
            item_seq = interaction[self.ITEM_SEQ][look_up_indices]
            item_seq_len = interaction[self.ITEM_SEQ_LEN][look_up_indices]
            item_seq_len_list = list(item_seq_len.detach().cpu().numpy())
            if isinstance(item_seq_len_all, list):
                item_seq_len_all.extend(item_seq_len_list)
            else:
                item_seq_len_all = item_seq_len_list
                
            seq_output = self.forward(item_seq, item_seq_len)
            tar_items = interaction[self.POS_ITEM_ID][look_up_indices]
            tar_items_emb = self.get_item_embedding(tar_items)
            user_id_cans = list(interaction[self.USER_ID][look_up_indices].detach().cpu().numpy())
            
            # 累积知识 - 三项内容：序列表征、原始交互序列、目标嵌入
            if isinstance(seq_emb_knowledge, np.ndarray):
                seq_emb_knowledge = np.concatenate((seq_emb_knowledge, seq_output.detach().cpu().numpy()), 0)
            else:
                seq_emb_knowledge = seq_output.detach().cpu().numpy()
                
            # 新增：累积原始交互序列
            if isinstance(item_seq_knowledge, np.ndarray):
                item_seq_knowledge = np.concatenate((item_seq_knowledge, item_seq.detach().cpu().numpy()), 0)
            else:
                item_seq_knowledge = item_seq.detach().cpu().numpy()
                
            if isinstance(tar_emb_knowledge, np.ndarray):
                tar_emb_knowledge = np.concatenate((tar_emb_knowledge, tar_items_emb.detach().cpu().numpy()), 0)
            else:
                tar_emb_knowledge = tar_items_emb.detach().cpu().numpy()
            
            # 累积目标物品ID
            if isinstance(tar_item_knowledge, np.ndarray):
                tar_item_knowledge = np.concatenate((tar_item_knowledge, tar_items.detach().cpu().numpy()), 0)
            else:
                tar_item_knowledge = tar_items.detach().cpu().numpy()
                
            if isinstance(user_id_list, list):
                user_id_list.extend(user_id_cans)
            else:
                user_id_list = user_id_cans
        
        # 处理验证集
        for batch_idx, batched_data in enumerate(val_dataset):
            interaction, history_index, swap_row, swap_col_after, swap_col_before = batched_data
            interaction = interaction.to("cuda")
            item_seq = interaction[self.ITEM_SEQ]
            item_seq_len = interaction[self.ITEM_SEQ_LEN]
            item_seq_len_list = list(item_seq_len.detach().cpu().numpy())
            if isinstance(item_seq_len_all, list):
                item_seq_len_all.extend(item_seq_len_list)
            else:
                item_seq_len_all = item_seq_len_list

            seq_output = self.forward(item_seq, item_seq_len)
            tar_items = interaction[self.POS_ITEM_ID]
            tar_items_emb = self.get_item_embedding(tar_items)
            user_id_cans = list(interaction[self.USER_ID].detach().cpu().numpy())
            
            # 累积知识 - 三项内容
            if isinstance(seq_emb_knowledge, np.ndarray):
                seq_emb_knowledge = np.concatenate((seq_emb_knowledge, seq_output.detach().cpu().numpy()), 0)
            else:
                seq_emb_knowledge = seq_output.detach().cpu().numpy()
                
            # 新增：累积原始交互序列
            if isinstance(item_seq_knowledge, np.ndarray):
                item_seq_knowledge = np.concatenate((item_seq_knowledge, item_seq.detach().cpu().numpy()), 0)
            else:
                item_seq_knowledge = item_seq.detach().cpu().numpy()
                
            if isinstance(tar_emb_knowledge, np.ndarray):
                tar_emb_knowledge = np.concatenate((tar_emb_knowledge, tar_items_emb.detach().cpu().numpy()), 0)
            else:
                tar_emb_knowledge = tar_items_emb.detach().cpu().numpy()
            
            # 累积目标物品ID
            if isinstance(tar_item_knowledge, np.ndarray):
                tar_item_knowledge = np.concatenate((tar_item_knowledge, tar_items.detach().cpu().numpy()), 0)
            else:
                tar_item_knowledge = tar_items.detach().cpu().numpy()
                
            if isinstance(user_id_list, list):
                user_id_list.extend(user_id_cans)
            else:
                user_id_list = user_id_cans
                
        # 保存知识库 - 四项内容
        self.user_id_list = user_id_list
        self.item_seq_len_all = item_seq_len_all
        self.seq_emb_knowledge = seq_emb_knowledge  # 序列表征
        self.item_seq_knowledge = item_seq_knowledge  # 原始交互序列  
        self.tar_emb_knowledge = tar_emb_knowledge  # 目标嵌入
        self.tar_item_knowledge = tar_item_knowledge  # 目标物品ID
        
        # 构建Faiss索引
        self._build_faiss_index()
        
        # 标记知识库已构建
        self.knowledge_built = True
        print(f"验证集知识库构建完成，包含 {len(user_id_list)} 个序列样本")
        print(f"知识库四项内容：序列表征维度 {self.seq_emb_knowledge.shape}，原始序列维度 {self.item_seq_knowledge.shape}，目标嵌入维度 {self.tar_emb_knowledge.shape}，目标物品ID维度 {self.tar_item_knowledge.shape}")

    def _build_faiss_index(self):
        """构建Faiss检索索引"""
        d = self.hidden_size
        nlist = 128
        
        # 构建序列嵌入索引
        seq_emb_knowledge_copy = np.array(self.seq_emb_knowledge, copy=True)
        normalize_L2(seq_emb_knowledge_copy)
        seq_emb_quantizer = faiss.IndexFlatL2(d) 
        self.seq_emb_index = faiss.IndexIVFFlat(seq_emb_quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT) 
        self.seq_emb_index.train(seq_emb_knowledge_copy)
        self.seq_emb_index.add(seq_emb_knowledge_copy)    
        self.seq_emb_index.nprobe = self.nprobe

        # 构建目标嵌入索引
        tar_emb_knowledge_copy = np.array(self.tar_emb_knowledge, copy=True)
        normalize_L2(tar_emb_knowledge_copy)
        tar_emb_quantizer = faiss.IndexFlatL2(d) 
        self.tar_emb_index = faiss.IndexIVFFlat(tar_emb_quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT) 
        self.tar_emb_index.train(tar_emb_knowledge_copy)
        self.tar_emb_index.add(tar_emb_knowledge_copy) 
        self.tar_emb_index.nprobe = self.nprobe

    # ============ 预测相关方法 ============
    def enable_retrieval(self):
        """启用检索增强功能"""
        self.use_retrieval = True

    def seq_augmented(self, seq_output, batch_user_id, batch_seq_len, mode="train", enhanced_sequences=None):
        """序列增强 - 使用训练后的交叉注意力层进行索引融合"""
        if enhanced_sequences is not None:
            retrieval_enhanced_output = self.fusion_forward(seq_output, enhanced_sequences)
        else:
            retrieved_seqs, retrieved_item_seqs, retrieved_tar_items = self.retrieve_seq_tar(
                seq_output, batch_user_id, batch_seq_len, topk=self.topk, mode=mode
            )
            # 直接使用检索到的目标 item 嵌入进行融合（最小改动）
            target_embs = self.get_item_embedding(retrieved_tar_items)  # [B, K, H]
            retrieval_enhanced_output = self.fusion_forward(seq_output, target_embs)
        
        all_items_emb = self.pretrained_model.item_embedding.weight  # [N, H]
        logits1 = torch.matmul(seq_output, all_items_emb.transpose(0, 1))  # [B, N]       
        logits2 = torch.matmul(retrieval_enhanced_output, all_items_emb.transpose(0, 1))  # [B, N]
        alpha = self.compute_alpha_weights(logits1, logits2)  # [B]
        # 将alpha从[B]扩展到[B, H]以匹配seq_output和retrieval_enhanced_output的维度
        alpha_expanded = alpha.unsqueeze(-1)  # [B, 1] -> [B, H]
        augmented_output = seq_output * alpha_expanded + retrieval_enhanced_output * (1 - alpha_expanded)
        
        return augmented_output

    def predict(self, interaction):
        """预测单个物品的得分"""
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.get_item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        """全排序预测 - 根据训练状态决定是否使用检索增强"""
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        
        # 根据训练状态决定是否使用检索增强
        if self.use_retrieval:
            batch_user_id = interaction[self.USER_ID].detach().cpu().numpy()
            batch_seq_len = item_seq_len.detach().cpu().numpy()
            seq_output = self.seq_augmented(
                seq_output, batch_user_id, batch_seq_len, mode="test"
            )
        
        # 区分不同预训练类型的评估路径
        pm_type = None
        if hasattr(self, 'config') and 'pretrained_model_type' in self.config:
            pm_type = str(self.config['pretrained_model_type']).lower()

        if pm_type == 'bert4rec':
            # BERT4Rec：去除mask列（仅[:n_items]），并加上输出偏置
            test_items_emb = self.pretrained_model.item_embedding.weight[: self.n_items]
            scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, n_items]
            output_bias = getattr(self.pretrained_model, 'output_bias', None)
            if output_bias is not None:
                scores = scores + output_bias[: self.n_items]
        else:
            # 其他（SASRec/DuoRec等）：使用完整embedding（包含padding对齐为0列，保持与原模型一致）
            test_items_emb = self.pretrained_model.item_embedding.weight
            scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, n_items]
        return scores