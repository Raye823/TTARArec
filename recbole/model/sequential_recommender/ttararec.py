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
        self.pretrained_model = PretrainedModelLoader.load_duorec_model(config, dataset)
        
        # ========== 2. 从预训练模型获取基础架构参数 ==========
        self.hidden_size = self.pretrained_model.hidden_size
        self.hidden_act = getattr(self.pretrained_model, 'hidden_act', config['hidden_act'] if 'hidden_act' in config else 'gelu')
        self.initializer_range = config['initializer_range']

        # ========== 3. 设置检索相关参数 ==========
        # 检索配置参数
        self.topk = config['top_k'] if 'top_k' in config else 10
        self.nprobe = config['nprobe'] if 'nprobe' in config else 1
        
        # 序列长度过滤参数
        self.len_lower_bound = config["len_lower_bound"] if "len_lower_bound" in config else -1
        self.len_upper_bound = config["len_upper_bound"] if "len_upper_bound" in config else -1
        self.len_bound_reverse = config["len_bound_reverse"] if "len_bound_reverse" in config else True
        self.low_popular = config['low_popular'] if 'low_popular' in config else 100

        # ========== 4. 设置训练相关参数 ==========
        # 温度系数和损失权重
        self.retriever_temperature = config['retriever_temperature'] if 'retriever_temperature' in config else 0.1
        self.recommendation_temperature = config['recommendation_temperature'] if 'recommendation_temperature' in config else 0.1
        
        # 新增损失函数权重和融合权重参数
        self.kl_loss_weight = config['kl_loss_weight'] if 'kl_loss_weight' in config else 0.6
        self.fusion_weight = config['fusion_weight'] if 'fusion_weight' in config else 0.5

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
        self.seq_emb_index = None
        self.tar_emb_index = None
        
        # 训练状态控制
        self.use_retrieval = False  # 初始时不使用检索增强

    def _build_retrieval_components(self, config):
        """构建检索器MLP层和交叉注意力融合组件"""
        # 检索器编码器相关参数
        self.retriever_layers = config['retriever_layers'] if 'retriever_layers' in config else 1
        self.retriever_dropout = config['retriever_dropout'] if 'retriever_dropout' in config else 0.1
                
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
        self.fusion_n_heads = config['fusion_n_heads'] if 'fusion_n_heads' in config else 2
        self.fusion_inner_size = config['fusion_inner_size'] if 'fusion_inner_size' in config else 256
        self.fusion_dropout_prob = config['fusion_dropout_prob'] if 'fusion_dropout_prob' in config else 0.1
        self.fusion_layer_norm_eps = config['fusion_layer_norm_eps'] if 'fusion_layer_norm_eps' in config else 1e-12
        self.attn_tau = config['attn_tau'] if 'attn_tau' in config else 0.2  # 注意力温度系数
        
        # 交叉注意力融合机制组件
        self.seq_tar_fusion = CrossMultiHeadAttention(
            n_heads=self.fusion_n_heads,
            hidden_size=self.hidden_size,
            hidden_dropout_prob=self.fusion_dropout_prob,
            attn_dropout_prob=self.fusion_dropout_prob,
            layer_norm_eps=self.fusion_layer_norm_eps,
            attn_tau=self.attn_tau
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
        self.seq_tar_fusion.apply(self._init_weights)
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
    
    def fusion_forward(self, seq_output, retrieved_seqs, retrieved_tars):
        """交叉注意力融合机制前向传播"""
        # 交叉注意力融合：query是当前序列，key是检索序列表征，value是目标嵌入
        seq_output_expanded = seq_output.unsqueeze(1)  # [B, 1, H]
        
        # 使用交叉注意力层：query=当前序列，key=检索序列表征，value=目标嵌入
        fused_output = self.seq_tar_fusion(
            seq_output_expanded,        # input_query: [B, 1, H]
            retrieved_seqs,            # input_key: [B, K, H] 
            retrieved_tars             # input_value: [B, K, H]
        )  # fused_output: [B, 1, H]
        
        fused_output = fused_output.squeeze(1)  # [B, H]
        
        # 前馈神经网络进一步处理
        fused_output = self.fusion_ffn(fused_output)  # [B, H]
        
        return fused_output

    def retrieve_seq_tar(self, queries, batch_user_id, batch_seq_len, topk=5, mode="train"):
        """检索相似序列和对应的目标嵌入以及原始交互序列"""
        queries_cpu = queries.detach().cpu().numpy()
        normalize_L2(queries_cpu)
        _, I1 = self.seq_emb_index.search(queries_cpu, 4*topk)
        
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
        
        # 获取检索结果 - 三项内容：序列表征、原始交互序列、目标嵌入
        retrieval_seqs = self.seq_emb_knowledge[I1_filtered]  # 序列表征
        retrieval_item_seqs = self.item_seq_knowledge[I1_filtered]  # 原始交互序列
        retrieval_tars = self.tar_emb_knowledge[I1_filtered]  # 目标嵌入
        
        return (
            torch.tensor(retrieval_seqs).to("cuda"), 
            torch.tensor(retrieval_item_seqs).to("cuda"),  # 增加返回原始交互序列
            torch.tensor(retrieval_tars ).to("cuda"), 

        )   

    # ============ 损失计算相关方法 ============

    def compute_retrieval_scores(self, retrieved_item_seqs, retrieved_tars, pos_items, item_seq, item_seq_len, batch_seq_len):
        """计算检索评分 - 基于原始序列拼接和重新编码的相似度"""
        batch_size, n_retrieved, _ = retrieved_tars.size()
        pos_items_emb = self.get_item_embedding(pos_items)  # [batch_size, hidden_size]
        
        fusion_scores = []
        
        # 对每个检索到的原始序列进行处理
        for i in range(n_retrieved):
            batch_new_seqs = []
            batch_new_seq_lens = []
            
            # 批量处理所有样本 - 大幅优化内层循环
            
            # 预分配结果张量
            max_seq_len = item_seq.size(1)
            batch_new_seqs = torch.zeros(batch_size, max_seq_len, dtype=item_seq.dtype, device=item_seq.device)
            batch_new_seq_lens = torch.zeros(batch_size, dtype=torch.long, device=item_seq.device)
            
            # 批量计算检索序列的有效长度
            retrieved_seqs_i = retrieved_item_seqs[:, i, :]  # [batch_size, max_seq_len]
            
            # 使用向量化操作找到有效长度
            nonzero_masks = (retrieved_seqs_i != 0)  # [batch_size, max_seq_len]
            # 找到每行最后一个非零位置
            seq_lens = torch.sum(nonzero_masks, dim=1)  # [batch_size] 直接计算非零元素数量
            
            # 完全向量化处理 - 真正消除batch_size循环！
            current_seq_lens = torch.from_numpy(batch_seq_len).to(item_seq.device)
            
            # 方法：使用gather和scatter操作进行向量化拼接
            # 1. 计算新序列长度
            total_lens = current_seq_lens + seq_lens
            new_seq_lens = torch.clamp(total_lens, max=max_seq_len)
            
            # 2. 创建一个大的拼接张量 [batch_size, 2*max_seq_len]
            # 前半部分放原序列，后半部分放检索序列
            concat_seqs = torch.zeros(batch_size, 2 * max_seq_len, dtype=item_seq.dtype, device=item_seq.device)
            concat_seqs[:, :max_seq_len] = item_seq
            concat_seqs[:, max_seq_len:] = retrieved_seqs_i
            
            # 3. 为每个样本创建索引，从concat_seqs中选择正确的元素
            batch_indices = torch.arange(batch_size, device=item_seq.device).unsqueeze(1)
            
            # 4. 创建序列索引：对于每个位置，决定从原序列还是检索序列取值
            position_indices = torch.arange(max_seq_len, device=item_seq.device).unsqueeze(0).expand(batch_size, -1)
            
            # 原序列部分的掩码
            current_mask = position_indices < current_seq_lens.unsqueeze(1)
            # 检索序列部分的掩码（从原序列长度开始）
            retrieved_start_pos = current_seq_lens.unsqueeze(1)
            retrieved_mask = (position_indices >= retrieved_start_pos) & (position_indices < new_seq_lens.unsqueeze(1))
            
            # 5. 构建最终序列
            batch_new_seqs = torch.zeros_like(item_seq)
            
            # 复制原序列
            batch_new_seqs[current_mask] = item_seq[current_mask]
            
            # 添加检索序列（需要计算正确的检索序列位置）
            if retrieved_mask.any():
                # 计算检索序列中的相对位置
                retrieved_relative_pos = position_indices - retrieved_start_pos
                retrieved_relative_pos = torch.clamp(retrieved_relative_pos, min=0, max=max_seq_len-1)
                
                # 从检索序列中获取正确的值
                retrieved_values = retrieved_seqs_i[batch_indices.expand(-1, max_seq_len), retrieved_relative_pos]
                batch_new_seqs[retrieved_mask] = retrieved_values[retrieved_mask]
            
            batch_new_seq_lens = new_seq_lens
            
            # 使用DuoRec编码器重新编码新序列
            new_seq_output = self.forward(batch_new_seqs, batch_new_seq_lens)  # [batch_size, hidden_size]
            
            # 从DuoRec输出中分离梯度，然后重新启用梯度用于检索器MLP训练
            new_seq_output = new_seq_output.detach().requires_grad_(True)
            
            # 使用检索器MLP进一步处理序列表征
            enhanced_seq_output = self.retriever_forward(new_seq_output)  # [batch_size, hidden_size]
            
            # 计算增强后序列表征与真实下一项的相似度
            similarity_score = torch.sum(enhanced_seq_output * pos_items_emb, dim=-1)  # [batch_size]
            fusion_scores.append(similarity_score)
        
        # 将所有检索结果的相似度堆叠
        stacked_scores = torch.stack(fusion_scores, dim=1)  # [batch_size, n_retrieved]
        
        # 应用温度缩放并转换为概率分布
        retrieval_logits = stacked_scores / self.retriever_temperature
        retrieval_probs = torch.softmax(retrieval_logits, dim=1)
        
        return retrieval_probs  # [batch_size, n_retrieved]

    def compute_attention_scores(self, seq_output, retrieved_seqs, retrieved_tars):
        """计算注意力评分 - 直接计算注意力权重"""
        # 扩展当前序列输出用于注意力计算
        seq_output_expanded = seq_output.unsqueeze(1)  # [B, 1, H]
        
        # 手动计算交叉注意力权重（不使用value，只计算query与key的注意力）
        mixed_query_layer = self.seq_tar_fusion.query(seq_output_expanded)  # [B, 1, H]
        mixed_key_layer = self.seq_tar_fusion.key(retrieved_seqs)  # [B, K, H]
        
        # 转换为多头形状
        query_layer = self.seq_tar_fusion.transpose_for_scores(mixed_query_layer)  # [B, n_heads, 1, head_size]
        key_layer = self.seq_tar_fusion.transpose_for_scores(mixed_key_layer)  # [B, n_heads, K, head_size]
        
        # 计算注意力得分
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # [B, n_heads, 1, K]
        attention_scores = attention_scores / math.sqrt(self.seq_tar_fusion.attention_head_size)
        
        # 平均所有注意力头的得分
        attention_scores = attention_scores.mean(dim=1)  # [B, 1, K]
        attention_scores = attention_scores.squeeze(1)  # [B, K]
        
        # 使用温度缩放
        attention_probs = torch.softmax(attention_scores / self.recommendation_temperature, dim=-1)
        
        return attention_probs

    def compute_kl_loss(self, attention_probs, retrieval_probs):
        """计算注意力分布与检索分布之间的KL散度损失"""
        # 确保两个分布的维度完全匹配
        assert attention_probs.shape == retrieval_probs.shape, \
            f"形状不匹配: attention_probs {attention_probs.shape} vs retrieval_probs {retrieval_probs.shape}"
        
        # 避免数值问题
        epsilon = 1e-8
        attention_probs = attention_probs + epsilon
        retrieval_probs = retrieval_probs + epsilon
        
        # KL散度计算: KL(attention_probs || retrieval_probs)
        # 优化注意力分布使其接近检索分布
        kl_div = torch.sum(attention_probs * torch.log(attention_probs / retrieval_probs), dim=-1)
        
        # 返回批次平均损失
        return kl_div.mean()

    def calculate_loss(self, interaction):
        """计算训练损失 - KL散度损失 + 推荐损失"""
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        # 减少CPU-GPU传输：一次性转换所有数据，避免list()转换
        batch_user_id = interaction[self.USER_ID].detach().cpu().numpy()
        batch_seq_len = item_seq_len.detach().cpu().numpy()        

        seq_output_aug = self.seq_augmented(seq_output, batch_user_id, batch_seq_len)
            
        # 计算推荐损失
        test_item_emb = self.pretrained_model.item_embedding.weight.requires_grad_(True)
        logits = torch.matmul(seq_output_aug, test_item_emb.transpose(0, 1))
        rec_loss = self.pretrained_model.loss_fct(logits, pos_items)

        
        # 检索相似序列和目标嵌入
        retrieved_seqs, retrieved_item_seqs, retrieved_tars = self.retrieve_seq_tar(
            seq_output,
            batch_user_id, 
            batch_seq_len,
            topk=self.topk
        )

        # 计算检索评分：基于原始序列拼接和重新编码的相似度
        retrieval_probs = self.compute_retrieval_scores(
            retrieved_item_seqs, retrieved_tars, pos_items, item_seq, item_seq_len, batch_seq_len
        )  # [B, K]
        
        # 计算注意力评分：基于交叉注意力权重
        attention_probs = self.compute_attention_scores(
            seq_output, retrieved_seqs, retrieved_tars
        )  # [B, K]
        
        # 计算KL散度损失（注意力评分向检索评分对齐）
        kl_loss = self.compute_kl_loss(attention_probs, retrieval_probs)
        
        # 总损失 = KL散度损失 * 权重 + 推荐损失 * 权重
        total_loss = kl_loss * self.kl_loss_weight + rec_loss * (1-self.kl_loss_weight)
        
        return total_loss

    # ============ 知识库构建相关方法 ============
    
    def precached_knowledge(self):
        """预缓存知识库 - 构建检索索引"""
        print("开始构建检索知识库...")
        seq_emb_knowledge, item_seq_knowledge, tar_emb_knowledge, user_id_list = None, None, None, None
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
            
            if isinstance(user_id_list, list):
                user_id_list.extend(user_id_cans)
            else:
                user_id_list = user_id_cans
        
        # 保存知识库 - 三项内容
        self.user_id_list = user_id_list
        self.item_seq_all = item_seq_all
        self.item_seq_len_all = item_seq_len_all
        self.seq_emb_knowledge = seq_emb_knowledge  # 序列表征
        self.item_seq_knowledge = item_seq_knowledge  # 原始交互序列
        self.tar_emb_knowledge = tar_emb_knowledge  # 目标嵌入
        
        # 构建Faiss索引
        self._build_faiss_index()
        
        # 标记知识库已构建
        self.knowledge_built = True
        print(f"知识库构建完成，包含 {len(user_id_list)} 个序列样本")
        print(f"知识库三项内容：序列表征维度 {self.seq_emb_knowledge.shape}，原始序列维度 {self.item_seq_knowledge.shape}，目标嵌入维度 {self.tar_emb_knowledge.shape}")

    def precached_knowledge_val(self, val_dataset):
        """为验证集构建知识库"""
        print("为验证集构建检索知识库...")
        seq_emb_knowledge, item_seq_knowledge, tar_emb_knowledge, user_id_list = None, None, None, None
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
                
            if isinstance(user_id_list, list):
                user_id_list.extend(user_id_cans)
            else:
                user_id_list = user_id_cans
                
        # 保存知识库 - 三项内容
        self.user_id_list = user_id_list
        self.item_seq_len_all = item_seq_len_all
        self.seq_emb_knowledge = seq_emb_knowledge  # 序列表征
        self.item_seq_knowledge = item_seq_knowledge  # 原始交互序列  
        self.tar_emb_knowledge = tar_emb_knowledge  # 目标嵌入
        
        # 构建Faiss索引
        self._build_faiss_index()
        
        # 标记知识库已构建
        self.knowledge_built = True
        print(f"验证集知识库构建完成，包含 {len(user_id_list)} 个序列样本")
        print(f"知识库三项内容：序列表征维度 {self.seq_emb_knowledge.shape}，原始序列维度 {self.item_seq_knowledge.shape}，目标嵌入维度 {self.tar_emb_knowledge.shape}")

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

    def seq_augmented(self, seq_output, batch_user_id, batch_seq_len, mode="train"):
        """序列增强 - 使用训练后的交叉注意力层进行索引融合"""
        # 检索相似序列和目标嵌入
        retrieved_seqs, retrieved_item_seqs, retrieved_tars = self.retrieve_seq_tar(
            seq_output, batch_user_id, batch_seq_len, topk=self.topk, mode=mode
        )
        
        # 使用交叉注意力层获得检索信息的融合表征
        retrieval_enhanced_output = self.fusion_forward(seq_output, retrieved_seqs, retrieved_tars)
        
        # 与原始序列表征进行加权融合
        augmented_output = seq_output * self.fusion_weight + retrieval_enhanced_output * (1 - self.fusion_weight)
        
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
            batch_user_id = list(interaction[self.USER_ID].detach().cpu().numpy())
            batch_seq_len = list(item_seq_len.detach().cpu().numpy())
            seq_output = self.seq_augmented(seq_output, batch_user_id, batch_seq_len, mode="test")
        
        # 计算与所有物品的得分
        test_items_emb = self.pretrained_model.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, n_items]
        return scores