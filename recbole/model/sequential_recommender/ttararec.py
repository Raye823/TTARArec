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
from recbole.model.layers import activation_layer
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

        # 检索相关参数
        self.len_lower_bound = config["len_lower_bound"] if "len_lower_bound" in config else -1
        self.len_upper_bound = config["len_upper_bound"] if "len_upper_bound" in config else -1
        self.len_bound_reverse = config["len_bound_reverse"] if "len_bound_reverse" in config else True
        self.nprobe = config['nprobe'] if 'nprobe' in config else 1
        self.topk = config['top_k'] if 'top_k' in config else 10
        self.alpha = config['alpha'] if 'alpha' in config else 0.8
        self.low_popular = config['low_popular'] if 'low_popular' in config else 100

        # 检索器编码器相关参数
        self.retriever_layers = config['retriever_layers'] if 'retriever_layers' in config else 1
        self.retriever_temperature = config['retriever_temperature'] if 'retriever_temperature' in config else 0.1
        self.recommendation_temperature = config['recommendation_temperature'] if 'recommendation_temperature' in config else 0.1
        self.retriever_dropout = config['retriever_dropout'] if 'retriever_dropout' in config else 0.1
        self.kl_weight = config['kl_weight'] if 'kl_weight' in config else 0.1

        # 加载预训练模型
        self.pretrained_model = PretrainedModelLoader.load_duorec_model(config, dataset)

        # 从预训练模型获取配置信息
        self.hidden_size = self.pretrained_model.hidden_size
        
        # 从预训练模型获取架构参数（用于构建检索器编码器）
        self.hidden_act = getattr(self.pretrained_model, 'hidden_act', config['hidden_act'] if 'hidden_act' in config else 'gelu')
        self.initializer_range = config['initializer_range']      
        # 构建检索器编码器组件
        self._build_retriever_encoder()

        # 初始化检索相关组件
        self.dataset = dataset
        self._init_retrieval_components()

    def _build_retriever_encoder(self):
        """构建检索器编码器 - 简单MLP架构"""
        self.retriever_mlp = nn.ModuleList()
        self.retriever_layer_norms = nn.ModuleList()
        
        for i in range(self.retriever_layers):
            self.retriever_mlp.append(
                nn.Linear(self.hidden_size, self.hidden_size)
            )
            self.retriever_layer_norms.append(
                nn.LayerNorm(self.hidden_size)
            )
        
        # 激活函数和dropout
        self.retriever_act_fn = activation_layer(self.hidden_act)
        self.retriever_dropout_layer = nn.Dropout(self.retriever_dropout)
        
        # 初始化检索器参数（只初始化检索器组件，不影响预训练模型）
        self._init_retriever_modules()

    def _init_retriever_modules(self):
        """只初始化检索器组件，不影响预训练模型"""
        # 初始化检索器MLP层
        for layer in self.retriever_mlp:
            if isinstance(layer, nn.Linear):
                layer.weight.data.normal_(mean=0.0, std=self.initializer_range)
                if layer.bias is not None:
                    layer.bias.data.zero_()
        
        # 初始化检索器LayerNorm层
        for layer_norm in self.retriever_layer_norms:
            if isinstance(layer_norm, nn.LayerNorm):
                layer_norm.bias.data.zero_()
                layer_norm.weight.data.fill_(1.0)

    def _init_retriever_weights(self, module):
        """初始化检索器权重"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _init_retrieval_components(self):
        """初始化检索相关组件"""
        # 这些属性将在precached_knowledge中设置
        self.user_id_list = None
        self.item_seq_all = None
        self.item_seq_len_all = None
        self.seq_emb_knowledge = None
        self.item_seq_knowledge = None  # 新增：原始交互序列知识库
        self.tar_emb_knowledge = None
        self.seq_emb_index = None
        self.tar_emb_index = None

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

    def get_item_embedding(self, item_ids):
        """获取物品嵌入 - 直接调用预训练模型"""
        with torch.no_grad():
            return self.pretrained_model.item_embedding(item_ids)

    @property
    def item_embedding(self):
        """兼容性属性：访问预训练模型的物品嵌入层"""
        return self.pretrained_model.item_embedding


    def seq_augmented(self, seq_output, batch_user_id, batch_seq_len, mode="train"):
        """使用检索增强进行序列增强"""
        # 检索相似序列和目标嵌入
        retrieved_seqs1, retrieved_item_seqs1, retrieved_tars1, retrieved_seqs2, retrieved_item_seqs2, retrieved_tars2 = self.retrieve_seq_tar(
            seq_output, batch_user_id, batch_seq_len, topk=self.topk, mode=mode
        )

        # 使用检索器编码器调整原始序列表示
        retriever_encoded_seq = self.retriever_forward(seq_output)
        
        # 计算检索器编码后序列与相似序列的相似度作为权重
        similarities = torch.sum(retriever_encoded_seq.unsqueeze(1) * retrieved_seqs1, dim=-1)  # [B, K]
        weights = torch.softmax(similarities, dim=-1)  # [B, K]
        
        # 使用权重来融合对应的目标嵌入
        weighted_retrieved = torch.sum(weights.unsqueeze(-1) * retrieved_tars1, dim=1)  # [B, H]
        
        # 与原始序列进行加权组合
        alpha = self.alpha
        seq_output = alpha * seq_output + (1 - alpha) * weighted_retrieved
        return seq_output

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
        retrieval_seq1 = self.seq_emb_knowledge[I1_filtered]  # 序列表征
        retrieval_item_seq1 = self.item_seq_knowledge[I1_filtered]  # 原始交互序列
        retrieval_tar1 = self.tar_emb_knowledge[I1_filtered]  # 目标嵌入
        
        retrieval_seq2 = self.seq_emb_knowledge[I1_filtered]
        retrieval_item_seq2 = self.item_seq_knowledge[I1_filtered]
        retrieval_tar2 = self.tar_emb_knowledge[I1_filtered]
        
        return (
            torch.tensor(retrieval_seq1).to("cuda"), 
            torch.tensor(retrieval_item_seq1).to("cuda"),  # 新增返回原始交互序列
            torch.tensor(retrieval_tar1).to("cuda"), 
            torch.tensor(retrieval_seq2).to("cuda"), 
            torch.tensor(retrieval_item_seq2).to("cuda"),  # 新增返回原始交互序列
            torch.tensor(retrieval_tar2).to("cuda")
        )

    # ============ 损失计算相关方法 ============
    
    def calculate_loss(self, interaction):
        """计算训练损失 - 基于KL散度对齐检索分布和推荐分布"""
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        batch_user_id = list(interaction[self.USER_ID].detach().cpu().numpy())
        batch_seq_len = list(item_seq_len.detach().cpu().numpy())
        
        # 检索相似序列和目标嵌入
        retrieved_seqs1, retrieved_item_seqs1, retrieved_tars1, retrieved_seqs2, retrieved_item_seqs2, retrieved_tars2 = self.retrieve_seq_tar(
            seq_output,
            batch_user_id, 
            batch_seq_len,
            topk=self.topk
        )
        
        # 计算检索分布：基于检索器编码后序列与目标项的融合相似度
        retrieval_probs = self.compute_retrieval_scores(
            seq_output, retrieved_seqs1, retrieved_tars1, pos_items
        )  # [B, K]
        
        # 计算推荐分布：基于原序列与检索序列的点积相似度
        recommendation_probs = self.compute_recommendation_scores(
            seq_output, retrieved_seqs2, retrieved_tars2
        )  # [B, K]
        
        # 计算KL散度损失
        kl_loss = self.compute_kl_loss(retrieval_probs, recommendation_probs)
        
        # 总损失 = KL散度损失 * 权重
        total_loss = self.kl_weight * kl_loss
        
        return total_loss

    def compute_retrieval_scores(self, seq_output, retrieved_seqs, retrieved_tars, pos_items):
        """计算检索评分 - 基于检索器编码后序列与目标嵌入的融合相似度"""
        batch_size, n_retrieved, hidden_size = retrieved_seqs.size()
        pos_items_emb = self.get_item_embedding(pos_items)  # [batch_size, hidden_size]
        
        # 使用检索器编码器处理原始序列
        retriever_encoded_seq = self.retriever_forward(seq_output)
        
        # 计算融合目标嵌入后的表示与目标项的相似度
        fusion_scores = []
        
        # 对每个检索到的目标嵌入进行融合并计算相似度
        for i in range(n_retrieved):
            # 获取当前检索结果的目标嵌入
            current_tar_emb = retrieved_tars[:, i, :]  # [batch_size, hidden_size]
            
            # 加权融合：检索器编码后序列 + 检索目标嵌入
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
        """计算推荐分布 - 模拟预训练模型的full_sort_predict逻辑"""
        batch_size, n_retrieved, hidden_size = retrieved_tars.size()
        
        with torch.no_grad():
            # 重塑retrieved_tars进行批量矩阵乘法
            retrieved_tars_t = retrieved_tars.transpose(1, 2)  # [batch_size, hidden_size, n_retrieved]

            # 计算序列表示与检索目标项的相似度得分
            scores = torch.bmm(seq_output.unsqueeze(1), retrieved_tars_t).squeeze(1)  # [B, n_retrieved]
            
            # 使用softmax将分数转换为概率分布
            recommendation_scores = torch.softmax(scores, dim=-1)
            
        return recommendation_scores

    def compute_kl_loss(self, retrieval_probs, recommendation_probs):
        """计算检索分布与推荐分布之间的KL散度损失"""
        # 确保两个分布的维度完全匹配
        assert retrieval_probs.shape == recommendation_probs.shape, \
            f"形状不匹配: retrieval_probs {retrieval_probs.shape} vs recommendation_probs {recommendation_probs.shape}"
        
        # 避免数值问题
        epsilon = 1e-8
        retrieval_probs = retrieval_probs + epsilon
        recommendation_probs = recommendation_probs + epsilon
        
        # KL散度计算: KL(retrieval_probs || recommendation_probs)
        # 优化检索分布使其接近推荐分布
        kl_div = torch.sum(retrieval_probs * torch.log(retrieval_probs / recommendation_probs), dim=-1)
        
        # 返回批次平均损失
        return kl_div.mean()

    # ============ 预测相关方法 ============
    
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
        """全排序预测 - 使用检索增强"""
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        batch_user_id = list(interaction[self.USER_ID].detach().cpu().numpy())
        batch_seq_len = list(item_seq_len.detach().cpu().numpy())
        
        # 序列增强
        seq_output_aug = self.seq_augmented(seq_output, batch_user_id, batch_seq_len, mode="test")
        
        # 根据序列长度决定是否使用增强
        seq_output_aug = torch.where(
            (item_seq_len > self.low_popular).unsqueeze(-1).repeat(1, self.hidden_size), 
            seq_output, 
            seq_output_aug
        )
        
        # 计算与所有物品的得分
        test_items_emb = self.pretrained_model.item_embedding.weight
        scores = torch.matmul(seq_output_aug, test_items_emb.transpose(0, 1))  # [B, n_items]
        return scores

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