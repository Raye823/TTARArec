# -*- coding: utf-8 -*-
# @Time    : 2025/1/1
# @Author  : Xinping Zhao
# @Email   : zhaoxinping@stu.hit.edu.cn

"""
RARec (Retrieval Augmented Recommender)
################################################

基于检索增强的序列推荐模型，加载预训练模型并应用检索增强进行评估。
检索增强逻辑：从构建的知识库中基于表征检索相似度最高的一个序列，
然后将该序列拼接到原始序列上，得到增强序列，进行推荐评估。

"""

import torch
import faiss
import numpy as np
from faiss import normalize_L2
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.sequential_recommender.pretrained_model_loader import PretrainedModelLoader


class RARec(SequentialRecommender):
    """
    基于检索增强的序列推荐模型
    
    主要功能：
    1. 加载预训练模型作为特征提取器
    2. 构建序列知识库
    3. 检索相似序列并拼接增强
    4. 进行推荐评估
    """

    def __init__(self, config, dataset):
        super(RARec, self).__init__(config, dataset)

        # 加载预训练模型
        self.pretrained_model = PretrainedModelLoader.load_duorec_model(config, dataset)
        
        # 从预训练模型获取基础参数
        self.hidden_size = self.pretrained_model.hidden_size

        # 检索配置参数
        self.topk = config['top_k']  # 简化为只检索1个最相似的序列
        self.nprobe = config['nprobe']
        
        # 序列长度过滤参数
        self.len_lower_bound = config["len_lower_bound"]
        self.len_upper_bound = config["len_upper_bound"]
        self.len_bound_reverse = config["len_bound_reverse"]

        # 初始化检索知识库相关变量
        self.dataset = dataset
        self.user_id_list = None
        self.item_seq_len_all = None
        self.seq_emb_knowledge = None
        self.item_seq_knowledge = None  # 原始交互序列知识库
        self.seq_emb_index = None
        
        # 检索增强状态控制
        self.use_retrieval = False
        self.knowledge_built = False

    def get_item_embedding(self, item_ids):
        """获取物品嵌入"""
        with torch.no_grad():
            return self.pretrained_model.item_embedding(item_ids)
            
    def forward(self, item_seq, item_seq_len):
        """序列编码"""
        with torch.no_grad():
            return self.pretrained_model.forward(item_seq, item_seq_len)

    def retrieve_similar_seq(self, queries, batch_user_id, batch_seq_len):
        """检索相似序列 - 简化为只返回最相似的一个序列"""
        queries_cpu = queries.detach().cpu().numpy()
        normalize_L2(queries_cpu)
        _, I1 = self.seq_emb_index.search(queries_cpu, self.topk * 4)  # 搜索更多候选以便过滤
        
        # 过滤掉同用户的相同长度序列，只保留最相似的一个
        filtered_indices = []
        for i, I_entry in enumerate(I1):
            current_user = batch_user_id[i]
            current_length = batch_seq_len[i]
            
            # 找到第一个满足条件的序列索引
            for idx in I_entry:
                if (self.user_id_list[idx] != current_user or 
                    (self.user_id_list[idx] == current_user and self.item_seq_len_all[idx] < current_length)):
                    filtered_indices.append(idx)
                    break
            else:
                # 如果没有找到合适的，使用第一个
                filtered_indices.append(I_entry[0])
        
        # 获取检索到的原始交互序列
        retrieval_item_seqs = self.item_seq_knowledge[filtered_indices]
        
        return torch.tensor(retrieval_item_seqs).to(queries.device)

    def seq_augmented(self, seq_output, item_seq, item_seq_len, batch_user_id, batch_seq_len):
        """序列增强 - 检索相似序列并拼接到原始序列"""
        # 检索相似序列
        retrieved_seqs = self.retrieve_similar_seq(seq_output, batch_user_id, batch_seq_len)  # [B, max_seq_len]
        
        batch_size = item_seq.size(0)
        max_seq_len = item_seq.size(1)
        
        # 创建增强序列
        augmented_seqs = torch.zeros_like(item_seq)
        augmented_seq_lens = torch.zeros_like(item_seq_len)
        
        for i in range(batch_size):
            original_seq = item_seq[i]
            original_len = batch_seq_len[i]
            retrieved_seq = retrieved_seqs[i]
            
            # 计算检索序列的有效长度
            retrieved_len = torch.sum(retrieved_seq != 0).item()
            
            # 拼接序列：原始序列 + 检索序列
            new_seq_len = min(original_len + retrieved_len, max_seq_len)
            
            # 填充原始序列
            augmented_seqs[i, :original_len] = original_seq[:original_len]
            
            # 填充检索序列
            remaining_space = max_seq_len - original_len
            if remaining_space > 0 and retrieved_len > 0:
                copy_len = min(retrieved_len, remaining_space)
                augmented_seqs[i, original_len:original_len + copy_len] = retrieved_seq[:copy_len]
            
            augmented_seq_lens[i] = new_seq_len
        
        # 重新编码增强序列
        augmented_seq_output = self.forward(augmented_seqs, augmented_seq_lens)
        
        return augmented_seq_output

    def precached_knowledge_val(self, val_dataset):
        """构建检索知识库"""
        print("构建检索知识库...")
        seq_emb_knowledge = []
        item_seq_knowledge = []
        user_id_list = []
        item_seq_len_all = []
        
        # 处理训练集
        for interaction in self.dataset:
            interaction = interaction.to("cuda")
            
            # 序列长度过滤
            if self.len_lower_bound != -1 or self.len_upper_bound != -1:
                if self.len_lower_bound != -1 and self.len_upper_bound != -1:
                    look_up_indices = ((interaction[self.ITEM_SEQ_LEN] >= self.len_lower_bound) & 
                                     (interaction[self.ITEM_SEQ_LEN] <= self.len_upper_bound))
                elif self.len_upper_bound != -1:
                    look_up_indices = interaction[self.ITEM_SEQ_LEN] < self.len_upper_bound
                else:
                    look_up_indices = interaction[self.ITEM_SEQ_LEN] > self.len_lower_bound
                    
                if self.len_bound_reverse:
                    look_up_indices = ~look_up_indices
            else:
                look_up_indices = interaction[self.ITEM_SEQ_LEN] > 0
            
            item_seq = interaction[self.ITEM_SEQ][look_up_indices]
            item_seq_len = interaction[self.ITEM_SEQ_LEN][look_up_indices]
            
            if len(item_seq) == 0:
                continue
                
            seq_output = self.forward(item_seq, item_seq_len)
            user_ids = interaction[self.USER_ID][look_up_indices]
            
            # 累积知识库
            seq_emb_knowledge.append(seq_output.detach().cpu().numpy())
            item_seq_knowledge.append(item_seq.detach().cpu().numpy())
            user_id_list.extend(user_ids.detach().cpu().numpy().tolist())
            item_seq_len_all.extend(item_seq_len.detach().cpu().numpy().tolist())
        
        # 处理验证集
        for batched_data in val_dataset:
            interaction = batched_data[0].to("cuda")
            item_seq = interaction[self.ITEM_SEQ]
            item_seq_len = interaction[self.ITEM_SEQ_LEN]
            user_ids = interaction[self.USER_ID]
            
            seq_output = self.forward(item_seq, item_seq_len)
            
            # 累积知识库
            seq_emb_knowledge.append(seq_output.detach().cpu().numpy())
            item_seq_knowledge.append(item_seq.detach().cpu().numpy())
            user_id_list.extend(user_ids.detach().cpu().numpy().tolist())
            item_seq_len_all.extend(item_seq_len.detach().cpu().numpy().tolist())
        
        # 合并所有知识库
        self.seq_emb_knowledge = np.concatenate(seq_emb_knowledge, axis=0)
        self.item_seq_knowledge = np.concatenate(item_seq_knowledge, axis=0)
        self.user_id_list = user_id_list
        self.item_seq_len_all = item_seq_len_all
        
        # 构建Faiss索引
        self._build_faiss_index()
        
        self.knowledge_built = True
        print(f"知识库构建完成，包含 {len(user_id_list)} 个序列样本")
        print(f"序列表征维度: {self.seq_emb_knowledge.shape}, 原始序列维度: {self.item_seq_knowledge.shape}")

    def _build_faiss_index(self):
        """构建Faiss检索索引"""
        d = self.hidden_size
        nlist = min(128, len(self.seq_emb_knowledge) // 10)  # 根据数据量调整
        
        # 构建序列嵌入索引
        seq_emb_knowledge_copy = np.array(self.seq_emb_knowledge, copy=True)
        normalize_L2(seq_emb_knowledge_copy)
        
        if len(seq_emb_knowledge_copy) < nlist:
            # 数据量太小，使用简单的平面索引
            self.seq_emb_index = faiss.IndexFlatIP(d)
        else:
            # 使用IVF索引
            quantizer = faiss.IndexFlatIP(d)
            self.seq_emb_index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
            self.seq_emb_index.train(seq_emb_knowledge_copy)
            self.seq_emb_index.nprobe = self.nprobe
        
        self.seq_emb_index.add(seq_emb_knowledge_copy)

    def enable_retrieval(self):
        """启用检索增强功能"""
        self.use_retrieval = True

    def predict(self, interaction):
        """预测单个物品的得分"""
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        
        seq_output = self.forward(item_seq, item_seq_len)
        
        # 如果启用检索增强
        if self.use_retrieval and self.knowledge_built:
            batch_user_id = interaction[self.USER_ID].detach().cpu().numpy().tolist()
            batch_seq_len = item_seq_len.detach().cpu().numpy().tolist()
            seq_output = self.seq_augmented(seq_output, item_seq, item_seq_len, batch_user_id, batch_seq_len)
        
        test_item_emb = self.get_item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        """全排序预测"""
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        
        seq_output = self.forward(item_seq, item_seq_len)
        
        # 如果启用检索增强
        if self.use_retrieval and self.knowledge_built:
            batch_user_id = interaction[self.USER_ID].detach().cpu().numpy().tolist()
            batch_seq_len = item_seq_len.detach().cpu().numpy().tolist()
            seq_output = self.seq_augmented(seq_output, item_seq, item_seq_len, batch_user_id, batch_seq_len)
        
        # 计算与所有物品的得分
        test_items_emb = self.pretrained_model.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores