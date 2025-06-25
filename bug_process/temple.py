    def retriever_fine_tuning_step(self, interaction):
        """
        执行一次检索器微调步骤
        
        Args:
            interaction: 交互数据
            
        Returns:
            检索器优化损失
        """
        # 获取序列和长度
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        batch_user_id = list(interaction[self.USER_ID].detach().cpu().numpy())
        batch_seq_len = list(item_seq_len.detach().cpu().numpy())
        
        # 获取序列表示
        seq_output = self.forward(item_seq, item_seq_len)
        
        # 检索相似序列和目标项
        torch_retrieval_seq_embs, torch_retrieval_tar_embs, _, _ = self.retrieve_seq_tar(
            seq_output, batch_user_id, batch_seq_len, topk=self.topk, mode="train"
        )
        
        # 计算检索似然分布
        retrieval_scores = self.compute_retrieval_scores(
            seq_output.unsqueeze(1), torch_retrieval_seq_embs, temperature=self.retriever_temperature
        )
        
        # 计算推荐模型的评分分布
        recommendation_scores = self.compute_recommendation_scores(
            seq_output, torch_retrieval_seq_embs, torch_retrieval_tar_embs
        )
        
        # 计算KL散度损失
        kl_loss = self.compute_kl_loss(retrieval_scores, recommendation_scores)
        
        return kl_loss

    def compute_recommendation_scores(self, seq_output, retrieved_seqs, retrieved_tars):
        """计算推荐分布 - 提取预训练DuoRec模型中Transformer自注意力的注意力权重
        
        目标：获取预训练推荐模型(Transformer)对检索序列的注意力权重分布
        这代表预训练模型认为哪些检索序列更重要/相关
        """
        batch_size, n_retrieved, hidden_size = retrieved_seqs.size()
        
        # 获取预训练模型的第一个注意力层
        first_attention_layer = self.trm_encoder.layer[0].multi_head_attention
        
        attention_weights = []
        
        with torch.no_grad():  # 冻结预训练模型参数，只提取注意力权重
            for i in range(n_retrieved):
                current_seq_emb = retrieved_seqs[:, i, :]  # [batch_size, hidden_size]
                
                # 使用预训练模型的Query/Key变换
                # 原序列作为query，检索序列作为key
                query = first_attention_layer.query(seq_output.unsqueeze(1))  # [B, 1, H]
                key = first_attention_layer.key(current_seq_emb.unsqueeze(1))    # [B, 1, H]
                
                # 计算注意力分数（模拟multi-head attention的计算过程）
                # 重新整理为多头格式
                query = query.view(batch_size, 1, first_attention_layer.num_attention_heads, 
                                 first_attention_layer.attention_head_size)  # [B, 1, heads, head_dim]
                key = key.view(batch_size, 1, first_attention_layer.num_attention_heads, 
                             first_attention_layer.attention_head_size)      # [B, 1, heads, head_dim]
                
                # 转置以便矩阵乘法
                query = query.transpose(1, 2)  # [B, heads, 1, head_dim]
                key = key.transpose(1, 2)      # [B, heads, 1, head_dim]
                
                # 计算注意力分数
                attention_scores = torch.matmul(query, key.transpose(-1, -2))  # [B, heads, 1, 1]
                attention_scores = attention_scores / (first_attention_layer.attention_head_size ** 0.5)
                
                # 平均所有注意力头的分数并压缩维度
                attention_score = attention_scores.mean(dim=1).squeeze(-1).squeeze(-1)  # [B]
                attention_weights.append(attention_score)
        
        stacked_weights = torch.stack(attention_weights, dim=1)  # [batch_size, n_retrieved]
        logits = stacked_weights / self.recommendation_temperature
        recommendation_probs = torch.softmax(logits, dim=1)
        
        return recommendation_probs


    def compute_recommendation_scores(self, seq_output, retrieved_seqs, retrieved_tars):
        """利用预训练DuoRec模型的多层注意力权重计算推荐分布"""
        batch_size, n_retrieved, hidden_size = retrieved_seqs.size()
        
        # 保存原始状态并设置推理模式
        original_training_mode = self.trm_encoder.training
        self.trm_encoder.eval()
        
        original_requires_grad = {}
        for name, param in self.trm_encoder.named_parameters():
            original_requires_grad[name] = param.requires_grad
            param.requires_grad = False
        
        recommendation_scores = []
        
        with torch.no_grad():
            for i in range(n_retrieved):
                current_seq_emb = retrieved_seqs[:, i, :]  # [B, H]
                
                layer_attention_scores = []
                
                for layer in self.trm_encoder.layer:
                    attention_layer = layer.multi_head_attention
                    
                    # 使用预训练模型的Query/Key变换
                    query = attention_layer.query(seq_output.unsqueeze(1))      # [B, 1, H]
                    key = attention_layer.key(current_seq_emb.unsqueeze(1))     # [B, 1, H]
                    
                    # 重塑为多头格式
                    query = attention_layer.transpose_for_scores(query)  # [B, num_heads, 1, head_size]
                    key = attention_layer.transpose_for_scores(key)      # [B, num_heads, 1, head_size]
                    
                    # 计算多头注意力分数
                    attention_scores = torch.matmul(query, key.transpose(-1, -2))  # [B, num_heads, 1, 1]
                    attention_scores = attention_scores / math.sqrt(attention_layer.attention_head_size)
                    
                    # 平均所有注意力头
                    attention_score = attention_scores.mean(dim=1).squeeze()  # [B]
                    layer_attention_scores.append(attention_score)
                
                # 加权平均所有层的注意力分数
                final_attention_score = torch.stack(layer_attention_scores, dim=0).mean(dim=0)  # [B]
                recommendation_scores.append(final_attention_score)
        
        # 恢复原始状态
        self.trm_encoder.train(original_training_mode)
        for name, param in self.trm_encoder.named_parameters():
            param.requires_grad = original_requires_grad[name]
        
        # 转换为概率分布
        recommendation_scores = torch.stack(recommendation_scores, dim=1)  # [B, n_retrieved]
        return F.softmax(recommendation_scores, dim=-1)

    def compute_recommendation_scores(self, seq_output, retrieved_seqs, retrieved_tars):
        """使用预训练DuoRec模型预测目标item，然后与检索结果计算相似度
        
        逻辑：
        1. 将seq_output传给预训练DuoRec，预测一个目标item嵌入
        2. 将预测的item嵌入与k个检索目标项计算相似度
        3. 相似度作为各检索结果的推荐评分
        """
        batch_size, n_retrieved, hidden_size = retrieved_seqs.size()
        
        with torch.no_grad():
            # 步骤1：使用预训练DuoRec模型预测目标item
            # 方法：计算seq_output与所有item嵌入的相似度，找到最相似的item
            all_item_emb = self.item_embedding.weight  # [n_items, H] - 所有item的嵌入
            
            # 计算序列输出与所有item的相似度分数（这就是DuoRec.full_sort_predict的逻辑）
            prediction_scores = torch.matmul(seq_output, all_item_emb.transpose(0, 1))  # [B, n_items]
            
            # 获取预测的item ID（取分数最高的item）
            predicted_item_ids = torch.argmax(prediction_scores, dim=-1)  # [B]
            
            # 获取预测item的嵌入
            predicted_item_emb = self.item_embedding(predicted_item_ids)  # [B, H]
            
            # 步骤2：计算预测item与k个检索目标项的相似度
            recommendation_scores = []
            
            for i in range(n_retrieved):
                retrieved_target_emb = retrieved_tars[:, i, :]  # [B, H]
                
                # 计算预测item与检索目标项的相似度
                similarity_score = torch.sum(predicted_item_emb * retrieved_target_emb, dim=-1)  # [B]
                recommendation_scores.append(similarity_score)
            
            recommendation_scores = torch.stack(recommendation_scores, dim=1)  # [B, n_retrieved]
        
        return F.softmax(recommendation_scores, dim=-1)

    def compute_recommendation_scores(self, seq_output, retrieved_seqs, retrieved_tars):
        """模拟DuoRec的full_sort_predict，然后与检索结果计算相似度
        
        逻辑：
        1. 使用seq_output（已经是DuoRec.forward的输出）
        2. 与所有item嵌入计算相似度，得到预测分布
        3. 基于预测分布与检索目标项计算相关性评分
        """
        batch_size, n_retrieved, hidden_size = retrieved_tars.size()
        
        with torch.no_grad():
            # 步骤1：模拟DuoRec的full_sort_predict
            # seq_output已经是self.forward()的输出 [B, H]
            all_items_emb = self.item_embedding.weight  # [n_items, H]
            prediction_scores = torch.matmul(seq_output, all_items_emb.transpose(0, 1))  # [B, n_items]
            
            # 步骤2：将预测分布转换为预测的item嵌入
            # 使用softmax获得概率分布，然后加权平均得到预测的item嵌入
            prediction_probs = torch.softmax(prediction_scores, dim=-1)  # [B, n_items]
            predicted_item_emb = torch.matmul(prediction_probs, all_items_emb)  # [B, H]
            
            # 步骤3：计算预测item嵌入与k个检索目标项的相似度
            recommendation_scores = []
            for i in range(n_retrieved):
                target_emb = retrieved_tars[:, i, :]  # [B, H]
                # 计算相似度（点积，与DuoRec保持一致）
                similarity = torch.sum(predicted_item_emb * target_emb, dim=-1)  # [B]
                recommendation_scores.append(similarity)
            
            # 转换为概率分布
            recommendation_scores = torch.stack(recommendation_scores, dim=1)  # [B, n_retrieved]
            recommendation_scores = torch.softmax(recommendation_scores, dim=-1)
        
        return recommendation_scores