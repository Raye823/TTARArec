import torch
import numpy as np
import logging
import os

logger = logging.getLogger("TTARArec")
logger.setLevel(logging.INFO)
logger.propagate = False  # 不向root传播，避免控制台处理器

# 不自动绑定文件处理器，等待外部配置


def compute_retrieval_effectiveness_vectorized(model, retrieved_item_seqs, pos_items, item_seq, item_seq_len, batch_seq_len, retrieved_seqs=None, retrieved_tar_items=None, enhanced_sequences=None):
    """完全向量化的检索效果计算 - 避免所有CPU-GPU调用"""
    batch_size = item_seq.size(0)
    max_seq_len = item_seq.size(1)
    n_retrieved = retrieved_item_seqs.size(1)
    pos_items_emb = model.get_item_embedding(pos_items)  # [B, H]
    
    # 将batch_seq_len转换为GPU张量
    current_seq_lens = torch.from_numpy(batch_seq_len).to(item_seq.device)  # [B]
    
    # 批量计算所有检索结果的相似度
    if enhanced_sequences is not None:
        # 复用已编码的增强序列，直接与正样本embedding做点积
        all_similarities = torch.sum(enhanced_sequences * pos_items_emb.unsqueeze(1), dim=-1)  # [B, K]
    
    # 找到最佳检索结果（完全GPU操作）
    max_similarities, best_augment_indices = torch.max(all_similarities, dim=1)  # [B], [B]
    
    # 计算索引一致性（完全GPU操作）
    augment_retrieval_consistency = 0.0
    fusion_retrieval_consistency = 0.0
    augment_fusion_consistency = 0.0
    top_retrieval_similarity = 0.0
    
    if retrieved_seqs is not None:
        # 计算表征相似度
        seq_output = model.forward(item_seq, item_seq_len)  # [B, H]
        seq_output_expanded = seq_output.unsqueeze(1)  # [B, 1, H]
        retrieval_similarities = torch.sum(seq_output_expanded * retrieved_seqs, dim=-1)  # [B, K]
        _, top_retrieval_indices = torch.max(retrieval_similarities, dim=1)  # [B]
        
        # 索引一致性计算（GPU操作）
        augment_consistency_matches = (best_augment_indices == top_retrieval_indices).float()
        augment_retrieval_consistency = augment_consistency_matches.mean().item()
        
        if retrieved_tar_items is not None:
            # 如果有增强序列表征，优先使用；否则使用原有逻辑
            if enhanced_sequences is not None:
                ret_attn = model.compute_attention_scores(seq_output, None, None, enhanced_sequences=enhanced_sequences)
            else:
                ret_attn = model.compute_attention_scores(seq_output, retrieved_seqs, retrieved_tar_items)
            if isinstance(ret_attn, tuple):
                fusion_attention_probs = ret_attn[0]
            else:
                fusion_attention_probs = ret_attn
            _, top_fusion_indices = torch.max(fusion_attention_probs, dim=1)
            fusion_consistency_matches = (top_fusion_indices == top_retrieval_indices).float()
            fusion_retrieval_consistency = fusion_consistency_matches.mean().item()
            # 新增：最佳增强效果索引 与 融合权重最高索引 一致性
            augment_fusion_matches = (best_augment_indices == top_fusion_indices).float()
            augment_fusion_consistency = augment_fusion_matches.mean().item()
        
        # 计算表征相似度最高索引的效果（向量化）- 使用目标物品拼接
        # 使用gather操作批量获取对应的目标物品ID
        if retrieved_tar_items is not None:
            top_target_items = torch.gather(
                retrieved_tar_items, 1, 
                top_retrieval_indices.unsqueeze(1)
            ).squeeze(1)  # [B]
            
            # 计算新序列长度（原序列长度 + 1个目标物品）
            top_new_seq_lens = torch.clamp(current_seq_lens + 1, max=max_seq_len)
            
            # 创建新序列，将目标物品拼接到原始序列末尾
            top_batch_new_seqs = torch.zeros_like(item_seq)
            
            # 向量化复制原始序列并添加目标物品
            top_batch_new_seqs = item_seq.clone()  # 先复制整个原始序列
            
            # 向量化添加目标物品到序列末尾
            batch_indices = torch.arange(batch_size, device=item_seq.device)
            valid_append_mask = current_seq_lens < max_seq_len  # 哪些样本可以添加目标物品
            
            if valid_append_mask.any():
                valid_batch_indices = batch_indices[valid_append_mask]
                valid_positions = current_seq_lens[valid_append_mask]
                valid_target_items = top_target_items[valid_append_mask]
                
                # 批量设置目标物品
                top_batch_new_seqs[valid_batch_indices, valid_positions] = valid_target_items
        else:
            # 回退到原有逻辑：使用检索序列拼接
            top_retrieval_seqs = torch.gather(
                retrieved_item_seqs, 1, 
                top_retrieval_indices.unsqueeze(1).unsqueeze(2).expand(-1, 1, max_seq_len)
            ).squeeze(1)  # [B, max_seq_len]
            
            # 向量化计算检索序列长度
            top_seq_lens = torch.sum(top_retrieval_seqs != 0, dim=1)  # [B]
            
            # 向量化序列拼接
            top_total_lens = current_seq_lens + top_seq_lens
            top_new_seq_lens = torch.clamp(top_total_lens, max=max_seq_len)
            
            top_batch_new_seqs = torch.zeros_like(item_seq)
            
            # 原序列掩码
            position_indices = torch.arange(max_seq_len, device=item_seq.device).unsqueeze(0).expand(batch_size, -1)
            top_current_mask = position_indices < current_seq_lens.unsqueeze(1)
            top_batch_new_seqs[top_current_mask] = item_seq[top_current_mask]
            
            # 检索序列掩码
            top_retrieved_start_pos = current_seq_lens.unsqueeze(1)
            top_retrieved_mask = (position_indices >= top_retrieved_start_pos) & (position_indices < top_new_seq_lens.unsqueeze(1))
            
            if top_retrieved_mask.any():
                top_batch_indices = torch.arange(batch_size, device=item_seq.device).unsqueeze(1)
                top_retrieved_relative_pos = position_indices - top_retrieved_start_pos
                top_retrieved_relative_pos = torch.clamp(top_retrieved_relative_pos, min=0, max=max_seq_len-1)
                top_retrieved_values = top_retrieval_seqs[top_batch_indices.expand(-1, max_seq_len), top_retrieved_relative_pos]
                top_batch_new_seqs[top_retrieved_mask] = top_retrieved_values[top_retrieved_mask]
        
        # 批量重新编码
        with torch.no_grad():
            top_new_seq_outputs = model.forward(top_batch_new_seqs, top_new_seq_lens)  # [B, H]
            top_similarities = torch.sum(top_new_seq_outputs * pos_items_emb, dim=-1)  # [B]
            top_retrieval_similarity = top_similarities.mean().item()

    # ========== 调试信息（每129个batch打印一次） ==========
    try:
        if hasattr(model, 'batch_count') and (model.batch_count % 129 == 0):
            # 安全获取B、K
            B = item_seq.size(0)
            K = retrieved_item_seqs.size(1)
            if B > 0 and K > 0:
                # 随机选择一个样本索引
                rand_idx = torch.randint(low=0, high=B, size=(1,), device=item_seq.device).item()
                # 确保有seq_output用于打印原始表征
                if 'seq_output' not in locals():
                    seq_output = model.forward(item_seq, item_seq_len)
                # 取该样本的原始表征与目标项
                sample_seq_repr = seq_output[rand_idx]
                sample_pos_item = pos_items[rand_idx]
                # 取该样本检索到的K个序列表征与目标项
                if retrieved_seqs is not None:
                    sample_ret_seq_repr = retrieved_seqs[rand_idx]  # [K, H]
                else:
                    # 若未提供retrieved_seqs，则回退为空（不打印表征）
                    sample_ret_seq_repr = None
                sample_ret_tar_items = None
                if retrieved_tar_items is not None:
                    sample_ret_tar_items = retrieved_tar_items[rand_idx]  # [K]

                # 构造需要打印的“物品序列ID列表”（尽量只搬一条到CPU）
                # 原始序列（按真实长度截断）
                sample_len = int(item_seq_len[rand_idx].item()) if torch.is_tensor(item_seq_len) else int(item_seq_len[rand_idx])
                sample_item_seq_ids = item_seq[rand_idx].detach().cpu().tolist()
                sample_item_seq_ids = sample_item_seq_ids[:sample_len] if sample_len > 0 else []
                # 检索到的K个序列（完整打印，含padding 0）
                sample_ret_item_seqs = retrieved_item_seqs[rand_idx].detach().cpu().tolist()
                # 目标项
                sample_pos_item_cpu = sample_pos_item.detach().cpu().item() if torch.is_tensor(sample_pos_item) else int(sample_pos_item)
                sample_ret_tar_items_cpu = sample_ret_tar_items.detach().cpu().tolist() if sample_ret_tar_items is not None else None

                # 打印
                logger.info("\n--- 调试样本（每129个batch一次）---")
                # 打印样本索引与用户ID；样本索引是本batch内的下标 [0, B)
                sample_user = None
                # 优先从模型暂存的当前batch用户ID中获取（由训练侧传入的numpy数组）
                cb_users = getattr(model, '_current_batch_user_ids', None)
                if cb_users is not None and len(cb_users) > rand_idx:
                    try:
                        sample_user = int(cb_users[rand_idx])
                    except Exception:
                        sample_user = None
                logger.info(f"样本索引: {rand_idx} | user: {sample_user if sample_user is not None else '?'}")
                logger.info(f"原始物品序列: {sample_item_seq_ids}")
                logger.info(f"目标项ID: {sample_pos_item_cpu}")
                logger.info(f"检索到的K个物品序列 (K={K}):")
                # 为每个检索序列附上所属用户ID（通过在知识库中匹配序列获得）
                kb_seqs = getattr(model, 'item_seq_knowledge', None)
                kb_users = getattr(model, 'user_id_list', None)
                for kk, seq_ids in enumerate(sample_ret_item_seqs):
                    user_of_seq = None
                    if kb_seqs is not None and kb_users is not None:
                        try:
                            seq_np = np.array(seq_ids)
                            # 直接全行匹配（包含padding）
                            matches = np.where((kb_seqs == seq_np).all(axis=1))[0]
                            if matches.size > 0:
                                user_of_seq = kb_users[matches[0]]
                        except Exception:
                            user_of_seq = None
                    if user_of_seq is None:
                        logger.info(f"  #{kk}: {seq_ids} | user: ?")
                    else:
                        logger.info(f"  #{kk}: {seq_ids} | user: {int(user_of_seq)}")
                if sample_ret_tar_items_cpu is not None:
                    logger.info(f"检索到的K个目标项ID: {sample_ret_tar_items_cpu}")
                else:
                    logger.info("检索到的目标项ID: 未提供retrieved_tar_items，跳过目标项打印")
                logger.info("--------------------------------\n")
    except Exception as e:
        # 调试打印不影响主流程
        logger.warning(f"[调试打印异常忽略] {e}")
    
    return max_similarities.mean().item(), augment_retrieval_consistency, fusion_retrieval_consistency, top_retrieval_similarity, augment_fusion_consistency


def print_diagnostic_info_optimized(model, rec_loss, kl_loss, retrieval_probs, attention_probs, 
                          seq_output, seq_output_aug, pos_items, retrieval_effectiveness, 
                          augment_retrieval_consistency, fusion_retrieval_consistency, top_retrieval_similarity, augment_fusion_consistency):
    """优化的诊断信息输出 - 减少GPU-CPU传输"""
    with torch.no_grad():
        # 批量计算所有GPU指标，最后一次性传输到CPU
        gpu_metrics = {}
        
        # 基础损失指标
        gpu_metrics['rec_loss'] = rec_loss
        gpu_metrics['kl_loss'] = kl_loss
        gpu_metrics['total_loss'] = kl_loss * model.kl_loss_weight + rec_loss 
        # 评分分布分析
        gpu_metrics['retrieval_entropy'] = -torch.sum(retrieval_probs * torch.log(retrieval_probs + 1e-8), dim=-1).mean()
        gpu_metrics['attention_entropy'] = -torch.sum(attention_probs * torch.log(attention_probs + 1e-8), dim=-1).mean()
        gpu_metrics['retrieval_std'] = retrieval_probs.std()
        gpu_metrics['attention_std'] = attention_probs.std()
        
        # 相关性分析
        retrieval_flat = retrieval_probs.view(-1)
        attention_flat = attention_probs.view(-1)
        gpu_metrics['correlation'] = torch.corrcoef(torch.stack([retrieval_flat, attention_flat]))[0, 1]
        
        # 排序一致性分析
        attention_ranks = torch.argsort(torch.argsort(attention_probs, dim=-1, descending=True), dim=-1)
        retrieval_ranks = torch.argsort(torch.argsort(retrieval_probs, dim=-1, descending=True), dim=-1)
        gpu_metrics['rank_correlation'] = torch.corrcoef(torch.stack([retrieval_ranks.view(-1).float(), attention_ranks.view(-1).float()]))[0, 1]
        
        # Top-1一致性
        top1_retrieval = torch.argmax(retrieval_probs, dim=-1)
        top1_attention = torch.argmax(attention_probs, dim=-1)
        gpu_metrics['top1_consistency'] = (top1_retrieval == top1_attention).float().mean()
        
        # 增强效果分析
        gpu_metrics['seq_similarity'] = torch.cosine_similarity(seq_output, seq_output_aug, dim=-1).mean()
        pos_items_emb = model.get_item_embedding(pos_items)
        gpu_metrics['original_sim'] = torch.sum(seq_output * pos_items_emb, dim=-1).mean()
        gpu_metrics['augmented_sim'] = torch.sum(seq_output_aug * pos_items_emb, dim=-1).mean()
        gpu_metrics['sim_improvement'] = gpu_metrics['augmented_sim'] - gpu_metrics['original_sim']
        # 新增：排序间隔指标（完全在GPU上计算）
        test_item_emb = model.pretrained_model.item_embedding.weight  # [n_items, H]
        batch_size_local = seq_output.size(0)
        batch_indices = torch.arange(batch_size_local, device=seq_output.device)
        # 原始与增强的全物品打分
        original_logits_full = torch.matmul(seq_output, test_item_emb.transpose(0, 1))  # [B, N]
        augmented_logits_full = torch.matmul(seq_output_aug, test_item_emb.transpose(0, 1))  # [B, N]
        # 正样本分数
        pos_scores_original = original_logits_full[batch_indices, pos_items]
        pos_scores_augmented = augmented_logits_full[batch_indices, pos_items]
        # 负样本屏蔽（将正样本位置置为极小值）
        original_logits_masked = original_logits_full.clone()
        original_logits_masked[batch_indices, pos_items] = -1e9
        augmented_logits_masked = augmented_logits_full.clone()
        augmented_logits_masked[batch_indices, pos_items] = -1e9
        # Top-1负样本
        top1_neg_original = torch.max(original_logits_masked, dim=1).values
        top1_neg_augmented = torch.max(augmented_logits_masked, dim=1).values
        # 间隔（Top-1负样本）
        gpu_metrics['original_margin_top1'] = (pos_scores_original - top1_neg_original).mean()
        gpu_metrics['augmented_margin_top1'] = (pos_scores_augmented - top1_neg_augmented).mean()
        gpu_metrics['margin_top1_improvement'] = gpu_metrics['augmented_margin_top1'] - gpu_metrics['original_margin_top1']
        # Top-10负样本均值间隔（可选）
        topk = 10 if augmented_logits_masked.size(1) >= 10 else max(1, int(augmented_logits_masked.size(1) // 100))
        if topk > 1:
            topk_neg_original = torch.topk(original_logits_masked, k=topk, dim=1).values.mean(dim=1)
            topk_neg_augmented = torch.topk(augmented_logits_masked, k=topk, dim=1).values.mean(dim=1)
            gpu_metrics['original_margin_topk'] = (pos_scores_original - topk_neg_original).mean()
            gpu_metrics['augmented_margin_topk'] = (pos_scores_augmented - topk_neg_augmented).mean()
            gpu_metrics['margin_topk_improvement'] = gpu_metrics['augmented_margin_topk'] - gpu_metrics['original_margin_topk']
        else:
            gpu_metrics['original_margin_topk'] = gpu_metrics['original_margin_top1']
            gpu_metrics['augmented_margin_topk'] = gpu_metrics['augmented_margin_top1']
            gpu_metrics['margin_topk_improvement'] = gpu_metrics['margin_top1_improvement']
        
        # 一次性将所有GPU指标传输到CPU
        cpu_metrics = {k: v.item() if torch.is_tensor(v) else v for k, v in gpu_metrics.items()}
        
        # 输出结果
        logger.info(f"\n========== 诊断信息 (Batch {model.batch_count}) ==========")
        logger.info(f"推荐损失: {cpu_metrics['rec_loss']:.6f}")
        logger.info(f"KL散度损失: {cpu_metrics['kl_loss']:.6f}")
        logger.info(f"总损失: {cpu_metrics['total_loss']:.6f}")
        
        logger.info(f"\n--- 评分分布 ---")
        logger.info(f"检索评分标准差: {cpu_metrics['retrieval_std']:.6f}")
        logger.info(f"检索评分熵: {cpu_metrics['retrieval_entropy']:.6f}")
        
        logger.info(f"注意力评分标准差: {cpu_metrics['attention_std']:.6f}")
        logger.info(f"注意力评分熵: {cpu_metrics['attention_entropy']:.6f}")
        logger.info(f"检索评分与注意力评分相关性: {cpu_metrics['correlation']:.6f}")
        logger.info(f"检索评分与注意力评分排序相关性: {cpu_metrics['rank_correlation']:.6f}")
        logger.info(f"Top-1选择一致性: {cpu_metrics['top1_consistency']:.6f}")
        
        logger.info(f"\n--- 增强效果 ---")
        logger.info(f"原始序列与增强序列相似度: {cpu_metrics['seq_similarity']:.6f}")
        logger.info(f"原始序列与目标项相似度: {cpu_metrics['original_sim']:.6f}")
        logger.info(f"增强序列与目标项相似度: {cpu_metrics['augmented_sim']:.6f}")
        logger.info(f"增强带来的相似度提升: {cpu_metrics['sim_improvement']:.6f}")
        logger.info(f"原始序列Top-1负样本间隔: {cpu_metrics['original_margin_top1']:.6f}")
        logger.info(f"增强序列Top-1负样本间隔: {cpu_metrics['augmented_margin_top1']:.6f}")
        logger.info(f"Top-1间隔提升: {cpu_metrics['margin_top1_improvement']:.6f}")
        logger.info(f"原始序列Top-10均值负样本间隔: {cpu_metrics['original_margin_topk']:.6f}")
        logger.info(f"增强序列Top-10均值负样本间隔: {cpu_metrics['augmented_margin_topk']:.6f}")
        logger.info(f"Top-10间隔提升: {cpu_metrics['margin_topk_improvement']:.6f}")
        
        logger.info(f"\n--- 检索效果 ---")
        logger.info(f"最佳检索序列拼接后相似度: {retrieval_effectiveness:.6f}")
        logger.info(f"表征相似度最高索引拼接后相似度: {top_retrieval_similarity:.6f}")
        
        
        logger.info(f"\n--- 索引一致性分析 ---")
        logger.info(f"最佳增强效果索引与表征相似度最高索引一致性: {augment_retrieval_consistency:.6f} ({augment_retrieval_consistency*100:.2f}%)")
        logger.info(f"融合权重最高索引与表征相似度最高索引一致性: {fusion_retrieval_consistency:.6f} ({fusion_retrieval_consistency*100:.2f}%)")
        logger.info(f"最佳增强效果索引与融合权重最高索引一致性: {augment_fusion_consistency:.6f} ({augment_fusion_consistency*100:.2f}%)")
        logger.info(f"========================================\n")
