import torch
import numpy as np
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender.ttararec import TTARArec
from recbole.model.sequential_recommender.pretrained_model_loader import PretrainedModelLoader

# 设置配置
config_dict = {
    'model': 'TTARArec',
    'dataset': 'Amazon_Beauty',
    'data_path': './recbole/dataset',
    'load_col': {'inter': ['user_id', 'item_id', 'timestamp']},
    'USER_ID_FIELD': 'user_id',
    'ITEM_ID_FIELD': 'item_id',
    'TIME_FIELD': 'timestamp', 
    'seq_len': 50,
    'MAX_ITEM_LIST_LENGTH': 50,
    'train_batch_size': 256,
    'eval_batch_size': 256,
    'embedding_size': 64,
    'hidden_size': 64,
    'inner_size': 256,
    'n_layers': 2,
    'n_heads': 2,
    'hidden_dropout_prob': 0.5,
    'attn_dropout_prob': 0.5,
    'hidden_act': 'gelu',
    'layer_norm_eps': 1e-12,
    'initializer_range': 0.02,
    'loss_type': 'CE',
    'training_neg_sample_num': 0,
    'pretrained_path': './log/DuoRec/Amazon_Beauty/bs1024-lmd0.1-sem0.1-us_x-Mar-19-2025_21-16-57-lr0.001-l20-tau1-dot-DPh0.5-DPa0.5/model.pth',
    'alpha': 0.8,
    'topk': 5,
    'pooling_mode': 'mean',
    'device': 'cuda',
    'gpu_id': 0
}

config = Config(model='TTARArec', dataset='Amazon_Beauty', config_dict=config_dict)

print("=== 1. 创建数据集 ===")
dataset = create_dataset(config)
train_data, valid_data, test_data = data_preparation(config, dataset)

print("=== 2. 创建TTARArec模型 ===")
ttararec_model = TTARArec(config, dataset).cuda()

print("=== 3. 创建独立的预训练模型 ===")
direct_model = PretrainedModelLoader.load_duorec_model(config, dataset).cuda()

print("=== 4. 获取测试数据 ===")
# 获取一批测试数据
test_iter = iter(test_data)
interaction = next(test_iter)
item_seq = interaction['item_id_list'].cuda()
item_seq_len = interaction['length'].cuda()

print(f"测试序列形状: {item_seq.shape}")
print(f"序列长度: {item_seq_len[:5]}")

print("\n=== 5. 对比前向传播 ===")

# 直接模型前向传播
print("5.1 直接模型前向传播")
with torch.no_grad():
    direct_output = direct_model.forward(item_seq, item_seq_len)
print(f"直接模型输出形状: {direct_output.shape}")
print(f"直接模型输出范围: {direct_output.min().item():.4f} ~ {direct_output.max().item():.4f}")

# TTARArec的forward方法
print("5.2 TTARArec的forward方法")
with torch.no_grad():
    ttararec_forward_output = ttararec_model.forward(item_seq, item_seq_len)
print(f"TTARArec forward输出形状: {ttararec_forward_output.shape}")
print(f"TTARArec forward输出范围: {ttararec_forward_output.min().item():.4f} ~ {ttararec_forward_output.max().item():.4f}")

# 检查forward输出是否一致
forward_diff = torch.abs(direct_output - ttararec_forward_output).max().item()
print(f"Forward输出差异: {forward_diff:.8f}")

print("\n=== 6. 检查预训练模型参数差异 ===")
param_diffs = []
for (name1, param1), (name2, param2) in zip(direct_model.named_parameters(), ttararec_model.pretrained_model.named_parameters()):
    if name1 == name2:
        diff = torch.abs(param1 - param2).max().item()
        if diff > 1e-8:
            param_diffs.append((name1, diff))

print(f"发现 {len(param_diffs)} 个参数有差异:")
for name, diff in param_diffs[:10]:  # 只显示前10个
    print(f"  {name}: {diff:.8f}")

print("\n=== 7. 检查检索器编码器 ===")
print("7.1 检索器前向传播")
with torch.no_grad():
    retriever_output = ttararec_model.retriever_forward(ttararec_forward_output)
print(f"检索器输出形状: {retriever_output.shape}")
print(f"检索器输出范围: {retriever_output.min().item():.4f} ~ {retriever_output.max().item():.4f}")

# 检查检索器是否改变了序列表示
retriever_diff = torch.abs(ttararec_forward_output - retriever_output).max().item()
print(f"检索器前后差异: {retriever_diff:.8f}")

print("\n=== 8. 检查物品嵌入 ===")
test_items = torch.arange(1, 101).cuda()  # 测试前100个物品

print("8.1 直接模型物品嵌入")
with torch.no_grad():
    direct_item_emb = direct_model.item_embedding(test_items)
print(f"直接模型物品嵌入形状: {direct_item_emb.shape}")
print(f"直接模型物品嵌入范围: {direct_item_emb.min().item():.4f} ~ {direct_item_emb.max().item():.4f}")

print("8.2 TTARArec物品嵌入")
with torch.no_grad():
    ttararec_item_emb = ttararec_model.get_item_embedding(test_items)
print(f"TTARArec物品嵌入形状: {ttararec_item_emb.shape}")
print(f"TTARArec物品嵌入范围: {ttararec_item_emb.min().item():.4f} ~ {ttararec_item_emb.max().item():.4f}")

# 检查物品嵌入是否一致
item_emb_diff = torch.abs(direct_item_emb - ttararec_item_emb).max().item()
print(f"物品嵌入差异: {item_emb_diff:.8f}")

print("\n=== 9. 计算推荐得分 ===")
print("9.1 直接模型推荐得分")
with torch.no_grad():
    # 使用直接模型计算得分
    direct_scores = torch.matmul(direct_output, direct_item_emb.transpose(0, 1))
print(f"直接模型得分形状: {direct_scores.shape}")
print(f"直接模型得分范围: {direct_scores.min().item():.4f} ~ {direct_scores.max().item():.4f}")

print("9.2 TTARArec推荐得分（无检索增强）")
with torch.no_grad():
    # 使用TTARArec但不进行检索增强
    ttararec_scores_no_aug = torch.matmul(ttararec_forward_output, ttararec_item_emb.transpose(0, 1))
print(f"TTARArec得分（无增强）形状: {ttararec_scores_no_aug.shape}")
print(f"TTARArec得分（无增强）范围: {ttararec_scores_no_aug.min().item():.4f} ~ {ttararec_scores_no_aug.max().item():.4f}")

# 检查得分是否一致
scores_diff = torch.abs(direct_scores - ttararec_scores_no_aug).max().item()
print(f"得分差异（无增强）: {scores_diff:.8f}")

print("\n=== 总结 ===")
if forward_diff < 1e-6 and item_emb_diff < 1e-6 and scores_diff < 1e-6:
    print("✓ TTARArec的基础功能与直接模型完全一致")
    print("问题可能出现在检索增强逻辑中")
else:
    print("✗ TTARArec的基础功能与直接模型不一致")
    print(f"  Forward差异: {forward_diff:.8f}")
    print(f"  物品嵌入差异: {item_emb_diff:.8f}")
    print(f"  得分差异: {scores_diff:.8f}")
    print("问题出现在基础模型加载或前向传播中") 