import torch
import numpy as np
from recbole.model.sequential_recommender.ttararec import TTARArec
from recbole.model.sequential_recommender.pretrained_model_loader import PretrainedModelLoader

# 模拟配置对象
class MockConfig:
    def __init__(self):
        self.config_dict = {
            'model': 'TTARArec',
            'dataset': 'Amazon_Beauty',
            'USER_ID_FIELD': 'user_id',
            'ITEM_ID_FIELD': 'item_id',
            'TIME_FIELD': 'timestamp',
            'LIST_SUFFIX': '_list',
            'ITEM_LIST_LENGTH_FIELD': 'length',
            'MAX_ITEM_LIST_LENGTH': 50,
            'NEG_PREFIX': 'neg_',
            'LABEL_FIELD': 'label',
            'threshold': {'rating': 4},
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
            'pretrained_model_path': './log/DuoRec/Amazon_Beauty/bs1024-lmd0.1-sem0.1-us_x-Mar-19-2025_21-16-57-lr0.001-l20-tau1-dot-DPh0.5-DPa0.5/model.pth',
            'train_batch_size': 256,
            'eval_batch_size': 256,
            'alpha': 0.8,
            'topk': 5,
            'device': 'cuda',
            'gpu_id': 0
        }
    
    def __contains__(self, key):
        return key in self.config_dict
    
    def __getitem__(self, key):
        return self.config_dict[key]
    
    def __setitem__(self, key, value):
        self.config_dict[key] = value

# 模拟数据集对象
class MockDataset:
    def __init__(self):
        self.field2type = {}
        self.field2source = {}
        self.field2id_token = {}
        self.field2token_id = {}
        self.num_items = 12102  # 根据Amazon_Beauty数据集
        self.num_users = 22364
        
    def num(self, field):
        if field == 'item_id':
            return self.num_items
        elif field == 'user_id':
            return self.num_users
        else:
            return 1

config = MockConfig()
dataset = MockDataset()

print("=== 1. 创建独立的预训练模型 ===")
try:
    direct_model = PretrainedModelLoader.load_duorec_model(config, dataset).cuda()
    print("✓ 独立预训练模型创建成功")
    print(f"模型参数数量: {sum(p.numel() for p in direct_model.parameters())}")
except Exception as e:
    print(f"✗ 独立预训练模型创建失败: {e}")
    exit(1)

print("\n=== 2. 创建TTARArec模型 ===")
try:
    ttararec_model = TTARArec(config, dataset).cuda()
    print("✓ TTARArec模型创建成功")
    print(f"模型参数数量: {sum(p.numel() for p in ttararec_model.parameters())}")
except Exception as e:
    print(f"✗ TTARArec模型创建失败: {e}")
    exit(1)

print("\n=== 3. 创建测试数据 ===")
batch_size = 4
seq_len = 20
max_item_id = 100

# 创建模拟的序列数据
item_seq = torch.randint(1, max_item_id + 1, (batch_size, seq_len)).cuda()
item_seq_len = torch.randint(5, seq_len + 1, (batch_size,)).cuda()

print(f"测试序列形状: {item_seq.shape}")
print(f"序列长度: {item_seq_len}")

print("\n=== 4. 对比前向传播 ===")

# 直接模型前向传播
print("4.1 独立预训练模型前向传播")
with torch.no_grad():
    direct_output = direct_model.forward(item_seq, item_seq_len)
print(f"独立模型输出形状: {direct_output.shape}")
print(f"独立模型输出范围: {direct_output.min().item():.4f} ~ {direct_output.max().item():.4f}")
print(f"独立模型输出均值: {direct_output.mean().item():.4f}")

# TTARArec的forward方法
print("\n4.2 TTARArec的forward方法")
with torch.no_grad():
    ttararec_forward_output = ttararec_model.forward(item_seq, item_seq_len)
print(f"TTARArec forward输出形状: {ttararec_forward_output.shape}")
print(f"TTARArec forward输出范围: {ttararec_forward_output.min().item():.4f} ~ {ttararec_forward_output.max().item():.4f}")
print(f"TTARArec forward输出均值: {ttararec_forward_output.mean().item():.4f}")

# 检查forward输出是否一致
forward_diff = torch.abs(direct_output - ttararec_forward_output).max().item()
print(f"\nForward输出最大差异: {forward_diff:.8f}")
forward_mean_diff = torch.abs(direct_output - ttararec_forward_output).mean().item()
print(f"Forward输出平均差异: {forward_mean_diff:.8f}")

print("\n=== 5. 检查预训练模型参数差异 ===")
param_diffs = []
total_params = 0

for (name1, param1), (name2, param2) in zip(direct_model.named_parameters(), ttararec_model.pretrained_model.named_parameters()):
    total_params += 1
    if name1 == name2:
        diff = torch.abs(param1 - param2).max().item()
        if diff > 1e-8:
            param_diffs.append((name1, diff))

print(f"总共检查了 {total_params} 个参数")
print(f"发现 {len(param_diffs)} 个参数有显著差异 (>1e-8):")
for name, diff in param_diffs[:10]:  # 只显示前10个
    print(f"  {name}: {diff:.8f}")

print("\n=== 6. 检查物品嵌入 ===")
test_items = torch.arange(1, 21).cuda()  # 测试前20个物品

print("6.1 独立模型物品嵌入")
with torch.no_grad():
    direct_item_emb = direct_model.item_embedding(test_items)
print(f"独立模型物品嵌入形状: {direct_item_emb.shape}")
print(f"独立模型物品嵌入范围: {direct_item_emb.min().item():.4f} ~ {direct_item_emb.max().item():.4f}")

print("\n6.2 TTARArec物品嵌入")
with torch.no_grad():
    ttararec_item_emb = ttararec_model.get_item_embedding(test_items)
print(f"TTARArec物品嵌入形状: {ttararec_item_emb.shape}")
print(f"TTARArec物品嵌入范围: {ttararec_item_emb.min().item():.4f} ~ {ttararec_item_emb.max().item():.4f}")

# 检查物品嵌入是否一致
item_emb_diff = torch.abs(direct_item_emb - ttararec_item_emb).max().item()
print(f"\n物品嵌入最大差异: {item_emb_diff:.8f}")

print("\n=== 7. 计算推荐得分对比 ===")
print("7.1 独立模型推荐得分")
with torch.no_grad():
    # 使用独立模型计算得分
    direct_scores = torch.matmul(direct_output, direct_item_emb.transpose(0, 1))
print(f"独立模型得分形状: {direct_scores.shape}")
print(f"独立模型得分范围: {direct_scores.min().item():.4f} ~ {direct_scores.max().item():.4f}")

print("\n7.2 TTARArec推荐得分（无检索增强）")
with torch.no_grad():
    # 使用TTARArec但不进行检索增强
    ttararec_scores_no_aug = torch.matmul(ttararec_forward_output, ttararec_item_emb.transpose(0, 1))
print(f"TTARArec得分（无增强）形状: {ttararec_scores_no_aug.shape}")
print(f"TTARArec得分（无增强）范围: {ttararec_scores_no_aug.min().item():.4f} ~ {ttararec_scores_no_aug.max().item():.4f}")

# 检查得分是否一致
scores_diff = torch.abs(direct_scores - ttararec_scores_no_aug).max().item()
print(f"\n得分最大差异（无增强）: {scores_diff:.8f}")

print("\n=== 8. 检查检索器编码器 ===")
print("8.1 检索器前向传播")
with torch.no_grad():
    retriever_output = ttararec_model.retriever_forward(ttararec_forward_output)
print(f"检索器输出形状: {retriever_output.shape}")
print(f"检索器输出范围: {retriever_output.min().item():.4f} ~ {retriever_output.max().item():.4f}")

# 检查检索器是否改变了序列表示
retriever_diff = torch.abs(ttararec_forward_output - retriever_output).max().item()
print(f"检索器前后最大差异: {retriever_diff:.8f}")

print("\n=== 总结 ===")
if forward_diff < 1e-6 and item_emb_diff < 1e-6 and scores_diff < 1e-6:
    print("✓ TTARArec的基础功能与独立模型完全一致")
    print("问题可能出现在检索增强逻辑中")
elif len(param_diffs) > 0:
    print("✗ 预训练模型参数存在差异")
    print("问题出现在预训练模型加载过程中")
else:
    print("✗ TTARArec的基础功能与独立模型不一致")
    print(f"  Forward差异: {forward_diff:.8f}")
    print(f"  物品嵌入差异: {item_emb_diff:.8f}")
    print(f"  得分差异: {scores_diff:.8f}")
    print("问题出现在模型前向传播过程中") 