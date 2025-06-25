import torch

checkpoint_path = './log/DuoRec/Amazon_Beauty/bs1024-lmd0.1-sem0.1-us_x-Mar-19-2025_21-16-57-lr0.001-l20-tau1-dot-DPh0.5-DPa0.5/model.pth'

print("=== 检查checkpoint内容 ===")
try:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"Checkpoint类型: {type(checkpoint)}")
    
    if isinstance(checkpoint, dict):
        print(f"Checkpoint包含键: {list(checkpoint.keys())}")
        
        # 检查state_dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print(f"\nstate_dict包含 {len(state_dict)} 个参数:")
            for i, (name, tensor) in enumerate(state_dict.items()):
                print(f"  {i+1:2d}. {name} : {tensor.shape} (范围: {tensor.min().item():.4f} ~ {tensor.max().item():.4f})")
        elif isinstance(checkpoint, dict):
            # 直接是参数字典
            print(f"\n直接参数字典包含 {len(checkpoint)} 个参数:")
            for i, (name, tensor) in enumerate(checkpoint.items()):
                if isinstance(tensor, torch.Tensor):
                    print(f"  {i+1:2d}. {name} : {tensor.shape} (范围: {tensor.min().item():.4f} ~ {tensor.max().item():.4f})")
                else:
                    print(f"  {i+1:2d}. {name} : {type(tensor)} = {tensor}")
    else:
        print("Checkpoint不是字典类型")
        
except Exception as e:
    print(f"加载checkpoint失败: {e}")

print("\n=== 检查DuoRec模型参数名称 ===")
from recbole.model.sequential_recommender.duorec import DuoRec

# 模拟配置
class MockConfig:
    def __init__(self):
        self.config_dict = {
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
        }
    
    def __contains__(self, key):
        return key in self.config_dict
    
    def __getitem__(self, key):
        return self.config_dict[key]
    
    def __setitem__(self, key, value):
        self.config_dict[key] = value

class MockDataset:
    def __init__(self):
        self.num_items = 12102
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

try:
    duorec_model = DuoRec(config, dataset)
    print(f"\nDuoRec模型参数:")
    for i, (name, param) in enumerate(duorec_model.named_parameters()):
        print(f"  {i+1:2d}. {name} : {param.shape}")
        
except Exception as e:
    print(f"创建DuoRec模型失败: {e}") 