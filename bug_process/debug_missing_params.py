import sys
import torch
sys.path.append('.')

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender.duorec import DuoRec

def debug_missing_params():
    """检查哪些参数没有被checkpoint覆盖"""
    
    config = Config(model='TTARArec', dataset='Amazon_Beauty', config_file_list=['ttararec_config.yaml'])
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    
    # 加载checkpoint
    model_path = config['pretrained_model_path']
    checkpoint = torch.load(model_path, map_location='cuda', weights_only=False)
    
    if 'config' in checkpoint:
        pretrained_config = checkpoint['config']
        for key in ['n_layers', 'n_heads', 'hidden_size', 'inner_size', 
                   'hidden_dropout_prob', 'attn_dropout_prob', 'hidden_act',
                   'layer_norm_eps', 'initializer_range', 'loss_type',
                   'lmd', 'lmd_sem', 'contrast', 'tau', 'sim']:
            try:
                config[key] = pretrained_config[key]
            except KeyError:
                pass
    
    # 创建两个模型
    print("=== 创建模型1 ===")
    model1 = DuoRec(config, train_data.dataset).to('cuda')
    
    print("=== 创建模型2 ===")
    model2 = DuoRec(config, train_data.dataset).to('cuda')
    
    # 检查初始化时的差异
    print("\n=== 初始化差异检查 ===")
    state1_before = model1.state_dict()
    state2_before = model2.state_dict()
    
    init_diff_count = 0
    for key in state1_before.keys():
        diff = (state1_before[key] - state2_before[key]).abs().max().item()
        if diff > 1e-6:
            print(f"初始化差异: {key} -> {diff}")
            init_diff_count += 1
    
    print(f"总共 {init_diff_count} 个参数在初始化时就不同")
    
    # 加载相同权重
    print("\n=== 加载相同checkpoint ===")
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print(f"Checkpoint包含 {len(state_dict)} 个参数")
        print("前10个参数:", list(state_dict.keys())[:10])
        
        missing_keys1, unexpected_keys1 = model1.load_state_dict(state_dict, strict=False)
        missing_keys2, unexpected_keys2 = model2.load_state_dict(state_dict, strict=False)
        
        print(f"模型1缺失参数: {missing_keys1}")
        print(f"模型1意外参数: {unexpected_keys1}")
        print(f"模型2缺失参数: {missing_keys2}")
        print(f"模型2意外参数: {unexpected_keys2}")
    
    # 检查加载后的差异
    print("\n=== 加载后差异检查 ===")
    model1.eval()
    model2.eval()
    
    state1_after = model1.state_dict()
    state2_after = model2.state_dict()
    
    final_diff_count = 0
    for key in state1_after.keys():
        diff = (state1_after[key] - state2_after[key]).abs().max().item()
        if diff > 1e-6:
            print(f"最终差异: {key} -> {diff}")
            final_diff_count += 1
    
    print(f"总共 {final_diff_count} 个参数在加载checkpoint后仍然不同")
    
    # 测试输出差异
    print("\n=== 输出差异测试 ===")
    test_iter = iter(test_data)
    batched_data = next(test_iter)
    interaction, _, _, _, _ = batched_data
    interaction = interaction.to('cuda')
    
    with torch.no_grad():
        output1 = model1.forward(
            interaction[model1.ITEM_SEQ], 
            interaction[model1.ITEM_SEQ_LEN]
        )
        
        output2 = model2.forward(
            interaction[model2.ITEM_SEQ], 
            interaction[model2.ITEM_SEQ_LEN]
        )
        
        print(f"模型1输出范围: {output1.min():.4f} ~ {output1.max():.4f}")
        print(f"模型2输出范围: {output2.min():.4f} ~ {output2.max():.4f}")
        print(f"输出差异: {(output1 - output2).abs().max():.8f}")

if __name__ == "__main__":
    debug_missing_params() 