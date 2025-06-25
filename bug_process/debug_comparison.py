import sys
import torch
sys.path.append('.')

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender.ttararec import TTARArec
from recbole.model.sequential_recommender.duorec import DuoRec
from recbole.trainer import Trainer
from recbole.utils import init_logger, init_seed

def debug_model_comparison():
    """对比TTARArec和DuoRec的输出差异"""
    
    # 配置
    config = Config(model='TTARArec', dataset='Amazon_Beauty', config_file_list=['ttararec_config.yaml'])
    init_seed(config['seed'], config['reproducibility'])
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # 创建TTARArec模型
    ttararec_model = TTARArec(config, train_data.dataset).to('cuda')
    
    # 加载原始DuoRec模型
    duorec_checkpoint = torch.load(config['pretrained_model_path'], 
                                  map_location='cuda', weights_only=False)
    
    # 创建原始DuoRec模型
    if 'config' in duorec_checkpoint:
        duorec_config = duorec_checkpoint['config']
        # 合并必要参数
        for key in ['n_layers', 'n_heads', 'hidden_size', 'inner_size', 
                   'hidden_dropout_prob', 'attn_dropout_prob', 'hidden_act',
                   'layer_norm_eps', 'initializer_range', 'loss_type',
                   'lmd', 'lmd_sem', 'contrast', 'tau', 'sim']:
            if key in duorec_config:
                config[key] = duorec_config[key]
    
    duorec_model = DuoRec(config, train_data.dataset).to('cuda')
    if 'state_dict' in duorec_checkpoint:
        duorec_model.load_state_dict(duorec_checkpoint['state_dict'])
    
    duorec_model.eval()
    ttararec_model.eval()
    
    # 测试相同输入的输出
    print("=== 模型对比诊断 ===")
    
    # 获取一个batch的测试数据
    test_iter = iter(test_data)
    batched_data = next(test_iter)
    interaction, history_index, swap_row, swap_col_after, swap_col_before = batched_data
    interaction = interaction.to('cuda')
    
    with torch.no_grad():
        # TTARArec预训练模型输出
        ttararec_seq_output = ttararec_model.pretrained_model.forward(
            interaction[ttararec_model.ITEM_SEQ], 
            interaction[ttararec_model.ITEM_SEQ_LEN]
        )
        
        # 原始DuoRec输出  
        duorec_seq_output = duorec_model.forward(
            interaction[duorec_model.ITEM_SEQ], 
            interaction[duorec_model.ITEM_SEQ_LEN]
        )
        
        print(f"TTARArec预训练模型seq_output范围: {ttararec_seq_output.min():.4f} ~ {ttararec_seq_output.max():.4f}")
        print(f"原始DuoRec seq_output范围: {duorec_seq_output.min():.4f} ~ {duorec_seq_output.max():.4f}")
        print(f"输出差异(最大绝对差): {(ttararec_seq_output - duorec_seq_output).abs().max():.6f}")
        
        # 对比物品嵌入
        ttararec_item_emb = ttararec_model.pretrained_model.item_embedding.weight
        duorec_item_emb = duorec_model.item_embedding.weight
        
        print(f"TTARArec物品嵌入范围: {ttararec_item_emb.min():.4f} ~ {ttararec_item_emb.max():.4f}")
        print(f"原始DuoRec物品嵌入范围: {duorec_item_emb.min():.4f} ~ {duorec_item_emb.max():.4f}")
        print(f"物品嵌入差异(最大绝对差): {(ttararec_item_emb - duorec_item_emb).abs().max():.6f}")
        
        # 对比full_sort_predict结果
        ttararec_scores = torch.matmul(ttararec_seq_output, ttararec_item_emb.transpose(0, 1))
        duorec_scores = duorec_model.full_sort_predict(interaction)
        
        print(f"TTARArec计算得分范围: {ttararec_scores.min():.4f} ~ {ttararec_scores.max():.4f}")
        print(f"原始DuoRec得分范围: {duorec_scores.min():.4f} ~ {duorec_scores.max():.4f}")
        print(f"得分差异(最大绝对差): {(ttararec_scores - duorec_scores).abs().max():.6f}")
        
        # 检查是否完全一致
        if torch.allclose(ttararec_scores, duorec_scores, atol=1e-5):
            print("✓ TTARArec和DuoRec的得分计算完全一致!")
        else:
            print("✗ TTARArec和DuoRec的得分计算存在差异!")
            
        # 检查模型参数是否一致
        ttararec_params = sum(p.numel() for p in ttararec_model.pretrained_model.parameters())
        duorec_params = sum(p.numel() for p in duorec_model.parameters())
        print(f"TTARArec预训练模型参数数量: {ttararec_params}")
        print(f"原始DuoRec参数数量: {duorec_params}")
        
        # 检查具体的权重差异
        print("\n=== 详细权重对比 ===")
        ttararec_state_dict = ttararec_model.pretrained_model.state_dict()
        duorec_state_dict = duorec_model.state_dict()
        
        max_diff = 0
        diff_layers = []
        for key in duorec_state_dict.keys():
            if key in ttararec_state_dict:
                diff = (ttararec_state_dict[key] - duorec_state_dict[key]).abs().max().item()
                if diff > 1e-6:
                    diff_layers.append((key, diff))
                    max_diff = max(max_diff, diff)
                    
        if diff_layers:
            print(f"发现 {len(diff_layers)} 个层有权重差异:")
            for layer, diff in diff_layers[:5]:  # 只显示前5个
                print(f"  {layer}: 最大差异 {diff:.6f}")
            if len(diff_layers) > 5:
                print(f"  ... 还有 {len(diff_layers)-5} 个层")
            print(f"总体最大权重差异: {max_diff:.6f}")
        else:
            print("✓ 所有权重完全一致!")
            
        # 检查模型状态
        print(f"TTARArec预训练模型training状态: {ttararec_model.pretrained_model.training}")
        print(f"原始DuoRec training状态: {duorec_model.training}")

if __name__ == "__main__":
    debug_model_comparison() 