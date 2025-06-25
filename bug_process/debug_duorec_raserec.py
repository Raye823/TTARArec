import sys
import torch
sys.path.append('.')

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender.duorec import DuoRec
from recbole.model.sequential_recommender.raserec import RaSeRec
from recbole.trainer import Trainer
from recbole.utils import init_logger, init_seed

def debug_duorec_raserec_comparison():
    """对比DuoRec和RaSeRec的输出差异"""
    
    # 配置
    config = Config(model='DuoRec', dataset='Amazon_Beauty', config_file_list=['ttararec_config.yaml'])
    init_seed(config['seed'], config['reproducibility'])
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

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
    
    # 创建RaSeRec模型（使用相同的预训练权重）
    raserec_model = RaSeRec(config, train_data.dataset).to('cuda')
    # RaSeRec加载相同的DuoRec权重
    if 'state_dict' in duorec_checkpoint:
        # 过滤掉RaSeRec不需要的参数
        state_dict = duorec_checkpoint['state_dict']
        filtered_state_dict = {}
        for key, value in state_dict.items():
            # 只加载基础Transformer和embedding参数
            if any(base_key in key for base_key in ['item_embedding', 'position_embedding', 
                                                   'trm_encoder', 'LayerNorm']):
                filtered_state_dict[key] = value
        
        raserec_model.load_state_dict(filtered_state_dict, strict=False)
    
    raserec_model.eval()
    
    # 测试相同输入的输出
    print("=== DuoRec vs RaSeRec 对比诊断 ===")
    
    # 获取一个batch的测试数据
    test_iter = iter(test_data)
    batched_data = next(test_iter)
    interaction, history_index, swap_row, swap_col_after, swap_col_before = batched_data
    interaction = interaction.to('cuda')
    
    with torch.no_grad():
        # DuoRec输出  
        duorec_seq_output = duorec_model.forward(
            interaction[duorec_model.ITEM_SEQ], 
            interaction[duorec_model.ITEM_SEQ_LEN]
        )
        
        # RaSeRec输出
        raserec_seq_output = raserec_model.forward(
            interaction[raserec_model.ITEM_SEQ], 
            interaction[raserec_model.ITEM_SEQ_LEN]
        )
        
        print(f"DuoRec seq_output范围: {duorec_seq_output.min():.4f} ~ {duorec_seq_output.max():.4f}")
        print(f"RaSeRec seq_output范围: {raserec_seq_output.min():.4f} ~ {raserec_seq_output.max():.4f}")
        print(f"序列输出差异(最大绝对差): {(duorec_seq_output - raserec_seq_output).abs().max():.6f}")
        
        # 对比物品嵌入
        duorec_item_emb = duorec_model.item_embedding.weight
        raserec_item_emb = raserec_model.item_embedding.weight
        
        print(f"DuoRec物品嵌入范围: {duorec_item_emb.min():.4f} ~ {duorec_item_emb.max():.4f}")
        print(f"RaSeRec物品嵌入范围: {raserec_item_emb.min():.4f} ~ {raserec_item_emb.max():.4f}")
        print(f"物品嵌入差异(最大绝对差): {(duorec_item_emb - raserec_item_emb).abs().max():.6f}")
        
        # 对比full_sort_predict结果（不使用检索增强）
        duorec_scores = duorec_model.full_sort_predict(interaction)
        
        # RaSeRec需要先构建知识库才能正常工作，这里我们直接计算基础得分
        raserec_basic_scores = torch.matmul(raserec_seq_output, raserec_item_emb.transpose(0, 1))
        
        print(f"DuoRec得分范围: {duorec_scores.min():.4f} ~ {duorec_scores.max():.4f}")
        print(f"RaSeRec基础得分范围: {raserec_basic_scores.min():.4f} ~ {raserec_basic_scores.max():.4f}")
        print(f"基础得分差异(最大绝对差): {(duorec_scores - raserec_basic_scores).abs().max():.6f}")
        
        # 检查模型参数数量
        duorec_params = sum(p.numel() for p in duorec_model.parameters())
        raserec_params = sum(p.numel() for p in raserec_model.parameters())
        print(f"DuoRec参数数量: {duorec_params}")
        print(f"RaSeRec参数数量: {raserec_params}")
        
        # 现在测试RaSeRec的评估性能
        print("\n=== 评估性能对比 ===")
        
        # 创建trainer
        duorec_trainer = Trainer(config, duorec_model)
        raserec_trainer = Trainer(config, raserec_model)
        
        # 评估DuoRec
        print("评估DuoRec...")
        duorec_result = duorec_trainer.evaluate(test_data, load_best_model=False, show_progress=True)
        print(f"DuoRec Recall@10: {duorec_result['recall@10']:.4f}")
        
        # 构建RaSeRec知识库
        print("构建RaSeRec知识库...")
        raserec_model.build_knowledge_base()
        
        # 评估RaSeRec
        print("评估RaSeRec...")
        raserec_result = raserec_trainer.evaluate(test_data, load_best_model=False, show_progress=True)
        print(f"RaSeRec Recall@10: {raserec_result['recall@10']:.4f}")
        
        print(f"性能差异: {raserec_result['recall@10'] - duorec_result['recall@10']:.4f}")

if __name__ == "__main__":
    debug_duorec_raserec_comparison() 