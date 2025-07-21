import argparse
import torch
import logging
import wandb
import pandas as pd
import numpy as np
from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, get_model, get_trainer, init_seed
from recbole.utils.utils import set_color


def create_ordered_metrics_tables(best_valid_result, final_test_result):
    """创建有序的指标表格，确保按5、10、20、50顺序排列"""
    # 提取指标类型（hit, ndcg, recall等）
    metric_types = set()
    for key in best_valid_result.keys():
        if '@' in key:
            metric_type = key.split('@')[0]
            metric_types.add(metric_type)
    
    # 创建验证和测试结果表格数据
    valid_data = []
    test_data = []
    
    # 按指标类型分组
    for metric_type in sorted(metric_types):
        # 收集该指标类型的所有K值
        valid_values = {}
        test_values = {}
        
        for k in [5, 10, 20, 50]:
            key = f"{metric_type}@{k}"
            if key in best_valid_result:
                valid_values[k] = best_valid_result[key]
            if key in final_test_result:
                test_values[k] = final_test_result[key]
        
        # 添加到表格数据
        valid_row = {"metric": metric_type}
        valid_row.update({f"K={k}": valid_values.get(k, np.nan) for k in [5, 10, 20, 50]})
        valid_data.append(valid_row)
        
        test_row = {"metric": metric_type}
        test_row.update({f"K={k}": test_values.get(k, np.nan) for k in [5, 10, 20, 50]})
        test_data.append(test_row)
    
    # 创建pandas DataFrame
    valid_df = pd.DataFrame(valid_data)
    test_df = pd.DataFrame(test_data)
    
    return valid_df, test_df


def run_ttararec_with_wandb(model=None, dataset=None, config_file_list=None, config_dict=None, saved=True):
    """运行TTARArec模型并记录到wandb"""
    
    # 初始化wandb - 支持离线模式
    import os
    os.environ["WANDB_MODE"] = "offline"
    
    wandb.init(
        project="ttararec-experiments",
        name=f"{model}_{dataset}",
        config=config_dict,
        tags=[model, dataset]
    )
    
    # RecBole流程
    config = Config(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
    init_logger(config)
    logger = getLogger()
    log_dir = os.path.dirname(logger.handlers[0].baseFilename)
    config['log_dir'] = log_dir
    
    logger.info("="*50)
    logger.info("TTARArec 检索增强推荐模型 - 使用wandb记录")
    logger.info("="*50)
    
    # 数据集和模型初始化
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    model = get_model(config['model'])(config, train_data).to(config['device'])
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    
    # 构建知识库
    logger.info("正在构建TTARArec检索知识库...")
    model.precached_knowledge()
    logger.info("检索知识库构建完成!")
    
    # 启用检索功能
    model.enable_retrieval()
    
    # 定义callback函数记录训练过程
    def training_callback(epoch_idx, valid_score):
        wandb.log({
            "training/epoch": epoch_idx + 1,
            "training/valid_score": valid_score,
        })
        logger.info(f"Epoch {epoch_idx + 1} - Valid Score: {valid_score:.4f}")
    
    # 训练模型
    logger.info("开始训练...")
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=saved, show_progress=config['show_progress'],
        callback_fn=training_callback
    )
    
    # 最终评估
    logger.info("为最终评估重建知识库...")
    model.precached_knowledge_val(valid_data)
    final_test_result = trainer.evaluate(test_data, load_best_model=saved, show_progress=config['show_progress'])
    
    # 记录最佳分数到summary
    wandb.summary.update({"best_valid_score": best_valid_score})
    
    # 创建有序的指标表格
    valid_df, test_df = create_ordered_metrics_tables(best_valid_result, final_test_result)
    
    # 记录表格到wandb
    wandb.log({
        "best_valid_metrics": wandb.Table(dataframe=valid_df),
        "final_test_metrics": wandb.Table(dataframe=test_df)
    })
    
    # 结束wandb记录
    wandb.finish()
    
    # 在日志中显示结果
    logger.info(f"recall@10最佳验证分数: {best_valid_score:.4f}")
    
    logger.info("\n验证结果:")
    for _, row in valid_df.iterrows():
        metric = row['metric']
        values = [f"K={k}: {row[f'K={k}']:.4f}" for k in [5, 10, 20, 50] if f'K={k}' in row and not pd.isna(row[f'K={k}'])]
        logger.info(f"  {metric}: {', '.join(values)}")
    
    logger.info("\n测试结果:")
    for _, row in test_df.iterrows():
        metric = row['metric']
        values = [f"K={k}: {row[f'K={k}']:.4f}" for k in [5, 10, 20, 50] if f'K={k}' in row and not pd.isna(row[f'K={k}'])]
        logger.info(f"  {metric}: {', '.join(values)}")
    
    logger.info("")
    logger.info("实验已记录到本地wandb,上传到网站请运行:wandb sync wandb/offline-run-*")
    
    return {
        'best_valid_score': best_valid_score,
        'best_valid_result': best_valid_result,
        'test_result': final_test_result
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='Amazon_Beauty', help='数据集')
    parser.add_argument('--pretrained_model_path', type=str, required=True, help='预训练模型路径')
    parser.add_argument('--alpha', type=float, default=0.5, help='检索权重alpha')
    parser.add_argument('--top_k', type=int, default=10, help='检索top_k')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--train_batch_size', type=int, default=4096, help='训练批大小')
    parser.add_argument('--fusion_weight', type=float, default=0.8, help='融合权重')
    parser.add_argument('--kl_loss_weight', type=float, default=0.8, help='KL损失权重')
    parser.add_argument('--nprobe', type=int, default=10, help='检索nprobe')
    parser.add_argument('--retriever_layers', type=int, default=1, help='检索层数')
    parser.add_argument('--retriever_dropout', type=float, default=0.3, help='检索dropout')
    parser.add_argument('--retriever_temperature', type=float, default=0.1, help='检索温度')    
    parser.add_argument('--recommendation_temperature', type=float, default=0.1, help='推荐温度')
    parser.add_argument('--fusion_n_heads', type=int, default=1, help='融合头数')
    parser.add_argument('--fusion_inner_size', type=int, default=256, help='融合内部维度')
    parser.add_argument('--fusion_dropout_prob', type=float, default=0.3, help='融合dropout')
    parser.add_argument('--fusion_layer_norm_eps', type=float, default=1e-12, help='融合layer norm epsilon')
    parser.add_argument('--attn_tau', type=float, default=0.5, help='注意力温度')
    
    args = parser.parse_args()
    
    # 配置超参数
    config_dict = {
        'pretrained_model_path': args.pretrained_model_path,
        'alpha': args.alpha,
        'top_k': args.top_k,
        'learning_rate': args.learning_rate,
        'train_batch_size': args.train_batch_size,
        'nprobe': args.nprobe,
        'retriever_layers': args.retriever_layers,
        'retriever_dropout': args.retriever_dropout,
        'retriever_temperature':args.retriever_temperature,       
        'recommendation_temperature':args.recommendation_temperature,  
        'fusion_n_heads':args.fusion_n_heads,               
        'fusion_inner_size': args.fusion_inner_size,          
        'fusion_dropout_prob': args.fusion_dropout_prob,        
        'fusion_layer_norm_eps': args.fusion_layer_norm_eps,    
        'attn_tau': args.attn_tau,                   
        'kl_loss_weight': args.kl_loss_weight,             
        'fusion_weight': args.fusion_weight,
    }
    
    print(f"开始实验，超参数: {config_dict}")
    
    run_ttararec_with_wandb(
        model='TTARArec', 
        dataset=args.dataset, 
        config_file_list=['ttararec_config.yaml'], 
        config_dict=config_dict
    ) 