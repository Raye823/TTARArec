# -*- coding: utf-8 -*-
# @Time    : 2025/1/1
# @Author  : Xinping Zhao
# @Email   : zhaoxinping@stu.hit.edu.cn

"""
RARec 运行脚本
先对预训练模型进行评估，然后启用检索增强再评估
"""

import argparse
import logging
import os
from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.utils import init_logger, get_trainer, init_seed
from recbole.utils.utils import set_color
from recbole.model.sequential_recommender.rarec import RARec


def run_ratest(model=None, dataset=None, config_file_list=None, config_dict=None):
    """运行RARec模型评估"""
    
    # 配置初始化
    config = Config(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])
    
    # 初始化日志
    init_logger(config)
    logger = getLogger()
    
    logger.info(set_color('开始RARec评估', 'green'))
    logger.info(config)
    
    # 创建数据集
    dataset = create_dataset(config)
    logger.info(dataset)
    
    # 数据准备
    train_data, valid_data, test_data = data_preparation(config, dataset)
    
    # 创建模型
    model = RARec(config, train_data.dataset).to(config['device'])
    logger.info(model)
    
    # 创建训练器
    trainer = Trainer(config, model)
    
    # 构建检索知识库
    logger.info(set_color('构建检索知识库', 'yellow'))
    model.precached_knowledge_val(valid_data)
    
    # ========== 第一阶段：预训练模型评估 ==========
    logger.info(set_color('第一阶段：预训练模型评估（未启用检索增强）', 'blue'))
    
    # 验证集评估
    valid_score, valid_result = trainer._valid_epoch(valid_data, show_progress=config['show_progress'])
    logger.info(set_color('预训练模型验证结果', 'blue') + f': {valid_score}')
    logger.info(set_color('详细验证结果', 'blue') + f': {valid_result}')
    
    # 测试集评估
    test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=config['show_progress'])
    logger.info(set_color('预训练模型测试结果', 'blue') + f': {test_result}')
    
    # ========== 第二阶段：检索增强评估 ==========
    logger.info(set_color('第二阶段：启用检索增强功能', 'red'))
    model.enable_retrieval()
    
    # 验证集评估（检索增强）
    valid_score_aug, valid_result_aug = trainer._valid_epoch(valid_data, show_progress=config['show_progress'])
    logger.info(set_color('检索增强验证结果', 'red') + f': {valid_score_aug}')
    logger.info(set_color('详细验证结果', 'red') + f': {valid_result_aug}')
    
    # 测试集评估（检索增强）
    test_result_aug = trainer.evaluate(test_data, load_best_model=False, show_progress=config['show_progress'])
    logger.info(set_color('检索增强测试结果', 'red') + f': {test_result_aug}')
    
    # ========== 结果对比 ==========
    logger.info(set_color('=' * 50, 'green'))
    logger.info(set_color('评估结果对比', 'green'))
    logger.info(set_color('=' * 50, 'green'))
    
    logger.info(set_color('预训练模型测试结果', 'blue') + f': {test_result}')
    logger.info(set_color('检索增强测试结果', 'red') + f': {test_result_aug}')
    
    # 计算性能提升
    for metric in test_result:
        if metric in test_result_aug:
            improvement = test_result_aug[metric] - test_result[metric]
            improvement_pct = (improvement / test_result[metric]) * 100 if test_result[metric] != 0 else 0
            logger.info(set_color(f'{metric} 提升', 'yellow') + f': {improvement:.4f} ({improvement_pct:.2f}%)')
    
    logger.info(set_color('RARec评估完成', 'green'))
    
    return test_result, test_result_aug


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='Amazon_Beauty', help='数据集')
    parser.add_argument('--pretrained_model_path', type=str, required=True, help='预训练模型路径')
    
    args = parser.parse_args()
    
    # 创建日志目录
    log_dir = './log/RARec'
    os.makedirs(log_dir, exist_ok=True)
    
    # 配置参数
    config_dict = {
        'pretrained_model_path': args.pretrained_model_path,
        'log_dir': log_dir,  # 设置日志目录
        'checkpoint_dir': log_dir,  # 设置检查点目录
    }
    
    run_ratest(
        model='RARec', 
        dataset=args.dataset, 
        config_file_list=['ttararec_config.yaml'], 
        config_dict=config_dict
    ) 