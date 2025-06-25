import argparse
import torch
import logging
from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, get_model, get_trainer, init_seed
from recbole.utils.utils import set_color


def run_recbole_with_evaluation(model=None, dataset=None, config_file_list=None, config_dict=None, saved=True):
    """运行RecBole，但在训练前先进行一次评估
    
    Args:
        model (str): 模型名称
        dataset (str): 数据集名称
        config_file_list (list): 用于修改实验参数的配置文件列表
        config_dict (dict): 用于修改实验参数的参数字典
        saved (bool): 是否保存模型
    """
    # 配置初始化
    config = Config(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
    
    # 日志初始化
    init_logger(config)
    logger = getLogger()
    
    # 获取日志目录
    import os
    log_dir = os.path.dirname(logger.handlers[0].baseFilename)
    config['log_dir'] = log_dir
    
    logger.info(config)
    
    # 数据集过滤
    dataset = create_dataset(config)
    logger.info(dataset)
    
    # 数据集分割
    train_data, valid_data, test_data = data_preparation(config, dataset)
    
    # 模型加载和初始化
    model = get_model(config['model'])(config, train_data).to(config['device'])
    
    # 如果指定了预训练模型路径，加载预训练模型
    if "pre_training_ckt" in config:
        logger.info(f"加载预训练模型: {config['pre_training_ckt']}")
        checkpoint = torch.load(config['pre_training_ckt'])
        # 使用非严格模式加载，允许缺少新添加组件的权重
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    
    # 如果是RaSeRec模型，初始化相关组件
    if config['model'] == 'RaSeRec':
        logger.info("初始化RaSeRec模型的知识库...")
        model.precached_knowledge()
    
    logger.info(model)
    
    # 加载trainer
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    
    # 在训练前先进行一次评估
    logger.info("在训练前进行评估...")
    valid_score, valid_result = trainer._valid_epoch(valid_data, show_progress=config['show_progress'])
    logger.info(set_color('初始评估结果', 'blue') + f': {valid_score}')
    logger.info(set_color('详细评估结果', 'blue') + f': {valid_result}')
    
    # 测试集评估
    test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=config['show_progress'])
    logger.info(set_color('测试集评估结果', 'blue') + f': {test_result}')
    
    # 开始训练
    logger.info("开始训练...")
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=saved, show_progress=config['show_progress']
    )
    
    # 训练结束后，使用最佳模型进行测试集评估
    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=config['show_progress'])
    
    logger.info(set_color('最佳验证结果', 'blue') + f': {best_valid_result}')
    logger.info(set_color('测试结果', 'blue') + f': {test_result}')
    
    return {
        'best_valid_score': best_valid_score,
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='RaSeRec', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='ml-100k', help='name of datasets')
    parser.add_argument('--config_files', type=str, default='seq.yaml', help='config files')
    parser.add_argument('--pre_training_ckt', type=str, default=None, help='path to pre-trained model checkpoint')

    args, _ = parser.parse_known_args()

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    config_dict = {'pre_training_ckt': args.pre_training_ckt} if args.pre_training_ckt else None
    
    run_recbole_with_evaluation(model=args.model, dataset=args.dataset, config_file_list=config_file_list, config_dict=config_dict)
