import argparse
import torch
import logging
from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, get_model, get_trainer, init_seed
from recbole.utils.utils import set_color


def run_ttararec(model=None, dataset=None, config_file_list=None, config_dict=None, saved=True):
    """运行TTARArec模型
    
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
    
    logger.info("="*50)
    logger.info("TTARArec 检索增强推荐模型")
    logger.info("="*50)
    logger.info(config)
    
    # 数据集过滤
    dataset = create_dataset(config)
    logger.info(dataset)
    
    # 数据集分割
    train_data, valid_data, test_data = data_preparation(config, dataset)
    
    # 模型加载和初始化
    logger.info("正在初始化TTARArec模型...")
    model = get_model(config['model'])(config, train_data).to(config['device'])
    
    logger.info(f"预训练模型类型: {config['pretrained_model_type']}")
    logger.info(f"预训练模型路径: {config['pretrained_model_path']}")
    logger.info(f"检索参数 - alpha: {config['alpha']}, top_k: {config['top_k']}")
    logger.info(f"检索器编码器层数: {config['retriever_layers']}")
    logger.info(f"KL散度损失权重: {config['kl_weight']}")
    
    logger.info(model)
    
    # 初始化TTARArec的知识库
    logger.info("正在构建TTARArec检索知识库...")
    try:
        model.precached_knowledge()
        logger.info("检索知识库构建完成!")
    except Exception as e:
        logger.error(f"构建检索知识库时出错: {e}")
        raise e
    
    # 加载trainer
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    
    # 在训练前进行一次评估
    try:
        valid_score, valid_result = trainer._valid_epoch(valid_data, show_progress=config['show_progress'])
        logger.info(set_color('初始验证结果', 'blue') + f': {valid_score}')
        logger.info(set_color('详细验证结果', 'blue') + f': {valid_result}')
        
        # 测试集评估
        test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=config['show_progress'])
        logger.info(set_color('初始测试结果', 'blue') + f': {test_result}')
    except Exception as e:
        logger.warning(f"初始评估失败: {e}")
        logger.info("跳过初始评估，直接开始训练...")
    
    # 开始训练
    logger.info("="*30)
    logger.info("开始训练TTARArec")
    logger.info("="*30)
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=saved, show_progress=config['show_progress']
    )
    
    # 训练结束后，使用最佳模型进行测试集评估
    logger.info("="*30)
    logger.info("最佳模型评估")
    logger.info("="*30)
    test_result = trainer.evaluate(test_data, load_best_model=saved, show_progress=config['show_progress'])
    
    logger.info("="*50)
    logger.info("TTARArec 训练完成!")
    logger.info("="*50)
    logger.info(set_color('最佳验证结果', 'green') + f': {best_valid_result}')
    logger.info(set_color('最终测试结果', 'green') + f': {test_result}')
    
    return {
        'best_valid_score': best_valid_score,
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='Amazon_Beauty', help='数据集')
    parser.add_argument('--pretrained_model_path', type=str, required=True, help='预训练模型路径')
    
    args = parser.parse_args()
    
    # 简单配置
    config_dict = {
        'pretrained_model_path': args.pretrained_model_path,
    }
    
    run_ttararec(
        model='TTARArec', 
        dataset=args.dataset, 
        config_file_list=['ttararec_config.yaml'], 
        config_dict=config_dict
    )