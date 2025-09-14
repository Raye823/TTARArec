import argparse
import torch
import logging
from logging import getLogger
from typing import Dict, List, Optional
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, get_model, get_trainer, init_seed
from recbole.utils.utils import set_color
from recbole.utils.ttararec_diagnostics import get_attention_grad_norms


def print_grad_norms(
    model,
    batch_idx: int,
    every: int = 129,
    names_of_interest: Optional[List[str]] = None,
) -> None:
    """在 backward() 后、optimizer.step() 前调用，按频率打印梯度范数。"""
    if (batch_idx % every) != 0:
        return

    norms: Dict[str, float] = {}
    norms = get_attention_grad_norms(model)
    # 兜底：遍历参数梯度
    if not norms:
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            norms[name] = p.grad.data.norm().item()

    # 可选过滤
    if names_of_interest:
        norms = {k: v for k, v in norms.items() if any(tag in k for tag in names_of_interest)}

    print(f"Batch {batch_idx} 梯度范数:")
    for name, norm in norms.items():
        print(f"  {name}: {norm:.6f}")
    print()


class GradNormTrainer:
    """梯度范数打印的Trainer混入类"""
    
    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        """重写训练epoch方法，添加梯度范数打印"""
        from tqdm import tqdm
        
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                ncols=100,
                desc=f'Training epoch {epoch_idx}'
            ) if show_progress else train_data
        )
        
        for batch_idx, interaction in enumerate(iter_data):
            interaction = interaction.to(self.device)
            self.optimizer.zero_grad()
            losses = loss_func(interaction)
            
            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            
            self._check_nan(loss)
            loss.backward()
            
            # 添加梯度范数打印
            print_grad_norms(self.model, batch_idx, every=129)
            
            if self.clip_grad_norm:
                from torch.nn.utils.clip_grad import clip_grad_norm_
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            
            self.optimizer.step()
            
        return total_loss


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
    
    # 配置TTARArec专用日志器，复用当前run的log.txt
    import logging
    tt_logger = logging.getLogger("TTARArec")
    tt_logger.setLevel(logging.INFO)
    tt_logger.propagate = False
    # 从当前logger获取文件路径
    if logger.handlers:
        for h in logger.handlers:
            if hasattr(h, 'baseFilename'):
                fh = logging.FileHandler(h.baseFilename, mode="a", encoding="utf-8")
                fh.setLevel(logging.INFO)
                fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%y-%m-%d %H:%M"))
                tt_logger.addHandler(fh)
                break
    
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
    
    # 为trainer添加梯度范数打印功能
    trainer_class = trainer.__class__
    class CustomTrainer(GradNormTrainer, trainer_class):
        pass
    
    # 使用新的trainer类创建实例
    trainer = CustomTrainer(config, model)
    
    # 在训练前进行一次评估

    valid_score, valid_result = trainer._valid_epoch(valid_data, show_progress=config['show_progress'])
    logger.info(set_color('初始验证结果', 'blue') + f': {valid_score}')
    logger.info(set_color('详细验证结果', 'blue') + f': {valid_result}')
        
    # 测试集评估
    test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=config['show_progress'])
    logger.info(set_color('初始测试结果', 'blue') + f': {test_result}')

    
    # 启用检索增强功能（在初始评估后，训练前）
    logger.info("启用检索增强功能")
    model.enable_retrieval()
    
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
    model.precached_knowledge_val(valid_data)
    test_result = trainer.evaluate(test_data, load_best_model=saved, show_progress=config['show_progress'])
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
