# -*- coding: utf-8 -*-
# @Time    : 2025/1/1
# @Author  : Xinping Zhao
# @Email   : zhaoxinping@stu.hit.edu.cn

"""
预训练模型加载器 - 简化版
################################################

用于加载DuoRec预训练模型供TTARArec使用
"""

import torch
from recbole.model.sequential_recommender.duorec import DuoRec


class PretrainedModelLoader:
    """简化的预训练模型加载器"""
    
    @staticmethod
    def load_duorec_model(config, dataset):
        """
        加载DuoRec预训练模型
        
        Args:
            config: 配置字典
            dataset: 数据集对象
            
        Returns:
            加载好的DuoRec模型实例
        """
        print("正在加载DuoRec预训练模型...")
        
        # 加载预训练权重和配置
        model_path = config['pretrained_model_path']
        if model_path:
            checkpoint = torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu', weights_only=False)
            
            # 提取预训练模型的配置
            pretrained_config = None
            state_dict = None
            
            if isinstance(checkpoint, dict):
                if 'config' in checkpoint:
                    pretrained_config = checkpoint['config']
                
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # 合并配置：使用预训练模型的架构配置
            if pretrained_config:
                architecture_params = [
                    'n_layers', 'n_heads', 'hidden_size', 'inner_size',
                    'hidden_dropout_prob', 'attn_dropout_prob', 'hidden_act',
                    'layer_norm_eps', 'initializer_range', 'loss_type',
                    'lmd', 'lmd_sem', 'contrast', 'tau', 'sim'
                ]
                
                for param in architecture_params:
                    if param in pretrained_config:
                        config[param] = pretrained_config[param]
            
            # 创建DuoRec模型实例
            model = DuoRec(config, dataset)
            
            # 加载预训练权重
            if state_dict:
                model.load_state_dict(state_dict, strict=False)
        else:
            model = DuoRec(config, dataset)
        
        # 设置为推理模式并冻结参数
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        
        print(f"DuoRec模型加载成功!")
        
        return model 