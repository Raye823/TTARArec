# -*- coding: utf-8 -*-
# @Time    : 2025/1/1
# @Author  : Xinping Zhao
# @Email   : zhaoxinping@stu.hit.edu.cn

"""
预训练模型加载器 - 通用版
################################################

用于加载多种预训练模型供TTARArec使用，支持DuoRec、SASRec、GRU4Rec、CL4SRec

使用方法：
1. 在配置文件中设置 pretrained_model_type: 'sasrec'  # 可选: duorec, sasrec, gru4rec, cl4srec
2. 设置 pretrained_model_path: '/path/to/checkpoint.pth'
3. TTARArec会自动根据模型类型加载对应的预训练模型

支持的模型类型及其主要参数：
- duorec: Transformer + 对比学习，参数包括 n_layers, n_heads, hidden_size, lmd, tau 等
- sasrec: 标准 Self-Attention，参数包括 n_layers, n_heads, hidden_size, inner_size 等  
- gru4rec: GRU-based，参数包括 embedding_size, hidden_size, num_layers 等
- cl4srec: Transformer + 对比学习，参数包括 n_layers, n_heads, hidden_size, lmd, tau, sim 等
"""

import torch
from recbole.model.sequential_recommender.duorec import DuoRec
from recbole.model.sequential_recommender.newmodel import NewModel



class PretrainedModelLoader:
    """通用的预训练模型加载器"""
    
    # 模型类映射
    MODEL_CLASSES = {
        'duorec': DuoRec,
        'newmodel': NewModel,

    }
    
    # 每个模型的架构参数
    MODEL_ARCHITECTURE_PARAMS = {
        'duorec': [
            'n_layers', 'n_heads', 'hidden_size', 'inner_size',
            'hidden_dropout_prob', 'attn_dropout_prob', 'hidden_act',
            'layer_norm_eps', 'initializer_range', 'loss_type',
            'lmd', 'lmd_sem', 'contrast', 'tau', 'sim'
        ],
        'newmodel': [
            'n_layers', 'n_heads', 'hidden_size',
            'hidden_dropout_prob', 'norm_first', 'hidden_act',
            'initializer_range'
        ]
    }
    
    @staticmethod
    def load_model(config, dataset, model_type=None):
        """
        通用模型加载方法
        
        Args:
            config: 配置字典
            dataset: 数据集对象
            model_type: 模型类型，如果未指定则从config中获取
            
        Returns:
            加载好的模型实例
        """
        if model_type is None:
            model_type = config['pretrained_model_type'].lower()
        
        if model_type not in PretrainedModelLoader.MODEL_CLASSES:
            raise ValueError(f"不支持的模型类型: {model_type}. 支持的类型: {list(PretrainedModelLoader.MODEL_CLASSES.keys())}")
        
        print(f"正在加载{model_type.upper()}预训练模型...")
        
        model_class = PretrainedModelLoader.MODEL_CLASSES[model_type]
        architecture_params = PretrainedModelLoader.MODEL_ARCHITECTURE_PARAMS[model_type]
        
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
                for param in architecture_params:
                    if param in pretrained_config:
                        config[param] = pretrained_config[param]
            
            # 创建模型实例
            model = model_class(config, dataset)
            
            # 加载预训练权重
            if state_dict:
                model.load_state_dict(state_dict, strict=False)
        else:
            model = model_class(config, dataset)
        
        # 设置为推理模式并冻结参数
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        
        print(f"{model_type.upper()}模型加载成功!")
        
        return model
    
    @staticmethod
    def load_duorec_model(config, dataset):
        """
        加载DuoRec预训练模型（为了保持向后兼容性）
        
        Args:
            config: 配置字典
            dataset: 数据集对象
            
        Returns:
            加载好的DuoRec模型实例
        """
        return PretrainedModelLoader.load_model(config, dataset, 'duorec')
    
    @staticmethod
    def load_sasrec_model(config, dataset):
        """
        加载SASRec预训练模型
        
        Args:
            config: 配置字典
            dataset: 数据集对象
            
        Returns:
            加载好的SASRec模型实例
        """
        return PretrainedModelLoader.load_model(config, dataset, 'sasrec')
    