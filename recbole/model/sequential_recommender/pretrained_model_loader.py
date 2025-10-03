# -*- coding: utf-8 -*-
# @Time    : 2025/1/1
# @Author  : Xinping Zhao
# @Email   : zhaoxinping@stu.hit.edu.cn

"""
预训练模型加载器 - 通用版
################################################

用于加载多种预训练模型供TTARArec使用，支持DuoRec、SASRec、BERT4Rec、GRU4Rec、CL4SRec

使用方法：
1. 在配置文件中设置 pretrained_model_type: 'sasrec'  # 可选: duorec, sasrec, bert4rec, gru4rec, cl4srec
2. 设置 pretrained_model_path: '/path/to/checkpoint.pth'
3. TTARArec会自动根据模型类型加载对应的预训练模型

支持的模型类型及其主要参数：
- duorec: Transformer + 对比学习，参数包括 n_layers, n_heads, hidden_size, lmd, tau 等
- sasrec: 标准 Self-Attention，参数包括 n_layers, n_heads, hidden_size, inner_size 等  
- bert4rec: 双向Transformer + MLM，参数包括 n_layers, n_heads, hidden_size, mask_ratio, loss_type 等
- gru4rec: GRU-based，参数包括 embedding_size, hidden_size, num_layers 等
- cl4srec: Transformer + 对比学习，参数包括 n_layers, n_heads, hidden_size, lmd, tau, sim 等
"""

import torch
from recbole.model.sequential_recommender.duorec import DuoRec
from recbole.model.sequential_recommender.sasrec import SASRec
from recbole.model.sequential_recommender.bert4rec import BERT4Rec
from torch import nn
import torch


class BERT4RecAdapter(nn.Module):
    """BERT4Rec适配器，使其接口与其他模型一致，并在打印中显示。"""
    
    def __init__(self, bert4rec_model):
        super().__init__()
        # 注册为子模块，便于日志打印
        self.backbone = bert4rec_model
        self.add_module('backbone', self.backbone)

        # 包装 item_embedding，使任何 .weight 访问都只暴露前 n_items（去掉padding），避免外部代码取到含padding的12103列
        class _SlicedEmbeddingView(nn.Module):
            def __init__(self, orig_emb: nn.Embedding, n_items: int):
                super().__init__()
                self.orig = orig_emb
                self.n_items = n_items
            @property
            def weight(self):
                return self.orig.weight[: self.n_items]
            def forward(self, input_ids: torch.Tensor):
                return self.orig(input_ids)
        # 仅当存在属性且尚未被包装时进行替换
        if isinstance(getattr(self.backbone, 'item_embedding', None), nn.Embedding):
            self.backbone.item_embedding = _SlicedEmbeddingView(self.backbone.item_embedding, self.backbone.n_items)
        
        # 为BERT4Rec添加缺失的loss_fct属性
        if not hasattr(self.backbone, 'loss_fct'):
            self.loss_fct = nn.CrossEntropyLoss()
            self.backbone.loss_fct = self.loss_fct
    
    def forward(self, item_seq, item_seq_len=None):
        """适配forward方法，确保与BERT4Rec原生行为一致"""
        # 仅在评估/预测时重构测试数据（训练阶段不重构）
        reconstruct = (not self.training) and (item_seq_len is not None) and hasattr(self.backbone, 'reconstruct_test_data')
        if reconstruct:
            item_seq = self.backbone.reconstruct_test_data(item_seq, item_seq_len)

        output = self.backbone.forward(item_seq)  # [B, L, H] 或 [B, H]

        # 聚合为 [B, H]：若提供了长度，则使用 gather_indexes(item_seq_len-1)
        if item_seq_len is not None and len(output.shape) == 3:
            return self.backbone.gather_indexes(output, item_seq_len - 1)
        if len(output.shape) == 3:
            # 无长度信息时退化为最后位置
            return output[:, -1, :]
        return output
    
    def to(self, device):
        self.backbone = self.backbone.to(device)
        return self
    
    def eval(self):
        self.backbone.eval()
        return self
    
    def full_sort_predict(self, interaction):
        """直接委托给 BERT 原生 full_sort_predict，确保评估通道完全一致。"""
        return self.backbone.full_sort_predict(interaction)
    
    def predict(self, interaction):
        return self.backbone.predict(interaction)
    
    def calculate_loss(self, interaction):
        return self.backbone.calculate_loss(interaction)
    
    def __getattr__(self, name):
        # 将未知属性转发到backbone，保留原有字段（如 hidden_size, item_embedding 等）
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.backbone, name)



class PretrainedModelLoader:
    """通用的预训练模型加载器"""
    
    # 模型类映射
    MODEL_CLASSES = {
        'duorec': DuoRec,
        'sasrec': SASRec,
        'bert4rec': BERT4Rec,
    }
    
    # 每个模型的架构参数
    MODEL_ARCHITECTURE_PARAMS = {
        'duorec': [
            'n_layers', 'n_heads', 'hidden_size', 'inner_size',
            'hidden_dropout_prob', 'attn_dropout_prob', 'hidden_act',
            'layer_norm_eps', 'initializer_range', 'loss_type',
            'lmd', 'lmd_sem', 'contrast', 'tau', 'sim'
        ],
        'sasrec': [
            'n_layers', 'n_heads', 'hidden_size', 'inner_size',
            'hidden_dropout_prob', 'attn_dropout_prob', 'hidden_act',
            'layer_norm_eps', 'initializer_range', 'loss_type'
        ],
        'bert4rec': [
            'n_layers', 'n_heads', 'hidden_size', 'inner_size',
            'hidden_dropout_prob', 'attn_dropout_prob', 'hidden_act',
            'layer_norm_eps', 'initializer_range', 'mask_ratio', 'loss_type'
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
            
            # 移动模型到正确设备
            device = config['device']
            model = model.to(device)
            
            # 加载预训练权重（调试：打印缺失/多余/形状不匹配信息）
            if state_dict:
                model.load_state_dict(state_dict, strict=False)
        else:
            model = model_class(config, dataset)
            # 移动模型到正确设备
            device = config['device']
            model = model.to(device)
        
        # 设置为推理模式并冻结参数
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        
        # 为BERT4Rec添加兼容适配器
        if model_type == 'bert4rec':
            model = BERT4RecAdapter(model)
        
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
    
    @staticmethod
    def load_bert4rec_model(config, dataset):
        """
        加载BERT4Rec预训练模型
        
        Args:
            config: 配置字典
            dataset: 数据集对象
            

        Returns:
            加载好的BERT4Rec模型实例
        """
        return PretrainedModelLoader.load_model(config, dataset, 'bert4rec')
    
