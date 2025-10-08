# -*- coding: utf-8 -*-
# @Time    : 2025/1/1
# @Author  : Xinping Zhao
# @Email   : zhaoxinping@stu.hit.edu.cn


import torch
from recbole.model.sequential_recommender.duorec import DuoRec
from recbole.model.sequential_recommender.sasrec import SASRec
from recbole.model.sequential_recommender.bert4rec import BERT4Rec
from recbole.model.sequential_recommender.gru4rec import GRU4Rec
from torch import nn

class PretrainedModelLoader:
    
    MODEL_CLASSES = {
        'duorec': DuoRec,
        'sasrec': SASRec,
    }
    
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
        ]
    }
    
    @staticmethod
    def load_model(config, dataset, model_type=None):
        if model_type is None:
            model_type = config['pretrained_model_type'].lower()
        
        if model_type not in PretrainedModelLoader.MODEL_CLASSES:
            raise ValueError(f"Unsupported model type: {model_type}. Supported types: {list(PretrainedModelLoader.MODEL_CLASSES.keys())}")
        
        model_class = PretrainedModelLoader.MODEL_CLASSES[model_type]
        architecture_params = PretrainedModelLoader.MODEL_ARCHITECTURE_PARAMS[model_type]

        model_path = config['pretrained_model_path']
        if model_path:
            checkpoint = torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu', weights_only=False)

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
            
            if pretrained_config:
                for param in architecture_params:
                    if param in pretrained_config:
                        config[param] = pretrained_config[param]
            
            model = model_class(config, dataset)

            device = config['device']
            model = model.to(device)
            
            if state_dict:
                model.load_state_dict(state_dict, strict=False)
        else:
            model = model_class(config, dataset)
            device = config['device']
            model = model.to(device)
        
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        
        print(f"{model_type.upper()} pretrained model loaded!")
        
        return model
    
    @staticmethod
    def load_duorec_model(config, dataset):
        return PretrainedModelLoader.load_model(config, dataset, 'duorec')
    
    @staticmethod
    def load_sasrec_model(config, dataset):
        return PretrainedModelLoader.load_model(config, dataset, 'sasrec')
