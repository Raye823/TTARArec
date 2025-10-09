# -*- coding: utf-8 -*-
# @Time    : 2025/1/10
# @Author  : Xinping Zhao

"""
TTARArec utility functions for evaluation and inference
"""

import torch
from logging import getLogger


def eval_ttararec(model_name='TTARArec', dataset=None, config_file_list=None, 
                  config_dict=None, model_path=None):
    """
    Evaluate a trained TTARArec model
    
    Args:
        model_name (str): Model name, default 'TTARArec'
        dataset (str): Dataset name (optional, can be loaded from checkpoint)
        config_file_list (list): Configuration file list (optional)
        config_dict (dict): Configuration dictionary (optional, will override checkpoint config)
        model_path (str): Path to the trained model checkpoint
    
    Returns:
        dict: Evaluation results including validation and test metrics
    
    Note:
        If dataset/config are not provided, they will be loaded from the checkpoint.
        The checkpoint contains the complete config used during training.
    """
    # Import here to avoid circular import
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.utils.logger import init_logger
    from recbole.utils.utils import get_model, get_trainer, set_color, init_seed
    
    # Load checkpoint to get saved config
    logger_temp = getLogger()
    logger_temp.info(f"Loading checkpoint from: {model_path}")
    checkpoint = torch.load(model_path)
    saved_config = checkpoint['config']
    
    # Use saved config as base, override with provided config_dict if any
    if config_dict is None:
        config_dict = {}
    
    # Merge: saved config as base, user provided config_dict overrides it
    final_config_dict = dict(saved_config.final_config_dict)
    
    # Remove dataset_path from saved config to avoid path conflicts
    # Let Config regenerate the correct path based on dataset name
    if 'data_path' in final_config_dict:
        del final_config_dict['data_path']
    if 'dataset_path' in final_config_dict:
        del final_config_dict['dataset_path']
    
    final_config_dict.update(config_dict)
    
    # Use dataset from checkpoint if not provided
    if dataset is None:
        dataset = saved_config['dataset']
    
    # Create config
    config = Config(model=model_name, dataset=dataset, 
                   config_file_list=config_file_list, config_dict=final_config_dict)
    init_logger(config)
    logger = getLogger()
    
    # Initialize random seed for reproducibility
    init_seed(config['seed'], config['reproducibility'])
    logger.info(f"Random seed set to {config['seed']} with reproducibility={config['reproducibility']}")
    
    logger.info("="*50)
    logger.info("TTARArec Model Evaluation Mode")
    logger.info("="*50)
    logger.info(f"Loaded config from checkpoint: {model_path}")
    logger.info(f"Pretrained model type: {config['pretrained_model_type']}")
    logger.info(f"Pretrained model path: {config['pretrained_model_path']}")
    logger.info(config)
    
    # Load dataset
    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # Initialize model
    logger.info("Initializing TTARArec model...")
    model = get_model(config['model'])(config, train_data).to(config['device'])
    
    # Load trained model checkpoint
    logger.info("Loading model state dict from checkpoint...")
    model.load_state_dict(checkpoint['state_dict'])
    logger.info("Model loaded successfully!")
    
    # Build collaborative knowledge base
    logger.info("Building collaborative knowledge base...")
    model.build_collaborative_knowledge()
    logger.info("Collaborative knowledge base built successfully!")
    
    # Initialize trainer
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    
    # Evaluate original model (without retrieval augmentation)
    logger.info("="*50)
    logger.info("Original Model Evaluation (without retrieval)")
    logger.info("="*50)
    original_test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=config['show_progress'])
    logger.info(set_color('Original model test result', 'yellow') + f': {original_test_result}')
    
    # Enable retrieval augmentation
    logger.info("="*50)
    logger.info("Enabling retrieval augmentation...")
    logger.info("="*50)
    model.enable_retrieval()
    
    # Build knowledge base with validation data
    logger.info("Building knowledge base with validation data...")
    model.build_collaborative_knowledge_val(valid_data)
    
    # Evaluate with retrieval augmentation
    logger.info("="*50)
    logger.info("Retrieval-Augmented Model Evaluation")
    logger.info("="*50)
    augmented_test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=config['show_progress'])
    logger.info(set_color('Retrieval-augmented test result', 'green') + f': {augmented_test_result}')
    
    return {
        'original_test_result': original_test_result,
        'augmented_test_result': augmented_test_result
    }

