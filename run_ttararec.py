import argparse
from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, get_model, get_trainer, init_seed, eval_ttararec
from recbole.utils.utils import set_color

def run_ttararec(model=None, dataset=None, config_file_list=None, config_dict=None, saved=True):
    config = Config(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
    init_logger(config)
    logger = getLogger()
    import os
    log_dir = os.path.dirname(logger.handlers[0].baseFilename)
    config['log_dir'] = log_dir
    
    logger.info("="*50)
    logger.info("TTARArec Retrieval-Augmented Recommendation Model")
    logger.info("="*50)
    logger.info(config)
    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    logger.info("Initializing TTARArec model...")
    model = get_model(config['model'])(config, train_data).to(config['device'])
    logger.info(f"Pretrained model type: {config['pretrained_model_type']}")
    logger.info(f"Pretrained model path: {config['pretrained_model_path']}")
    logger.info(model)
    logger.info("Building TTARArec collaborative knowledge base...")
    model.build_collaborative_knowledge()
    logger.info("Collaborative knowledge base built successfully!")

    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    valid_score, valid_result = trainer._valid_epoch(valid_data, show_progress=config['show_progress'])
    logger.info(set_color('Detailed validation result', 'blue') + f': {valid_result}')
    test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=config['show_progress'])
    logger.info(set_color('Initial test result', 'blue') + f': {test_result}')
    logger.info("Enabling retrieval augmentation")
    model.enable_retrieval()
    logger.info("="*30)
    logger.info("Start training TTARArec")
    logger.info("="*30)
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=saved, show_progress=config['show_progress']
    )
    logger.info("="*30)
    logger.info("Best model evaluation")
    logger.info("="*30)
    model.build_collaborative_knowledge_val(valid_data)
    test_result = trainer.evaluate(test_data, load_best_model=saved, show_progress=config['show_progress'])
    logger.info(set_color('Best validation result', 'green') + f': {best_valid_result}')
    logger.info(set_color('Final test result', 'green') + f': {test_result}')
    
    return {
        'best_valid_score': best_valid_score,
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default=None, help='Dataset name (optional in eval mode, required in training mode)')
    parser.add_argument('--pretrained_model_path', '-pp', type=str, default=None, help='Pretrained model path (optional in eval mode, required in training mode)')
    parser.add_argument('--pretrained_model_type', '-pt', type=str, default=None, help='Pretrained model type (optional in eval mode, required in training mode)')
    parser.add_argument('--eval', '-e', action='store_true', help='Evaluation mode: load trained model and evaluate')
    parser.add_argument('--model_path', '-mp', type=str, default=None, help='Trained TTARArec model path (required in eval mode)')
    args = parser.parse_args()
    
    if args.eval:
        # Evaluation mode
        if args.model_path is None:
            raise ValueError("--model_path is required in evaluation mode. Please specify the trained model path.")
        
        # Build config_dict only with provided arguments (non-None values)
        config_dict = {}
        if args.pretrained_model_path is not None:
            config_dict['pretrained_model_path'] = args.pretrained_model_path
        if args.pretrained_model_type is not None:
            config_dict['pretrained_model_type'] = args.pretrained_model_type
        
        eval_ttararec(
            model_name='TTARArec',
            dataset=args.dataset,  # Can be None, will be loaded from checkpoint
            config_file_list=['ttararec_config.yaml'],
            config_dict=config_dict if config_dict else None,
            model_path=args.model_path
        )
    else:
        # Training mode - require all parameters
        if args.dataset is None:
            raise ValueError("--dataset is required in training mode.")
        if args.pretrained_model_path is None:
            raise ValueError("--pretrained_model_path is required in training mode.")
        if args.pretrained_model_type is None:
            raise ValueError("--pretrained_model_type is required in training mode.")
        
        config_dict = {
            'pretrained_model_path': args.pretrained_model_path,
            'pretrained_model_type': args.pretrained_model_type
        }
        
        run_ttararec(
            model='TTARArec', 
            dataset=args.dataset, 
            config_file_list=['ttararec_config.yaml'], 
            config_dict=config_dict
        )
