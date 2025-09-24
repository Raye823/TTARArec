import argparse
from logging import getLogger
from recbole.utils import init_logger, init_seed
from recbole.trainer import Trainer
from recbole.utils import get_model
from recbole.config import Config
from recbole.data import create_dataset, data_preparation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='SASRec', help='model name')
    parser.add_argument('--dataset', type=str, default='Amazon_Beauty', help='dataset name')
    parser.add_argument('--config_files', type=str, default='sasrec.yaml', help='config files')
    parser.add_argument('--saved', action='store_true', help='whether to save model')
    args = parser.parse_args()
    # 默认保存最佳模型（无需手动传 --saved）
    args.saved = True

    # config initialization
    config = Config(
        model='SASRec', 
        dataset=args.dataset,
        config_file_list=[args.config_files] if args.config_files else None
    )
    
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()

    logger.info(config)

    # ensure log_dir matches the actual log file directory
    if ('log_dir' not in config) or (config['log_dir'] is None):
        log_dir = None
        for h in logger.handlers:
            if hasattr(h, 'baseFilename'):
                import os
                log_dir = os.path.dirname(h.baseFilename)
                break
        if log_dir is None:
            log_dir = f"log/{args.model}"
        config['log_dir'] = log_dir

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization via registry
    model_cls = get_model('SASRec')
    model = model_cls(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = Trainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=args.saved, show_progress=config['show_progress']
    )

    # model evaluation
    test_result = trainer.evaluate(
        test_data, load_best_model=args.saved, show_progress=config['show_progress']
    )

    logger.info('best valid result: {}'.format(best_valid_result))
    logger.info('test result: {}'.format(test_result))


if __name__ == '__main__':
    main()
