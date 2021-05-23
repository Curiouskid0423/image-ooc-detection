'''
Out-of-class anomaly detection model.
Main script in enable one-click training on virtual machine.
'''

import numpy as np
import os
import yaml
import argparse
import torch
import random
from models.trainer import Trainer
from inference import inference

def same_seeds(seed):
    # Fix seed for reproducibility (random, numpy, and pytorch)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':

    """ Process config file """
    parser = argparse.ArgumentParser(description="Anomaly Detection Model")
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()
    with open(args.config) as stream:
        conf = yaml.safe_load(stream)
    for key, value in conf['common'].items():
        setattr(args, key, value)

    """ Load in data """
    same_seeds(19530615)
    train = np.load(args.train_data_path, allow_pickle=True)
    test = np.load(args.test_data_path, allow_pickle=True)

    """ Default Training """
    if args.train_mode == 'default':
        print("Runnning default training")
        trainer = Trainer(
            num_epochs=args.num_epochs, batch_size=args.batch_size, lr=args.lr, data_path=train, 
            ckpt_save_path=args.ckpt_save_path, model_type=args.model_type, schedule=args.schedule)
        trainer.train()
    elif args.train_mode == 'hp_search':
        print("Runnning grid search over hyperparams")
        with open('hyperparams.txt', 'w') as hp_file:
            hp_file.write("Loss, batch_size, learning_rate, schedule, gamma\n")
            
        for batch in args.hp['batch_size']:
            for lr in args.hp['lr']:
                for sch in args.hp['schedule']:
                    for g in args.hp['gamma']:
                        trainer = Trainer(
                            num_epochs=args.num_epochs, batch_size=batch, lr=lr, data_path=train,
                            ckpt_save_path=args.ckpt_save_path, model_type=args.model_type, 
                            schedule=sch, gamma=g, train_mode=args.train_mode)
                        curr_loss = trainer.train()
                        print(f'-- Loss: {curr_loss} | batch: {batch}, lr: {lr}, schedule: {sch}, gamma: {g}\n')
                        with open('hyperparams.txt', 'a') as hp_file:
                            hp_file.write(f'{curr_loss}, {batch}, {lr}, {sch}, {g}\n')


    """ Inference """
    ckpt = os.path.join(args.ckpt_save_path, f'last_model_{args.model_type}.ckpt')
    out_file = 'PREDICTION_FILE.csv'
    inference(
        test_path=test, eval_batch_size=200, checkpoint_path=ckpt, 
        out_file=out_file, model_type=args.model_type)
    print(f'Completed training! Inference file stored as {out_file}.')

    
