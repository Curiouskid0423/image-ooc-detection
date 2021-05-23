"""
Trainer file that assembles building blocks.
"""
import os
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, random_split
from torch.optim import Adam, lr_scheduler
from data.dataloader import CustomTensorDataset
from models.cnn_encoder import CNN_AutoEncoder
from models.vae import VAE, loss_vae

class Trainer:

    def __init__(self, num_epochs=50, batch_size=512, lr=1e-3, data_path=None, \
        ckpt_save_path='.', model_type='cnn', schedule=0, gamma=0.5,  train_mode='default'):
        self.train_mode = train_mode
        self.ckpt_save_path = ckpt_save_path
        print(f'Checkpoints will be stored at {ckpt_save_path}')

        """ Hyperparameters """
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.criterion = nn.MSELoss()
        """ Set up DataLoader & Sampler """
        x = torch.from_numpy(data_path)
        self.dataset = CustomTensorDataset(x)

        t_len = int(len(self.dataset) * 0.9)
        v_len = len(self.dataset) - t_len
        train_set, valid_set = random_split(self.dataset, [t_len, v_len])
        print(f'Train set: {len(train_set)} | Validation set: {len(valid_set)}')
        self.train_sampler = RandomSampler(train_set)
        self.valid_sampler = SequentialSampler(valid_set)
        self.train_dataloader = DataLoader(train_set, sampler=self.train_sampler, batch_size=self.batch_size)
        self.valid_dataloader = DataLoader(valid_set, sampler=self.valid_sampler, batch_size=self.batch_size)
        
        """ Select model """
        model_choices = {
            'cnn': CNN_AutoEncoder(),
            'vae': VAE(),
        }
        self.model_type = model_type
        self.model = model_choices[self.model_type].cuda()
        print(f'Training model: {model_type}')
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = None
        self.schedule = schedule
        self.gamma = gamma
        if schedule != 0: # Enable lr_scheduler
            self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.schedule, gamma=self.gamma)
            print(f"Enabled lr_scheduler with step_size={schedule}, gamma={gamma}")

    def train(self):
        best_loss = np.inf
        epoch_loss_record = []

        for epoch in tqdm(range(self.num_epochs)):
            tot_train_loss, tot_valid_loss = [], []

            """ Train set """
            self.model.train()
            for data in self.train_dataloader:
                self.optimizer.zero_grad()

                img = data.float().cuda()
                output = self.model(img)
                loss = None
                if self.model_type in ['vae']:
                    loss = loss_vae(output[0], img, output[1], output[2], self.criterion)
                else:
                    loss = self.criterion(output, img)

                tot_train_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()
            
            """ Validation set """
            self.model.eval()
            for data in self.valid_dataloader:
                img = data.float().cuda()
                with torch.no_grad():
                    output = self.model(img)
                loss = None
                if self.model_type in ['vae']:
                    loss = loss_vae(output[0], img, output[1], output[2], self.criterion)
                else:
                    loss = self.criterion(output, img)
                tot_valid_loss.append(loss.item())

            if self.scheduler is not None:
                self.scheduler.step()

            mean_loss = sum(tot_train_loss) / len(tot_train_loss)
            mean_valid_loss = sum(tot_valid_loss) / len(tot_valid_loss)
            epoch_loss_record.append([mean_loss, mean_valid_loss])
            print(f' Loss :: {mean_loss:.5f} | Val Loss :: {mean_valid_loss:.5f} | Epoch :: {epoch + 1:.0f}/{self.num_epochs:.0f}')


            if mean_loss < best_loss:
                best_loss = mean_loss
                torch.save(self.model, os.path.join(self.ckpt_save_path, 'best_model_{}.ckpt'.format(self.model_type)))
                
            # ===================save_last========================
            if self.train_mode == 'default':
                torch.save(self.model, os.path.join(self.ckpt_save_path, 'last_model_{}.ckpt'.format(self.model_type)))
            else:
                torch.save(self.model, os.path.join(self.ckpt_save_path, f'last_model_{self.batch_size}_{self.lr}_s{self.schedule}.ckpt'))    
        # ==============Save the loss record===================
        with open('loss_history.txt', 'w') as filehandle:
            for train, val in epoch_loss_record:
                filehandle.write(f'{train}, {val}\n')
        
        return best_loss # Return best loss to be used when doing hyperparameter search.

