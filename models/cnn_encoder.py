'''
Convolutional Autoencoder model.
'''
import torch
from torch import nn

class CNN_AutoEncoder(nn.Module):
    def __init__(self):
        super(CNN_AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            
            nn.Conv2d(3, 12, 4, stride=2, padding=1),  
            nn.BatchNorm2d(12), 
            nn.ReLU(),

            nn.Conv2d(12, 24, 4, stride=2, padding=1),  
            nn.BatchNorm2d(24),       
            nn.ReLU(),

			nn.Conv2d(24, 48, 4, stride=2, padding=1), 
            nn.BatchNorm2d(48),         
            nn.ReLU(),

        )
        self.decoder = nn.Sequential(

			nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1), 
            nn.BatchNorm2d(24), 
            nn.ReLU(),

			nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),
            nn.BatchNorm2d(12), 
            nn.ReLU(),

            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x