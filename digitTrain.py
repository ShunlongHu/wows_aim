# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 21:37:28 2022

@author: HU
"""

import pytorch_lightning as pl
import torch
from torch import nn
import pickle
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from argparse import ArgumentParser
import os

class MyModule(nn.Module):
    def __init__(self, act_fn, input_size=11*7*3, num_classes=11, hidden_sizes=[256,64,16]):
        """
        Args:
            act_fn: Object of the activation function that should be used as non-linearity in the network.
            input_size: Size of the input images in pixels
            num_classes: Number of classes we want to predict
            hidden_sizes: A list of integers specifying the hidden layer sizes in the NN
        """
        super().__init__()

        # Create the network based on the specified hidden sizes
        layers = []
        layer_sizes = [input_size] + hidden_sizes
        for layer_index in range(1, len(layer_sizes)):
            layers += [nn.Linear(layer_sizes[layer_index - 1], layer_sizes[layer_index]), act_fn]
        layers += [nn.Linear(layer_sizes[-1], num_classes)]
        # A module list registers a list of modules as submodules (e.g. for parameters)
        self.layers = nn.ModuleList(layers)

        self.config = {
            "act_fn": act_fn.__class__.__name__,
            "input_size": input_size,
            "num_classes": num_classes,
            "hidden_sizes": hidden_sizes,
        }

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x)
        return x
    
class Identity(nn.Module):
    def forward(self, x):
        return x

class MyDataSet(Dataset):
    def __init__(self, label, img):
        self.label = label
        self.img = img
    def __len__(self):
        return len(self.label)
    def __getitem__(self, idx):
        batch= { 
                'img': self.img[:,:,:,idx].astype('float32'),
                 'label': self.label[idx]
                 }
        return batch
        

class MyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        f = open('digitLabel.pickle',"rb")
        label = pickle.load(f)
        f = open('digitData.pickle',"rb")
        img = (pickle.load(f)/255-0.5)*2
        
        trainLen = len(label)//10*8
        valLen = len(label)//10
        
        self.train_dataset = MyDataSet(label[:trainLen], img[:,:,:,:trainLen])
        self.val_dataset = MyDataSet(label[trainLen:trainLen+valLen], 
                                     img[:,:,:,trainLen:trainLen+valLen])
        self.test_dataset = MyDataSet(label[trainLen+valLen:], 
                                     img[:,:,:,trainLen+valLen:])
        self.batch_size = 32
        self.model = MyModule(nn.Sigmoid())
        self.loss = nn.CrossEntropyLoss()
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True)
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False)
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),weight_decay=0.001)
        # try CosineAnnealingLR
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                mode='min', 
                                                factor=0.7,
                                                patience=20, 
                                                verbose=True,
                                                min_lr=0.0001)
        return {'optimizer': optimizer, 
                'lr_scheduler': scheduler,
                'monitor': 'val/rec'}
    def forward(self, batch):
        return self.model(batch['img']), batch['label']
    
    def training_step(self, train_batch, batch_idx):
        label = train_batch['label']
        out, target = self.forward(train_batch)
        rec_loss = self.loss(out, target)
        self.log('train/rec', rec_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        
        max_idx = torch.argmax(out,dim=1)
        acc = torch.mean((max_idx == label).type(torch.FloatTensor))
        self.log('train/acc', acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return rec_loss
    
    def validation_step(self, val_batch, batch_idx):
        label = val_batch['label']
        out, target = self.forward(val_batch)
        rec_loss = self.loss(out, target)
        self.log('val/rec', rec_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        
        max_idx = torch.argmax(out,dim=1)
        acc = torch.mean((max_idx == label).type(torch.FloatTensor))
        self.log('val/acc', acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return rec_loss

    def test_step(self, test_batch, batch_idx, dataloader_idx=None):
        label = test_batch['label']
        out, target = self.forward(test_batch)
        rec_loss = self.loss(out, target)
        
        self.log('test/rec', rec_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        max_idx = torch.argmax(out,dim=1)
        acc = torch.mean((max_idx == label).type(torch.FloatTensor))
        self.log('test/acc', acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)


class BestValidationCallback(pl.callbacks.base.Callback):
    # logs the best validation loss and other stuff
    def __init__(self, monitor):
        super().__init__()
        self.monitor = monitor
        self.best_val_loss = np.Inf


    def on_validation_end(self, trainer, pl_module):
        if trainer.running_sanity_check:
            return
        losses = trainer.logger_connector.callback_metrics
        print('cur: '+str(losses['val/rec']) + ' best: '+ str(self.best_val_loss))
        if (losses[self.monitor] < self.best_val_loss):
            self.best_val_loss = losses[self.monitor]


class TestEndCallback(pl.callbacks.base.Callback):
    # logs the best validation loss and other stuff
    def __init__(self):
        super().__init__()

    def on_test_end(self, trainer, pl_module):
        acc = trainer.logger_connector.callback_metrics['test/acc']
        print('accuracy: ', acc)

if __name__ == '__main__':
    pl.seed_everything(0)
    
    model = MyModel()
    lr_monitor_callback = pl.callbacks.LearningRateMonitor()
    logger = pl.loggers.TensorBoardLogger(save_dir='./log', name='digit')
    checkpoint_dir = logger.log_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
                    dirpath=checkpoint_dir,
                    filename='{epoch}',
                    monitor='val/rec', 
                    save_top_k=1,verbose=True, mode='min',
                    save_last=False)
    best_validation_callback = BestValidationCallback('val/rec')
    test_end_callback = TestEndCallback()
    
    trainer = pl.Trainer(
                    logger=logger,
                    max_epochs = 1000,
                    callbacks=[
                        checkpoint_callback,
                        lr_monitor_callback,
                        best_validation_callback,
                        test_end_callback,
                                ],
                    deterministic=True,
                    check_val_every_n_epoch=1
                    )
    
    trainer.fit(model)
    trainer.test()