import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf

# from simulation import Simulation
# from contact_planner import ContactPlanner
from utils import get_plan, get_des_velocities, get_estimated_com, \
                    construct_goal, compute_goal_reaching_error, rotate_jacobian
# import pinocchio as pin
from database import Database

import numpy as np
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')
import random
import hydra
import os
from tqdm import tqdm
from datetime import datetime
import h5py
import pickle
import sys
import time
import tkinter as tk
from tkinter.filedialog import asksaveasfilename


# set random seet for reproducability
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


class TestTrainPolicy():  

    def __init__(self, cfg):        
        # configuration file (containing the hyper/parameters)
        self.cfg = cfg
        
        # Data related parameters 
        self.n_state = cfg.n_state
        self.n_action = cfg.n_action
        self.goal_horizon = cfg.goal_horizon
        self.normalize_policy_input = True
        
        # Dataset parameters
        self.max_replay_buffer_size = cfg.max_replay_buffer_size
        self.size_dataset_per_training_iter = cfg.size_dataset_per_training_iter
        
        assert self.size_dataset_per_training_iter <= self.max_replay_buffer_size, 'size of dataset per training iter larger than buffer size!'
        
        # Policy Network Properties
        # WATCHOUT: phi, vdes, wdes and gait type is excluded from state!
        self.input_size = (self.n_state) + (self.goal_horizon * 3 * 4)  # goal is goal_horizon * (time + xy) * n_eff.
        self.output_size = self.n_action
        self.network = None
        self.criterion = nn.L1Loss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Nvidia GPU availability is ' + str(torch.cuda.is_available()))
        
        # Training properties
        self.n_epoch = 150
        self.batch_size = 256
        self.n_train_frac = 0.9
        self.learning_rate = 0.002
        self.kl_div_reg_weight = 0.0
        self.prev_policy = None
        
        
        # Declare Database
        # self.database = BehaviorCloningMemory(limit=cfg.max_replay_buffer_size)
        self.database = Database(limit=cfg.max_replay_buffer_size, norm_input=self.normalize_policy_input)

        # Tensorboard logging
        tensorboard_path = './tensorboard/logs'
        os.makedirs(tensorboard_path, exist_ok=True)
        self.tb_writer = SummaryWriter(tensorboard_path)
    
    
    def initialize_network(self):
        """
        load policy network and determine input and output sizes
        """        
        from networks import GoalConditionedPolicyNet
        
        self.network = GoalConditionedPolicyNet(self.input_size, self.output_size, num_hidden_layer=3, hidden_dim=512, batch_norm=True).to(self.device)
        print("Policy Network initialized")
    

    def train_network(self, current_iter=0, plot_loss=False):
        """
        Train and validate the policy network with samples from the current dataset

        Args:
            dataset (Pytorch Dataset): the sampled and splitted dataset for training
            current_iter (int): current simulation step (Not Time!)
            plot_loss (bool, optional): (Non-Blocking) Plot the Training and validation 
            loss. Defaults to False.

        Returns:
            loss_history (np.array): array of training loss
            test_loss_history (np.array): array of validation loss
        """        
        
        # get the training dataset size (use whole dataset)
        train_set_size = len(self.database)
        # train_set_size = min(len(self.database), self.size_dataset_per_training_iter)
        print("Dataset size: " + str(train_set_size))
        

        # define training and test set size
        n_train = int(self.n_train_frac*train_set_size)
        n_test = train_set_size - n_train
        # n_batches = int(n_train/self.batch_size)
        # n_batches_test = int(n_test/self.batch_size)
        
        # Split data into training and validation
        train_data, test_data = torch.utils.data.random_split(self.database, [n_train, n_test])
        train_loader = DataLoader(train_data, self.batch_size, shuffle=True)
        test_loader = DataLoader(test_data, self.batch_size, shuffle=True)
        
        # define training optimizer
        
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
            
        tepoch = tqdm(range(self.n_epoch))
        
        # loss history variable
        loss_history = np.full(self.n_epoch, np.nan)
        test_loss_history = np.full(self.n_epoch, np.nan)
        
        # main training loop
        for epoch in tepoch:
            # set network to training mode
            self.network.train()
            
            # training loss
            train_running_loss = 0
            
            # train network
            for x, y in train_loader:
                self.optimizer.zero_grad()
                x, y = x.to(self.device).float(), y.to(self.device).float()
                y_pred = self.network(x)
                loss = self.criterion(y_pred, y)
                
                loss.backward()
                self.optimizer.step()
                train_running_loss += loss.item()
            

            # test network
            test_running_loss = 0
            self.network.eval()
            for z, w in test_loader:
                z, w = z.to(self.device).float(), w.to(self.device).float()
                w_pred = self.network(z)
                test_loss = self.criterion(w_pred, w)
                test_running_loss += test_loss.item()
                
            tepoch.set_postfix({'training loss': train_running_loss/len(train_loader),
                                'validation loss': test_running_loss/len(test_loader)})
            
            # save 
            loss_history[epoch] = train_running_loss/len(train_loader)
            test_loss_history[epoch] = test_running_loss/len(test_loader)


            # # # Tensorboard
            # for name, param in self.network.named_parameters():
            #     self.tb_writer.add_histogram(name, param, epoch)
            # self.tb_writer.add_scalar('Loss/train', loss_history[epoch], epoch)
            # self.tb_writer.add_scalar('Loss/test', test_loss_history[epoch], epoch)
            

            # # Plot loss
            # if plot_loss is True:
            #     self.ax_training.clear()
            #     self.ax_training.plot(test_loss_history, label='test_loss')
            #     self.ax_training.plot(loss_history, label='training_loss')
            #     self.ax_training.set_title("Training and Validation Loss - Iteration " + str(current_iter))
            #     self.ax_training.legend()
            #     self.ax_training.set_xbound(0, self.n_epoch)
            #     self.fig_training.canvas.draw()
            #     self.fig_training.canvas.flush_events()
            
            # save current network as prev
            self.prev_policy = self.network
            self.prev_policy.eval()

        # return loss_history, test_loss_history
    
    
    def save_network(self, iter):
        """
        Save trained policy network

        Args:
            iter (int): current algorithm iteration
        """
        # root =tk.Tk()
        # root.withdraw()
        
        # savepath = asksaveasfilename(defaultextension='.pth', filetypes=[('path files', '.pth')])
        
        savepath = '/home/atari_ws/data/goal_cond_iterative_algorithm/trot/min_mean_max/test_network.pth'
        
        payload = {'network': self.network,
                   'optimizer': self.optimizer,
                   'norm_policy_input': None}
        
        if self.normalize_policy_input:
            payload['norm_policy_input'] = self.database.get_database_mean_std()
        
        torch.save(payload, savepath)
        print('Network Snapshot saved for iteration ' + str(iter))
        
        
    def run(self):   
        '''
        Run the iterative algorithm
        '''

        # Initialize policy network
        self.initialize_network()
        
        # load saved database
        self.database.load_saved_database(filename='/home/atari_ws/data/goal_cond_iterative_algorithm/trot/min_mean_max/dataset/database_0.hdf5')
        
        # Activate matplotlib interactive plot for non-blocking plotting
        # plt.ion()
        # self.fig_training, self.ax_training = plt.subplots()
        # plt.show()
        
        # NOTE: Train Policy
        print('=== Training Policy ===')
        
        # train network
        self.train_network(plot_loss=True)
        
        self.save_network(0)
            
        # Close tensorboard logger
        self.tb_writer.close()
                
        
@hydra.main(config_path='cfgs', config_name='config')
def main(cfg):
    icc = TestTrainPolicy(cfg)
    icc.run() 

if __name__ == '__main__':
    main()
    
        


    


