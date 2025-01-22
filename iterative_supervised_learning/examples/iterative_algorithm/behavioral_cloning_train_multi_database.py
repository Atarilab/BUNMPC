import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from simulation import Simulation
from contact_planner import ContactPlanner
import utils
import pinocchio as pin
from database import Database

import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')
import random
import hydra
import os
from tqdm import tqdm
from datetime import datetime
import h5py
import pickle
import wandb

# set random seet for reproducability
# seed = 42
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)

# login to wandb
wandb.login()

project_name = 'bc_benchmark_train_3'


class BehavioralCloning():  

    def __init__(self, cfg):        
        # configuration file (containing the hyper/parameters)
        self.cfg = cfg
        
        # Model Parameters
        self.action_type = cfg.action_type
        self.normalize_policy_input = cfg.normalize_policy_input
        
        # Data related parameters 
        self.n_state = cfg.n_state
        self.n_action = cfg.n_action
        self.goal_horizon = cfg.goal_horizon
        
        # Network related parameters
        self.criterion = nn.L1Loss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Nvidia GPU availability is ' + str(torch.cuda.is_available()))
        
        # Training properties
        self.n_epoch = cfg.n_epoch  # per iteration
        self.batch_size = cfg.batch_size
        self.n_train_frac = cfg.n_train_frac
        self.learning_rate = cfg.learning_rate

    
    def initialize_network(self, input_size=0, output_size=0, num_hidden_layer=3, hidden_dim=512, batch_norm=True):
        """initialize policy network

        Args:
            input_size (int, optional): input dimension size (state + goal). Defaults to 0.
            output_size (int, optional): output dimension size (action). Defaults to 0.
            num_hidden_layer (int, optional): number of hidden layers. Defaults to 3.
            hidden_dim (int, optional): number of nodes per hidden layer. Defaults to 512.
            batch_norm (bool, optional): if 1D Batch normalization should be performed. Defaults to True.

        Returns:
            network: the created pytorch policy network
        """           
        from networks import GoalConditionedPolicyNet
        
        network = GoalConditionedPolicyNet(input_size, output_size, num_hidden_layer=num_hidden_layer, 
                                                hidden_dim=hidden_dim, batch_norm=batch_norm).to(self.device)
        print("Policy Network initialized")
        return network
    

    def train_network(self, network, batch_size=256, learning_rate=0.002, n_epoch=150):
        """Train the policy network

        Args:
            network (_type_): policy network to train
            batch_size (int, optional): training batch size. Defaults to 256.
            learning_rate (float, optional): training learning rate. Defaults to 0.002.
            n_epoch (int, optional): training epochs. Defaults to 150.

        Returns:
            network: the trained policy network
        """             
        
        # get the training dataset size (use whole dataset)
        train_set_size = len(self.database)

        print("Dataset size: " + str(train_set_size))
        print(f'Batch size: {batch_size}')
        print(f'learning rate: {learning_rate}')
        print(f'num of epochs: {n_epoch}')

        # define training and test set size
        n_train = int(self.n_train_frac*train_set_size)
        n_test = train_set_size - n_train
        
        print(f'training data size: {n_train}')
        print(f'validation data size: {n_test}')
        
        # Split data into training and validation
        train_data, test_data = torch.utils.data.random_split(self.database, [n_train, n_test])
        train_loader = DataLoader(train_data, batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_data, batch_size, shuffle=True, drop_last=True)
        
        # define training optimizer
        
        self.optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
            
        tepoch = tqdm(range(n_epoch))
        
        # main training loop
        for epoch in tepoch:
            # set network to training mode
            network.train()
            
            # training loss
            train_loss, valid_loss = [], []
            
            # train network
            for x, y in train_loader:
                self.optimizer.zero_grad()
                x, y = x.to(self.device).float(), y.to(self.device).float()
                y_pred = network(x)
                loss = self.criterion(y_pred, y)
                
                loss.backward()
                self.optimizer.step()
                train_loss.append(loss.item())
            

            # test network
            test_running_loss = 0
            network.eval()
            for z, w in test_loader:
                z, w = z.to(self.device).float(), w.to(self.device).float()
                w_pred = network(z)
                test_loss = self.criterion(w_pred, w)
                valid_loss.append(test_loss.item())
                
            train_loss_avg = np.mean(train_loss)
            valid_loss_avg = np.mean(valid_loss)
            tepoch.set_postfix({'training loss': train_loss_avg,
                                'validation loss': valid_loss_avg})
            
            # wandb log
            wandb.log({'Training Loss': train_loss_avg,
                       'Validation Loss': valid_loss_avg})
            
            # if epoch>0 and epoch%network_save_frequency==0:
            #     self.save_network(network, name=self.database.goal_type+'_'+str(epoch))
        
        # save final network
        # self.save_network(network, name=self.database.goal_type+'_'+str(n_epoch))    
            
        return network
    
    
    def save_network(self, network, name='policy'):
        """save trained network

        Args:
            network (_type_): trained policy network
            name (str, optional): name of the network to save. Defaults to 'policy'.
        """        
        
        savepath =  self.network_savepath + "/"+name+".pth"
        
        payload = {'network': network,
                #    'optimizer': self.optimizer,
                #    'scheduler': self.scheduler,
                   'norm_policy_input': None}
        
        # save normalization parameters
        if self.normalize_policy_input:
            payload['norm_policy_input'] = self.database.get_database_mean_std()
        
        torch.save(payload, savepath)
        print('Network Snapshot saved')
        
        
    def run(self):
        """run training
        """           
        
        # database_dir = '/home/atari_ws/data/behavior_cloning/trot/bc_benchmark_3/dataset'
        database_dir = '/home/atari_ws/iterative_supervised_learning/examples/iterative_algorithm/data/behavior_cloning/trot/Jan_22_2025_10_33_55/dataset'
        raw_files = os.listdir(database_dir)
        
        files = []
        for file in raw_files:
            if file[:8] == 'database':
                files.append(file)
        
        # NOTE: Iterate over networks from each training iteration
        for file in files:
        
            # NOTE: Initialize Network
            # self.cc_input_size = self.n_state + (self.goal_horizon * 3 * 4)
            self.vc_input_size = self.n_state + 5  # phi, vx, vy, w
            
            self.output_size = self.n_action
            
            # Initialize policy network
            self.vc_network = self.initialize_network(input_size=self.vc_input_size, output_size=self.output_size, 
                                                        num_hidden_layer=self.cfg.num_hidden_layer, hidden_dim=self.cfg.hidden_dim,
                                                        batch_norm=True)
            
            # self.cc_network = self.initialize_network(input_size=self.cc_input_size, output_size=self.output_size, 
            #                                             num_hidden_layer=self.cfg.num_hidden_layer, hidden_dim=self.cfg.hidden_dim,
            #                                             batch_norm=True)
            
            # NOTE: Load database
            self.database = Database(limit=self.cfg.database_size, norm_input=self.normalize_policy_input)
            filename = os.path.join(database_dir, file)
            self.database.load_saved_database(filename=filename)
            
            # Network saving
            directory_path = os.path.dirname(filename)
            self.network_savepath = directory_path + '/../network'
            os.makedirs(self.network_savepath, exist_ok=True)
            
            # # NOTE: Train Policy
            # wandb.init(project='bc_single_gait_multi_goal_with_stop', config={'goal_type':'cc'}, name='cc_training')
            # print('=== Training CC Policy ===')
            # self.database.set_goal_type('cc')
            # self.cc_network = self.train_network(self.cc_network, batch_size=self.batch_size, learning_rate=self.learning_rate, n_epoch=self.n_epoch, network_save_frequency=10)
            # # self.save_network(self.cc_network, name='cc_policy')
            # wandb.finish()
            
            wandb.init(project=project_name, config={'goal_type':'vc', 'database_size': len(self.database)}, name='vc_training')
            print('=== Training VC Policy for datasize ', str(len(self.database)), ' ===')
            self.database.set_goal_type('vc')
            self.vc_network = self.train_network(self.vc_network, batch_size=self.batch_size, learning_rate=self.learning_rate, n_epoch=self.n_epoch)
            self.save_network(self.vc_network, name='vc_policy_'+str(len(self.database)))
            wandb.finish()
            
        
@hydra.main(config_path='cfgs', config_name='bc_config')
def main(cfg):
    icc = BehavioralCloning(cfg) 
    icc.run() 

if __name__ == '__main__':
    main()
    
        


    


