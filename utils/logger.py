import datetime
from tensorboardX import SummaryWriter
time_board = "{0:%Y-%m-%d %H:%M:%S/}".format(datetime.datetime.now())
import torch
import os.path as osp, time, atexit, os


import os
import pickle
import torch
from .output import println
import numpy as np

class Logger:

    def __init__(self, hyperparams):
        self.data = {'time': 0,
                         'MinR': [],
                         'MaxR': [],
                         'AvgR': [],
                         'MinC': [],
                         'MaxC': [],
                         'AvgC': [],
                         'nu': [],
                         'running_stat': None,
                         'MaxRatio': [],
                         'MinRatio': [],
                         }

        self.models = {'iter': None,
                       'policy_params': None,
                       'value_params': None,
                       'cvalue_params': None,
                       'pi_optimizer': None,
                       'vf_optimizer': None,
                       'cvf_optimizer': None,
                       'pi_loss': None,
                       'vf_loss': None,
                       'cvf_loss': None}

        self.hyperparams = hyperparams
        self.writer = SummaryWriter('./logs/CUP/'+ time_board)
        
        filename1 = '_'.join([self.hyperparams["file_prefix"],self.hyperparams["algo"],self.hyperparams["env_id"], 'seed_', str(self.hyperparams["seed"]),"data"]) + '.txt'
        output_dir = "./Data"
        self.output_dir = output_dir or "/tmp/experiments/%i"%int(time.time())
        if osp.exists(self.output_dir):
            print("Warning: Log dir %s already exists! Storing info there anyway."%self.output_dir)
        else:
            os.makedirs(self.output_dir)
        self.output_file = open(osp.join(self.output_dir, filename1), 'w')
        atexit.register(self.output_file.close)

    def update(self, key, value):
        if not (key in self.data.keys()):
            self.data[key] = []

        if type(self.data[key]) is list:
            self.data[key].append(value)
        else:
            self.data[key] = value

    def save_model(self, component, params):
        self.models[component] = params

    def dump(self, iter):


        print("seed:",self.hyperparams["seed"])
        batch_size = self.hyperparams['batch_size']
        # Print results
        println('Results for Iteration:', iter + 1)
        println('Number of Samples:', (iter + 1) * batch_size)
        println('Time: {:.2f}'.format(self.data['time']))
        println('MinR: {:.2f}| MaxR: {:.2f}| AvgR: {:.2f}'.format(self.data['MinR'][-1],
                                                                  self.data['MaxR'][-1],
                                                                  self.data['AvgR'][-1]))
        println('MinC: {:.2f}| MaxC: {:.2f}| AvgC: {:.2f}'.format(self.data['MinC'][-1],
                                                                  self.data['MaxC'][-1],
                                                                  self.data['AvgC'][-1]))

        println('--------------------------------------------------------------------')
        self.prefix = '_'.join([self.hyperparams["env_id"], 'seed', str(self.hyperparams["seed"])])
        self.writer.add_scalar("/".join([self.prefix, "_".join(["Reward",self.hyperparams["algo"]])]), self.data["AvgR"][-1], iter)
        self.writer.add_scalar("/".join([self.prefix, "_".join(["Cost",self.hyperparams["algo"]])]), self.data["AvgC"][-1], iter)

        vals=[
            self.data['AvgR'][-1], 
            self.data['AvgC'][-1], 
            self.data['nu'][-1],
            self.data['AvgC'][-1]/1000.0
        ]
        self.output_file.write("\t".join(map(str,vals))+"\n")
        vals.clear()

    def save_data(self, file_Path):
        # Save Logger
        env_id = self.hyperparams['env_id']
        constraint = self.hyperparams['constraint']
        seed = self.hyperparams['seed']
        kl_coef = self.hyperparams['kl_coef']
        envname = env_id.partition(':')[-1] if ':' in env_id else env_id
        prefix = self.hyperparams["algo"]
        now_time = "{0:%Y-%m-%d %H:%M:%S}".format(datetime.datetime.now())
        
        if not os.path.exists(file_Path):
            os.mkdir(file_Path)
        
