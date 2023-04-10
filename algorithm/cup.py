import argparse
import gym
import torch
import torch.nn as nn
import time
from utils.data_generator import DataGenerator
from model.models import GaussianPolicy, Value
from utils.environment import get_threshold
from utils.trans import graph_detach, to_dytype_device,graph_detach
from utils.math import gaussian_kl
from utils.output import println
import numpy as np
from collections import deque

class CUP:

    def __init__(self,
                 env,
                 policy_net,
                 value_net,
                 cvalue_net,
                 pi_optimizer,
                 vf_optimizer,
                 cvf_optimizer,
                 num_epochs,
                 mb_size,
                 c_gamma,
                 lam,
                 delta,
                 eta,
                 nu,
                 nu_lr,
                 nu_max,
                 cost_lim,
                 l2_reg,
                 score_queue,
                 cscore_queue,
                 logger,
                 gae_lam,
                 c_gae_lam,
                 kl_coef,
                 clip):

        self.env = env

        self.policy = policy_net
        self.value_net = value_net
        self.cvalue_net = cvalue_net

        self.pi_optimizer = pi_optimizer
        self.vf_optimizer = vf_optimizer
        self.cvf_optimizer = cvf_optimizer

        self.pi_loss = None
        self.vf_loss = None
        self.cvf_loss = None

        self.num_epochs = num_epochs
        self.mb_size = mb_size

        self.c_gamma = c_gamma
        self.lam = lam
        self.delta = delta
        self.eta = eta
        self.cost_lim = cost_lim

        self.nu = nu
        self.nu_lr = nu_lr
        self.nu_max = nu_max

        self.l2_reg = l2_reg

        self.logger = logger
        self.score_queue = score_queue
        self.cscore_queue = cscore_queue

        self.max_ratio = -9999999
        self.min_ratio = 9999999
        self.betta = 0.3
        self.gae_lam = gae_lam
        self.c_gae_lam = c_gae_lam
        self.kl_coef = kl_coef
        self.clip = clip

    def update_params(self, rollout, dtype, device):

        # Convert data to tensor
        obs = torch.Tensor(rollout['states']).to(dtype).to(device)
        act = torch.Tensor(rollout['actions']).to(dtype).to(device)
        vtarg = torch.Tensor(rollout['v_targets']).to(dtype).to(device).detach()
        adv = torch.Tensor(rollout['advantages']).to(dtype).to(device).detach()
        cvtarg = torch.Tensor(rollout['cv_targets']).to(dtype).to(device).detach()
        cadv = torch.Tensor(rollout['c_advantages']).to(dtype).to(device).detach()

        # Get log likelihood, mean, and std of current policy
        old_logprob, old_mean, old_std = self.policy.logprob(obs, act)
        old_logprob, old_mean, old_std = to_dytype_device(dtype, device, old_logprob, old_mean, old_std)
        old_logprob, old_mean, old_std = graph_detach(old_logprob, old_mean, old_std)

        # Store in TensorDataset for minibatch updates
        dataset = torch.utils.data.TensorDataset(obs, act, vtarg, adv, cvtarg, cadv,
                                                 old_logprob, old_mean, old_std)
        loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.mb_size, shuffle=True)
        avg_cost = rollout['avg_cost']
        # Older
        self.nu += self.nu_lr * (avg_cost - self.cost_lim)
        for epoch in range(self.num_epochs):
            for _, (obs_b, act_b, vtarg_b, adv_b, cvtarg_b, cadv_b,
                    old_logprob_b, old_mean_b, old_std_b) in enumerate(loader):

                # Update reward critic
                mse_loss = nn.MSELoss()
                vf_pred = self.value_net(obs_b)
                self.vf_loss = mse_loss(vf_pred, vtarg_b)
                # weight decay
                for param in self.value_net.parameters():
                    self.vf_loss += param.pow(2).sum() * self.l2_reg
                self.vf_optimizer.zero_grad()
                self.vf_loss.backward()
                self.vf_optimizer.step()

                # Update cost critic
                cvf_pred = self.cvalue_net(obs_b)
                self.cvf_loss = mse_loss(cvf_pred, cvtarg_b)
                # weight decay
                for param in self.cvalue_net.parameters():
                    self.cvf_loss += param.pow(2).sum() * self.l2_reg
                self.cvf_optimizer.zero_grad()
                self.cvf_loss.backward()
                self.cvf_optimizer.step()

                # Update policy
                logprob, mean, std = self.policy.logprob(obs_b, act_b)
                kl_new_old = gaussian_kl(mean, std, old_mean_b, old_std_b)
                ratio = torch.exp(logprob - old_logprob_b)
                pi_loss = ratio * adv_b
                pi_loss_clip = torch.clamp(ratio, 1-self.clip, 1+self.clip)*adv_b
                self.pi_loss = -torch.min(pi_loss, pi_loss_clip).mean()
                self.pi_optimizer.zero_grad()
                self.pi_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 40)
                self.pi_optimizer.step()
            # Early stopping
            logprob, mean, std = self.policy.logprob(obs, act)
            kl_val = gaussian_kl(mean, std, old_mean, old_std).mean().item()
            if kl_val > self.delta:
                println('Break at epoch {} because KL value {:.4f} larger than {:.4f}'.format(epoch + 1, kl_val,
                                                                                              self.delta))
                break

        old_logprob_2, old_mean_2, old_std_2 = self.policy.logprob(obs, act)
        old_logprob_2, old_mean_2, old_std_2 = to_dytype_device(dtype, device, old_logprob_2, old_mean_2, old_std_2)
        old_logprob_2, old_mean_2, old_std_2 = graph_detach(old_logprob_2, old_mean_2, old_std_2)

        # Store in TensorDataset for minibatch updates
        dataset = torch.utils.data.TensorDataset(obs, act, old_logprob, cvtarg, cadv,
                                                 old_logprob_2, old_mean_2, old_std_2)
        loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.mb_size, shuffle=True)
        for epoch in range(self.num_epochs):
            for _, (obs_b, act_b, old_logprob_b, cvtarg_b, cadv_b,
                    old_logprob_2_b, old_mean_2_b, old_std_2_b) in enumerate(loader):


                logprob, mean, std = self.policy.logprob(obs_b, act_b)
                kl_new_old = gaussian_kl(mean, std, old_mean_2_b, old_std_2_b)
                ratio = torch.exp(logprob - old_logprob_b)

                c_loss_coef = (1 - self.c_gamma * self.c_gae_lam) / (1 - self.c_gamma)

                if self.nu < 0:
                    self.nu = 0
                elif self.nu > self.nu_max:
                    self.nu = self.nu_max

                c_loss = self.nu * c_loss_coef * ratio * cadv_b
                totla_c_loss = (kl_new_old + c_loss).mean()
                self.pi_optimizer.zero_grad()
                totla_c_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 40)
                self.pi_optimizer.step()
                temp_max = torch.max(ratio).detach().cpu().numpy()
                temp_min = torch.min(ratio).detach().cpu().numpy()
                if temp_max > self.max_ratio:
                    self.max_ratio = temp_max
                if temp_min < self.min_ratio:
                    self.min_ratio = temp_min
            # Early stopping
            logprob, mean, std = self.policy.logprob(obs, act)
            kl_val = gaussian_kl(mean, std, old_mean, old_std).mean().item()
            if kl_val > self.delta:
                println('Break at epoch {} because KL value {:.4f} larger than {:.4f}'.format(epoch + 1, kl_val,
                                                                                              self.delta))
                break

        # Store everything in log
        self.logger.update('MinR', np.min(self.score_queue))
        self.logger.update('MaxR', np.max(self.score_queue))
        self.logger.update('AvgR', np.mean(self.score_queue))
        self.logger.update('MinC', np.min(self.cscore_queue))
        self.logger.update('MaxC', np.max(self.cscore_queue))
        self.logger.update('AvgC', np.mean(self.cscore_queue))
        self.logger.update('MaxRatio', self.max_ratio)
        self.logger.update('MinRatio', self.min_ratio)
        self.logger.update('nu', self.nu)

        # Save models
        self.logger.save_model('policy_params', self.policy.state_dict())
        self.logger.save_model('value_params', self.value_net.state_dict())
        self.logger.save_model('cvalue_params', self.cvalue_net.state_dict())
        self.logger.save_model('pi_optimizer', self.pi_optimizer.state_dict())
        self.logger.save_model('vf_optimizer', self.vf_optimizer.state_dict())
        self.logger.save_model('cvf_optimizer', self.cvf_optimizer.state_dict())
        self.logger.save_model('pi_loss', self.pi_loss)
        self.logger.save_model('vf_loss', self.vf_loss)
        self.logger.save_model('cvf_loss', self.cvf_loss)
