import argparse
import gym
import safety_gym
import torch.nn as nn
import time
import torch
import numpy as np
from utils.data_generator import DataGenerator
from model.models import GaussianPolicy, Value
from utils.environment import get_threshold
from utils.logger import Logger
from utils.running_stats import RunningStats
from collections import deque
from algorithm.cup import CUP

def train(args):
    # Initialize data type
    dtype = torch.float32
    torch.set_default_dtype(dtype)
    device = torch.device('cuda', args.cuda) if torch.cuda.is_available() else torch.device('cpu')

    # Initialize environment
    env = gym.make(args.env_id)
    envname = env.spec.id
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    # act_dim = env.action_space.n

    # Initialize random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)

    # Initialize neural nets
    args.hidden_size = (args.hidden_size, args.hidden_size)
    policy = GaussianPolicy(obs_dim, act_dim, args.hidden_size, args.activation, args.logstd)
    value_net = Value(obs_dim, args.hidden_size, args.activation)
    cvalue_net = Value(obs_dim, args.hidden_size, args.activation)
    policy.to(device)
    value_net.to(device)
    cvalue_net.to(device)

    # Initialize optimizer
    pi_optimizer = torch.optim.Adam(policy.parameters(), args.pi_lr)
    vf_optimizer = torch.optim.Adam(value_net.parameters(), args.vf_lr)
    cvf_optimizer = torch.optim.Adam(cvalue_net.parameters(), args.cvf_lr)

    # Initialize learning rate scheduler
    lr_lambda = lambda it: max(1.0 - it / args.max_iter_num, 0)
    pi_scheduler = torch.optim.lr_scheduler.LambdaLR(pi_optimizer, lr_lambda=lr_lambda)
    vf_scheduler = torch.optim.lr_scheduler.LambdaLR(vf_optimizer, lr_lambda=lr_lambda)
    cvf_scheduler = torch.optim.lr_scheduler.LambdaLR(cvf_optimizer, lr_lambda=lr_lambda)

    # Store hyperparameters for log
    hyperparams = vars(args)

    # Initialize RunningStat for state normalization, score queue, logger
    running_stat = RunningStats(clip=5)
    score_queue = deque(maxlen=100)
    cscore_queue = deque(maxlen=100)
    logger = Logger(hyperparams)
    # Get constraint bounds
    #cost_lim = get_threshold(envname, constraint=args.constraint)
    cost_lim = logger.hyperparams["cost_lim"]
    # Initialize and train CUP agent
    agent = CUP(env, policy, value_net, cvalue_net,
                       pi_optimizer, vf_optimizer, cvf_optimizer,
                       args.num_epochs, args.mb_size,
                       args.c_gamma, args.lam, args.delta, args.eta,
                       args.nu, args.nu_lr, args.nu_max, cost_lim,
                       args.l2_reg, score_queue, cscore_queue, logger,
                       args.gae_lam, args.c_gae_lam, args.kl_coef,args.clip)
    start_time = time.time()

    for iter in range(args.max_iter_num):
        # Update iteration for model
        agent.logger.save_model('iter', iter)

        # Collect trajectories
        data_generator = DataGenerator(obs_dim, act_dim, args.batch_size, args.max_eps_len)
        rollout = data_generator.run_traj(env, agent.policy, agent.value_net, agent.cvalue_net,
                                          running_stat, agent.score_queue, agent.cscore_queue,
                                          args.gamma, args.c_gamma, args.gae_lam, args.c_gae_lam,
                                          dtype, device, args.constraint)
        # Update Agent parameters
        agent.update_params(rollout, dtype, device)
        # Update learning rates
        pi_scheduler.step()
        vf_scheduler.step()
        cvf_scheduler.step()

        # Update time and running stat
        agent.logger.update('time', time.time() - start_time)
        agent.logger.update('running_stat', running_stat)

        # Save and print values
        agent.logger.dump(iter)

    agent.logger.save_data(file_Path = "./Data")

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='PyTorch CUP Implementation')
    parser.add_argument('--cost-lim',type=float, default=20.0,
                        help='Data File Prefix')
    parser.add_argument('--file-prefix', default='None',
                        help='Data File Prefix')
    parser.add_argument('--env-id', default='Swimmer-v3',
                        help='Name of Environment (default: Swimmer-v3)')
    parser.add_argument('--constraint', default='velocity',
                        help='Constraint setting (default: velocity')
    parser.add_argument('--activation', default="tanh",
                        help='Activation function for policy/critic network (Default: tanh)')
    parser.add_argument('--hidden-size', type=int, default=64,
                        help='Tuple of size of hidden layers for policy/critic network (Default: (64, 64))')
    parser.add_argument('--logstd', type=float, default=-0.5,
                        help='Log std of Policy (Default: -0.5)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor for reward (Default: 0.99)')
    parser.add_argument('--c-gamma', type=float, default=0.99,
                        help='Discount factor for cost (Default: 0.99)')
    parser.add_argument('--gae-lam', type=float, default=0.95,
                        help='Lambda value for GAE for reward (Default: 0.95)')
    parser.add_argument('--c-gae-lam', type=float, default=0.95,
                        help='Lambda value for GAE for cost (Default: 0.95)')
    parser.add_argument('--l2-reg', type=float, default=1e-3,
                        help='L2 Regularization Rate (default: 1e-3)')
    parser.add_argument('--pi-lr', type=float, default=3e-4,
                        help='Learning Rate for policy (default: 3e-4)')
    parser.add_argument('--vf-lr', type=float, default=3e-4,
                        help='Learning Rate for value function (default: 3e-4)')
    parser.add_argument('--cvf-lr', type=float, default=3e-4,
                        help='Learning Rate for c-value function (default: 3e-4)')
    parser.add_argument('--lam', type=float, default=1.5,
                        help='Inverse temperature lambda (default: 1.5)')
    parser.add_argument('--delta', type=float, default=0.02,
                        help='KL bound (default: 0.02)')
    parser.add_argument('--eta', type=float, default=0.02,
                        help='KL bound for indicator function (default: 0.02)')
    parser.add_argument('--nu', type=float, default=0,
                        help='Cost coefficient (default: 0)')
    parser.add_argument('--nu_lr', type=float, default=0.01,
                        help='Cost coefficient learning rate (default: 0.01)')
    parser.add_argument('--nu_max', type=float, default=2.0,
                        help='Maximum cost coefficient (default: 2.0)')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random Seed (default: 0)')
    parser.add_argument('--max-eps-len', type=int, default=1000,
                        help='Maximum length of episode (default: 1000)')
    parser.add_argument('--mb-size', type=int, default=64,
                        help='Minibatch size per update (default: 64)')
    parser.add_argument('--cuda', type=int, default=0,
                        help='cuda')
    parser.add_argument('--batch-size', type=int, default=2048,
                        help='Batch Size per Update (default: 2048)')
    parser.add_argument('--num-epochs', type=int, default=10,
                        help='Number of passes through each minibatch per update (default: 10)')
    parser.add_argument('--max-iter-num', type=int, default=500,
                        help='Number of Main Iterations (default: 500)')
    parser.add_argument('--algo', type=str,default="CUP",
                        help='algo')
    parser.add_argument('--kl-coef', type=float, default=0.3, 
                        help='kl_coef')
    parser.add_argument('--clip', type=float, default=0.2, help='clip')
    args = parser.parse_args()
    train(args)
