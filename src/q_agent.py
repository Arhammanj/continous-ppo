#!/usr/bin/env python
# coding: utf-8

# In[3]:


"""
We use PyTorch for all agents:

- Linear model trained one sample at a time -> Easy to train, slow and results are not great.
- Linear model trained with batches of data. -> Faster to train, but results are still not good.
"""
import os
from pathlib import Path
from typing import Union, Callable, Tuple, List
import random
from argparse import ArgumentParser
import json
from pdb import set_trace as stop

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F

from src.model_factory import get_model
from src.agent_memory import AgentMemory
from src.utils import(
    get_agent_id,
    get_input_output_dims,
    get_epsilon_decay_fn,
    # load_default_hyperparameters,
    get_observation_samples,
    set_seed,
    get_num_model_parameters)

from src.loops import train
from src.config import TENSORBOARD_LOG_DIR, SAVED_AGENTS_DIR

# importing environment into Agent


class QAgent:

    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 1e-4,
        discount_factor: float = 0.99,
        batch_size: int = 64,
        memory_size: int = 10000,
        freq_steps_update_target: int = 200,
        n_steps_warm_up_memory: int = 500,
        freq_steps_train: int = 16,
        n_gradient_steps: int = 2,
        nn_hidden_layers: List[int] = None,
        max_grad_norm: int = 10,
        normalize_state: bool = False,
        use_double_dqn: bool = True,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
            steps_epsilon_decay: float = 60000,
        log_dir: str = None,
    ):
        """
        :param env:
        :param learning_rate: size of the updates in the SGD/Adam formula
        :param discount_factor: discount factor for future rewards
        :param batch_size: number of (s,a,r,s') experiences we use in each SGD
        update
        :param memory_size: number of experiences the agent keeps in the replay
        memory
        :param freq_steps_update_target: frequency at which we copy the
        parameter from the main model to the target model.
        :param n_steps_warm_up_memory: number of experiences we require to have
        in memory before we start training the agent.
        :param freq_steps_train: frequency at which we update the main model
        parameters
        :param n_gradient_steps: number of SGD/Adam updates we perform when we
        train the main model.
        :param nn_hidden_layers: architecture of the main and target models.
        :param max_grad_norm: used to clipped gradients if they become too
        large.
        :param normalize_state: True/False depending if you want to normalize
        the raw states before feeding them into the model.
        :param epsilon_start: starting exploration rate
        :param epsilon_end: final exploration rate
        :param steps_epsilon_decay: number of step in which epsilon decays from
        'epsilon_start' to 'epsilon_end'
        :param log_dir: Tensorboard logging folder
        """
        self.env = env

        # general hyper-parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        # replay memory we use to sample experiences and update parameters
        # `memory_size` defines the maximum number of past experiences we want the
        # agent remember.
        self.memory_size = memory_size
        self.memory = AgentMemory(memory_size)

        # number of experiences we take at once from `self.memory` to update parameters
        self.batch_size = batch_size

        # hyper-parameters to control exploration of the environment
        self.steps_epsilon_decay = steps_epsilon_decay
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_fn = get_epsilon_decay_fn(epsilon_start, epsilon_end, steps_epsilon_decay)
        self.epsilon = None

        # create q model(s). Plural because we use 2 models: main one, and the other for the target.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available:
            print('GPU is running')
        self.q_net, self.target_q_net = None, None
        self._init_models(nn_hidden_layers)
        print(f'{get_num_model_parameters(self.q_net):,} parameters')
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate) # Adam optimizer is a safe and standard choice
        self.max_grad_norm = max_grad_norm

        # hyper-parameters to control how often or when we do certain things, like
        # - update the main net parameters
        self.freq_steps_train = freq_steps_train
        # - update the target net parameters
        self.freq_steps_update_target = freq_steps_update_target
        # - start training until the memory is big enough  
        # Assertion error     
        assert n_steps_warm_up_memory > batch_size, 'batch_size must be smaller than n_steps_warm_up_memory'
        self.n_steps_warm_up_memory = n_steps_warm_up_memory
        # - number of gradient steps we perform every time we update the main net parameters
        self.n_gradient_steps = n_gradient_steps

        # state variable we use to keep track of the number of calls to `observe()`
        self._step_counter = 0

        # input normalizer
        self.normalize_state = normalize_state
        if normalize_state:
            state_samples = get_observation_samples(env, n_samples=10000)
            self.mean_states = state_samples.mean(axis=0)
            self.std_states = state_samples.std(axis=0)

        # Double DQN decouples next-action selection from value evaluation,
        # reducing overestimation bias (targets the peak-then-decline failure mode).
        self.use_double_dqn = use_double_dqn

        # create a tensorboard logger if `log_dir` was provided
        # logging becomes crucial to understand what is not working in our code.
        self.log_dir = log_dir
        if log_dir:
            self.logger = SummaryWriter(log_dir)

        # save hyper-parameters
        self.hparams = {
            'learning_rate': learning_rate,
            'discount_factor': discount_factor,
            'batch_size': batch_size,
            'memory_size': memory_size,
            'freq_steps_update_target': freq_steps_update_target,
            'n_steps_warm_up_memory': n_steps_warm_up_memory,
            'freq_steps_train': freq_steps_train,
            'n_gradient_steps': n_gradient_steps,
            'nn_hidden_layers': nn_hidden_layers,
            'max_grad_norm': max_grad_norm,
            'normalize_state': normalize_state,
            'use_double_dqn': use_double_dqn,
            'epsilon_start': epsilon_start,
            'epsilon_end': epsilon_end,
            'steps_epsilon_decay': steps_epsilon_decay,
        }

    def _init_models(self, nn_hidden_layers):

        # state is a vector of dimension 4, and 2 are the possible actions
        input_dim, output_dim = get_input_output_dims('MgProject1')
        self.q_net = get_model(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_layers=nn_hidden_layers,
        )
        self.q_net.to(self.device)

        # target q-net
        self.target_q_net = get_model(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_layers=nn_hidden_layers,
        )
        self.target_q_net.to(self.device)

        # copy parameters from the `self.q_net`
        self._copy_params_to_target_q_net()

    def _copy_params_to_target_q_net(self):
        """
        Copies parameters from q_net to target_q_net
        """
        for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(param.data)

    def _normalize_state(self, state: np.array) -> np.array:
        """"""
        # return (state - self.min_states) / (self.max_states - self.min_states)
        return (state - self.mean_states) / (self.std_states)

    def _preprocess_state(self, state: np.array) -> np.array:

        # state = np.copy(state_)

        if len(state.shape) == 1:
            # add extra dimension to make sure it is 2D
            s = state.reshape(1, -1)
        else:
            s = state

        if self.normalize_state:
            s = self._normalize_state(s)

        return s

    def act(self, state: np.array, epsilon: float = None) -> int:
        """
        Behavioural policy
        """
        if epsilon is None:
            # update epsilon
            self.epsilon = self.epsilon_fn(self._step_counter)
            epsilon = self.epsilon
            # print(f'Epsilon: {epsilon}')
        
        #print(f'epsilon: {epsilon}')

        if random.uniform(0, 1) < epsilon:
            # Explore action space by taking random action
            action = self.env.action_space.sample()
            return action

        # Exploit from learned (Exploitation case)
        # make sure s is a numpy array with 2 dimensions,
        # and normalize it if `self.normalize_state = True`
        s = self._preprocess_state(state)

        # forward pass through the net to compute q-values for the 3 actions
        s = torch.from_numpy(s).float().to(self.device)
        q_values = self.q_net(s)

        # extract index max q-value and reshape tensor to dimensions (1, 1)
        action = q_values.max(1)[1].view(1, 1)
        #print(action)
        # tensor to float
        action = action.item()

        return action

    def get_state_q_stats(self, state: np.array) -> Tuple[float, float]:
        """Return mean and max Q-values for a given state."""
        s = self._preprocess_state(state)
        s = torch.from_numpy(s).float().to(self.device)
        with torch.no_grad():
            q_values = self.q_net(s)

        q_mean = float(q_values.mean().item())
        q_max = float(q_values.max().item())
        return q_mean, q_max

    def observe(self, state, action, reward, next_state, done) -> None:

        # preprocess state
        s = self._preprocess_state(state)
        ns = self._preprocess_state(next_state)

        # store new experience in the agent's memory.
        self.memory.push(s, action, reward, ns, done)

        self._step_counter += 1

    def replay(self) -> None:

        if self._step_counter % self.freq_steps_train != 0:
            # update parameters every `self.freq_steps_update_target`
            # this way we add inertia to the agent actions, as they are more sticky
            return

        if self._step_counter < self.n_steps_warm_up_memory:
            # memory needs to be larger, no training yet
            return

        if self._step_counter % self.freq_steps_update_target == 0:
            # we update the target network parameters
            # self.target_nn.load_state_dict(self.nn.state_dict())
            self._copy_params_to_target_q_net()

        losses = []
        for i in range(0, self.n_gradient_steps):

            # get batch of experiences from the agent's memory.
            batch = self.memory.sample(self.batch_size)

            # A bit of plumbing to transform numpy arrays to PyTorch tensors
            state_batch = torch.cat([torch.from_numpy(s).float().view(1, -1) for s in batch.state]).to(self.device)
            action_batch = torch.cat([torch.tensor([[a]]).long().view(1, -1) for a in batch.action]).to(self.device)
            reward_batch = torch.cat([torch.tensor([r]).float() for r in batch.reward]).to(self.device)
            next_state_batch = torch.cat([torch.from_numpy(s).float().view(1, -1) for s in batch.next_state]).to(self.device)
            done_batch = torch.tensor(batch.done).float().to(self.device)   # getting 1 or 0s corresponding to True and False

            # q_values for all 55 actions
            q_values = self.q_net(state_batch)
            #print(q_values.size())

            # keep only q_value for the chosen action in the trajectory, i.e. `action_batch`, 
            # here .gather() select the q values against the index of action taken only
            q_values = q_values.gather(1, action_batch)

            '''
            with torch.no_grad():, 
            it temporarily disables gradient computation for all tensors within that block.
            '''

            with torch.no_grad():
                if self.use_double_dqn:
                    # Double DQN: pick the next action with the ONLINE net, then
                    # evaluate its value with the TARGET net. Decoupling action
                    # selection from value estimation reduces overestimation bias,
                    # which prevents the peak-then-decline seen with plain DQN.
                    next_q_online = self.q_net(next_state_batch)
                    next_actions = next_q_online.max(dim=1)[1].unsqueeze(1)
                    target_next_q_values = self.target_q_net(next_state_batch).gather(1, next_actions).squeeze(1)
                else:
                    # Plain DQN: max over the target net only.
                    target_next_q_values = self.target_q_net(next_state_batch).max(dim=1)[0]

                # TD target
                target_q_values = (1 - done_batch) * target_next_q_values * self.discount_factor + reward_batch     # (1 - done_batch -> 0/1, here no target Q values for terminal state )

            
            # compute loss
            loss = F.mse_loss(q_values.squeeze(1), target_q_values)
            losses.append(loss.item())

            # backward step to adjust network parameters
            # i) clears the gradients of all model parameters for each batch
            # ii) compute the gradient of the losses
            self.optimizer.zero_grad()
            loss.backward()
            # to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
            # update the model params using computed gradients
            self.optimizer.step()

        if self.log_dir:
            self.logger.add_scalar('train/loss', np.mean(losses), self._step_counter)

    def save_to_disk(self, path: Path) -> None:
        """"""
        if not path.exists():
            os.makedirs(path)

        # save hyper-parameters in a json file
        with open(path / 'hparams.json', 'w') as f:
            json.dump(self.hparams, f)

        if self.normalize_state:
            np.save(path / 'mean_states.npy', self.mean_states)
            np.save(path / 'std_states.npy', self.std_states)

        # save main model (state_dict so it loads safely with torch.load default weights_only=True)
        torch.save(self.q_net.state_dict(), path / 'model')

    @classmethod
    def load_from_disk(cls, env: gym.Env, path: Path):
        """
        We recover all necessary variables to be able to evaluate the agent.

        NOTE: training state is not stored, so it is not possible to resume
        an interrupted training run as it was.
        """
        # load hyper-params
        with open(path / 'hparams.json', 'r') as f:
            hparams = json.load(f)

        # generate Python object
        agent = cls(env, **hparams)

        agent.normalize_state = hparams['normalize_state']
        if hparams['normalize_state']:
            agent.mean_states = np.load(path / 'mean_states.npy')
            agent.std_states = np.load(path / 'std_states.npy')

        # Load the model. Newer checkpoints store a state_dict (loads cleanly);
        # older ones stored a full pickled module, which requires weights_only=False.
        raw = torch.load(path / 'model', map_location=agent.device, weights_only=False)
        if isinstance(raw, dict):
            agent.q_net.load_state_dict(raw)
        else:
            agent.q_net = raw
        agent.q_net.to(agent.device)
        agent.q_net.eval()

        return agent


# In[ ]:




