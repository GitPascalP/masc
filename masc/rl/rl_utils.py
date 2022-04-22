""" basic implementations for creating replay-buffer, policy networks, ... """

import random
from collections import deque, namedtuple
import numpy as np

import torch
import torch.nn as nn


class ReplayBuffer:
    """ Ringbuffer to store experienced data during learning """

    def __init__(self, size, element_keys, seed=None):
        """
        args:
            size(int): size of replay-buffer
            element_keys(tuple): names of entries stored in buffer
            seed (int): random seed
        """
        self.size = size
        self.entry = self.create_entry("entry", element_keys)
        self.buffer = deque(maxlen=size)

        self.len = len(self.buffer)
        self.seed = seed

    @staticmethod
    def create_entry(name, elements):
        """ create namedtuple as buffer entry """
        return namedtuple(name, elements, defaults=(None,)*len(elements))

    def store(self, observation):
        """ store observation in replay-buffer """
        obs_entry = self.entry(*observation)
        self.buffer.append(obs_entry)
        self.len = len(self.buffer)

    def sample(self, batch_size):
        """ sample mini-batch from replay-buffer """
        np.random.seed(self.seed)
        batch_samples = random.sample(self.buffer, batch_size)
        batch = self.entry(*zip(*batch_samples))
        return batch


def get_tensors(batch, device):
    """ convert batch data to corresponing torch-tensors """
    # convert batch form deque to array to allow faster tensor conversion
    obs_batch = np.asarray([np.squeeze(bo) for bo in batch.obs])
    next_obs_batch = np.asarray([np.squeeze(bo) for bo in batch.next_obs])
    action_batch = np.asarray(batch.action)
    reward_batch = np.asarray(batch.reward)
    done_batch = np.asarray(batch.done)

    obs_tensor = torch.FloatTensor(obs_batch).to(device)
    next_obs_tensor = torch.FloatTensor(next_obs_batch).to(device)
    action_tensor = torch.FloatTensor(action_batch).to(device)
    reward_tensor = torch.FloatTensor(reward_batch).to(device)
    done_tensor = 1 - torch.FloatTensor(done_batch).to(device)

    return obs_tensor, next_obs_tensor, action_tensor, reward_tensor, done_tensor


def basic_network(sizes, activation, output_activation):
    layers = []

    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    network = nn.Sequential(*layers)

    return network


class Network(nn.Module):
    """ mlp as function-approximator """

    def __init__(
            self,
            sizes,
            activation=nn.ReLU,
            output_activation=nn.Identity,
            clipping=False,
            limit=1,
    ):
        """
        args:
            sizes(list): sizes of layers, first and last entry are
                         in and output dimensions
            activation(torch activation): activation for hidden layers
            output_activation(torch activation): output-layer activation
            limit: output-layer limits
        """
        super().__init__()
        self.layer_stack = basic_network(sizes, activation, output_activation)
        self.limit = limit
        self.clipping = clipping

    def forward(self, x):
        """   """
        if len(x.size()) >= 3:
            x = x.squeeze(-1)

        if self.clipping:
            out = torch.clamp(self.layer_stack(x), -int(self.limit), int(self.limit))
        else:
            out = self.layer_stack(x) * torch.FloatTensor(self.limit)

        return out


class DoubleNetwork(nn.Module):
    """ mlp function-approximator with 2 networks calculating to 2 outputs """

    def __init__(
            self,
            sizes,
            activation=nn.ReLU,
            output_activation=nn.Identity,
    ):
        """
        args:
            sizes(list): sizes of layers, first and last entry are
                         in and output dimensions
            activation(torch activation): activation for hidden layers
            output_activation(torch activation): output-layer activation
        """
        super().__init__()
        self.layer_stack1 = basic_network(sizes, activation, output_activation)
        self.layer_stack2 = basic_network(sizes, activation, output_activation)

    def forward(self, input1, input2):
        input2 = input2.view(-1, 1)
        x_ = torch.cat((input1, input2), dim=1).float()
        out1 = self.layer_stack1(x_)
        out2 = self.layer_stack2(x_)
        return out1, out2

    def get_output(self, input1, input2):
        x_ = torch.cat((input1, input2), dim=1).float()
        out1 = self.layer_stack1(x_)
        return out1
