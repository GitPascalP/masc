""" Various RL-Agent implementations """
#TODO: add paper for agents
import sys
import pathlib
import numpy as np
import gym
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

dir_list = str(pathlib.Path().resolve()).split("/")
parent_idx = np.where(np.array(dir_list) == "masc")[0][0]
parent_path = pathlib.Path().resolve()
for idx in range(len(dir_list) - parent_idx - 1):
    parent_path = parent_path.parent
sys.path.append(str(parent_path) + "/")

from masc.rl.rl_utils import get_tensors


class Agent:
    """ basic structure for an RL-agent """

    def __init__(self,):
        """
        args:

        """
        raise NotImplementedError

    def preprocess(func):
        """ preprosseing observations for feature engineering """

        def wrapper(self, observation, *args, **kwargs):
            observation_new = observation
            return func(self, observation_new, *args, **kwargs)

        return wrapper

    @preprocess
    def controle(self, observation):
        """ give control output in reaction to observation """
        raise NotImplementedError

    @preprocess
    def fit(self, observation, reward, done=False):
        """ fit rl function approximator(s) and learn optimal behaviour """
        raise NotImplementedError

    def save_agent(self,):
        """ save network model for agent """
        raise NotImplementedError

    def load_agent(self,):
        """ load pretrained network model for agent """
        raise NotImplementedError


class DQNAgent:
    """
    Deep Q-Network Agent

    references:

    """

    def __init__(
        self,
        environment,
        replay_buffer,
        model,
        target_model,
        double_dqn=False,
        batch_size=120,
        update_frequency=20,
        gamma=0.99,
        warm_up=1000,
        eps_start=0.9,
        eps_end=0.05,
        eps_decay=1000,
        learning_rate=0.001,
        normalize=False,
        device="cpu",
    ):
        """
        args:
            environment :
            replay_buffer :
            model
            target_model
            double_dqn=False
            batch_size
            update_frequency
            gamma
            warm_up
            eps_start
            eps_end
            eps_decay
            learning_rate
            normalize
            device
        """
        self.env = environment
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n
        self.actions, self.last_obs = None, None
        self.obs_limits = (
            self.env.observation_space.low,
            self.env.observation_space.high,
        )
        self.act_limits = (self.env.action_space.low, self.env.action_space.high)

        # dqn hyperparameters
        self.double_dqn = double_dqn
        self.batch_size, self.k_w = batch_size, update_frequency
        self.gamma, self.warm_up = gamma, warm_up
        # exploration
        self.eps_start, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay

        self.experience_buffer = replay_buffer
        self.Experience = namedtuple(
            "Expericence",
            ("obs", "action", "next_obs", "reward", "done"),
            defaults=(None,) * 5,
        )

        # Get cpu or gpu device for training.
        self.device = device
        self.normalize = normalize or True
        # print("Using {} device".format(self.device))

        self.model, self.target_model = (
            model.to(self.device),
            target_model.to(self.device),
        )
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fct = nn.MSELoss()

        self.iter = 0
        self.training_mode = True

    def preprocess(func):
        """ preprosseing observations for feature engineering """

        def wrapper(self, obs, *args, **kwargs):
            if self.normalize:
                upper, lower = 1, -1
                obs_new = (upper - lower) * (obs - self.obs_limit[0]) / (
                    self.obs_limit[1] - self.obs_limit[0]
                ) + upper
            else:
                obs_new = obs
            return func(self, obs_new, *args, **kwargs)

        return wrapper

    @preprocess
    def controle(self, observation, **kwargs):
        """ give control output in reaction to observation """
        # epsilon-greedy decision with epsilon-decay for better exploration
        epsilon = self.eps_end + (self.eps_start - self.eps_end) * np.exp(
            -1.0 * self.iter / self.eps_decay
        )

        if self.training_mode:
            if epsilon <= np.random.uniform(0, 1):
                # greedy action
                obs = (
                    torch.tensor(observation, dtype=torch.float)
                    .to(self.device)
                    .view(1, self.n_states)
                )
                with torch.no_grad():
                    action = int(self.model(obs).argmax(1))
            else:
                # random action
                rnd_action = np.random.choice(np.arange(self.n_actions))
                action = int(rnd_action)
        else:
            obs = (
                torch.tensor(observation, dtype=torch.float)
                .to(self.device)
                .view(1, self.n_states)
            )
            with torch.no_grad():
                action = int(self.model(obs).argmax(1))
        # remember action and observation
        self.action, self.last_obs = action, observation
        return action

    @preprocess
    def fit(self, observation, reward=0, done=False):
        """ fit rl function approximator(s) and learn optimal behaviour """
        # remember experience of this iteration
        experience = self.Experience(
            obs=self.last_obs,
            action=self.action,
            next_obs=observation,
            reward=reward,
            done=done,
        )
        # and store it in replay-buffer
        self.experience_buffer.store(experience)
        # collect enough experience
        if len(self.experience_buffer.buffer) < self.warm_up:
            return

        # list of tuples converted in tuple of lists
        batch_samples = self.experience_buffer.sample(self.batch_size)
        batch = self.Experience(*zip(*batch_samples))
        # convert batches in torch tensors
        obs_tensor = torch.FloatTensor(batch.obs).to(self.device)
        next_obs_tensor = torch.FloatTensor(batch.next_obs).to(self.device)
        action_tensor = torch.LongTensor(batch.action).to(self.device)
        action_tensor = action_tensor.unsqueeze(1)
        reward_tensor = torch.FloatTensor(batch.reward).to(self.device)
        # used for filtering finalstates out in q-calculation
        done_mask = 1 - torch.FloatTensor(batch.done).to(self.device)

        if self.double_dqn:
            # using best action from model-network to choose q for q-target estimation
            with torch.no_grad():
                q_target = self.target_model(next_obs_tensor)
                best_action = (self.model(next_obs_tensor).argmax(1)).unsqueeze(1)
                q_target = q_target.gather(1, best_action)
            target = reward_tensor + (done_mask * self.gamma * q_target.squeeze())
        else:
            # using best q from target-network for q-target estimation
            with torch.no_grad():
                q_target = self.target_model(next_obs_tensor)
            target = reward_tensor + done_mask * self.gamma * q_target.max(1)[0]

        # estimated q for eps-greedy action
        q_hat = self.model(obs_tensor).gather(1, action_tensor)

        # calc loss
        loss = self.loss_fct(target.squeeze(), q_hat.squeeze())
        # optimize gradients
        self.optimizer.zero_grad()
        loss.backward()
        # clamp parameters to avoid exploding gradients
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        if (self.iter % self.k_w) == 0:
            # target-network update
            self.target_model.load_state_dict(self.model.state_dict())

        if done:
            # if episode terminates, update target-network
            self.target_model.load_state_dict(self.model.state_dict())

        self.iter += 1

    def save_agent(self,):
        """ save network model for agent """
        raise NotImplementedError

    def load_agent(self,):
        """ load pretrained network model for agent """
        raise NotImplementedError


class DDPGAgent:
    """
    Deep Deterministic Policy Gradient Agent with possible TD3 extension

    references:

    """

    def __init__(
        self,
        environment,
        actor_model,
        actor_target_model,
        critic_model,
        critic_target_model,
        warm_up=5000,
        batch_size=128,
        delay=1,
        mue=0,
        policy_noise=0.2,
        noise_clipper=0.5,
        action_noise=0.1,
        noise_decay=False,
        noise_start=1,
        noise_end=0.1,
        noise_decay_rate=0.98,
        noise_decay_steps=1000,
        tau=0.005,
        gamma=0.99,
        learning_rate=1e-3,
        lr_decay=0.1,
        lr_decay_step=100,
        TD3=False,
        normalize=None,
        device="cpu",
    ):
        """
        args:
            environment,
            actor_model,
            actor_target_model,
            critic_model,
            critic_target_model,
            warm_up=5000,
            batch_size=128,
            delay=1,
            mue=0,
            policy_noise=0.2,
            noise_clipper=0.5,
            action_noise=0.1,
            noise_decay=False,
            noise_start=1,
            noise_end=0.1,
            noise_decay_rate=0.98,
            noise_decay_steps=1000,
            tau=0.005,
            gamma=0.99,
            learning_rate=1e-3,
            lr_decay=0.1,
            lr_decay_step=100,
            TD3=False,
            normalize=None,
            device="cpu",
        """
        self.env = environment
        if isinstance(self.env.observation_space, gym.spaces.tuple.Tuple):
            # if observations are constisting of state and reference
            self.n_states = self.env.observation_space[0].shape[0]
            self.s_low = self.env.observation_space[0].low
            self.s_high = self.env.observation_space[0].high
        else:
            self.n_states = self.env.observation_space.shape[0]
            self.s_low = self.env.observation_space.low
            self.s_high = self.env.observation_space.high

        self.n_actions = self.env.action_space.shape[0]
        self.action, self.last_obs = None, None
        self.a_low, self.a_high = (
            self.env.action_space.low[0],
            self.env.action_space.high[0],
        )

        self.device = device
        self.TD3 = TD3

        # hyperparameter
        self.mue, self.action_noise = mue, action_noise
        self.policy_noise, self.noise_clipper = (
            policy_noise,
            noise_clipper,
        )
        self.noise_decay, self.start_value, self.end_value = (
            noise_decay,
            noise_start,
            noise_end,
        )
        self.decay_steps, self.decay_rate = noise_decay_steps, noise_decay_rate
        self.action_noise_ = None
        self.warm_up, self.batch_size = warm_up, batch_size
        self.tau, self.gamma = tau, gamma

        if self.TD3:
            self.delay = delay
        else:
            self.delay = 1

        # setup function-approximator
        self.actor, self.target_actor = (
            actor_model.to(self.device),
            actor_target_model.to(self.device),
        )
        self.critic, self.target_critic = (
            critic_model.to(self.device),
            critic_target_model.to(self.device),
        )
        # init target parameters/ weights and set in eval-mode?
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.eval()
        self.target_critic.eval()

        # set up optimizers
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.lr_scheduler_actor = optim.lr_scheduler.StepLR(
            self.actor_opt, step_size=lr_decay_step, gamma=lr_decay
        )
        self.lr_scheduler_critic = optim.lr_scheduler.StepLR(
            self.critic_opt, step_size=lr_decay_step, gamma=lr_decay
        )

        self.normalize = normalize or False
        self.iter = 0
        self.training_mode = True

    def preprocess(func):
        """ preprosseing observations for feature engineering """

        def wrapper(self, obs, *args, **kwargs):
            if self.normalize:
                upper, lower = 1, -1
                obs_new = (upper - lower) * (obs - self.s_low) / (
                    self.s_high - self.s_low
                ) + upper
            else:
                # possible feature engineering can be implemented here
                obs_new = obs
            return func(self, obs_new, *args, **kwargs)

        return wrapper

    @preprocess
    def control(self, observation, **kwargs):
        """ give control output in reaction to observation """
        if self.training_mode:
            self.action_noise_ = self.action_noise
            if self.noise_decay:
                # decaying actor-noise
                self.action_noise_ = (
                    self.start_value - self.end_value
                ) * self.decay_rate ** (self.iter / self.decay_steps) + self.end_value
            # get action from actor and add noise for exploration
            with torch.no_grad():
                self.actor.eval()
                observation_ = torch.FloatTensor(observation).to(self.device)
                action = self.actor(observation_.unsqueeze(0))
                action = action + torch.normal(
                    self.mue, self.action_noise_, size=action.size()
                ).to(self.device)
        else:
            with torch.no_grad():
                self.actor.eval()
                observation_ = torch.FloatTensor(observation).to(self.device)
                observation_ = observation_.view(1, -1)
                action = self.actor(observation_).to(self.device)

        # clip actions to maximal action-space bounds
        action = ((torch.clip(action, self.a_low, self.a_high)).cpu()).numpy()
        self.action, self.last_obs = action, observation
        return action

    @preprocess
    def fit(self, observation, reward, done=False, update_steps=1, buffer=None):
        """ fit rl function approximator(s) and learn optimal behaviour """
        if buffer.len < self.warm_up:
            return None, None, None

        for j in range(update_steps):
            # list of tuples converted in tuple of lists
            batch = buffer.sample(self.batch_size)
            # convert batches in torch tensors
            obs, next_obs, action, r, d = get_tensors(batch, self.device)

            with torch.no_grad():
                if self.TD3:
                    # compute next action with additional clipped noise on policy/ actor
                    clipped_noise = np.clip(
                        np.random.normal(self.mue, self.policy_noise, self.n_actions),
                        -self.noise_clipper,
                        self.noise_clipper,
                    )
                    clipped_noise = torch.FloatTensor(clipped_noise).to(self.device)
                    target_action_next = (
                        torch.clip(
                            self.target_actor(next_obs) + clipped_noise,
                            self.a_low,
                            self.a_high,
                        )
                    ).float()
                    # calculate action-values from both critics
                    target_q1, target_q2 = self.target_critic(
                        next_obs, target_action_next
                    )
                    # and select the smaller one
                    target_q = torch.min(target_q1, target_q2)
                else:
                    target_action_next = self.target_actor(next_obs)
                    target_q, _ = self.target_critic(next_obs, target_action_next)

                # estimate target
                y_target = r.squeeze() + (self.gamma * d * target_q.squeeze())

            self.critic.train()
            q1_hat, q2_hat = self.critic(obs, action)

            # calculate loss for critic-update
            if self.TD3:
                loss_c = F.mse_loss(q1_hat.squeeze(), y_target) + F.mse_loss(
                    q2_hat.squeeze(), y_target
                )
            else:
                loss_c = F.mse_loss(q1_hat.squeeze(), y_target)

            # critic optimization
            self.critic_opt.zero_grad()
            loss_c.backward()
            self.critic_opt.step()

            # delayed actor update
            if (self.iter % self.delay) == 0:
                self.actor.train()
                # Q as actor-loss, negative to achieve gradient-ascent
                loss_a = -self.critic.get_output(obs, self.actor(obs)).mean()

                # actor optimization
                self.actor_opt.zero_grad()
                loss_a.backward()
                self.actor_opt.step()

                # smooth target-network updates
                for tc_params, c_params in zip(
                    self.target_critic.parameters(), self.critic.parameters()
                ):
                    # update target-critic
                    tc_params.data.mul_(1 - self.tau)
                    tc_params.data.add_((self.tau) * c_params.data)

                for ta_params, a_params in zip(
                    self.target_actor.parameters(), self.actor.parameters()
                ):
                    # update target-actor
                    ta_params.data.mul_(1 - self.tau)
                    ta_params.data.add_((self.tau) * a_params)

        self.iter = self.iter + 1

    def save_agent(self, name, path):
        """ save network model for agent """
        torch.save(self.actor.state_dict(), path + name + '_actor.pth')
        torch.save(self.critic.state_dict(), path + name + '_critic.pth')
        print('successfully saved')

    def load_agent(self, name, path):
        """ load pretrained network model for agent """
        self.actor.load_state_dict(torch.load(path + name + '_actor.pth'))
        self.critic.load_state_dict(torch.load(path + name + '_critic.pth'))
        # write loaded models in target networks
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        print('successfully loaded')
