""" implementations for runner classes, for executing RL-training loops """

import os
import sys
import pathlib
from tqdm import tqdm
from tqdm.notebook import tqdm as notetqdm

import numpy as np
from torch.utils.tensorboard import SummaryWriter

dir_list = str(pathlib.Path().resolve()).split("/")
parent_idx = np.where(np.array(dir_list) == "masc")[0][0]
parent_path = pathlib.Path().resolve()
for idx in range(len(dir_list) - parent_idx - 1):
    parent_path = parent_path.parent
sys.path.append(str(parent_path) + "/")

from masc import misc
from masc.mpc import control_utils


class FullRunner:
    """ runner for train and test loop for whole configurable system """

    def __init__(
            self,
            env,
            agent,
            replay_buffer=None,
            safeguard=None,
            model_fit=None,
            model_delta=0.001,
            state_space_mats=None,
            feasible_set_Nmax=250,
            feasible_set_slack=0.2,
            guard_update_steps=10000,
            safeguard_penalty=-0.0,
            log_keys=None,
            notebook_mode=False,
            use_tensorboard=False,
            tensorboard_log='runs/',
            mpc_run=False,
            save_model_interval=1000000,
            savepath=None,
            savename=None,
            obs_limits=None,
            act_limits=None,
            debug_log=False,
            denormalize_env=False,
            seed=None,
            **kwargs
    ):
        """
        args:


        """
        self.save_model_interval = save_model_interval

        if savepath is not None:
            self.savepath = savepath + '/'
            self.savename = savename
            self.model_save_name = savename + '/' + str(seed)
            self.model_save_path = savepath + '/agent_models/'
        else:
            self.savepath = None
            self.savename = None
            self.model_save_name = None
            self.model_save_path = None

        self.seed = seed

        # standard rl runner blocks
        self.env = env
        self.agent = agent
        self.replay_buffer = replay_buffer

        # additional blocks for safe rl without model-knowledge
        self.safeguard = safeguard
        self.state_space_mats = state_space_mats
        self.feasible_N_max = feasible_set_Nmax
        self.feasible_slack = feasible_set_slack
        self.safeguard_penalty = safeguard_penalty
        self.guard_update_steps = guard_update_steps
        self.model_fit = model_fit
        self.model_delta = model_delta
        self.model_fit_log = {
            'error': [],
            'variance': [],
            'total_steps': 0,
            'model_delta': [],
            'estimated_model': [],
            'updated_episode': [],
        }
        if model_fit is not None:
            self.init_model_fit()

        self.training_mode = True
        self.agent.training_mode = self.training_mode
        self.buffer_filled = False

        self.logger_keys = log_keys
        self.obs_log, self.train_log, self.test_log = self.setup_logs()
        self.writer = None

        self.mpc_run = mpc_run

        self.denormalize_env = denormalize_env
        self.obs_limits = obs_limits
        self.act_limits = act_limits
        self.total_steps = 0

        # deleted later
        self.debug_log = debug_log

        # setup logging features
        if use_tensorboard:
            self.writer = SummaryWriter(tensorboard_log)
        self.notebook_mode = notebook_mode
        if notebook_mode:
            self.tqdm_bar = notetqdm
        else:
            self.tqdm_bar = tqdm

    def init_model_fit(self):
        """ init constraint for safeguard, using initial model matrices """
        # calculate initial safeguard-constraints, based on rnd-system matrices
        n = self.env.observation_space.shape[0]
        m = self.env.action_space.shape[0]
        self.A_prev, self.B_prev = np.random.random((n, n)), np.random.random(
            (n, m))
        if self.safeguard is not None:
            eigs = np.linalg.eigvals(self.A_prev)
            while not (np.abs(eigs) <= 1).all():
                # use only stable system matrice with eigvals within unit-circle
                self.A_prev, self.B_prev = np.random.random(
                    (n, n)), np.random.random((n, m))
                eigs = np.linalg.eigvals(self.A_prev)

            Mx, wx, Mu, wu, Q, R = self.state_space_mats
            feasible_init = control_utils.calc_feasible_set(
                self.A_prev, self.B_prev, Q, R, Mx, wx, Mu, wu,
                project_dim=[1, 2, 3],
                N_max=self.feasible_N_max,
                tol=self.feasible_slack,
                discount=0.9,
                N_start=20,
            )

            self.safeguard.update(
                constraints=feasible_init,
                mode='update',
                #fit_error_margin=np.array([0.5, 0.5]),
            )

    def update(
            self,
            model_estimate=None,
            estimate_variance=None,
            episode=None,
        ):
        """
        update safeguard model, model-fit parameter, ...

        args:

        """
        A_hat, B_hat = model_estimate
        Mx, wx, Mu, wu, Q, R = self.state_space_mats
        var_y1, var_y2 = estimate_variance

        modulo_steps = self.total_steps % self.guard_update_steps
        if (modulo_steps == 0) and (self.total_steps != 0):
            # difference to previous model-update
            delta_A = np.linalg.norm(
                A_hat - self.A_prev) / np.linalg.norm(
                self.A_prev)
            delta_B = np.linalg.norm(
                B_hat - self.B_prev) / np.linalg.norm(
                self.B_prev)
            #print(f'delta A: {delta_A},  steps: {self.total_steps}')

            # update safeguard constraint im change in model-est is significant
            if delta_A >= self.model_delta:
                #print(f'\n\n  {delta_A} \nAold: {self.A_prev}, \nA: {A_hat}')

                feasible_set = control_utils.calc_feasible_set(
                    A_hat, B_hat, Q, R, Mx, wx, Mu, wu,
                    project_dim=[1, 2, 3],
                    N_max=self.feasible_N_max,
                    tol=self.feasible_slack,
                    discount=0.8,
                    N_start=25,
                    notebook_bar=self.notebook_mode,
                    progress_bar=(not self.notebook_mode),
                    verbose=1,
                )

                self.safeguard.update(
                    constraints=feasible_set,
                    fit_error_margin=np.array([var_y1, var_y2]),
                    mode='update',
                )

                self.A_prev, self.B_prev = A_hat, B_hat
                self.model_fit_log['updated_episode'].append((episode))

    def observe(self, episodes=100, max_len=200, max_total_steps=None, progress_bar=True):
        """ collect data and fill buffer without training the agent """
        self.training_mode = True
        self.agent.training_mode = self.training_mode
        model_error = 0
        est_mats, delta_mats, cov_mats = [], [], []
        active_safeguards = 0

        # state space matrices/vector for model-fit log
        A_prev, B_prev = None, None

        if progress_bar:
            pbar = self.tqdm_bar(total=episodes)

        for ep in range(episodes):
            k, done, ref = 0, False, None
            state_log, action_log, reference_log, reward_log = [], [], [], []
            safeguard_actives, error_log = [], []
            sg_active = False

            obs = self.env.reset()

            # check if env returns state, reference tuple
            if type(obs) == tuple:
                state, ref = obs
                if self.denormalize_env:
                    state = state * self.obs_limits
                    ref = ref * self.obs_limits[0]
            else:
                state = obs
                if self.denormalize_env:
                    state = state * self.obs_limits

            while not done:
                action_rl = self.agent.control(state, reference=ref)

                # use safeguard if action will violate constraints
                if self.safeguard is None:
                    action = np.atleast_1d(np.squeeze(action_rl))
                else:
                    action_safe, sg_active = self.safeguard.guide(action_rl, state)
                    action = np.atleast_1d(np.squeeze(action_safe))
                    if sg_active:
                        active_safeguards += 1

                next_obs, reward, done, _ = self.env.step(action)
                if bool(sg_active):
                    reward += self.safeguard_penalty

                if type(next_obs) == tuple:
                    next_state, ref = next_obs
                    if self.denormalize_env:
                        next_state = next_state * self.obs_limits
                        ref = ref * self.obs_limits[0]
                else:
                    next_state = next_obs
                    if self.denormalize_env:
                        next_state = next_state * self.obs_limits

                # fit model on observations
                if self.model_fit is not None:
                    _, model_est = self.model_fit.fit(
                        (state, action), next_state, return_mode='all'
                    )
                    y_hat = self.model_fit.predict((state, action))
                    model_error = self.model_fit.calc_error(metric='Error')
                    P1 = self.model_fit.estimator[0].R
                    P2 = self.model_fit.estimator[1].R
                    # calculate estimation-variance
                    xi = np.concatenate([state, action], axis=0)
                    var_y1 = xi.T @ P1 @ xi
                    var_y2 = xi.T @ P2 @ xi
                    var_estimate = (var_y1, var_y2)

                    # update feasible set based on estimated model parameters
                    self.update(model_est, var_estimate, episode=ep)

                    self.model_fit_log['error'].append(model_error)
                    self.model_fit_log['variance'].append(var_estimate)
                    self.model_fit_log['total_steps'] += 1
                    self.model_fit_log['estimated_model'].append(model_est)

                done = done or k == (max_len - 1)

                self.replay_buffer.store(
                    (state, action, next_state, reward, done, model_error, k)
                )

                k += 1
                self.total_steps += 1
                # safeguard_actives.append(int(sg_active))
                error_log.append(model_error)
                state_log.append(state)
                reference_log.append(ref)
                action_log.append(action)
                reward_log.append(reward)
                error_log.append(model_error)
                state = next_state

            if self.debug_log:
                data2log = [None, k, state_log, action_log, None, None, None]
            else:
                data2log = [None, k, state_log, action_log, active_safeguards]
            self.obs_log.store(data2log)

            if progress_bar:
                pbar.update(1)
                pbar.set_postfix_str(f"total steps: {int(self.total_steps):.3f}")

            if self.total_steps >= max_total_steps:
                break

        if progress_bar:
            pbar.close()

    def train(self, episodes=100, max_len=200, progress_bar=True, visualize=False, **kwargs):
        self.training_mode = True
        self.agent.training_mode = self.training_mode
        model_error = 0

        if progress_bar:
            pbar = self.tqdm_bar(total=episodes)

        for ep in range(episodes):
            k, done, cumulated_reward, sg_active = 0, False, 0, False
            active_safeguards = 0
            state_log, action_log, reference_log, reward_log = [], [], [], []
            safeguard_actives, error_log = [], []
            loss_a_log, loss_c_log, q_val_log = [], [], []
            loss_c , loss_a, q1_hat = None, None, None

            obs = self.env.reset()
            # check if env returns state, reference tuple
            if type(obs) == tuple:
                state, ref = obs
                if self.denormalize_env:
                    state = state * self.obs_limits
                    ref = ref * self.obs_limits[0]
            else:
                state = obs
                if self.denormalize_env:
                    state = state * self.obs_limits

            while not done:
                action_rl = self.agent.control(state, reference=ref)
                action_rl = np.atleast_1d(np.squeeze(action_rl))

                # use safeguard if action will violate constraints
                if self.safeguard is None:
                    action = np.atleast_1d(np.squeeze(action_rl))
                else:
                    action_safe, sg_active = self.safeguard.guide(action_rl, state)
                    action = np.atleast_1d(np.squeeze(action_safe))
                    if sg_active:
                        active_safeguards += 1

                next_obs, reward, done, _ = self.env.step(action)

                # issue safeguard penalty if SG was active
                if bool(sg_active):
                    reward_rl = self.safeguard_penalty
                else:
                    reward_rl = reward

                if type(next_obs) == tuple:
                    next_state, ref = next_obs
                    if self.denormalize_env:
                        next_state = next_state * self.obs_limits
                        ref = ref * self.obs_limits[0]
                else:
                    next_state = next_obs
                    if self.denormalize_env:
                        next_state = next_state * self.obs_limits

                # fit model on observations
                if self.model_fit is not None:
                    _, model_est = self.model_fit.fit(
                        (state, action), next_state, return_mode='all'
                    )
                    y_hat = self.model_fit.predict((state, action))
                    model_error = self.model_fit.calc_error(metric='Error')
                    # update system components
                    # self.update(state, action, 'variance')
                    P1 = self.model_fit.estimator[0].R
                    P2 = self.model_fit.estimator[1].R
                    rls_cov = [P1, P2]
                    # calculate estimation-variance
                    xi = np.concatenate([state, action], axis=0)
                    # variance for one predicted state
                    var_y1 = xi.T @ P1 @ xi
                    var_y2 = xi.T @ P2 @ xi
                    var_estimate = (var_y1, var_y2)

                    # update feasible set based on estimated model parameters
                    self.update(model_est, var_estimate, episode=ep)

                    self.model_fit_log['error'].append(model_error)
                    self.model_fit_log['variance'].append(var_estimate)
                    self.model_fit_log['total_steps'] += 1
                    self.model_fit_log['estimated_model'].append(model_est)

                self.agent.fit(
                    next_state, reward, done, buffer=self.replay_buffer)

                done = done or k == (max_len - 1)
                # store guarded tuple
                self.replay_buffer.store(
                    (state, action, next_state, reward, done, model_error, k)
                )
                if sg_active:
                    #store rl tuple if safeguard is active
                    self.replay_buffer.store(
                        (state, action_rl, next_state, reward_rl, done, model_error,
                         k)
                    )

                k += 1
                self.total_steps += 1
                cumulated_reward += reward_rl
                state_log.append(state)
                reference_log.append(ref)
                action_log.append(action)
                reward_log.append(reward_rl)
                error_log.append(model_error)
                loss_a_log.append(loss_a)
                loss_c_log.append(loss_c)
                q_val_log.append(q1_hat)

                state = next_state

                if visualize:
                    self.env.render()

            # decay agents learning rate for actor and critic
            # self.agent.lr_scheduler_actor.step()
            # self.agent.lr_scheduler_critic.step()

            if self.debug_log:
                # log additional values for debugging
                data2log = [cumulated_reward, k, state_log, action_log, loss_a_log, loss_c_log, q_val_log]
            else:
                data2log = [
                    cumulated_reward,
                    k,
                    state_log,
                    action_log,
                    active_safeguards,
                    reward_log
                ]

            self.train_log.store(data2log)

            if self.writer is not None:
                self.writer.add_scalar('Training/reward', cumulated_reward, ep)
                self.writer.add_scalar('Training/episode-length', k, ep)
                self.writer.add_scalar(
                    'Training/action-Noise', self.agent.action_noise_, ep
                )
                self.writer.add_scalar(
                    'Training/learning-rate', float(self.agent.lr_scheduler_actor.get_last_lr()[0]), ep
                )
                if self.safeguard is not None:
                    self.writer.add_scalar('Training/safeguard-actives',
                                           active_safeguards, ep)
                if self.model_fit is not None:
                    self.writer.add_scalar('Training/mean-fitting-error',
                                           np.mean(error_log), ep)

            if progress_bar:
                pbar.update(1)
                pbar.set_postfix_str(f"cumulated rew: {(cumulated_reward):.3f}")

            if not(bool(ep % self.save_model_interval)) and (ep != 0):
                try:
                    os.mkdir(self.model_save_path + self.model_save_name + '/')
                except FileExistsError:
                    print('Folder exits already')

                self.agent.save_agent('ep_' + str(ep),
                                      self.model_save_path + self.model_save_name + '/')

        # save latest model
        if self.model_save_name is not None:
            try:
                os.mkdir(self.model_save_path + self.model_save_name + '/')
            except FileExistsError:
                print('Folder exits already')

            self.agent.save_agent('ep_' + str(ep),
                                  self.model_save_path + self.model_save_name + '/')

        if self.writer is not None:
            self.writer.close()

        if progress_bar:
            pbar.close()

    def test(self,
             episodes=10,
             max_len=200,
             progress_bar=True,
             visualize=False,
             deactivate_safeguard=False,
             deactivate_model_fit=False,
             ):
        self.training_mode = False
        self.agent.training_mode = self.training_mode
        model_error = 0

        if progress_bar:
            pbar = self.tqdm_bar(total=episodes)

        for ep in range(episodes):
            k, done, cumulated_reward, sg_active = 0, False, 0, False
            active_safeguards = 0
            state_log, action_log, reference_log, reward_log = [], [], [], []
            safeguard_actives, error_log = [], []

            obs = self.env.reset()
            # check if env returns state, reference tuple
            if type(obs) == tuple:
                state, ref = obs
            else:
                state, ref = obs, 0

            while not done:
                action_rl = self.agent.control(state, reference=ref)
                if (self.safeguard is None) or deactivate_safeguard:
                    action = np.atleast_1d(np.squeeze(action_rl))
                else:
                    action_safe, sg_active = self.safeguard.guide(action_rl, state)
                    action = np.atleast_1d(np.squeeze(action_safe))
                    if sg_active:
                        active_safeguards += 1
                        reward_rl = - float(self.safeguard_penalty)


                next_obs, reward, done, _ = self.env.step(action)
                if bool(sg_active):
                    reward += self.safeguard_penalty

                if type(next_obs) == tuple:
                    next_state, ref = next_obs
                else:
                    next_state = next_obs

                done = done or k == (max_len - 1)

                k += 1
                cumulated_reward += reward
                state_log.append(state)
                reference_log.append(ref)
                action_log.append(action)
                reward_log.append(reward)
                safeguard_actives.append(int(sg_active))
                error_log.append(model_error)

                state = next_state

                if visualize:
                    self.env.render()
                    # time.sleep(0.05)

            if self.debug_log:
                data2log = [cumulated_reward, k, state_log, action_log, None, None, None]
            else:
                data2log = [
                    cumulated_reward,
                    k,
                    state_log,
                    action_log,
                    active_safeguards,
                    reward_log
                ]
            self.test_log.store(data2log)

            if self.writer is not None:
                self.writer.add_scalar('Testing/reward', cumulated_reward, ep)
                self.writer.add_scalar('Testing/episode-length', k, ep)
                self.writer.add_scalar('Testing/safeguard-actives',
                                       active_safeguards, ep)
                if self.model_fit is not None:
                    self.writer.add_scalar('Testing/mean-fitting-error',
                                           np.mean(error_log), ep)
            if progress_bar:
                pbar.update(1)
                pbar.set_postfix_str(f"cumulated rew: {int(cumulated_reward):.3f}")

        if self.writer is not None:
            self.writer.close()
        if progress_bar:
            pbar.close()

        self.env.close()

    def setup_logs(self,):
        """ setup dictionary for logging training/ testing process """
        obs_log = misc.Logger(self.logger_keys)
        train_log = misc.Logger(self.logger_keys)
        test_log = misc.Logger(self.logger_keys)

        return obs_log, train_log, test_log


