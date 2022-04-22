""" implementation and wrappers for system-models and environments """
# TODO überarbeiten, aufräumen
import gym
import gym_electric_motor as gem
import warnings
from collections import deque
import numpy as np
import padasip as sip
from scipy.integrate import ode
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def constant_ref(t):
    """ constant equilibrium state reference """
    return np.array([5, 0])


def doube_integrator_ode(t, states, action, coeffs=(0, 0, 1)):
    a0, a1, a2 = coeffs
    b0 = 1 / a2

    x, x_dot = states
    states_dot = [x_dot, -a1 * x_dot - a0 * x + b0 * action]

    return states_dot


class DoubleIntegrator(gym.Env):
    """ double-integrator environment for cont. state and action spaces """

    def __init__(
        self,
        model=None,
        obs_limits=None,
        act_limits=None,
        critical_limits=None,
        init_limits=None,
        init_value=None,
        ref_init=None,
        feasible_set=None,
        reward_weights=None,
        reward_gamma=None,
        reward_bias=None,
        reward_exponent=None,
        reward_qoutient=None,
        sg_penalty=0.0,
        N=None,
        M=None,
        tau=None,
        max_episode_length=None,
        violation_penalty=None,
        violation_scaler=None,
        gamma=0.99,
        seed=None,
        normalize=False,
        noise_level=None,
        noise_sigma=None,
        model_type=None,
        rew_type='wse_reward',
    ):
        """
        args:
            ode_coeffs
            obs_limits(array-like): maximal observation-values
            action_limits(array-like): maximal action-values
            critical_limits
            init_limits
            feasible_set (polytope.Polytope): 2-D polytope
            reward_weights
            reward_gamma
            reward_bias
            reward_exponent
            N(int): state/observation dimension
            M(int): action dimension
            tau(float): time difference between timesteps
            max_episode_length
            violation_penalty
            seed
            normalize

        TODO: update docs
        """
        self.model_type = model_type or 'ode'
        self.model = model or (0, 0, 1)
        self.physical_ode = ode(doube_integrator_ode).set_integrator("dopri5")
        self.N, self.M = N, M
        self.normalize_env = normalize

        # defining action, observation space
        self.obs_limits = obs_limits
        self.act_limits = act_limits
        self.init_limits = init_limits
        self.init_value = init_value
        self.ref_init = ref_init
        self.feasible_set = feasible_set

        if critical_limits is not None:
            self.obs_limits = critical_limits

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.action_space = gym.spaces.Box(-self.act_limits,
                                               self.act_limits,
                                               dtype=np.float32)
            self.observation_space = gym.spaces.Box(-self.obs_limits,
                                                    self.obs_limits,
                                                    dtype=np.float32)

        self.reference_value = np.zeros(self.N)
        self.tau = tau
        self.max_episode_length = max_episode_length or 1000

        self.rew_weights = reward_weights
        if reward_weights is None:
            self.rew_weights = np.ones(self.N)

        self.rew_type = rew_type
        self.rew_bias = reward_bias or 0
        self.rew_exp = reward_exponent or 1
        self.rew_quotient = reward_qoutient or 1
        self.violation_penalty = violation_penalty or -1000
        self.violation_scaler = violation_scaler or 1
        self.state_span = 2*self.obs_limits
        self.sg_penalty_value = sg_penalty
        self._gamma = gamma

        self.noise_level = noise_level or 0
        self.noise_sigma = noise_sigma or 0

        self.seed(seed=seed)
        self.state = None
        self.done = None
        self.t = 0
        self.iter = 0

    def get_reference(self,):
        """ reference/target observation defined by reference function

        because the double-integrator will be mostly used for set-point
        control, the reference is set on a constant value within the feasible
        set (random for each episode)
        """
        return self.reference_value

    def get_reward(self, state, reference, safeguard_active):
        """ return mse/ weighted mse between state and reference as reward """
        safeguard_penalty = 0.0

        if self.done and (self.iter < self.max_episode_length):

            if self.rew_type == 'wse_reward':
                return self.violation_penalty

            elif self.rew_type == 'gamma_reward':
                # calculate violation penalty
                violation_factor = np.abs(state / self.obs_limits)
                reward = -1 * np.sum(violation_factor) * self.violation_scaler
                return reward

        if self.rew_type== 'wse_reward':
            norm_error = np.abs(state - reference) / self.state_span
            wse = -np.sum(self.rew_weights * norm_error**self.rew_exp) + self.rew_bias + safeguard_penalty
            wse = wse / self.rew_quotient
            return wse

        elif self.rew_type == 'gamma_reward':
            # calculate reward
            error = state - reference
            reward = 2 * (1 - self._gamma) * (1 - np.sum(np.abs(error / self.state_span))) - (1 - self._gamma)
            return reward

        else:
            return -np.mean(
                    (self.obs[self.target_idx] - self.reference[self.target_idx]) ** 2
                )

    def step(self, action, safeguard_active=False):
        """ take a step in environment """
        action = np.atleast_1d(np.squeeze(action))

        if self.model_type == 'ode':
            self.physical_ode.set_f_params(action, self.model)
            self.obs = self.physical_ode.integrate(self.physical_ode.t + self.tau)

        elif self.model_type == 'exact':
            A, B = self.model
            self.obs = np.matmul(A, self.obs) + np.matmul(B, action)

        self.reference = self.get_reference()

        # terminate if episode-end or constraint violation occured
        self.done = not (np.abs(self.obs) <= self.obs_limits).all()
        reward = self.get_reward(self.obs, self.reference, safeguard_active)

        self.t += self.tau
        self.iter += 1

        if self.normalize_env:
            return (self.normalize(np.array(self.noise_sigma * np.random.randn() + self.obs)), \
                   self.normalize(np.array(self.reference))), reward, self.done, {}
        else:
            return (np.array(self.noise_sigma * np.random.randn() + self.obs),
                    np.array(self.reference)), reward, self.done, {}

    def reset(self):
        """ reset a important parameters in environment """
        if self.init_limits is not None:
            limits = self.init_limits
        else:
            limits = self.obs_limits

        if self.init_value is not None:
            self.obs = self.init_value
        else:
            self.obs = self.np_random.uniform(-limits,
                                              limits)

        # ensure init state is in feasible set
        if self.feasible_set is not None:
            feasible_2d_set = self.feasible_set.project([1, 2])
            while not (self.obs in feasible_2d_set):
                self.obs = self.np_random.uniform(-limits,
                                                  limits)

        if self.ref_init is None:
            self.reference_value[0] = self.np_random.uniform(-limits[0],
                                                             limits[0])
            self.reference_value[1] = 0

            # ensure init state is in feasible set
            if self.feasible_set is not None:
                feasible_2d_set = self.feasible_set.project([1, 2])
                while not (self.reference_value in feasible_2d_set):
                    self.reference_value[0] = self.np_random.uniform(
                        -limits[0],
                        limits[0])
                    self.reference_value[1] = 0
        else:
            self.reference_value = self.ref_init

        self.reference = self.get_reference()
        self.physical_ode.set_initial_value(self.obs, 0.0)

        self.t = 0
        self.iter = 0
        self.done = False

        # add noise to env
        if self.noise_level is not None:
            noise = self.noise_level * np.random.rand()
        else:
            noise = 0

        if self.normalize_env:
            return (self.normalize(np.array(self.noise_sigma * np.random.randn() + self.obs)),
                    self.normalize(np.array(self.reference)))
        else:
            return (np.array(self.noise_sigma * np.random.randn(2) + self.obs),
                    np.array(self.reference))

        # if self.normalize_env:
        #     return (self.normalize(np.array(self.obs + noise)), \
        #            self.normalize(np.array(self.reference)))
        # else:
        #     return np.array(self.obs + noise), np.array(self.reference)

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def normalize(self, x, norm_limits=(-1, 1)):
        """ min-max normalize observations """
        scaler = norm_limits[1] - norm_limits[0]
        max_limits, min_limits = self.obs_limits, -self.obs_limits
        x_norm = (scaler * (x - min_limits) / (max_limits - min_limits)) + norm_limits[0]
        return x_norm


class LSFit:
    # TODO delete weil LS Fit eh nicht gebraucht
    """ class for linear model fitting of collected environment data"""

    def __init__(self,
                 test_split=None,
                 dimensions=None,
                 scaler=None,
                 algorithm=None,
                 seed=None):
        """
        args:
            test_split(float): part of data  for testing and error calculation
            dimensions(tuple): state space dimensions (state-dim, act-dim)
            scaler(str): used prescaling befor model fit
            algorithm(str): algorithm for model fit
            seed(int): random seed
        """
        self.test_split = test_split or 1/3
        self.scaler = scaler or None
        self.algorithm = algorithm or 'OLS'
        self.seed = seed
        self.n, self.m = dimensions

        self.estimator = None
        self.train_data = None
        self.test_data = None

    def scale(self, data):
        """ pre-scale data used for fitting """
        if self.scaler == 'normalize':
            raise NotImplementedError
        else:
            return data

    def restructure(self, batch):
        """ restructure data for fit from namedtuple or observation tuple """
        if self.algorithm == 'OLS':
            # TODO: right now LS needs namedtuple
            x_k = np.asarray(batch.obs)
            u_k = np.squeeze(np.asarray(batch.action))
            x_kn = np.asarray(batch.next_obs)

            # build regressor matrix and data vector
            Xi = np.concatenate(
                [x_k[:, 0, None], x_k[:, 1, None], u_k[:, None]], axis=-1
            )
            psi = x_kn
            # split in train and test data
            Xi_train, Xi_test, psi_train, psi_test = train_test_split(
                                                    Xi, psi,
                                                    test_size=self.test_split,
                                                    random_state=self.seed)
            return (Xi_train, psi_train), (Xi_test, psi_test)

        elif self.algorithm == 'RLS':
            # build measurement vector for current time step
            xi = np.concatenate(batch)

        else:
            raise NotImplementedError

    def fit(self, batch, next_obs=None, model='state_space'):
        """ fit data with selected algorithm
        args:
            data:
            model:

        return:
            fitted model as instance of estimator or state-space matrices
        """

        if self.algorithm == 'OLS':
            scaled_batch = self.scale(batch)
            train_data, test_data = self.restructure(scaled_batch)
            # LS fit model
            Xi, psi = train_data
            self.estimator = LinearRegression()
            self.estimator.fit(Xi, psi)

            # create estimated state space model
            A_hat = self.estimator.coef_[:, :self.n]
            B_hat = self.estimator.coef_[:, self.n:]

            return self.estimator.coef_, (A_hat, B_hat)

        elif self.algorithm == 'RLS':
            scaled_batch = self.scale(batch)
            xi = self.restructure(scaled_batch)

            state_hat1 = self.estimator[0].predict(xi)
            state_hat2 = self.estimator[1].predict(xi)
            state_hat = np.array([state_hat1, state_hat2])

            self.estimator[0].adapt(next_obs[0], xi)
            self.estimator[1].adapt(next_obs[1], xi)

            state_coeff1, state_coeff2 = self.estimator[0].w[:self.n], \
                                         self.estimator[1].w[:self.n]
            action_coeff1, action_coeff2 = self.estimator[0].w[self.n:], \
                                           self.estimator[1].w[self.n:]

            A_hat = np.concatenate(
                [state_coeff1[None, :], state_coeff2[None, :]], axis=0)
            B_hat = np.concatenate(
                [action_coeff1[None, :], action_coeff2[None, :]], axis=0)
        else:
            raise NotImplementedError

    def calc_error(self, metric='MSE'):
        """ calculate prediction errors """
        Xi, psi = self.test_data
        psi_hat = self.estimator.predict(Xi)
        error = psi - psi_hat
        mse = np.mean(error ** 2)

        return error, mse


class RLSFit:
    """ class utilizing existing RLS filter, adding a few helping features """

    def __init__(self, state_dim, action_dim, mu=0.5, buffer_len=1):
        """
        args:

        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        # setup measurement vector
        self.xi = None
        self.next_state = None

        self.estimator = [
            sip.filters.FilterRLS(n=state_dim+action_dim, mu=mu) for i in range(state_dim)
        ]

        self.error_buffer = deque(maxlen=buffer_len)

    def fit(self, state_act, next_state, return_mode=None):
        self.xi = np.concatenate(state_act)
        self.next_state = next_state

        for i in range(self.state_dim):
            self.estimator[i].adapt(next_state[i], self.xi)

        state_coeffs = [est.w[:self.state_dim][None, :] for est in self.estimator]
        action_coeffs = [est.w[self.state_dim:][None, :] for est in self.estimator]
        A_hat = np.concatenate(state_coeffs, axis=0)
        B_hat = np.concatenate(action_coeffs, axis=0)

        if return_mode == 'all':
            return (state_coeffs, action_coeffs), (A_hat, B_hat)
        elif return_mode == 'coeff':
            return (state_coeffs, action_coeffs)
        else:
            return

    def predict(self, state_act=None):
        if state_act is not None:
            xi = np.concatenate(state_act)
        else:
            xi = self.xi

        predictions = np.asarray([est.predict(xi) for est in self.estimator])
        return predictions

    def calc_error(self, metric='MSE'):
        if metric == 'AE':
            # calc absolute prediction error for each state
            predictions = self.predict()
            abs_error = np.abs(self.next_state - predictions)
            return abs_error

        elif metric == 'Error':
            # calc direct error
            predictions = self.predict()
            error = self.next_state - predictions
            return error

        elif metric == 'MSE':
            # TODO add other error metrics
            # calc mean squared error over error-buffer
            raise NotImplementedError

        else:
            raise NotImplementedError


# reward functions for GEM environment

class GammaReward(gem.reward_functions.WeightedSumOfErrors):
    """ modified reward function for the gym-electric-motor toolbox

    reference:
    Todo: add paper
    """

    def __init__(
            self,
            gamma=0.99,
            state_limits=np.array([1, 1]),
            state_filter=['omega', 'i'],
            violation_scaler=1,
            **__
    ):
        super().__init__(gamma=gamma)

        self.state_limits = state_limits
        self.state_filter = state_filter
        self.violation_scaler = violation_scaler

    def set_modules(self, physical_system, reference_generator, constraint_monitor):
        super().set_modules(physical_system, reference_generator, constraint_monitor)
        self.state_names = physical_system.state_names
        self.name2idx = physical_system.state_positions
        self.state_idx = np.asarray([self.name2idx[name] for name in self.state_filter])

    def reward(self, state, reference, k=None, action=None, violation_degree=0.0):
        """ compare to docstring of parent-class """
        if violation_degree != 0:
            # calculate violation penalty
            violation_factor = np.abs(state[self.state_idx] / self.state_limits)
            reward = -1 * np.sum(violation_factor) * self.violation_scaler
        else:
            # calculate normal reward
            error = state[self.name2idx['omega']] - reference[self.name2idx['omega']]
            reward = 2 * (1 - self._gamma) * (1 - np.abs(error / self._state_length[self.name2idx['omega']])) - (1 - self._gamma)

        return reward


class NormalizedReward(gem.reward_functions.WeightedSumOfErrors):
    """ modified reward function for the gym-electric-motor toolbox

    reference:
    TODO: add Max paper

    """

    def __init__(
            self,
            reward_weights=None,
            normed_reward_weights=False,
            violation_reward=None,
            gamma=0.99,
            reward_power=1,
            bias=0.0,
            rew_quotient=1,
            state_limits=np.array([1, 1]),
            state_filter=['omega', 'i'],
            **__
    ):
        super().__init__(
            reward_weights=reward_weights,
            normed_reward_weights=normed_reward_weights,
            violation_reward=violation_reward,
            gamma=gamma,
            reward_power=reward_power,
            bias=bias,
        )

        self.rew_quotient = rew_quotient
        self.state_limits = state_limits
        self.state_filter = state_filter

    def set_modules(self, physical_system, reference_generator, constraint_monitor):
        super().set_modules(physical_system, reference_generator, constraint_monitor)
        self.state_names = physical_system.state_names
        self.name2idx = physical_system.state_positions
        self.state_idx = np.asarray([self.name2idx[name] for name in self.state_filter])

    def reward(self, state, reference, k=None, action=None, violation_degree=0.0):
        """ compare to docstring of parent-class """
        wse_reward = (((1.0 - violation_degree) * self._wse_reward(state, reference))
                      * (1 / self.rew_quotient)) \
                     + violation_degree * self._violation_reward
        return wse_reward