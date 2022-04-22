import pathlib
import pandas as pd
import numpy as np


def checkpath(path):
    """ checks for correct path formatting """
    if path[-1] != "/":
        path += "/"
    return path


def path2parent(parent, dtype="str"):
    """ get path to parent directory one or more levels above function-call

    args:
        parent(str): name of parent directory
        dtype(str): str (string) or posix (PosixPath)

    returns:
        path to parent as str
    """
    path = pathlib.Path().resolve()
    dir_list = str(path).split("/")
    parent_idx = np.where(np.array(dir_list) == parent)[0][0]

    parent_path = path
    for idx in range(len(dir_list) - parent_idx - 1):
        parent_path = parent_path.parent

    if dtype == "str":
        parent_path = checkpath(str(parent_path))

    return parent_path


class Logger:
    """ Functions for logging data during control loop """

    def __init__(self, keys):
        self.keys = keys
        self.log = self.create_log()

    def create_log(self,):
        log_dict = {k: [] for k in self.keys}
        return log_dict

    def store(self, data):
        for k, d in zip(self.keys, data):
            self.log[k].append(d)

    def reset_log(self,):
        self.log = self.create_log()

    def get_log(self, dtype='list'):
        if dtype == 'list':
            return [self.log[k] for k in self.keys]
        elif dtype == 'dict':
            return self.log
        else:
            raise NotImplementedError


def save_logs(logs, log_keys, path='', name='test', saveformat='json', save=True):
    """ convert log recorded from runner to multilevel pd.DataFrame

    args:
        logs : logs recorded during rl-training
        log_keys : column names for data logged
        path : path to save logs to
        name : name of saved log
        saveformat : save log as 'json' or None
        save : if True log is save as file, if False Dataframe is returned

    returns:
        if save=False returns log in pandas Dataframe format
    """
    obs, train, test = logs

    if obs is not None:
        obs_episodes = len(obs[log_keys[0]])
        obs_idx = [['episode_' + str(i) for i in np.arange(obs_episodes)]]
        obs_df = pd.DataFrame(obs, index=obs_idx)

    if train is not None:
        train_episodes = len(train[log_keys[0]])
        train_idx = ['episode_' + str(i) for i in np.arange(train_episodes)]
        train_df = pd.DataFrame(train, index=train_idx)

    if test is not None:
        test_episodes = len(test[log_keys[0]])
        test_idx = ['episode_' + str(i) for i in np.arange(test_episodes)]
        test_df = pd.DataFrame(test, index=test_idx)

    if save:
        # save logs
        if saveformat == 'json':
            if obs is not None:
                obs_df.to_json(path + name + '_observer.json')
            if train is not None:
                train_df.to_json(path + name + '_training.json')
            if test is not None:
                test_df.to_json(path + name + '_testing.json')
        else:
            raise NotImplementedError

        print('data saved')
    else:
        # return log dataframes
        return train_df, test_df


def load_logs(path='', name='test', saveformat='json', read_obs=True):
    """ load json log and return as pandas Dataframe """
    if saveformat == 'json':
        if read_obs:
            obs_df = pd.read_json(path + name + '_observer.json')
        else:
            obs_df = None

        train_df = pd.read_json(path + name + '_training.json')
        test_df = pd.read_json(path + name + '_testing.json')
    else:
        raise NotImplementedError

    return train_df, test_df, obs_df


def read_stats(log, key, timesteps=1000):
    """ read column, pad and convert to array """
    col = []

    for row in np.arange(log.shape[0]):
        pad_len = timesteps - log.iloc[row]['episode-length']
        col_ = np.squeeze(np.asarray(log[key].iloc[row]))

        if col_.ndim == 1:
            col.append(np.pad(col_, (0, pad_len), constant_values=np.nan))
        elif col_.ndim == 2:
            col.append(np.pad(col_, ((0, pad_len), (0, 0)),
                              constant_values=np.nan))
    col = np.asarray(col)
    return col
