from os.path import join as join_path
from subprocess import Popen, PIPE
from tqdm import tqdm

import multiprocessing as mp
import numpy as np
import pickle
import time
import os


def append_cwd_to_python_path(cmd):
    return 'export PYTHONPATH=$PYTHONPATH:$CWD && {}'.format(cmd)


def copy_directory(source, destination):
    run_command('cp -r {} {}'.format(source, destination))


def delete_directory(path):
    try:
        for item in os.listdir(path):
            item = join_path(path, item)
            if os.path.isdir(item):
                delete_directory(item)
            else:
                os.remove(item)
        os.removedirs(path)
    except FileNotFoundError:
        pass


def get_time_stamp():
    return time.strftime('%b %d, %Y %l:%M:%S%p')


def load_array(file_name):
    return np.load(file_name)


def load_object(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def make_directory(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def make_dict(key_list, value_list):
    return dict([(key, value) for key, value in zip(key_list, value_list)])


def print_log(*content):
    print('{} - '.format(get_time_stamp()), end='')
    print(*content)


def print_log_header(args):
    print('# Running in {}'.format(os.uname()[1]))
    print(' '.join(args))
    print()


def print_time_stamp():
    print(get_time_stamp())


def remove_duplicates(args):
    _, unique_idx = np.unique(args[:, 0], return_index=True)
    return args[unique_idx, :], args.shape[0] - len(unique_idx)


def run_command(cmd, shell_=True):
    output, error = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=shell_).communicate()
    return output.decode("utf-8"), error.decode("utf-8") if error is not None else error


def run_parallel(func, args_list, n_workers=10, p_bar=True):
    pool = mp.Pool(n_workers)
    if p_bar:
        if type(args_list) is list:
            total_len = len(args_list)
        else:
            total_len = args_list.shape[0]
        out = tqdm(pool.imap(func, args_list), total=total_len)
    else:
        out = pool.map(func, args_list)
    pool.close()
    if out is not None:
        return list(out)


def save_array(file_name, obj):
    np.save(file_name, obj)


def save_batch_array(location, args_idx, obj, ext='.npy'):
    for i, arg in enumerate(args_idx):
        save_loc = os.path.join(location, arg + ext)
        save_array(save_loc, obj[i])


def save_object(file_name, obj):
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def split_dict(old_dict, keys):
    new_dict = dict()
    for key in keys:
        try:
            new_dict[key] = old_dict[key]
        except KeyError:
            raise Exception('{}: {} not found.'.format('SPLIT_DICT KeyError', key))
    return new_dict
