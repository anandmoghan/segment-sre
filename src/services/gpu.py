from subprocess import PIPE, Popen

import numpy as np
import os


def get_gpu_info():
    cmd = ['nvidia-smi', '--query-gpu=index,memory.total,memory.used,memory.free', '--format=csv,noheader,nounits']
    csv_output, _ = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=False).communicate()
    return np.array([np.fromstring(line, dtype=int, sep=',')[1:] for line in csv_output.decode('utf8').strip().split('\n')])


def set_gpu(gpu_id, min_free_fraction=0.33):
    free_fraction = np.array([gpu[2] / gpu[0] for gpu in get_gpu_info()])
    gpu_id = np.argmax(free_fraction) if gpu_id < 0 else gpu_id
    if free_fraction[gpu_id] <= min_free_fraction:
        raise Exception('Free memory available is less than {}%.'.format(min_free_fraction * 100))
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
