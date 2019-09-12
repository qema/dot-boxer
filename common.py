import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
import dotboxes
import numpy as np

device_cache = None
def get_device():
    global device_cache
    if not device_cache:
        device_cache = torch.device("cuda" if torch.cuda.is_available() else
            "cpu")
    return device_cache

def clear_debug_log():
    open("log.txt", "w").close()

def debug_log(name, *args):
    print(name, *args)
    with open("log.txt", "a") as f:
        print(name, *args, file=f)

if __name__ == "__main__":
    debug_log("Andrew", 1, 2, 3)
