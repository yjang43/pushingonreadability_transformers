import logging
import time
import os
import random

import torch
import numpy as np


# prepare logger
def get_logger():
    current_time = time.strftime("%Y-%m-%d-%H:%M:%S")
    os.makedirs('logs', exist_ok=True)
    log_path = os.path.join('logs', f'{current_time}-train.log')
    logging.basicConfig(
        filename=log_path,
        format="%(asctime)s | %(funcName)10s | %(message)s",
        datefmt="%Y-%m-%d-%H:%M:%S",
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)
    return logger

# set seed values
def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    