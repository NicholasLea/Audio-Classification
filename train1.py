import torch
import torchvision
import torch.nn as nn
import numpy as np
import json
import utils
import validate
import argparse
import models.densenet
import models.resnet
import models.inception
import time
import dataloaders.datasetaug
import dataloaders.datasetnormal


from tqdm import tqdm
from tensorboardX import SummaryWriter

# https://docs.python.org/zh-cn/3/library/argparse.html
parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str)
print(parser)