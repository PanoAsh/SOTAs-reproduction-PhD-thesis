import torch
from torch.optim import Adam
from torch.autograd import Variable
from model import build_model
import numpy as np
import cv2
import os
from torch.nn import functional as F
import time
import matplotlib.pyplot as plt


class SolverReTrain(object):
    def __init__(self, train_loader, test_loader, config):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
