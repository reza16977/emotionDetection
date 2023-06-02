from deep_emotion import Deep_Emotion

from __future__ import print_function
import argparse
import numpy  as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net=Deep_Emotion()
net.load_state_dict(torch.load('DeepEmotion_trainded1.pt'))
net.to(device)
