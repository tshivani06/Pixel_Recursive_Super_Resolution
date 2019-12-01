import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from net import Network
from skimage.io import imsave

'''
model = Network()

a = torch.randn(1, 3, 32, 32, dtype=torch.float)
b = torch.randn(1, 3, 8, 8, dtype = torch.float)

hr_images = a / 255.0 - 0.5
lr_images = b / 255.0 - 0.5

a, b = model.forward(hr_images, lr_images)

a = a.reshape(-1, 256)
b = torch.tensor(b, dtype = torch.int8)
print(a.shape)
print(b)
'''

x = np.random.rand(32,32,3) * 255
im = Image.fromarray(x.astype('uint8')).convert('RGB')
np_x = np.array(im)
print(np_x)
x = transforms.ToTensor()(im)
x *= 255
print(x)