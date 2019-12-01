import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
#from torch.utils.tensorboard import SummaryWriter # TensorBoard support

# import torchvision module to handle image manipulation
#import torchvision
#import torchvision.transforms as transforms

# calculate train time, writing train data to files etc.
import time
#import pandas as pd
import json
#from IPython.display import clear_output

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.prior_logits = prior_network()
        self.conditioning_logits = conditioning_network()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
    
    def forward(self, hr_images, lr_images):
        prior_logits = self.prior_logits(hr_images)
        conditioning_logits = self.conditioning_logits(lr_images)

        return prior_logits, conditioning_logits


class conditioning_network(nn.Sequential):
    def __init__(self):
        super(conditioning_network, self).__init__()

        self.add_module('CondMaskedConv1', MaskedConv2d(None,  in_channels = 3, out_channels = 32, kernel_size = 1, padding = 1//2))
        for i in range(2):
            for j in range(6):
                self.add_module('ResDeconv' + str(i) + str(j), ResBlock(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 3//2))
            self.add_module('Deconv' + str(i), nn.ConvTranspose2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 2, padding = 1, output_padding = 1))
            self.add_module('ReLU' + str(i), nn.ReLU(inplace = True))
        for i in range(6):
            self.add_module('Res' + str(i), ResBlock(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 3//2))
        self.add_module('CondMaskedConv2', MaskedConv2d(None,  in_channels = 32, out_channels = 3 * 256, kernel_size = 1, padding = 1//2))

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
        MaskedConv2d(None, in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, padding = padding),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace = True),
        MaskedConv2d(None, in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, padding = padding),
        nn.BatchNorm2d(in_channels))
    
    def forward(self, x):
        inputs = x
        x = self.block(x)
        x += inputs
        return x



class prior_network(nn.Module):
    def __init__(self):
        super(prior_network, self).__init__()
        self.MaskedConv1 = MaskedConv2d('A', in_channels = 3, out_channels = 64, kernel_size = 7, padding = 7//2)
        self.GatedConvLayers = GatedConv2dLayers(num_layers = 20, in_channels = 64, kernel_size = 5)
        self.final_layers = nn.Sequential(MaskedConv2d('B', in_channels = 64, out_channels = 1024, kernel_size = 1, padding = 1//2),
        nn.ReLU(inplace=True),
        MaskedConv2d('B', in_channels = 1024, out_channels = 3 * 256, kernel_size = 1, padding = 1//2))
    
    def forward(self, x):
        x = self.MaskedConv1(x)
        inputs = x
        state = x
        x = self.GatedConvLayers(inputs, state)
        x = self.final_layers(x)
        x = torch.cat((x[:, 0::3, :, :], x[:, 1::3, :, :], x[:, 2::3, :, :]), 1)
        return x


class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        self.register_buffer('mask', self.weight.data.clone())
        _, _, h_size, w_size = self.weight.size()

        center_h = h_size // 2
        center_w = w_size // 2
        self.mask.fill_(0)
        if mask_type is not None:
            self.mask[:, :, :center_h, :] = 1
            if mask_type == 'A':
                self.mask[:, :, center_h, :center_w] = 1
            if mask_type == 'B':
                self.mask[:, :, center_h, :(center_w+1)] = 1
        else:
            self.mask[:, :, :, :] = 1
    
    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)

class GatedConv2dLayers(nn.Sequential):
    def __init__(self, num_layers, in_channels, kernel_size):
        super(GatedConv2dLayers, self).__init__()
        for i in range(num_layers):
            self.add_module('GatedConv'+str(i), GatedConv2d(in_channels, kernel_size))
    def forward(self, inputs, state):
        for module in self._modules.values():
            #print(module)
            inputs, state = module(inputs, state)
        
        return inputs


class GatedConv2d(nn.Sequential):
    def __init__(self, in_channels, kernel_size):
        super(GatedConv2d, self).__init__()
        self.conv1_state = MaskedConv2d('C', in_channels = in_channels, out_channels = 2 * in_channels, kernel_size = kernel_size, padding = kernel_size//2)
        self.tanh_state = nn.Tanh()
        self.sigmoid_state = nn.Sigmoid()
        self.conv2_state = MaskedConv2d(None, in_channels = 2 * in_channels, out_channels = 2 * in_channels, kernel_size = 1, padding = 1//2)

        self.conv1_input = MaskedConv2d('B', in_channels = in_channels, out_channels = 2 * in_channels, kernel_size = (1, kernel_size), padding = (0,kernel_size//2))
        self.tanh_input = nn.Tanh()
        self.sigmoid_input = nn.Sigmoid()
        self.conv2_input = MaskedConv2d('B', in_channels = in_channels, out_channels = in_channels, kernel_size = 1, padding = 0)
    
    def forward(self, inputs, state):
        left = self.conv1_state(state)
        _, in_channel, _, _ = left.size()
        in_channel = in_channel // 2
        left1 = left[:, 0:in_channel, :, :]
        left2 = left[:, in_channel:, :, :]
        left1 = self.tanh_state(left1)
        left2 = self.sigmoid_state(left2)
        new_state = left1 * left2
        left2right = self.conv2_state(left)

        right = self.conv1_input(inputs)
        right = right + left2right
        right1 = right[:, 0:in_channel, :, :]
        right2 = right[:, in_channel:, :, :]
        right1 = self.tanh_input(right1)
        right2 = self.sigmoid_input(right2)
        up_right = right1 * right2
        up_right = self.conv2_input(up_right)
        outputs = inputs + up_right

        return outputs, new_state


