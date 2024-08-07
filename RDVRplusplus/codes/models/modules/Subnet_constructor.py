import torch
import torch.nn as nn
import torch.nn.functional as F
import models.modules.module_util as mutil

import time

from global_var import *

class D2DTInput(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier',\
         gc=32, bias=True,INN_init = True,is_res = False):
        super(D2DTInput, self).__init__()
        self.conv1 = nn.Conv3d(channel_in, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv2 = nn.Conv3d(channel_in + gc, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv3 = nn.Conv3d(channel_in + 2 * gc, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv4 = nn.Conv3d(channel_in + 3 * gc, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv5 = nn.Conv3d(channel_in + 4 * gc, channel_out, (3,1,1), 1, (1,0,0), bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        if INN_init:
            if init == 'xavier':
                mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
            else:
                mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
            mutil.initialize_weights(self.conv5, 0)

    def forward(self, x,io_type="2d"):
        if not io_type == '3d':
            io_type = "2d"
        if io_type == "2d":
            bt,c,w,h = x.size()
            # print(x.size())
            t = GlobalVar.get_Temporal_LEN()
            # t = 5
            b = bt//t
            x  = x.reshape(b,t,c,w,h).transpose(1,2)
        #print(x.shape)
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        #print('ppppppppppppppppppppppppppppp')
        #print(torch.cat((x, x1, x2, x3, x4), 1).shape)
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        #print(x5.shape)
        if io_type == "2d":
            x5 = x5.transpose(1,2).reshape(bt,-1,w,h)
        return x5
    
class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(channel_in + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias)

        self.conv5 = nn.Conv2d(channel_in + 4 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        else:
            mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        mutil.initialize_weights(self.conv5, 0)

        #for p in self.parameters():
        #    p.requires_grad=False
        #self.conv44 = nn.Conv2d(gc, gc, 3, 1, 1, bias=bias)

    def forward(self, x):

        #t0 = time.time()
        #print(x.shape)
        x1 = self.lrelu(self.conv1(x))
        #print(x1.shape)
        #t1 = time.time()
        #print("===> Timer: %.4f sec." % (t1 - t0))

        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        #x4 = self.conv44(x4)+x4
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        
        #print(x5.shape)
        #nnn=x5.shape[1]
        #x51=x5[:,0:3,:,:]
        #print(x51.shape)

        return x5

class DenseBlock2(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier',\
         gc=32, bias=True):
#,INN_init = True,is_res = False
        INN_init = True
        super(DenseBlock2, self).__init__()
        self.conv1 = nn.Conv3d(channel_in, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv2 = nn.Conv3d(channel_in + gc, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv3 = nn.Conv3d(channel_in + 2 * gc, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv4 = nn.Conv3d(channel_in + 3 * gc, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv5 = nn.Conv3d(channel_in + 4 * gc, channel_out, (3,1,1), 1, (1,0,0), bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        if INN_init:
            if init == 'xavier':
                mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
            else:
                mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
            mutil.initialize_weights(self.conv5, 0)

    def forward(self, x,io_type="2d"):
        if not io_type == '3d':
            io_type = "2d"
        if io_type == "2d":
            bt,c,w,h = x.size()
            # print(x.size())
            t = 5# GlobalVar.get_Temporal_LEN()
            # t = 5
            print(t)
            b = bt//t
            print(x.shape)
            x  = x.reshape(b,t,c,w,h).transpose(1,2)
        
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        if io_type == "2d":
            x5 = x5.transpose(1,2).reshape(bt,-1,w,h)
        return x5
def subnet(net_structure, init='xavier'):
    def constructor(channel_in, channel_out):
        if net_structure == 'DBNet':
            if init == 'xavier':
                return DenseBlock(channel_in, channel_out, init)
            else:
                return DenseBlock(channel_in, channel_out)
        else:
            return None

    return constructor
