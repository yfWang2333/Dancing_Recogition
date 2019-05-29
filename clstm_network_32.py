# pylint: disable=no-member
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ConvLSTM import *

"""
original: N = 5, C = 3, D = 32, W = 112, H = 112
5-dimensional tensor [N,C,D,W,H] for conv3d, bn3d, pool3d
4-dimensional tensor [N,C,W,H] for conv2d, bn2d, pool2d

"""

batch_size = 3
channels = 3
depth = 32
W = 112
H = 112
fc_input_W = (W / 16) // 2 + 1
fc_input_size = int(256 * (fc_input_W**2))
num_cls = 4

class CNet(nn.Module):
    def __init__(self):
        super(CNet, self).__init__()
        self.conv3d_1 = nn.Conv3d(3,64,3,1,1) # [3×32×112×112] -> [64×32×112×112]
        self.bn3d_1 = nn.BatchNorm3d(64)
        self.pool3d_1 = nn.MaxPool3d((1,2,2),(1,2,2)) # [64×32×112×112] -> [64×32×56×56]
        self.conv3d_2 = nn.Conv3d(64,128,3,1,1) # [64×32×56×56] -> [128×32×56×56]
        self.bn3d_2 = nn.BatchNorm3d(128)
        self.pool3d_2 = nn.MaxPool3d((2,2,2),(2,2,2)) # [128×32×56×56] -> [128×16×28×28]
        self.conv3d_3a = nn.Conv3d(128,256,3,1,1) # [128×16×28×28] -> [256×16×28×28]
        self.conv3d_3b = nn.Conv3d(256,256,3,1,1) # [256×16×28×28] -> [256×16×28×28]
        self.bn3d_3 = nn.BatchNorm3d(256)
        #self.pool3d_3 = nn.MaxPool3d((2,1,1),(2,1,1)) # [256×16×28×28] -> [256×8×28×28]
        self.convlstm = CLSTM((28,28),256,3,256,2) # shape: (W/4,H/4)
        self.conv2d_1 = nn.Conv2d(256,128,3,1,1) # [256×28×28] -> [128×28×28]
        self.bn2d_1 = nn.BatchNorm2d(128)
        self.pool2d_1 = nn.MaxPool2d((2,2),(2,2)) # [128×28×28] -> [128×14×14]
        self.conv2d_2 = nn.Conv2d(128,256,3,1,1) # [128×14×14] -> [256×14×14]
        self.bn2d_2 = nn.BatchNorm2d(256)
        self.pool2d_2 = nn.MaxPool2d((2,2),(2,2)) # [256×14×14] -> [256×7×7]
        self.conv2d_3 = nn.Conv2d(256,256,3,1,1) # [256×7×7] -> [256×7×7]
        self.bn2d_3 = nn.BatchNorm2d(256)
        self.pool2d_3 = nn.MaxPool2d((2,2),(2,2),1) # [256×7×7] -> [256×4×4]
        self.conv2d_4 = nn.Conv2d(256,1024,3,1,1) # [256×4×4] -> [1024×4×4]
        self.bn2d_4 = nn.BatchNorm2d(1024)
        self.pool2d_4 = nn.MaxPool2d((2,2),(2,2)) # [1024×4×4] -> [1024×2×2]
        self.conv2d_5 = nn.Conv2d(1024,4096,3,1,1) # [1024×2×2] -> [4096×2×2]
        self.bn2d_5 = nn.BatchNorm2d(4096)
        self.pool2d_5 = nn.MaxPool2d((2,2),(2,2)) # [4096×2×2] -> [4096×1×1] -> [4096]
        self.fc = nn.Linear(fc_input_size, num_cls) # [4096] -> [num_cls]

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight, gain = 1)
            if isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d)):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        x = self.conv3d_1(x)
        x = self.bn3d_1(x)
        x = F.relu(x, inplace = True)
        x = self.pool3d_1(x)
        x = self.conv3d_2(x)
        x = self.bn3d_2(x)
        x = F.relu(x, inplace = True)
        x = self.pool3d_2(x)
        x = self.conv3d_3a(x)
        x = self.conv3d_3b(x)
        x = self.bn3d_3(x)
        x = F.relu(x, inplace = True)
        x = x.transpose(1,2).contiguous()
        #self.convlstm.apply(weights_init)
        x = Variable(x).cuda()
        self.convlstm.cuda()
        hidden_state = self.convlstm.init_hidden(x)
        convlstmout = self.convlstm(x, hidden_state)
        x = convlstmout[1]
        x_ = torch.zeros(16,x.shape[1],256,4,4).cuda()
        for t in range(x.size(0)):
            temp = self.conv2d_1(x[t])
            temp = self.bn2d_1(temp)
            temp = F.relu(temp, inplace = True)
            temp = self.pool2d_1(temp)
            temp = self.conv2d_2(temp)
            temp = self.bn2d_2(temp)
            temp = F.relu(temp, inplace = True)
            temp = self.pool2d_2(temp)
            temp = self.conv2d_3(temp)
            temp = self.bn2d_3(temp)
            temp = F.relu(temp, inplace = True)
            temp = self.pool2d_3(temp)
            x_[t] = temp
        x_ = torch.mean(x_,0)
        x_ = self.conv2d_4(x_)
        x_ = self.bn2d_4(x_)
        x_ = F.relu(x_, inplace = True)
        x_ = self.pool2d_4(x_)
        x_ = self.conv2d_5(x_)
        x_ = self.bn2d_5(x_)
        x_ = F.relu(x_, inplace = True)
        x_ = self.pool2d_5(x_)
        x_ = x_.view(x_.size(0),-1)
        x_ = self.fc(x_)
        return x_

    def predict(self,x):
        pred = F.softmax(x, dim = 1)
        _, pred = torch.max(pred, 1)
        return pred

"""
x = torch.randn(4,3,32,112,112).cuda()
Net = CNet().cuda()
y = Net(x)
pred = Net.predict(y.data)
print(y.size())
print(pred, pred.size())
"""