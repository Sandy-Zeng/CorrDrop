'''ResNet18/34/50/101/152 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
from dropblock import DropBlock2D, LinearScheduler
from dropblock.corrdropblock import CorrDropBlockDB
from dropblock.channel_wise_drop import ChannelDrop


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, p=0.1, block_size=9):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class SEBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, p=0.1, block_size=9):
        super(SEBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

        self.channel_attention = ChannelAttention(inplanes=planes)



    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.channel_attention(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class DropoutBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, p=0.1, block_size=9):
        super(DropoutBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

        # self.dropout = nn.Dropout(p=p)
        self.dropout = nn.Dropout2d(p=p)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class CDBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, p=0.1, block_size=9):
        super(CDBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

        self.channel_drop = LinearScheduler(
            ChannelDrop(drop_prob=p),
            start_value=0.,
            stop_value=p,
            nr_steps=90
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        self.channel_drop.step()
        out = self.channel_drop(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SCBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, p=0.1, block_size=9):
        super(SCBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

        self.corrdrop_block = LinearScheduler(
            CorrDropBlockDB(drop_prob=p, block_size=block_size),
            start_value=0.,
            stop_value=p,
            nr_steps=90
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        self.corrdrop_block.step()
        out = self.corrdrop_block(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class DBBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, p=0.1, block_size=9):
        super(DBBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

        self.dropblock = LinearScheduler(
            DropBlock2D(drop_prob=p, block_size=block_size, att=True),
            start_value=0.,
            stop_value=p,
            nr_steps=90
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        self.dropblock.step()
        out = self.dropblock(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, p=0.1, dropway='no'):
        super(ResNet, self).__init__()
        if dropway == 'SCD':
            block2 = SCBlock
        elif dropway == 'CCD':
            block2 = CDBlock
        elif dropway == 'dropblock':
            block2 = DBBlock
        elif dropway == 'dropout' or dropway == 'spatialdropout':
            block2 = DropoutBlock
        else:
            block2 = block

        self.in_planes = 64

        self.conv1 = conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, p=p, block_size=7)
        self.layer3 = self._make_layer(block2, 256, num_blocks[2], stride=2, p=p, block_size=5)
        self.layer4 = self._make_layer(block2, 512, num_blocks[3], stride=2, p=p, block_size=3)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        self.dropway = dropway


    def _make_layer(self, block, planes, num_blocks, stride, p=0.1, block_size=9):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, p=p, block_size=block_size))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # print(out.shape)

        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        # print(out.shape)
        out = self.linear(out)
        return out


def ResNet18(num_classes=10,p=0.3,dropway='no'):
    return ResNet(BasicBlock, [2,2,2,2], num_classes,p=p, dropway=dropway)

def ResNet34(num_classes=10):
    return ResNet(BasicBlock, [3,4,6,3], num_classes)

def ResNet50(num_classes=10):
    return ResNet(Bottleneck, [3,4,6,3], num_classes)

def ResNet101(num_classes=10, p=0.3, dropway='no'):
    return ResNet(Bottleneck, [3,4,23,3], num_classes, p=p, dropway=dropway)

def ResNet152(num_classes=10):
    return ResNet(Bottleneck, [3,8,36,3], num_classes)

def test_resnet():
    net = ResNet50()
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())

# test_resnet()