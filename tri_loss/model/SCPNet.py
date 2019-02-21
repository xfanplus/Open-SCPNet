import torch
import torch.nn as nn
import torch.nn.functional as F

import math


# ------------------- Model Defination ---------------------------
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.row_pool = nn.AdaptiveAvgPool2d((4, 1))
        self.conv_feature = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.drop = nn.Dropout(0.75)

        # use conv to replace pool
        self.conv_pool1 = nn.Conv2d(2048, 2048, kernel_size=1, stride=1, padding=0)
        self.conv_pool2 = nn.Conv2d(2048, 2048, kernel_size=1, stride=1, padding=0)
        self.conv_pool3 = nn.Conv2d(2048, 2048, kernel_size=1, stride=1, padding=0)
        self.conv_pool4 = nn.Conv2d(2048, 2048, kernel_size=1, stride=1, padding=0)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # classification part
        self.fc_cls1 = nn.Linear(2048, num_classes)
        self.fc_cls2 = nn.Linear(2048, num_classes)
        self.fc_cls3 = nn.Linear(2048, num_classes)
        self.fc_cls4 = nn.Linear(2048, num_classes)

        self.conv_bn1 = nn.BatchNorm2d(2048)
        self.conv_bn2 = nn.BatchNorm2d(2048)
        self.conv_bn3 = nn.BatchNorm2d(2048)
        self.conv_bn4 = nn.BatchNorm2d(2048)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = list()
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # new feature
        row_x = self.row_pool(x)
        # row_x = self.conv_feature(row_x)
        N = x.size(0)
        row_f1 = row_x[:, :, 0].contiguous().view(N, -1)
        row_f2 = row_x[:, :, 1].contiguous().view(N, -1)
        row_f3 = row_x[:, :, 2].contiguous().view(N, -1)
        row_f4 = row_x[:, :, 3].contiguous().view(N, -1)
        row_f1 = row_f1.detach()
        row_f2 = row_f2.detach()
        row_f3 = row_f3.detach()
        row_f4 = row_f4.detach()

        # new feature
        conv_f1 = self.global_pool(self.conv_bn1(self.conv_pool1(x))).squeeze(3).squeeze(2)
        conv_f2 = self.global_pool(self.conv_bn2(self.conv_pool2(x))).squeeze(3).squeeze(2)
        conv_f3 = self.global_pool(self.conv_bn3(self.conv_pool3(x))).squeeze(3).squeeze(2)
        conv_f4 = self.global_pool(self.conv_bn4(self.conv_pool4(x))).squeeze(3).squeeze(2)

        # classification
        s1 = self.fc_cls1(self.drop(conv_f1))
        s2 = self.fc_cls2(self.drop(conv_f2))
        s3 = self.fc_cls3(self.drop(conv_f3))
        s4 = self.fc_cls4(self.drop(conv_f4))

        return [s1, s2, s3, s4], [conv_f1, conv_f2, conv_f3, conv_f4], [row_f1, row_f2, row_f3, row_f4]


def get_scp_model(nr_class):

    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=nr_class)
