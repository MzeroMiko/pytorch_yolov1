import torch
import torch.nn as nn
from torchvision import models


# modified from detnet_bottleneck_ori copied from https://github.com/xiongzihua/pytorch-YOLO-v1/resnet_yolo.py
class detnet_bottleneck(nn.Module):
    # no expansion
    # dilation = 2
    # type B use 1x1 conv
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, block_type='A'):
        super(detnet_bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=2, bias=False,dilation=2)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes or block_type=='B':
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = out + self.downsample(x)
        out = self.relu(out)
        return out


# one way to modify given model # use basic resnet50
class resnetYOLO(models.resnet.ResNet):
    
    def __init__(self, target_num, **kwargs):
        super(resnetYOLO, self).__init__(models.resnet.Bottleneck, [3, 4, 6, 3], **kwargs)
        del self.avgpool
        del self.fc
        self.layer5 = self._make_detnet_layer(in_channels=2048, out_channels=1024)
        self.conv_end = nn.Conv2d(1024, target_num, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_end = nn.BatchNorm2d(target_num)
        # init code copied from models.resnet.ResNet
        for m in self.modules(): 
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_detnet_layer(self, in_channels, out_channels):
        layers = []
        layers.append(detnet_bottleneck(in_planes=in_channels, planes=out_channels, block_type='B'))
        layers.append(detnet_bottleneck(in_planes=out_channels, planes=out_channels, block_type='A'))
        layers.append(detnet_bottleneck(in_planes=out_channels, planes=out_channels, block_type='A'))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x) # batch_size*2048*(intput/32)*(input/32)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        x = self.layer5(x) # batch_size*256*(intput/32)*(input/32)
        x = self.conv_end(x) # batch_size*target_num*(intput/32)*(input/32)
        x = self.bn_end(x) # batch_size*target_num*(intput/32)*(input/32)
        x = x.sigmoid().permute(0,2,3,1) # batch_size*(intput/32)*(input/32)*target_num
        return x


# must use it when net is in cpu, model_dict can be a dict or a filename
def loadModelDict(net, model_dict):
    load_dict = torch.load(model_dict) if isinstance(model_dict, str) else model_dict
    net_dict = net.state_dict()
    for key in load_dict.keys():
        if key in net_dict.keys():
            net_dict[key] = load_dict[key]
        elif key[7:] in net_dict.keys(): # remove gpu parallel prefix 'module.'
            net_dict[key[7:]] = load_dict[key]

    net.load_state_dict(net_dict)
    return net


def resYolo(target_num, pretrained=False, model_dict='', **kwargs):
    model = resnetYOLO(target_num)
    if model_dict != '': # load self-trained model
        model = loadModelDict(model, model_dict)
    elif pretrained: # load pre-trained model
        model = loadModelDict(model, models.resnet50(pretrained=True).state_dict())
    return model



