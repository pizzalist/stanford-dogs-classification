import torch
from torch import nn
import torchvision.models as models
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1 # 클래스 속성
    def __init__(self, in_channels, inner_channels, stride = 1, projection = None):
        super().__init__()

        self.residual = nn.Sequential(nn.Conv2d(in_channels, inner_channels, 3, stride=stride, padding=1, bias=False),
                                      nn.BatchNorm2d(inner_channels),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(inner_channels, inner_channels * self.expansion, 3, padding=1, bias=False),
                                      nn.BatchNorm2d(inner_channels))
        self.projection = projection
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        residual = self.residual(x)

        if self.projection is not None:
            shortcut = self.projection(x)
        else:
            shortcut = x

        out = self.relu(residual + shortcut) # 엉!
        return out

class full_pre_activation(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution
    # while original implementation places the stride at the first 1x1 convolution
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
    expansion = 4 # 클래스 속성
    def __init__(self, in_channels, inner_channels, stride = 1, projection = None):
        super().__init__()

        self.residual = nn.Sequential(
                                      nn.BatchNorm2d(in_channels),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(in_channels, inner_channels, 1, bias=False),
                                      nn.BatchNorm2d(inner_channels),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(inner_channels, inner_channels, 3, stride=stride, padding=1, bias=False),
                                      nn.BatchNorm2d(inner_channels),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(inner_channels, inner_channels * self.expansion, 1, bias=False))

        self.projection = projection

    def forward(self, x):
        residual = self.residual(x)

        if self.projection is not None:
            shortcut = self.projection(x)
        else:
            shortcut = x

       
        return shortcut

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution
    # while original implementation places the stride at the first 1x1 convolution
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
    expansion = 4 # 클래스 속성
    def __init__(self, in_channels, inner_channels, stride = 1, projection = None):
        super().__init__()

        self.residual = nn.Sequential(nn.Conv2d(in_channels, inner_channels, 1, bias=False),
                                      nn.BatchNorm2d(inner_channels),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(inner_channels, inner_channels, 3, stride=stride, padding=1, bias=False),
                                      nn.BatchNorm2d(inner_channels),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(inner_channels, inner_channels * self.expansion, 1, bias=False),
                                      nn.BatchNorm2d(inner_channels * self.expansion))

        self.projection = projection
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.residual(x)

        if self.projection is not None:
            shortcut = self.projection(x)
        else:
            shortcut = x

        out = self.relu(residual + shortcut)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_block_list, num_classes = 1000, zero_init_residual = True):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False) # bias = False는 논문에 설명 없다
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True) # 좀더 메모리 효율적
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layers(block, 64, num_block_list[0], stride=1)
        self.layer2 = self.make_layers(block, 128, num_block_list[1], stride=2)
        self.layer3 = self.make_layers(block, 256, num_block_list[2], stride=2)
        self.layer4 = self.make_layers(block, 512, num_block_list[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # GAP
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3%p according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, block):
                    nn.init.constant_(m.residual[-1].weight, 0) # block에 마지막 웨이트 값을 0으로

    def make_layers(self, block, inner_channels, num_blocks, stride = 1):
        #1. 사이즈 안맞거나 2. 채널 갯수 안맞으면 => 점선(projection shortcut
        if stride != 1 or self.in_channels != inner_channels * block.expansion:
            # stride = 1 이여도 채널 수가 다르면 (layer1의 첫번째 BottleNeck) projection 해야함 (이 때는 resoltion은 그대로, 채널 수만 늘어남)
            projection = nn.Sequential(
                nn.Conv2d(self.in_channels, inner_channels * block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(inner_channels * block.expansion)) # 점선 connection 임
        else:
            projection = None

        layers = []
        layers += [block(self.in_channels, inner_channels, stride, projection)] # projection은 첫 block에서만
        self.in_channels = inner_channels * block.expansion
        for _ in range(1, num_blocks):
            layers += [block(self.in_channels, inner_channels)]

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

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

def resnet152(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)

def resnet50_full_pre(**kwargs):
    return ResNet(full_pre_activation, [3, 4, 6, 3], **kwargs)

class ResNet152PretrainedModel(nn.Module):
    def __init__(self):
        super().__init__()
        # self.model = models.resnet152(pretrained=True)
        self.model = models.resnet152(weights = models.ResNet152_Weights.IMAGENET1K_V2)
        self.model.fc = nn.Linear(2048, 120)

        layer = 0
        for child in self.model.children():
            if layer <= 7: # 7, 8, 9번째 layer만 학습
                for param in child.parameters():
                    param.requires_grad = False
            layer += 1
class densenet201PretrainedModel(nn.Module):
    def __init__(self):
        super().__init__()
        # self.model = models.resnet152(pretrained=True)
        self.model = models.densenet201(weights = models.DenseNet201_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(2048, 120)

        layer = 0
        for child in self.model.children():
            if layer <= 7: # 7, 8, 9번째 layer만 학습
                for param in child.parameters():
                    param.requires_grad = False
            layer += 1
    def forward(self, x):
        return F.log_softmax(self.model(x), dim=0)