import torch.nn as nn
import torch.nn.functional as F
import torch


class Encoder:
    encoder = nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(3, 64, (3, 3)),
        nn.ReLU(),  # relu1-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),  # relu1-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 128, (3, 3)),
        nn.ReLU(),  # relu2-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),  # relu2-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 256, (3, 3)),
        nn.ReLU(),  # relu3-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 512, (3, 3)),
        nn.ReLU(),  # relu4-1, this is the last layer used
    )


class InceptionModule(nn.Module):
    def __init__(self, in_cahannels, n1xn1, n3xn3red, n3xn3, n5xn5red, n5xn5, pool_proj):
        super(InceptionModule, self).__init__()

        # 1x1 branch
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=in_cahannels, out_channels=n1xn1, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        # 1x1 into 3x3
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=in_cahannels, out_channels=n3xn3red, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=n3xn3red, out_channels=n3xn3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # 1x1 into 5x5
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels=in_cahannels, out_channels=n5xn5red, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=n5xn5red, out_channels=n5xn5, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )

        # 3x3 pool into 1x1 conv branch
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=in_cahannels, out_channels=pool_proj, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        return torch.cat([b1, b2, b3, b4], 1)


class ImageClassifier(nn.Module):
    def __init__(self, encoder, num_classes):
        super().__init__()
        self.encoder = encoder
        self.avgpool = nn.AdaptiveAvgPool2d(8)

        # reduce channel size back to 512
        self.expansion = nn.Sequential(
            nn.Conv2d(960, 512, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        self.inception1 = InceptionModule(512, 64, 96, 128, 16, 32, 32)
        self.inception2 = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.inception3 = InceptionModule(480, 192, 96, 208, 16, 48, 64)

        self.bn1 = nn.BatchNorm2d(num_features=256)
        self.bn2 = nn.BatchNorm2d(num_features=480)
        self.bn3 = nn.BatchNorm2d(num_features=512)

        self.classifier = nn.Sequential(
            nn.Linear(512 * 8 * 8, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=1024),
            nn.Dropout(0.4),
            nn.Linear(1024, num_classes)
        )
        # freeze encoder weights
        for param in self.encoder.parameters():
            param.requires_grad = False

        # need access to these intermediate encoder steps
        # for the AdaIN computation
        encoder_list = list(encoder.children())
        self.encoder_stage_1 = nn.Sequential(*encoder_list[:4])  # input -> relu1_1
        self.encoder_stage_2 = nn.Sequential(*encoder_list[4:11])  # relu1_1 -> relu2_1
        self.encoder_stage_3 = nn.Sequential(*encoder_list[11:18])  # relu2_1 -> relu3_1
        self.encoder_stage_4 = nn.Sequential(*encoder_list[18:31])  # relu3_1 -> relu4_1

    def encode(self, X):
        relu1_1 = self.encoder_stage_1(X)
        relu2_1 = self.encoder_stage_2(relu1_1)
        relu3_1 = self.encoder_stage_3(relu2_1)
        relu4_1 = self.encoder_stage_4(relu3_1)

        return relu4_1

    def forward(self, x):
        x = self.encode(x)

        x = self.inception1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.inception2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.inception3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.avgpool(x)

        h = x.view(x.shape[0], -1)
        x = self.classifier(h)
        return x, h
