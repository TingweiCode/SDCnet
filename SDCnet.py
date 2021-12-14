import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels, k=3, s=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(channels, channels, k, stride=s, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, k, stride=s, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        return self.bn2(self.conv2(y)) + x

class down(nn.Module):
    def __init__(self,in_channel,out_channel):
        super().__init__()
        self.model = nn.Conv2d(in_channel,out_channel,4,2,1)
    def forward(self,x):
        return self.model(x)

class up(nn.Module):
    def __init__(self,in_channel,out_channel):
        super().__init__()
        self.model = nn.ConvTranspose2d(in_channel,out_channel,4,2,1)
    def forward(self, x):
        return self.model(x)

class Resunet_base(nn.Module):
    def __init__(self):
        super().__init__()
        self.inc = nn.Sequential(nn.Conv2d(1,64,3,1,1),nn.BatchNorm2d(64),nn.ReLU())
        self.down1 = down(64,128)
        self.c1 = nn.Sequential(nn.Conv2d(128,128,1,1,0),nn.ReLU(),ResidualBlock(128), nn.Dropout(0.4))
        self.down2 = down(128,256)
        self.c2 = nn.Sequential(nn.Conv2d(256, 256, 1, 1, 0), nn.ReLU(), ResidualBlock(256), nn.Dropout(0.4))
        self.down3 = down(256,512)
        self.c3 = nn.Sequential(nn.Conv2d(512, 512, 1, 1, 0), nn.ReLU(), ResidualBlock(512), nn.Dropout(0.4))
        self.down4 = down(512,1024)
        self.c4 = nn.Sequential(nn.Conv2d(1024, 1024, 1, 1, 0), nn.ReLU(), ResidualBlock(1024), nn.Dropout(0.4))
        ## symmetry
        self. up5 = nn.Sequential(up(1024,512),nn.BatchNorm2d(512))
        self.c5 = nn.Sequential(nn.Conv2d(1024, 512, 1, 1, 0), nn.ReLU(), ResidualBlock(512), nn.Dropout(0.4))
        self.up6 = nn.Sequential(up(512,256), nn.BatchNorm2d(256))
        self.c6 = nn.Sequential(nn.Conv2d(512, 256, 1, 1, 0), nn.ReLU(), ResidualBlock(256), nn.Dropout(0.4))
        self.up7 = nn.Sequential(up(256,128), nn.BatchNorm2d(128))
        self.c7 = nn.Sequential(nn.Conv2d(256, 128, 1, 1, 0), nn.ReLU(), ResidualBlock(128), nn.Dropout(0.4))
        self.up8 = nn.Sequential(up(128,64), nn.BatchNorm2d(64))
        self.c8 = nn.Sequential(nn.Conv2d(128, 64, 1, 1, 0), nn.ReLU(), ResidualBlock(64), nn.Dropout(0.4))

        self.seg = nn.Sequential(nn.Conv2d(64, 1, 3, 1, 1), nn.Sigmoid())
    
    def forward(self,x):
        x1 = self.inc(x)#[1, 64, 256, 256]

        x2 = self.down1(x1)#[1, 64, 128, 128]
        x2 = self.c1(x2) #[1, 128, 128, 128]

        x3 = self.down2(x2)
        x3 = self.c2(x3) # [1, 256, 64, 64]

        x4 = self.down3(x3)
        x4 = self.c3(x4) #[1, 512, 32, 32]

        x5 = self.down4(x4)
        x5 = self.c4(x5)# [1, 1024, 16, 16]

        x = self.up5(x5)
        x = torch.cat([x,x4],dim=1)
        x = self.c5(x)

        x = self.up6(x)
        x = torch.cat([x, x3], dim=1)
        x = self.c6(x)

        x = self.up7(x)
        x = torch.cat([x,x2],dim=1)
        x = self.c7(x)

        x = self.up8(x)
        x = torch.cat([x,x1],dim=1)
        x = self.c8(x)

        x = self.seg(x)

        return x

