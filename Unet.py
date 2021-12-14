import torch
import torch.nn as nn
import torch.nn.functional as F
class DoubleConv(nn.Module):
    def __init__(self,in_channel,out_channel):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channel,0.1),
            nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel,0.1),
            nn.ReLU(True)
        )
    def forward(self,x):
        return self.model(x)

class DOWN(nn.Module):
    def __init__(self,in_channel,out_channel):
        super().__init__()
        self.model = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channel, out_channel)
        )
    def forward(self,x):
        return self.model(x)

class Up(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels

        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                nn.Conv2d(in_channels,out_channels,1,1)
                              )


    def forward(self, x):
        return self.up(x)

class UNet(nn.Module):
    def __init__(self):
        super(UNet,self).__init__()
        self.inc = DoubleConv(1,64)
        self.down1 = DOWN(64,128)
        self.down2 = DOWN(128,256)
        self.down3 = DOWN(256,512)
        self.down4 = DOWN(512,1024)
        self.up1 = Up(1024,512)
        self.conv1 = DoubleConv(1024,512)
        self.up2 = Up(512,256)
        self.conv2 = DoubleConv(512,256)
        self.up3 = Up(256,128)
        self.conv3 = DoubleConv(256, 128)
        self.up4 = Up(128, 64)
        self.conv4 = DoubleConv(128, 64)
        self.seg = nn.Conv2d(64,1,3,1,1)
    def forward(self,x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5)
     
        x = torch.cat([x,x4],dim=1) 
        x = self.conv1(x) 
        x = self.up2(x)  
        x = torch.cat([x,x3],dim=1)
        x = self.conv2(x) 
        x = self.up3(x) 
        x = torch.cat([x,x2],dim=1)
        x = self.conv3(x) 
        x = self.up4(x) 
        x = torch.cat([x, x1], dim=1) 
        x = self.conv4(x)
        x = self.seg(x)
        x = F.sigmoid(x)

        return x

