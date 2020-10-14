import torch
from torch import nn
from PIL import Image
from torchvision.transforms import ToTensor

class ConvNet1(nn.Module):
    def __init__(self):
        super(ConvNet1, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2)))

    def forward(self, x):
    	y = self.layer1(x)
    	y = self.layer2(y)
    	return y


class ConvNet2(nn.Module):
    def __init__(self):
        super(ConvNet2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2)))

    def forward(self, x):
    	y = self.layer1(x)
    	y = self.layer2(y)
    	return y

class ConvNet3(nn.Module):
    def __init__(self):
        super(ConvNet3, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2)))

    def forward(self, x):
    	y = self.layer1(x)
    	y = self.layer2(y)
    	return y

class ConvNet4(nn.Module):
    def __init__(self):
        super(ConvNet4, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU())

    def forward(self, x):
    	y = self.layer1(x)
    	y = self.layer2(y)
    	return y


# nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1)

class UpConv(nn.Module):
    def __init__(self):
        super(UpConv, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1024, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), nn.ConvTranspose2d(128, 64, 2, stride=2, padding=0))

    def forward(self, x):
    	y = self.layer1(x)
    	y = self.layer2(y)
    	return y

class UpConv2(nn.Module):
    def __init__(self):
        super(UpConv2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), nn.ConvTranspose2d(64, 32, 2, stride=2, padding=0))

    def forward(self, x):
    	y = self.layer1(x)
    	y = self.layer2(y)
    	return y

class UpConv3(nn.Module):
    def __init__(self, output_size):
        super(UpConv3, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(96, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), nn.ConvTranspose2d(32, output_size, 2, stride=2, padding=0), nn.Softmax(dim=1))

    def forward(self, x):
    	y = self.layer1(x)
    	y = self.layer2(y)
    	return y

class UNET(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
        self.conv1 = ConvNet1()
        self.conv2 = ConvNet2()
        self.conv3 = ConvNet3()
        self.conv4 = ConvNet4()
        self.upconv1 = UpConv()
        self.upconv2 = UpConv2()
        self.upconv3 = UpConv3(self.output_size)
    def forward(self, x):
    	y = self.conv1(x)
    	out1 = y
    	y = self.conv2(y)
    	out2 = y
    	#print('out1: ', out1.shape)
    	y = self.conv3(y)
    	#out3 = y
    	#print('out3: ', out3.shape)
    	y = self.conv4(y)
    	y = self.upconv1(y)
    	
    	y = torch.cat((y, out2), dim=1)
    	y = self.upconv2(y)
    	
    	y = torch.cat((y, out1), dim=1)
    	y = self.upconv3(y)
    	#print('here: ', y.shape)
    	#exit(0)
    	return y
