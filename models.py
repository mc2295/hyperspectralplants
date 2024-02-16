import torch
from efficientnet_pytorch import EfficientNet

class Inception2D(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inception_model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained = True)
        self.conv2D = torch.nn.Conv2d(10, 32, kernel_size=3, stride=1, padding=1)
        self.linear = torch.nn.Linear(1000, 6)
    def forward(self, x):
        x = self.conv2D(x)
        x = self.inception_model(x)
        out = self.linear(x.logits)
        return out

class Inception3D(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inception_model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained = True)
        self.conv3d = torch.nn.Conv3d(10,32, kernel_size = (3,3,3),padding = (1,1,1))
        self.linear = torch.nn.Linear(1000, 6)
         
    def forward(self, x):
        x = self.conv3d(x.permute(1,0,2,3))
        x = self.inception_model(x.permute(1,0,2,3))
        out = self.linear(x.logits)
        return out 
    

class EfficientNet2D(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.inception_model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained = True)
        self.efficient_model = EfficientNet.from_pretrained('efficientnet-b0')
        self.conv2d = torch.nn.Conv2d(10, 3, kernel_size=3, stride=1, padding=1)
        # self.conv3d = torch.nn.Conv3d(10,3, kernel_size = (3,3,3),padding = (1,1,1))
        self.linear = torch.nn.Linear(1000, 6)
         
    def forward(self, x):
        x = self.conv2d(x)
        x = self.efficient_model(x)
        out = self.linear(x)
        return out  

class EfficientNet3D(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.inception_model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained = True)
        self.efficient_model = EfficientNet.from_pretrained('efficientnet-b0')
        # self.conv1 = torch.nn.Conv2d(10, 32, kernel_size=3, stride=1, padding=1)
        self.conv3d = torch.nn.Conv3d(10,3, kernel_size = (3,3,3),padding = (1,1,1))
        self.linear = torch.nn.Linear(1000, 6)
         
    def forward(self, x):
        x = self.conv3d(x.permute(1,0,2,3))
        x = self.efficient_model(x.permute(1,0,2,3))
        out = self.linear(x)
        return out  
    
