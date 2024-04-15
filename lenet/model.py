import torch
import torch.nn as nn
import torchvision.transforms.functional as TF




class LeNet(nn.Module):
    def __init__(self,in_channel,num_classes):
        super(LeNet, self).__init__()
        self.relu=nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=in_channel,out_channels=6,kernel_size=(5,5),stride=(1,1),padding=(0,0))
        self.conv2 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=(5,5),stride=(1,1),padding=(0,0))
        self.conv3= nn.Conv2d(in_channels=16,out_channels=120,kernel_size=(5,5),stride=(1,1),padding=(0,0))
        self.pool = nn.AvgPool2d(kernel_size=(2,2),stride=(2,2))
        self.linear1= nn.Linear(120,84)
        self.linear2= nn.Linear(84,num_classes)
    def forward(self,x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x)) # num_examples x 120 x 1 x 1 --> num_examples x 120
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
# model = LeNet(1,10)
# x= torch.randn(64,1,28,28)
# print(model(x).shape)


"""print(model)
LeNet(
  (relu): ReLU()
  (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (conv3): Conv2d(16, 120, kernel_size=(5, 5), stride=(1, 1))
  (pool): AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
  (linear1): Linear(in_features=120, out_features=84, bias=True)
  (linear2): Linear(in_features=84, out_features=10, bias=True)
)
"""


    