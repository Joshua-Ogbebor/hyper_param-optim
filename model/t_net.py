import torch.nn as nn
import torch
import torchvision
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter


class t_net(nn.Module):
    '''
    param: channels= 3 for RGB, 1 for grayscale
        '''
    def __init__(self, channels=3):                                                           
        super(t_net, self).__init__()                                            

        self.C1 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=24, kernel_size=20, stride=2),                           
            nn.BatchNorm2d(num_features=24))
        self.P1 = nn.MaxPool2d(kernel_size=7, stride=2)
        self.C2 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=48, kernel_size=15, stride=2),
            nn.BatchNorm2d(num_features=48))
        self.P2 = nn.MaxPool2d(kernel_size=4, stride=2)
        self.C3 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=96, kernel_size=10, stride=2),
            nn.BatchNorm2d(num_features=96))                                 # Check if BatchNorm1d              

        self.DropOut = nn.Dropout()                                          # This is just 1d lol
        self.ReLu = nn.ReLU()                                                          #
        self.C4 = nn.Conv2d(in_channels=96, out_channels=2, kernel_size=1, stride=1)                   # Check if Conv1d
        #self.Linear = nn.Linear(in_features=4, out_features=2)

    def forward(self,N):
        N = self.C1(N)
        N = self.P1(N)
        N = self.C2(N)
        N = self.P2(N)
        N = self.C3(N)
        N = self.DropOut(N)
        N = self.ReLu(N)
        Output = self.C4(N)
        #Output = self.Linear(Output)
        return Output

def net():
    t_net_=t_net()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    # Convert model to be used on GPU do this before optimizer
    t_net_.to(device)

    print(summary(t_net_, (3, 256, 256)))
    return t_net_
