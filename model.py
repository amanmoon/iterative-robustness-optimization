import torch.nn as nn
import torch.nn.functional as F

class ClassifierCNN_128p(nn.Module):
    def __init__(self, num_classes=3):
        super(ClassifierCNN_128p, self).__init__()
        
        self.pool = nn.MaxPool2d(2, 2) 

        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(32 * 16 * 16, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = x.view(-1, 32 * 16 * 16)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
