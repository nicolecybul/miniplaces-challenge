import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from DataLoader import *

# Dataset Parameters
load_size = 256
fine_size = 224
c = 3
data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])

# load data

opt_data_train = {
    #'data_h5': 'miniplaces_256_train.h5',
    'data_root': '../../dataset/images/',   # MODIFY PATH ACCORDINGLY
    'data_list': '../../data/train.txt', # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': True
    }
opt_data_val = {
    #'data_h5': 'miniplaces_256_val.h5',
    'data_root': '../../dataset/images/',   # MODIFY PATH ACCORDINGLY
    'data_list': '../../data/val.txt',   # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': False
    }

loader_train = DataLoaderDisk(**opt_data_train)
loader_val = DataLoaderDisk(**opt_data_val)

# define model

class Net(nn.Module):

    def __init__(self):
        super().__init__()
        # LeNet CNN
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=20,kernel_size=(5, 5))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.fc1 = nn.Linear(in_features=140450, out_features=500)
        self.relu3 = nn.ReLU()

        self.fc2 = nn.Linear(in_features=500, out_features=100)

    # Defining the forward pass    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x


net = Net()
net = net.double()
print(net)

# define loss and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# train model

for epoch in range(2):

    count = 0
    batch_size = 100
    for i in range(int(100000/batch_size)):
        data = loader_train.next_batch(batch_size)
        # get the inputs; data is inputs, labels
        inputs, labels = data
        inputs = torch.tensor(inputs)
        labels = torch.tensor(labels)
        labels = labels.type(torch.LongTensor)

        # zero the parameter gradients
        optimizer.zero_grad()

        batch_loss = 0
        # forward + backward + optimize
        for one_input, one_label in zip(inputs, labels):
            output = net(one_input.reshape((1,3,224,224)))
            loss = criterion(output, one_label.reshape((1)))
            batch_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        print("done with batch ", count, " avg. loss = ", batch_loss/batch_size)
        count += 1
        


print('Finished Training')


# save trained model

PATH = './miniplaces_net.pth'
torch.save(net.state_dict(), PATH)

# validate model with validation set (TODO)

# generate and predictions for test set (TODO)