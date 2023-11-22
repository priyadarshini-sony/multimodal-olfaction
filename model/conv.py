import torch
import torch.nn as nn
import torch.optim as optim

class Conv1D(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Conv1D, self).__init__()
        self.net = []

        #each conv reduce dim by 6, 1388 - 1200 = 188
        for _ in range(200):
            self.net.extend([nn.Conv1d(1, 1, kernel_size=7),
                             nn.ReLU()])

        #pop last relu
        self.net.pop()

        # self.flatten = nn.Flatten()
        fc = nn.Linear(188, 91)
        self.net.append(fc)
        self.net.append(nn.Sigmoid())
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add the channel dimension
        x = self.net(x)
        x = torch.squeeze(x, dim=1)
        # print('x.size', x.size())
        # import pdb; pdb.set_trace()
        return x



# class Conv1D(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(Conv1D, self).__init__()
#         self.conv1 = nn.Conv1d(1, 32, kernel_size=3)
#         self.relu1 = nn.ReLU()
#         self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
#         self.relu2 = nn.ReLU()
#         self.pool = nn.MaxPool1d(kernel_size=2)
#         self.flatten = nn.Flatten()
#         self.fc = nn.Linear(1920, output_dim)
#     def forward(self, x):
#         print('x.size', x.size())
#         x = x.unsqueeze(1)
#         print('x.size after', x.size())
#         x = self.conv1(x)
#         x = self.relu1(x)
#         x = self.pool(x)
#         x = self.conv2(x)
#         x = self.relu2(x)
#         x = self.pool(x)
#         x = self.flatten(x)
#         x = self.fc(x)
#         return x

# class Conv1D(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(Conv1D, self).__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim

#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv1d(in_channels=32, out_channels=output_dim, kernel_size=3, padding=1)
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         print('x.size', x.size())
#         x = x.unsqueeze(1)
#         x = self.conv1(x)
#         import pdb; pdb.set_trace()
#         x = self.relu(x)
#         x = self.conv2(x)
#         x = self.relu(x)
#         x = self.conv3(x)
#         x = self.sigmoid(x)
#         return x