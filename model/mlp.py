import torch.nn as nn

# model architecture
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_prob=0.2):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob

        self.fcn1 = nn.Linear(input_dim, 512)
        self.fcn2 = nn.Linear(512, 256)
        self.fcn3 = nn.Linear(256, output_dim)

        # self.fcn1 = nn.Linear(input_dim, 512)
        # self.fcn2 = nn.Linear(512, 128)
        # self.fcn3 = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=self.dropout_prob)

    def forward(self, x):
        # x = self.dropout(self.relu(self.fcn1(x)))
        # x = self.dropout(self.relu(self.fcn2(x)))
        # x = self.sigmoid(self.fcn3(x))
        x = self.relu(self.fcn1(x))
        x = self.relu(self.fcn2(x))
        x = self.sigmoid(self.fcn3(x))
        return x


# import torch.nn as nn

# # model architecture
# class MLP(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(MLP, self).__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim

#         self.fcn1 = nn.Linear(input_dim, 512)
#         self.fcn2 = nn.Linear(512, 256)
#         self.fcn3 = nn.Linear(256, output_dim)
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x = self.relu(self.fcn1(x))
#         x = self.relu(self.fcn2(x))
#         x = self.sigmoid(self.fcn3(x))
#         return x

        # def model(self):
        #     model = Sequential()
        #     model.add(Dense(self.input_dim, activation="relu"))
        #     model.add(Dense(512, activation="relu"))
        #     model.add(Dense(256, activation="relu"))
        #     model.add(Dense(self.output_dim, activation="sigmoid", name="multi_o/p"))

        # return(model)

        # return x
