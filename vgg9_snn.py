import torch
import torch.nn as nn

class VGG9_SNN(nn.Module):
    def __init__(self):
        super(VGG9_SNN, self).__init__()

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        # Define the fully connected layers
        self.fc1 = nn.Linear(256 * 8 * 8, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)  # Assuming 10 classes for classification

        # Define activation functions
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.sigmoid(self.conv1(x))
        x = nn.MaxPool2d(2)(x)
        x = self.sigmoid(self.conv2(x))
        x = nn.MaxPool2d(2)(x)
        x = self.sigmoid(self.conv3(x))
        x = nn.MaxPool2d(2)(x)
        x = self.sigmoid(self.conv4(x))

        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.fc3(x)

        return x

# Create the model instance
model = VGG9_SNN()