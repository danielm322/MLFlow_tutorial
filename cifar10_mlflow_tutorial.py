#!/usr/bin/env python
# coding: utf-8

# # Building a custom CNNs model in PyTorch
#
# In this Jupyter notebook, I will be building a custom CNNs in PyTorch.
#
# **Dataset:** CIFAR10

# ### Libraries import

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import mlflow

# ### Hyperparameters definition

BATCH_SIZE = 64
LEARNING_RATE = 0.005
NUM_EPOCHS = 20
DROPOUT_RATE = 0.1
EXPERIMENT_NAME = "CIFAR10_Mlflow_tutorial"

# # Dataset Loading

# Define transforms
all_transforms = transforms.Compose([transforms.Resize((32, 32)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                          std=[0.2023, 0.1994, 0.2010])
                                     ])
# Create the full Training dataset
full_train_dataset = torchvision.datasets.CIFAR10(root='./data',
                                                  train=True,
                                                  transform=all_transforms,
                                                  download=True)

# Define the sizes for training and validation datasets
train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size

# Split the dataset
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

# Create train loader
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True)
# Create val loader
val_loader = DataLoader(dataset=val_dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=False)

# Create Testing dataset
test_dataset = torchvision.datasets.CIFAR10(root='./data',
                                            train=False,
                                            transform=all_transforms,
                                            download=True)

# Create test loader
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False)


# ### Define PyTorch CNN model, with dropout at input


# CNN class creation
class Custom_CNN(nn.Module):
    # Layers definition and order
    def __init__(self):
        super().__init__()
        self.conv_layer_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_layer_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_layer_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_layer_4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.max_pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc_layer_1 = nn.Linear(2 * 2 * 256, 512)

        self.dropout = nn.Dropout(DROPOUT_RATE)

        self.fc_layer_2 = nn.Linear(512, 10)

        self.softmax_af = nn.Softmax()

    # Data progress across layers
    def forward(self, output_layer):
        output_layer = self.conv_layer_1(output_layer)
        output_layer = F.relu(output_layer)
        output_layer = self.max_pool_1(output_layer)

        output_layer = self.conv_layer_2(output_layer)
        output_layer = F.relu(output_layer)
        output_layer = self.max_pool_2(output_layer)

        output_layer = self.conv_layer_3(output_layer)
        output_layer = F.relu(output_layer)
        output_layer = self.max_pool_3(output_layer)

        output_layer = self.conv_layer_4(output_layer)
        output_layer = F.relu(output_layer)
        output_layer = self.max_pool_4(output_layer)

        output_layer = torch.flatten(output_layer, 1)

        output_layer = self.fc_layer_1(output_layer)

        output_layer = self.dropout(output_layer)

        output_layer = self.fc_layer_2(output_layer)

        output_layer = self.softmax_af(output_layer)

        return output_layer


# ### Loss function definition

model = Custom_CNN()

# Set Loss function with criterion
criterion = nn.CrossEntropyLoss()

# ### Optimizer definition

# Stochastic Gradient Descent (SGD) optimizer
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=0.005, momentum=0.9)

# ### Total number of iterations

# Calculate the total number of steps per epoch
total_step = len(train_loader)

# ### Device selection (CPU/GPU) for training

# Device will determine whether to run the training on GPU or CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ### Model training with evaluation

# Model training with evaluation
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data.shape)

###############################################################
# Training
###############################################################

# Mlflow parameters
experiment_name = EXPERIMENT_NAME
mlflow.set_tracking_uri("http://10.8.33.50:5050")
existing_exp = mlflow.get_experiment_by_name(experiment_name)
if not existing_exp:
    mlflow.create_experiment(
        name=experiment_name,
    )
experiment = mlflow.set_experiment(experiment_name=experiment_name)
mlflow_run_name = f"lr_{LEARNING_RATE}_dropout_{int(DROPOUT_RATE*10)}"
with mlflow.start_run(
        experiment_id=experiment.experiment_id,
        run_name=mlflow_run_name
) as run:

    # Log parameters to mlflow
    mlflow.log_param("learning_rate", LEARNING_RATE)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("num_epochs", NUM_EPOCHS)
    mlflow.log_param("dropout_rate", DROPOUT_RATE)

    for epoch in range(NUM_EPOCHS):

        # Set the model to training mode
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, data in enumerate(train_loader):
            inputs, labels = data
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # For accuracy calculation
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Calculate the average training loss for this epoch
        avg_training_loss = running_loss / len(train_loader)
        training_accuracy = 100 * correct / total

        # Evaluate the model on the validation set
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # For accuracy calculation
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total

        print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], '
              f'Training Loss: {avg_training_loss:.4f}, '
              f'Training Accuracy: {training_accuracy:.2f}%, '
              f'Validation Loss: {avg_val_loss:.4f}, '
              f'Validation Accuracy: {val_accuracy:.2f}%')

        mlflow.log_metric("train_loss", avg_training_loss, step=epoch)
        mlflow.log_metric("train_accuracy", training_accuracy, step=epoch)
        mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
        mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)

    print('Training finished!')

    # Plot the Accuracy and Loss Values

    # ### Save the trained model


    path = './cifar.pth'
    torch.save(model.state_dict(), path)

    # ### Test the model


    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # For accuracy calculation
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = 100 * correct / total

    print(f'Final Test Loss: {avg_test_loss:.4f}, '
          f'Final Test Accuracy: {test_accuracy:.2f}%')

    mlflow.log_metric("final_test_loss", avg_test_loss, step=0)
    mlflow.log_metric("final_test_accuracy", test_accuracy, step=0)

    mlflow.end_run()