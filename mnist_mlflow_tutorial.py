#!/usr/bin/env python
# coding: utf-8

# In this tutorial we will learn how to use mlflow for experiment tracking in deep learning experiments

# ### Libraries import
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import mlflow

### Hyperparameters definition
BATCH_SIZE = 64
LEARNING_RATE = 0.08
NUM_EPOCHS = 10
DROPOUT_RATE = 0.5
EXPERIMENT_NAME = "MNIST_Mlflow_tutorial"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    # # Dataset Loading
    train_loader, val_loader, test_loader = get_mnist_datasets()

    # CNN class instantiation
    model = Net(dropout_rate=DROPOUT_RATE)
    model.to(device)

    # Set Loss function with criterion
    criterion = nn.CrossEntropyLoss()

    # ### Optimizer definition
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=0.005, momentum=0.9)

    ###############################################################
    # Mlflow experiment configuration
    ###############################################################
    experiment_name = EXPERIMENT_NAME
    mlflow.set_tracking_uri("http://10.8.33.50:5050")
    existing_exp = mlflow.get_experiment_by_name(experiment_name)
    if not existing_exp:
        mlflow.create_experiment(
            name=experiment_name,
        )
    experiment = mlflow.set_experiment(experiment_name=experiment_name)
    # mlflow_run_name = f"lr_{LEARNING_RATE}_dropout_{int(DROPOUT_RATE*10)}"
    with mlflow.start_run(
            experiment_id=experiment.experiment_id,
            # run_name=mlflow_run_name
    ) as run:

        # Log (hyper)-parameters to mlflow
        mlflow.log_param("batch_size", BATCH_SIZE)
        # TODO: Log other parameters
        ###############################################################
        # Training
        ###############################################################
        for epoch in range(NUM_EPOCHS):
            training_loss, training_accuracy, val_loss, val_accuracy = train_one_epoch(
                model, train_loader, val_loader, optimizer, criterion
            )
            print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], '
                  f'Training Loss: {training_loss:.4f}, '
                  f'Training Accuracy: {training_accuracy:.2f}%, '
                  f'Validation Loss: {val_loss:.4f}, '
                  f'Validation Accuracy: {val_accuracy:.2f}%')

            # Log metrics with mlflow
            mlflow.log_metric("train_loss", training_loss, step=epoch)
            # TODO: Log other metrics of interest

        print('Training finished!')

        # ### Optionally save locally the trained model
        # path = './mnist.pth'
        # torch.save(model.state_dict(), path)
        # TODO: Log model checkpoint with mlflow
        # inputs_sample, outputs_sample = get_model_signature(train_loader, model)
        # signature = mlflow.models.infer_signature(inputs_sample, outputs_sample)
        # mlflow.pytorch.log_model(model, "model", signature=signature)
        # ### Test the model
        test_loss, test_accuracy = test_model(model, test_loader, criterion)

        print(f'Final Test Loss: {test_loss:.4f}, '
              f'Final Test Accuracy: {test_accuracy:.2f}%')

        # Optionally log final metrics
        mlflow.log_metric("final_test_loss", test_loss, step=0)
        # TODO: Optionally log final test accuracy metric

        mlflow.end_run()


def get_mnist_datasets():
    # Define transforms
    all_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ]
    )
    # Create the full Training dataset
    full_train_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        transform=all_transforms,
        download=True
    )

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
    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        transform=all_transforms,
        download=True
    )
    # Create test loader
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=False)
    return train_loader, val_loader, test_loader


class Net(nn.Module):
    def __init__(self, dropout_rate):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=dropout_rate)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def train_one_epoch(model, train_loader, val_loader, optimizer, loss_fn):
    # Set the model to training mode
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

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
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            val_loss += loss.item()

            # For accuracy calculation
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total

    return avg_training_loss, training_accuracy, avg_val_loss, val_accuracy


def test_model(model, test_loader, loss_fn):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            test_loss += loss.item()

            # For accuracy calculation
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = 100 * correct / total
    return avg_test_loss, test_accuracy


def get_model_signature(train_loader, model):
    model.eval()
    with torch.no_grad():
        for data in train_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            outputs = model(inputs)
            break
    return inputs.cpu().numpy(), outputs.cpu().numpy()


if __name__ == '__main__':
    main()
